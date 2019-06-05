
import os
import gym
import numpy as np
import logging as log

from argparse import ArgumentParser
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Dense, InputLayer
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from ntext.envs.datasets.imdb import ImdbDataset
from agents.base_agent import BaseAgent


REP_MODEL_REGRESSION = 'regression'
REP_MODEL_FASTTEXT = 'fasttext'
RL_MODEL_FASTTEXT = 'clf-fasttext'  # just for comparison
RL_MODEL_NAIVE = 'rl-naive'
RL_MODEL_DEEPQ = 'rl-deepq'


class NtextAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        log.info('initializing ntext agent...')
        self.args = args
        #todo: when using naive method use tfidf
        self.env = gym.make(
            'ntext:NText-v0',
            max_episode_steps = args.max_episode_steps,
            max_sequence_len = args.max_sequence_len
        )
        self.num_episodes = args.num_episodes
        self.num_games = 5
        self.batch_size = 32
        # self.cloud_handler = CloudHandler(args)
        if not self.args.job_dir.startswith('gs://'):
            os.makedirs(self.args.job_dir, exist_ok=True)
        log.info('init NChain Agent...[ok]')

    def fasttext_classifier(self):
        # Set parameters:
        # ngram_range = 2 will add bi-grams features
        ngram_range = 1
        max_features = 20000
        #maxlen = 400

        #logdir = os.path.join(self.args.job_dir, "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        #tensorboard_callback = TensorBoard(log_dir=logdir)

        self.dataset = ImdbDataset()
        self.dataset.load()

        max_features = self.dataset.max_features
        maxlen = self.dataset.maxlen

        embedding_dims = self.args.embeddings_dim
        epochs = self.num_episodes

        x_train = self.dataset.x_train
        x_test = self.dataset.x_test
        y_train = self.dataset.y_train
        y_test = self.dataset.y_test

        print('Build model...')
        model = Sequential()

        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        model.add(Embedding(max_features,
                            embedding_dims,
                            input_length=maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(
            x_train, y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            #callbacks=[tensorboard_callback]
        )

        results=[]
        for metric, values in model.history.history.items():
            r = [[self.run_id, datetime.now().timestamp(), epoch, metric, val] for epoch, val in enumerate(values)]
            results.extend(r)
        self.write_results('clf_fasttext_', results)

        log.info('done')



    def naive_sum_reward(self):
        # this is the table that will hold our summated rewards for
        # each action in each state
        r_table = np.zeros((10,2))
        s_table = np.random.rand(10,400)
        r_avg_list = []

        for g in range(self.num_episodes):
            s = self.env.reset()
            done = False
            r_sum = 0

            while not done:
                ntext = self.env.render()
                #current_s = np.array(ntext)
                sim = cosine_similarity(s_table, ntext)
                s_ix = np.argmax(sim)

                if np.sum(r_table[s_ix,:])==0:
                    # make a random selection of actions
                    a = np.random.randint(0, 2)
                else:
                    # select the action with highest cummulative reward
                    a = np.argmax(r_table[s_ix,:])

                new_s, r, done, _ = self.env.step(a)
                r_table[s_ix, a] += r
                s_table[s_ix,:] = np.mean([ntext,s_table[s_ix,:]],axis=0)
                r_sum += r
                s = new_s

            r_avg_list.append(r_sum / self.env.max_episode_steps)

        # fname = os.path.join(self.args.job_dir, 'naive_sum_reward_agent.txt')
        # self.write_results(fname, r_avg_list)
        # self.cloud_handler.upload_jobdir()
        return r_table

    def get_representation_model(self):

        if self.args.rep_model == REP_MODEL_REGRESSION:
            return self.get_rep_model_regression()

        if self.args.rep_model == REP_MODEL_FASTTEXT:
            return self.get_rep_model_fasttext()

        raise NotImplemented('Representation model not implemented!')

    def get_rep_model_fasttext(self):
        model = Sequential()
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embeddings_size dimensions
        embeddings_size = self.args.embeddings_dim
        embeddings_layer = Embedding(
            input_dim=self.env.dataset.max_features,
            output_dim=embeddings_size,
            input_length=self.env.dataset.maxlen
        )
        model.add(embeddings_layer)
        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(GlobalAveragePooling1D())
        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(Dense(2, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        return model

    def get_rep_model_regression(self):
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, self.env.dataset.maxlen)))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(2, activation='linear'))
        # regression for 2 values (0, 1)
        model.compile(
            loss='mse',
            optimizer='adam',
            metrics=['mae']
        )
        return model

    def deep_q_learning(self):
        # create the keras model
        #logdir = os.path.join(self.args.job_dir, "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        #file_writer = tf.summary(logdir + "/metrics")
        #file_writer.set_as_default()
        #tensorboard_callback = TensorBoard(log_dir=logdir)

        model = self.get_representation_model()

        # now execute the q learning
        y = 0.95
        eps = 0.05
        decay_factor = 0.999
        r_avg_list = []
        history = {}

        for i in range(self.num_episodes):
            s = self.env.reset()
            eps *= decay_factor
            done = False
            r_sum = 0
            history_epoch = {}

            while not done:
                ntext = self.env.render()
                ntext = ntext.reshape(-1,1).T

                if np.random.random() < eps:
                    target_vec = model.predict(ntext)[0]
                    a = np.random.randint(0, 2)
                else:
                    target_vec = model.predict(ntext)[0]
                    a = np.argmax(target_vec)
                new_s, r, done, _ = self.env.step(a)
                new_ntext = self.env.render().reshape(-1,1).T
                target = r + y * np.max(model.predict(new_ntext))
                #todo: not working with target (delayed reward), check!!

                target_vec[a] = 1 if r == 1 else -1 # target
                target_vec = np.argsort(target_vec)

                model.fit(
                    ntext,
                    target_vec.reshape(-1, 2),
                    epochs=1,
                    verbose=0,
                    #callbacks=[tensorboard_callback]
                )

                for metric, values in model.history.history.items():
                    for epoch, val in enumerate(values):
                        if metric in history_epoch:
                            history_epoch[metric].append(val)
                        else:
                            history_epoch[metric] = [val]

                s = new_s
                r_sum += r

            # calculate over validation data
            y = to_categorical(self.env.dataset.y_test)
            metrics=model.evaluate(self.env.dataset.x_test, y, batch_size=self.batch_size, verbose=0)

            for metric, value in zip(model.metrics_names, metrics):
                history_epoch['val_'+metric] = value

            # calculate metrics
            mean_reward = r_sum / self.env.max_episode_steps
            r_avg_list.append( mean_reward)
            for metric, values in history_epoch.items():
                avg_val = np.mean(values)
                if metric in history:
                    history[metric].append(avg_val)
                else:
                    history[metric] = [avg_val]
            log.info('episode: {}/{}, mean reward: {}'.format(i, self.num_episodes, mean_reward))

        history['reward'] = r_avg_list

        results=[]
        for metric, values in history.items():
            r = [[self.run_id, datetime.now().timestamp(), epoch, metric, val] for epoch, val in enumerate(values)]
            results.extend(r)
        self.write_results('rl_deepq_', results)

    def run(self):
        log.info('Parameters')
        log.info(self.args)

        for _ in range(self.args.num_experiments):
            dateTimeObj = datetime.now()
            self.run_id = dateTimeObj.strftime("%Y%m%d%H%M%S")
            if self.args.model == RL_MODEL_FASTTEXT:
                self.fasttext_classifier()
            elif self.args.model == RL_MODEL_NAIVE:
                self.naive_sum_reward()
            elif self.args.model == RL_MODEL_DEEPQ:
                self.deep_q_learning()
            else:
                raise NotImplemented('RL model not implemented')



def parse_args():
    parser = ArgumentParser()
    # parser.add_argument(
    #     '--cloud-dir',
    #     help='cloud location to write checkpoints and export models',
    #     # required=True,
    #     type=lambda x: os.path.expanduser(x)
    # )
    # Saved model arguments
    parser.add_argument(
        '--job-dir',
        help='local location to write checkpoints and export models',
        required=True,
        type=lambda x: os.path.expanduser(x)
    )
    parser.add_argument(
        '--model',
        default=RL_MODEL_DEEPQ,
        type=str,
        choices=[RL_MODEL_NAIVE,RL_MODEL_FASTTEXT, RL_MODEL_DEEPQ]
    )
    parser.add_argument(
        '--rep-model',
        default=REP_MODEL_REGRESSION,
        type=str,
        choices=[REP_MODEL_REGRESSION,REP_MODEL_FASTTEXT]
    )
    parser.add_argument(
        '--max-episode-steps',
        default=50,
        type=int
    )
    parser.add_argument(
        '--num-episodes',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num-experiments',
        default=1,
        type=int
    )
    parser.add_argument(
        '--embeddings-dim',
        default=50,
        type=int
    )
    parser.add_argument(
        '--max-sequence-len',
        default=200,
        type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=log.DEBUG)
    agent  = NtextAgent(parse_args())
    agent.run()


