
import os
import csv
import gym
import numpy as np
import logging as log
import tensorflow as tf

from argparse import ArgumentParser
from datetime import datetime
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.callbacks import TensorBoard

from sklearn.metrics.pairwise import cosine_similarity
from ntext.envs.datasets.imdb import ImdbDataset
from base_agent import BaseAgent


class NtextAgent(BaseAgent):
    def __init__(self, args):
        super().__init__(args)
        log.info('initializing ntext agent...')
        self.args = args
        self.env = gym.make('ntext:NText-v0')
        self.env.max_episode_steps = args.num_episodes
        self.num_epochs = args.num_epochs
        self.num_games = 5
        dateTimeObj = datetime.now()
        self.run_id = dateTimeObj.strftime("%Y%m%d%H%M%S")
        # self.cloud_handler = CloudHandler(args)
        if not self.args.job_dir.startswith('gs://'):
            os.makedirs(self.args.job_dir, exist_ok=True)
        log.info('init NChain Agent...[ok]')

    def fasttext_classifier(self):
        # Set parameters:
        # ngram_range = 2 will add bi-grams features
        ngram_range = 1
        max_features = 20000
        maxlen = 400
        batch_size = 32
        embedding_dims = 50
        epochs = 5

        self.dataset = ImdbDataset()
        self.dataset.load()
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

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test))

    def naive_sum_reward(self):
        # this is the table that will hold our summated rewards for
        # each action in each state
        r_table = np.zeros((10,2))
        s_table = np.random.rand(10,400)
        r_avg_list = []

        for g in range(self.num_epochs):
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

    def deep_q_learning(self):
        # create the keras model
        logdir = os.path.join(self.args.job_dir, "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        #file_writer = tf.summary(logdir + "/metrics")
        #file_writer.set_as_default()

        #tensorboard_callback = TensorBoard(log_dir=logdir)

        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, 400)))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        # now execute the q learning
        y = 0.95
        eps = 0.5
        decay_factor = 0.999
        r_avg_list = []

        for i in range(self.num_epochs):
            s = self.env.reset()
            eps *= decay_factor
            done = False
            r_sum = 0

            while not done:
                ntext = self.env.render()

                if np.random.random() < eps:
                    a = np.random.randint(0, 2)
                else:
                    ta = model.predict(ntext)
                    a = np.argmax(ta)
                new_s, r, done, _ = self.env.step(a)
                new_ntext = self.env.render()
                target = r + y * np.max(model.predict(new_ntext))
                target_vec = model.predict(ntext)[0]
                target_vec[a] = target

                model.fit(
                    ntext,
                    target_vec.reshape(-1, 2),
                    epochs=1,
                    verbose=0,
                    #callbacks=[tensorboard_callback]
                )

                s = new_s
                r_sum += r

            mean_reward = r_sum / self.env.max_episode_steps
            r_avg_list.append([ datetime.now().timestamp(), mean_reward])
            log.info('epoch: {}/{}, mean reward: {}'.format(i, self.num_epochs, mean_reward))

        self.write_results('ntext_deep_q_learning', r_avg_list)

    def run(self):
        # self.fasttext_classifier()
        # self.naive_sum_reward()
        self.deep_q_learning()



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
        '--num-episodes',
        default=50,
        type=int
    )
    parser.add_argument(
        '--num-epochs',
        default=5,
        type=int
    )
    # parser.add_argument(
    #     '--downsample',
    #     default=0.1,
    #     type=float
    # )
    # parser.add_argument(
    #     '--optimize',
    #     action='store_true'
    # )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    log.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=log.DEBUG)
    agent  = NtextAgent(parse_args())
    agent.run()


