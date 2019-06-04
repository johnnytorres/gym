
#import zipfile
#import logging

import gym
import csv
import os
import numpy as np
import logging as log

from datetime import datetime
from argparse import ArgumentParser
#from google.cloud import storage
from tensorflow.python.lib.io import file_io

from keras.models import Sequential
from keras.layers import Dense, InputLayer


# class CloudHandler:
#     def __init__(self, args):
#         self.args = args
#
#     def upload_jobdir(self):
#
#         if self.args.cloud_dir is None:
#             return
#
#         folder_path = self.args.job_dir
#
#         logging.info('compressing output...')
#         ofolder, fname = os.path.split(folder_path)
#         output_path = os.path.join(ofolder, fname+ '.zip')
#         with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#             for root, dirs, files in os.walk(folder_path):
#                 for file in files:
#                     folder = root.replace(folder_path, '').strip('/')
#                     zip_path = os.path.join(folder, file)
#                     #logging.info(f'zipping {zip_path}')
#                     zipf.write(os.path.join(root, file), arcname=zip_path)
#
#         logging.info('compressing output...[OK]')
#         logging.info('uploading to the cloud...')
#         # for authentication first run: https://googleapis.github.io/google-cloud-python/latest/core/auth.html
#         bucket_tokens = self.args.cloud_dir.replace('gs://', '').split('/')
#         bucket_id = bucket_tokens[0]
#         bucket_folder = '/'.join(bucket_tokens[1:])
#         bucket = storage.Client().bucket(bucket_id)
#         fpath = os.path.join(bucket_folder, fname)
#         blob = bucket.blob(fpath)
#         blob.upload_from_filename(output_path)
#         logging.info('uploading to the cloud...[OK]')


class NChainAgent:
    def __init__(self, args):
        log.info('initializing NChain Agent...')
        self.args = args
        self.env = gym.make('NChain-v0')
        # env.env.slip = 0
        # self.env.env.spec.max_episode_steps=100
        self.num_episodes = args.num_episodes
        self.num_games = 5
        dateTimeObj = datetime.now()
        self.run_id = dateTimeObj.strftime("%Y%m%d%H%M%S")
        #self.cloud_handler = CloudHandler(args)
        if not self.args.job_dir.startswith('gs://'):
            os.makedirs(self.args.job_dir, exist_ok=True)
        log.info('init NChain Agent...[ok]')

    def write_results(self, fname, r_avg_list):
        with file_io.FileIO(fname, mode='w+') as f:
            writer = csv.writer(f)
            for r in r_avg_list:
                writer.writerow([self.run_id, r])

    def naive_sum_reward(self):
        # this is the table that will hold our summated rewards for
        # each action in each state
        r_table = np.zeros((5, 2))
        r_avg_list = []

        for g in range(self.num_episodes):
            s = self.env.reset()
            done = False
            r_sum = 0

            while not done:
                if np.sum(r_table[s, :]) == 0:
                    # make a random selection of actions
                    a = np.random.randint(0, 2)
                else:
                    # select the action with highest cummulative reward
                    a = np.argmax(r_table[s, :])
                new_s, r, done, _ = self.env.step(a)
                r_table[s, a] += r
                r_sum += r
                s = new_s

            r_avg_list.append(r_sum / self.env.env.spec.max_episode_steps)

        fname = os.path.join(self.args.job_dir, 'naive_sum_reward_agent.txt')
        self.write_results(fname, r_avg_list)
        # self.cloud_handler.upload_jobdir()
        return r_table



    def q_learning_with_table(self):
        y = 0.95
        lr = 0.8
        q_table = np.zeros((5, 2))
        r_avg_list = []

        for i in range(self.num_episodes):
            s = self.env.reset()
            done = False
            r_sum = 0
            while not done:
                if np.sum(q_table[s,:]) == 0:
                    # make a random selection of actions
                    a = np.random.randint(0, 2)
                else:
                    # select the action with largest q value in state s
                    a = np.argmax(q_table[s, :])
                new_s, r, done, _ = self.env.step(a)
                q_table[s, a] +=  lr*(r +y*np.max(q_table[new_s, :]) - q_table[s, a])
                r_sum += r
                s = new_s

            r_avg_list.append(r_sum / self.env.env.spec.max_episode_steps)

        fname = os.path.join(self.args.job_dir, 'q_learning_agent.txt')
        self.write_results(fname, r_avg_list)
        return q_table

    def eps_greedy_q_learning_with_table(self):
        q_table = np.zeros((5, 2))
        y = 0.95
        eps = 0.5
        lr = 0.8
        decay_factor = 0.999
        r_avg_list = []

        for i in range(self.num_episodes):
            s = self.env.reset()
            eps *= decay_factor
            done = False
            r_sum = 0

            while not done:
                if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                    a = np.random.randint(0, 2)
                else:
                    a = np.argmax(q_table[s, :])
                # pdb.set_trace()
                new_s, r, done, _ = self.env.step(a)
                q_table[s, a] +=  lr * (r +y * np.max(q_table[new_s, :]) - q_table[s, a])
                r_sum += r
                s = new_s

            r_avg_list.append(r_sum / self.env.env.spec.max_episode_steps)

        fname = os.path.join(self.args.job_dir, 'eps_greedy_q_learning.txt')
        self.write_results(fname, r_avg_list)
        return q_table

    def deep_q_learning(self):
        # create the keras model
        model = Sequential()
        model.add(InputLayer(batch_input_shape=(1, 5)))
        model.add(Dense(10, activation='sigmoid'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        # now execute the q learning
        y = 0.95
        eps = 0.5
        decay_factor = 0.999
        r_avg_list = []

        for i in range(self.num_episodes):
            s = self.env.reset()
            eps *= decay_factor
            log.info("Episode {} of {}".format(i + 1, self.num_episodes))
            done = False
            r_sum = 0
            while not done:
                if np.random.random() < eps:
                    a = np.random.randint(0, 2)
                else:
                    ta = model.predict(np.identity(5)[s:s + 1])
                    a = np.argmax(ta)
                new_s, r, done, _ = self.env.step(a)
                target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
                target_vec = model.predict(np.identity(5)[s:s + 1])[0]
                target_vec[a] = target
                model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
                s = new_s
                r_sum += r
            r_avg_list.append(r_sum / self.env.env.spec.max_episode_steps)

        # for i in range(5):
        #     print("State {} - action {}".format(i, model.predict(np.identity(5)[i:i + 1])))
        fname = os.path.join(self.args.job_dir, 'deep_q_learning.txt')
        self.write_results(fname, r_avg_list)
        # return q_table
#
# def run_game(table, env):
#     s = env.reset()
#     tot_reward = 0
#     done = False
#     while not done:
#         a = np.argmax(table[s, :])
#         s, r, done, _ = env.step(a)
#         tot_reward += r
#     return tot_reward
#
    def run(self):
        winner = np.zeros((3,))
        print(self.naive_sum_reward())
        print(self.q_learning_with_table())
        print(self.eps_greedy_q_learning_with_table())
        print(self.deep_q_learning())
        # for g in range(num_iterations):
        #     m0_table = naive_sum_reward_agent(env, 500)
        #     m1_table = q_learning_with_table(env, 500)
        #     m2_table = eps_greedy_q_learning_with_table(env, 500)
        #     m0 = run_game(m0_table, env)
        #     m1 = run_game(m1_table, env)
        #     m2 = run_game(m2_table, env)
        #     w = np.argmax(np.array([m0, m1, m2]))
        #     winner[w] += 1
        #     print("Game {} of {}: {}".format(g + 1, num_iterations,winner))


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
    agent  = NChainAgent(parse_args())
    agent.run()
