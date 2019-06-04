
import gym
import logging
from gym import error, spaces, utils
from gym.utils import seeding
from ntext.envs.datasets.base import NtextDataset
from ntext.envs.datasets.factory import FactoryDataset


class NtextEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps=None, n=5, slip=0.2, small=0, large=1):
        logging.info('initializing gym ntext environment...')
        self.dataset = FactoryDataset.get_dataset('imdb')
        self.dataset.load()
        self.x_train = self.dataset.x_train
        #self.x_train = self.dataset.get_tfidf()
        self.max_episode_steps = len(self.x_train) if max_episode_steps is None else max_episode_steps
        self.n = n
        self.slip = slip  # probability of 'slipping' an action
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        logging.info('max episode steps:{}'.format(self.max_episode_steps))
        logging.info('initializing gym ntext environment...[ok]')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        if self.state < self.max_episode_steps:
            return self.x_train[self.state]
        else:
            return self.x_train[self.state-1]

    def step(self, action):
        assert self.action_space.contains(action)
        # if self.np_random.rand() < self.slip:
        #     # agent slipped, reverse action taken
        #     action = not action

        # 'backwards': go back to the beginning, get small reward
        if action == self.dataset.y_train[self.state]:
            reward = self.large
        else:
            reward = self.small

        self.state += 1
        done = True if self.state >= self.max_episode_steps else False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

