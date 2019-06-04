
from ntext.envs.datasets.imdb import ImdbDataset

class FactoryDataset:
    @staticmethod
    def get_dataset(name):
        if name == 'imdb':
            return ImdbDataset()