import sys, os
from configparser import ConfigParser

DIR = os.path.dirname(os.path.abspath(__file__))

class Singleton(type):
    _instances = {} # type: dict
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class VarConfig(metaclass=Singleton):
    def __init__(self, **kwargs):
        print('Loading configs')
        for value in kwargs['dependencies']:
            print('Dependency ->', value)
            sys.path.append(value)        

class Config(metaclass=Singleton):
    def __init__(self):
        self.parser = ConfigParser()
        self.parser.read('setup.cfg')
        print('Loaded setup.cfg')
    
    def dataset_dir(self, dataset):
        dataset_dir = self.parser['datasets_directories'].get(dataset)
        dataset_dir = dataset_dir.replace('$(dir)', DIR)
        return dataset_dir

    def load_dependencies(self):
        for key, value in self.parser['dependencies'].items():
            print('Dependency ->', key, 'added from ->', value)
            sys.path.append(value)
    
    def model_url(self, network):
        model_url = self.parser['model_urls'].get(network, '')
        return model_url
    
    def checkpoint_dir(self, network):
        checkpoint_dir = self.parser['model_directories'].get(network, '')
        checkpoint_dir = checkpoint_dir.replace('$(dir)', DIR)
        return checkpoint_dir