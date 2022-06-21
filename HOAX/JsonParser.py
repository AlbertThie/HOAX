import json


class JsonParser():
    def __init__(self,configName):
        with open(configName) as f:
            config = json.loads(f.read())
        self.config = config['config']
        self.neuralnetwork = self.config['neural_network']
        self.optimizer = self.setOptimizer()


    def setOptimizer(self):
        if 'grid_search' in self.config:
            return 'grid_search'
        elif 'random_search' in self.config:
            return "random_search"
        elif 'simulated_annealing' in self.config:
            return "simulated_annealing"
        elif 'genetic_algorithm' in self.config:
            return "genetic_algorithm"
        else:
            return "no_optimizer"

    def getOptimizer(self):
        return self.optimizer

    def getDatabase(self):
        return self.config['database']

    def getLoggingFile(self):
        return self.neuralnetwork['logging_file']

    def getEpochs(self):
        return self.neuralnetwork['epochs']

    def getEpochStep(self):
        return self.neuralnetworkjson['epoch_step']

    def getConfig(self):
        return self.config