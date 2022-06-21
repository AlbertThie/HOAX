import json


class JsonParser():
    def __init__(self,configName):
        with open(configName) as f:
            config = json.loads(f.read())
        self.config = config['config']
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