from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork
from bayes_opt import BayesianOptimization, UtilityFunction
import math
import random
import numpy as np

class BayesianOptimizer(Optimizer):
    
    
    def __init__(self,jsonparser,database,logger):
        self.jsonparser =jsonparser
        self.database= database
        self.logger = logger
        self.nodesbounds = self.jsonparser.getConfig()['bayesian_optimization']["hiddenlayer_size"]
        self.layerbounds = self.jsonparser.getConfig()['bayesian_optimization']["hiddenlayer_number"]
        self.learningbounds = self.jsonparser.getConfig()['bayesian_optimization']["learning_rates"]
        self.batchsize = self.jsonparser.getConfig()['bayesian_optimization']['batch_size']

        self.utilityfunction = UtilityFunction( kind = self.jsonparser.getConfig()['bayesian_optimization']["utility_function"],
                                                kappa = self.jsonparser.getConfig()['bayesian_optimization']["kappa"],
                                                xi = self.jsonparser.getConfig()['bayesian_optimization']["xi"])
        self.iterations = self.jsonparser.getConfig()['bayesian_optimization']['iterations']
        
        self.nodes = [i for i in range(self.nodesbounds[0],self.nodesbounds[1],self.nodesbounds[2])]
        self.layers = [i for i in range(self.layerbounds[0],self.layerbounds[1],self.layerbounds[2])]

        self.searchlog = {}

        nbound = len(self.nodes) -1
        lbound =len(self.layers) -1
        ebound = len(self.learningbounds) -1
        bbound = len(self.batchsize) -1
        ##change the order of the parameters to use pythons autosorting dictionary
        self.upperbounds = [bbound,lbound,ebound,nbound]
        self.lowerbounds = [0,0,0,0]
        self.bayesianOptimizer = BayesianOptimization(f = None, 
                                 pbounds = { "Batch" : [self.lowerbounds[0],self.upperbounds[0]],
                            "Layers" : [self.lowerbounds[1],self.upperbounds[1]],
                            "Learning Rate" : [self.lowerbounds[2],self.upperbounds[2]],
                            "Neurons" : [self.lowerbounds[3],self.upperbounds[3]]}, 
                                 verbose = 2, random_state = self.jsonparser.getConfig()['bayesian_optimization']["random_state"])



    
    def getNewNetwork(self,positions):
        network = NeuralNetwork(self.jsonparser, self.database, self.logger,
                                                    hiddenlayer_size= self.nodes[positions[0]], hiddenlayer_number = self.layers[positions[1]], 
                                                    learning_rate=self.learningbounds[positions[2]],batch_size=self.batchsize[positions[3]]) 
        error = network.train()
        return error,network
      
    def run(self):
        lowest_error = 1000
        for n in range(self.iterations) :

            next_point = self.bayesianOptimizer.suggest(self.utilityfunction)
            next_points_int = dict(map(lambda x: (x[0],np.rint(x[1]).astype(np.int32)),next_point.items()))
            next_points_list = list(next_points_int.values())
            next_point_string = str(next_points_int)    
            next_points_list[0],next_points_list[3] = next_points_list[3],next_points_list[0]
            print(f"The next points list is {next_points_list}")

            if next_point_string in self.searchlog:
                error = self.searchlog[next_point_string]
                print(f"old error {error} from library")
            else:
                error,new_network = self.getNewNetwork(next_points_list) 
                self.searchlog[next_point_string] = error
                print(f"new error found {error} ")
            
            print("error is: " + str(error) + "  lowest error  " + str(lowest_error))
            if error < lowest_error:
                new_network.export()
                lowest_error = error

            try:
                # Update the optimizer with the evaluation results. 
                # This should be in try-except to catch any errors!
                print(1-error)
                print(next_point)
                self.bayesianOptimizer.register(params = next_point, target = 1-error)
            except:
                pass
            
            
   



