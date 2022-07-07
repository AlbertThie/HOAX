#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork
import math
import random

# In[ ]:



class SimulatedAnnealing(Optimizer):
    
    
    def __init__(self,jsonparser,database,logger):
        self.jsonparser =jsonparser
        self.database= database
        self.logger = logger
        self.nodesbounds = self.jsonparser.getConfig()['simulated_annealing']["hiddenlayer_size"]
        self.layerbounds = self.jsonparser.getConfig()['simulated_annealing']["hiddenlayer_number"]
        self.learningbounds = self.jsonparser.getConfig()['simulated_annealing']["learning_rates"]
        self.batchsize = self.jsonparser.getConfig()['simulated_annealing']['batch_size']

        self.temperature = self.jsonparser.getConfig()['simulated_annealing']["temperature"]
        self.nodes = [i for i in range(self.nodesbounds[0],self.nodesbounds[1],self.nodesbounds[2])]
        self.layers = [i for i in range(self.layerbounds[0],self.layerbounds[1],self.layerbounds[2])]
        self.iterations = self.jsonparser.getConfig()['simulated_annealing']['iterations']
        self.searchlog = {}

        nbound = len(self.nodes) -1
        lbound =len(self.layers) -1
        ebound = len(self.learningbounds) -1
        bbound = len(self.batchsize) -1

        self.upperbounds = [nbound,lbound,ebound,bbound]
        self.lowerbounds = [0,0,0,0]
   
      
    def move(self,positions):
        newpositions = self.trymove(positions.copy())
        while newpositions == positions:
            print(newpositions)
            print(positions)
            newpositions = self.trymove(positions.copy())
            
        return newpositions
        
    def trymove(self,positions):
        for i in range(len(positions)):
            positions[i] = self.getpos(positions[i],self.upperbounds[i],self.lowerbounds[i])
            print(positions[i])
        return positions


    def getpos(self,position,upperbound,lowerbound):
        newposition = position + random.randint(-1,1)
        if newposition <= lowerbound:
            return lowerbound
        elif newposition >= upperbound:
            return upperbound
        else:
            return newposition

    
    def getNewNetwork(self,positions):
        network = NeuralNetwork(self.jsonparser, self.database, self.logger,
                                                    hiddenlayer_size= self.nodes[positions[0]], hiddenlayer_number = self.layers[positions[1]], 
                                                    learning_rate=self.learningbounds[positions[2]],batch_size=self.batchsize[positions[3]]) 
        error = network.train()
        return error,network
      
    def run(self):


        positions = [random.randrange(self.upperbounds[0]),random.randrange(self.upperbounds[1]),random.randrange(self.upperbounds[2]),random.randrange(self.upperbounds[3])]
        network = NeuralNetwork(self.jsonparser, self.database, self.logger,
                                                    hiddenlayer_size= self.nodes[positions[0]], hiddenlayer_number = self.layers[positions[1]], 
                                                    learning_rate=self.learningbounds[positions[2]],batch_size=self.batchsize[positions[3]]) 
            
        lowest_error = network.train()
        best_network = network.export()
     
        for n in range(self.iterations) :
            
            trialpositions = self.move(positions).copy()
            
            if str(trialpositions) in self.searchlog:
                error = self.searchlog[str(trialpositions)]
                print(f"old error {error} from library")
            else:
                error,new_network = self.getNewNetwork(trialpositions) 
                self.searchlog[str(trialpositions)] = error
                print(f"new error found {error} ")
            
            print("error is: " + str(error) + "  lowest error  )" + str(lowest_error))
            acceptance_threshold = math.exp(-(error-lowest_error)/ (self.temperature / float(n + 1)))
            print(acceptance_threshold)
            if error < lowest_error :
                new_network.export()
                lowest_error = error
                positions = trialpositions.copy()

            elif random.random() < acceptance_threshold :
                lowest_error = error
                positions = trialpositions.copy()
                        
    
    
   



