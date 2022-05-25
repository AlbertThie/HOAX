#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork
import math
import random

# In[ ]:


class RandomSearch(Optimizer):
    
    
    def __init__(self,config,train_input,train_output,val_input,val_output):
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.val_input = val_input
        self.val_output = val_output
        self.nodesbounds = self.config['config']['random_search']["hiddenlayer_size"]
        self.layerbounds = self.config['config']['random_search']["hiddenlayer_number"]
        self.learningbounds = self.config['config']['random_search']["learning_rates"]
        self.batchsize = self.config['config']['random_search']['batch_size']


        self.nodes = [i for i in range(self.nodesbounds[0],self.nodesbounds[1],self.nodesbounds[2])]
        self.layers = [i for i in range(self.layerbounds[0],self.layerbounds[1],self.layerbounds[2])]
        self.searchlog = {}

        nbound = len(self.nodes) -1
        lbound =len(self.layers) -1
        ebound = len(self.learningbounds) -1
        bbound = len(self.batchsize) -1

        self.upperbounds = [nbound,lbound,ebound,bbound]
        self.lowerbounds = [0,0,0,0]
     

  
    def run(self):
        lowest_error, best_network = math.inf, None


        for n in range(self.config['config']['random_search']['iterations']) :
     
            trialpositions = [random.choice(self.nodes),random.choice(self.layers),random.choice(self.learningbounds),random.choice(self.batchsize)]
            print(trialpositions)
            if str(trialpositions) in self.searchlog:
                error = self.searchlog[str(trialpositions)]
                print(f"old error {error} from library")
            else:
                error = self.getNewNetwork(trialpositions) 
                self.searchlog[str(trialpositions)] = error
                print(f"new error found {error} ")
            
            if error < lowest_error:
                lowest_error = error
                lowest_positions = trialpositions
                print(f"Lowest positions are {lowest_positions}")


    def getNewNetwork(self,positions):
        network = NeuralNetwork(self.config,self.train_input, self.train_output,self.val_input,self.val_output,
                                                    hiddenlayer_size= positions[0], hiddenlayer_number = positions[1], 
                                                    learning_rate=positions[2],batch_size=positions[3]) 
        error = network.train()
        return error 





