#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork


# In[ ]:


class GridSearch(Optimizer):
    
    
    def __init__(self,jsonparser,database,logger):
        self.jsonparser =jsonparser
        self.database= database
        self.logger = logger

    def run(self):
        lowest_error = 1000
        hiddenlayer_size = self.jsonparser.getConfig()['grid_search']['hiddenlayer_size']
        hiddenlayer_number = self.jsonparser.getConfig()['grid_search']['hiddenlayer_number']
        for g in range(hiddenlayer_size[0],hiddenlayer_size[1],hiddenlayer_size[2]):
           for i in range(hiddenlayer_number[0],hiddenlayer_number[1],hiddenlayer_number[2]):
               for h in self.jsonparser.getConfig()['grid_search']['learning_rates']:
                    for j in self.jsonparser.getConfig()['grid_search']['batch_sizes']:
                
                        network = NeuralNetwork(self.jsonparser, self.database, self.logger, 
                                                        hiddenlayer_size=g,hiddenlayer_number=i, learning_rate=h,batch_size=j) 
                        temp_error = network.train()
                        if (temp_error < lowest_error):
                            lowest_error = temp_error
                            network.export()  

        

