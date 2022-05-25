#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork


# In[ ]:


class GridSearch(Optimizer):
    
    
    def __init__(self,config,train_input,train_output,val_input,val_output):
        self.config = config
        self.train_input = train_input
        self.train_output = train_output
        self.val_input = val_input
        self.val_output = val_output

  
    def run(self):
        lowest_error = 1000
        for g in range(self.config['config']['grid_search']['hiddenlayer_size'][0],self.config['config']['grid_search']['hiddenlayer_size'][1],self.config['config']['grid_search']['hiddenlayer_size'][2]):
           for i in range(self.config['config']['grid_search']['hiddenlayer_number'][0],self.config['config']['grid_search']['hiddenlayer_number'][1],self.config['config']['grid_search']['hiddenlayer_number'][2]):
               for h in self.config['config']['grid_search']['learning_rates']:
                
                    network = NeuralNetwork(self.config, self.train_input, self.train_output,self.val_input,self.val_output,
                                                    hiddenlayer_size=g,hiddenlayer_number=i, learning_rate=h,batch_size=64) 
                    temp_error = network.train()
                    if (temp_error < lowest_error):
                        lowest_error = temp_error
                        network.export()

    

