#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import torch
import math
import numpy as np
import pickle
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
import shutil
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import tables as tb

class NeuralNetwork():
    def __init__(self, jsonparser, database,logger,
                 hiddenlayer_size,hiddenlayer_number, learning_rate,batch_size ): 
       
        self.dataset = NNDataset(database.getInput(),database.getOutput())
        self.validationset = NNDataset(database.getValInput(),database.getValOutput())
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size,shuffle= 'True')
        self.hiddenlayer_size = hiddenlayer_size
        self.batch_size = batch_size
        self.hiddenlayer_number = hiddenlayer_number
        self.learning_rate = learning_rate
        self.logger = logger
        self.epochs = jsonparser.getEpochs()
        self.epoch_step = jsonparser.getEpochStep()
        self.weights_file = jsonparser.getWeightsFile()
        self.model_filename = jsonparser.getModelFile()
        self.loggingfile = jsonparser.getLoggingFile()
        network = NeuralNet( len(database.getInput()[0]), hiddenlayer_size, hiddenlayer_number, len(database.getOutput()[0]),  
                              jsonparser.getActivation())
        self.model = network.model
        self.loss_fn = self.get_loss_function(jsonparser.getLossFunction())
        self.model.double()
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        # create your dataset
        # create your dataloade






    def get_interpolators(self, db, properties):
        print("test")

    def save(self, filename):
        torch.save(self.model, filename)
    
    def _save(self):
        torch.save(self.model, self.weights_file)

    def loadweights(self, filename):
        self.model = torch.load(filename)
        self.model.eval()

    def get_interpolators_from_file(self, filename, properties):
        """Properties contains a tuple of [energy,gradient] """
        return {prop_name: self.db[prop_name].shape[1:] for prop_name in properties}


    def get(self, request):
        """Gives object with coordinates and desired properties"""
        pass
    def export(self):
        shutil.copy(self.weights_file, self.model_filename)
    def train(self):
        minimum_loss = math.inf
        printinglog = []
        for t in range(self.epochs):

            for index, data in enumerate(self.dataloader,0):
                local_batch, local_labels = data
                y_pred = self.model(local_batch)
                loss = self.loss_fn(y_pred, local_labels)               
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            if t % self.epoch_step == 0:
                
                lossprint = np.sqrt(self.validate_model(t)/len(self.validationset))
                if lossprint < minimum_loss:
                    minimum_loss = lossprint
                    self._save()
                print(f"{lossprint} {t} {self.hiddenlayer_size} {self.hiddenlayer_number} {self.learning_rate} {self.batch_size}")
                printinglog.append(lossprint)

        print("Done Training")
        filesaving = self.logger.getLogFile()
        #Rewrite this part to use logger class
        root = filesaving.root
        idnumber = 0
        table = root.NeuralNetworkRun.NeuralNetworkRun1
        for row in table.iterrows():
            idnumber +=1

        addTable=table.row 
        addTable['idNumber'] = idnumber
        addTable['hiddenLayers'] = self.hiddenlayer_number
        addTable['nodesPerLayer'] = self.hiddenlayer_size
        addTable['batchsize'] = self.batch_size
        addTable['learningRate'] = self.learning_rate
        addTable['validationError'] =np.array(printinglog)
        addTable.append()
        table.flush()
        filesaving.close()

        return minimum_loss
        
        

    def validate_model(self,n):
        model_predictions = []
        testpoint_positions = []
        losssquared = 0
        for local_batch, local_labels in self.validationset:
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(torch.flatten(local_batch))
                #print(y_pred)
                model_predictions.append(y_pred.tolist())
                # Compute and print loss
                testpoint_positions.append(local_batch.tolist())
                loss = self.loss_fn(y_pred, local_labels)
                losssquared += loss.item()
        valLoss =np.sqrt(losssquared /len(self.validationset))
        return losssquared
    
    def get_loss_function(self,loss):
        if loss == 'MSE':
            return torch.nn.MSELoss()
class NNDataset(torch.utils.data.Dataset):
    """Molecule Data set"""

    def __init__(self, coordinates, energyCurves):
        self.coordinates =torch.tensor(coordinates)
        self.energyCurves = torch.tensor(energyCurves)

    def __getitem__(self,index):
        coordinate = self.coordinates[index]
        curve = self.energyCurves[index]

        return coordinate, curve

    def __len__(self):
        return len(self.coordinates)
        
    def input_shape(self):
        return list(self.coordinates[0].size())[0] *3
    def output_shape(self):
        return list(self.energyCurves[0].size())[0]

class NeuralNet(torch.nn.Module):
    def __init__(self, inputsize, hiddensize, hiddennumber, outputsize, normalizer="tanh"):
        super(NeuralNet, self).__init__()
        self.hidden = torch.nn.ModuleList()
        if normalizer == "tanh":
            self.hidden.append(torch.nn.Linear(inputsize, hiddensize))
            self.hidden.append(torch.nn.Tanh())
            for k in range(hiddennumber):
                self.hidden.append(torch.nn.Linear(hiddensize, hiddensize))
                self.hidden.append(torch.nn.Tanh())
            self.hidden.append(torch.nn.Linear(hiddensize,outputsize))
        self.model = torch.nn.Sequential(*self.hidden)
        
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




