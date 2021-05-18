# -*- coding: utf-8 -*-
from pysurf import Interpolator
import torch
import numpy as np
import pickle
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets

def get_interpolators(self, db, properties):
    pass

class NeuralNetworkInterpolator(Interpolator):
    """cool class"""
    _questions = """
     inputsize = 3 :: int
     outputsize = 3 :: int
     hiddenlayer_sizes = [50] :: list
     learningrate = 1e-4:: float
     epochs = 30000 :: int
     epochstep = 50 :: int
     loggingfile = logfile.txt :: str
     plottingfile = plottingfile.txt :: str
     optimizer = adam :: str
     normalizer = tanh :: str
     
    """
    def __init__(self, db, properties, logger, inputsize, outputsize, energy_only=True, savefile='', inverse=False,
                 hiddenlayer_sizes=[50], learning_rate=1e-3, epochs=300,epochsteps=50, optimizing=False,loggingfile="",printingfile="",optimizer="adam", normalizer="tanh"):
        self.db = db
        self.vdb = vdb
        
        
        self.hiddenlayer_sizes = hiddenlayer_sizes
        self.dataset = NNDataset(torch.tensor(dbcoord),torch.tensor(np.copy(db['energy'])-np.amin(np.copy(db['energy']))))
        self.validationset = NNDataset(torch.tensor(vdbcoord),torch.tensor(np.copy(vdb['energy'])-np.amin(np.copy(db['energy']))))
        self.dataloader = DataLoader(self.dataset, batch_size=256,shuffle= 'True')
        print(self.dataset.energyCurves)
        self.N = 256
        
        #self.dataset.output_shape()
        self.epochs = epochs
        self.optimizing=optimizing
        self.epochsteps = epochsteps
        if optimizing:
            self.loggingfile = loggingfile
            self.printingfile = printingfile
        # Use the nn package to define our model and loss function.
        network = NeuralNet(inputsize, hiddenlayer_sizes, outputsize, optimizer, normalizer)
        self.model = network.hidden
        self.loss_fn = torch.nn.MSELoss()
        self.model.double()
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)



        super().__init__(db, properties, logger, energy_only, savefile, inverse=inverse)
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        # create your dataset
        # create your dataloade






    def get_interpolators(self, db, properties):
        print("test")

    def save(self, filename):
        torch.save(self.model, filename)

    def loadweights(self, filename):
        self.model = torch.load(filename)

    def get_interpolators_from_file(self, filename, properties):
        """Properties contains a tuple of [energy,gradient] """
        return {prop_name: self.db[prop_name].shape[1:] for prop_name in properties}


    def get(self, request):
        """Gives object with coordinates and desired properties"""
        pass

    def _train(self):
        for t in range(self.epochs):

            for index, data in enumerate(self.dataloader,0):
                local_batch, local_labels = data
                #print(local_batch)
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(local_batch)

                # Compute and print loss.
                #print(y_pred)
                #print(local_labels)
                loss = self.loss_fn(y_pred, local_labels)
                #print(loss)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                self.optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self.optimizer.step()
                

            if t % self.epochsteps == 0:
                
                lossprint = np.sqrt(self.validate_model(t)/len(self.validationset))
                print(f"{lossprint} {t} {self.hiddenlayer_sizes}")
                self.printingfile.append((self.hiddenlayer_sizes,t,lossprint))
                with open('printoutNormGroundBGrid.txt', 'wb') as f:
                    pickle.dump(self.printingfile,f)
                file = open(self.loggingfile,"a")
                file.write(f"Loss = {lossprint} \n Epochs = {self.epochs} \n Hiddenlayers = {self.hiddenlayer_sizes} \n Learningrate = {self.learning_rate} \n Validation_percent = {self.validation_percent} \n Batchsize = {self.N} \n \n")
                file.close()

    
                if self.optimizing == "Seperate":
                    loss = np.sqrt(self.validate_model()/len(self.validationset))
                    self.printingfile.append((self.hiddenlayer_sizes,self.epochs,loss))
                    with open('printoutsixlayer.txt', 'wb') as f:
                        pickle.dump(self.printingfile,f)
                    file = open(self.loggingfile,"a")
                    file.write(f"Loss = {loss} \n Epochs = {t} \n Hiddenlayers = {self.hiddenlayer_sizes} \n Learningrate = {self.learning_rate} \n Validation_percent = {self.validation_percent} \n Batchsize = {self.N} \n \n")
                    file.close()

        print("Done Training")
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
        #print(model_predictions)
        if np.sqrt(losssquared /len(self.validationset))< 0.001 :
            plt.plot(self.vdb['energy']-np.amin(self.vdb['energy']))
            plt.plot(model_predictions)
            
            plt.savefig(f"graph{n}{self.hiddenlayer_sizes}sixlayer.png")
            plt.show()
            plt.close()
        return losssquared

class NNDataset(torch.utils.data.Dataset):
    """Molecule Data set"""

    def __init__(self, coordinates, energyCurves):
        self.coordinates = coordinates
        self.energyCurves = energyCurves

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
    def __init__(self, D_in, H, D_out,optimizer="adam", normalizer="tanh"):
        H = [D_in] + H + []
        self.hidden = torch.nn.ModuleList()
        if normalizer == "tanh":
            for k in range(len(H)-2):
                self.hidden.append(torch.nn.Linear(H[k], H[k+1]))
                self.hidden.append(torch.nn.Tanh())
            self.hidden.append(torch.nn.Linear(H[-2],H[-1]))
    
        
