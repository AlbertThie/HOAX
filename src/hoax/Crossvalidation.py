from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork
from bayes_opt import BayesianOptimization, UtilityFunction
import math
import random
import numpy as np
from sklearn.model_selection import KFold 

class Crossvalidation(Optimizer):
    
    
    def __init__(self,jsonparser,database,logger):
        self.jsonparser =jsonparser
        self.database= database
        self.logger = logger
        self.nodes = self.jsonparser.getConfig()['crossvalidation']["hiddenlayer_size"]
        self.layers = self.jsonparser.getConfig()['crossvalidation']["hiddenlayer_number"]
        self.learningrate = self.jsonparser.getConfig()['crossvalidation']["learning_rates"]
        self.batchsize = self.jsonparser.getConfig()['crossvalidation']['batch_size']
        self.partition =self.jsonparser.getConfig()["crossvalidation"]['partition']

    
    def getNewNetwork(self):
        network = NeuralNetwork(self.jsonparser, self.database, self.logger,
                                                    hiddenlayer_size= self.nodes, hiddenlayer_number = self.layers, 
                                                    learning_rate=self.learningrate,batch_size=self.batchsize) 
        error = network.train()
        return error,network
      
    def run(self):
        X = self.database.getCoordinatesOut().copy()
        y = self.database.getEnergyOut().copy()
        kf = KFold(n_splits=self.partition) 
        kf.get_n_splits(X)
        errors = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.database.setInput(X_train)
            self.database.setValInput(X_test)
            self.database.setOutput(y_train)
            self.database.setValOutput(y_test)
            error,network = self.getNewNetwork()
            errors.append(error)
        print(errors)