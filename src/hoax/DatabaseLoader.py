import numpy as np
from netCDF4 import Dataset
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split


class DatabaseLoader:
    def __init__(self,databaseName,databaseConfig,databaseType='netcdf'):
        if databaseType == 'netcdf':
            self.database = Dataset(databaseName, mode='r')
            if databaseConfig["crdmode"] == "cartesian":
                dbset = np.copy(self.database['crd'])
                self.coordinatesout = []
                for j,i in enumerate(dbset):
                    if j == 0:
                        self.coordinatesout = [pdist(i)]
                    else:
                        self.coordinatesout = np.concatenate((self.coordinatesout,[pdist(i)]))
                
        self.energyout =np.copy(self.database['energy'])-np.amin(self.database['energy'])     
        self.input, self.val_input, self.output, self.val_output = train_test_split(self.coordinatesout, self.energyout , test_size=databaseConfig['validation_ratio'])


    def setInput(self,input):
        self.input = input

    def setValInput(self,val_input):
        self.val_input = val_input

    def setOutput(self,output):
        self.output = output

    def setValOutput(self,val_output):
        self.val_output = val_output

    def getInput(self):
        return self.input

    def getValInput(self):
        return self.val_input

    def getOutput(self):
        return self.output

    def getValOutput(self):
        return self.val_output

    def getCoordinatesOut(self):
        return self.coordinatesout

    def getEnergyOut(self):
        return self.energyout




