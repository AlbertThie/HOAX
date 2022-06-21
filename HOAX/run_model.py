import numpy as np

import random

import NeuralNetwork
from GridSearch import GridSearch
import tables as tbe
from ArgParser import ArgParser
from JsonParser import JsonParser
import math
from DatabaseLoader import DatabaseLoader



if __name__ == "__main__":
    parser = ArgParser()
    jsonparser = JsonParser(parser.getConfigName())
    database = DatabaseLoader(parser.getDatabaseName(),jsonparser.getDatabase())


    num = config['config']["neural_network"]["epochs"]/config['config']["neural_network"]["epoch_step"]
    print(num)
    class NeuralNetworkRun(tb.IsDescription):
        idNumber  = tb.Int64Col()
        hiddenLayers = tb.Int64Col()
        nodesPerLayer = tb.Int64Col()
        batchsize = tb.Int64Col()
        learningRate = tb.Int64Col()
        validationError = tb.Float64Col(shape=(num,))
        
         
        

    filesaving = tb.open_file(config['config']["neural_network"]["logging_file"],mode="w")
    root = filesaving.root
    group = filesaving.create_group(root,"NeuralNetworkRun")
         
    gRuns = root.NeuralNetworkRun 
    table = filesaving.create_table("/NeuralNetworkRun","NeuralNetworkRun1",NeuralNetworkRun,"Runs:"+"NeuralNetworkRun1")
    table.flush()
    print(filesaving)
    filesaving.close()

    if (any(i == 'grid_search' for i in config['config'])):
        if (any(j == 'grid_search' for j in config['config'])):
            optimizer = GridSearch(config,coordinates,  output,val_coordinates, val_output)
            optimizer.run()
            
            





# In[ ]:




