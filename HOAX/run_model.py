import numpy as np
import math
import random

import NeuralNetwork
from GridSearch import GridSearch
from ArgParser import ArgParser
from JsonParser import JsonParser
from Logger import Logger

from DatabaseLoader import DatabaseLoader



if __name__ == "__main__":
    parser = ArgParser()
    jsonparser = JsonParser(parser.getConfigName())
    database = DatabaseLoader(parser.getDatabaseName(),jsonparser.getDatabase())
    logger = Logger(jsonparser.getLoggingFile(),jsonparser.getEpochs(),jsonparser.getEpochStep())


    if JsonParser.getOptimizer() == "grid_search":
        optimizer = GridSearch(jsonparser,database,logger)
        optimizer.run()

    elif  JsonParser.getOptimizer() == "random_search":
        optimizer = RandomSearch(config,coordinates,  output,val_coordinates, val_output)
        optimizer.run()

    elif  JsonParser.getOptimizer() == "simulated_anneaing":
        optimizer = SimulatedAnnealing(config,coordinates,  output,val_coordinates, val_output)
        optimizer.run()

    elif  JsonParser.getOptimizer() ==  "genetic_algorithm":
        optimizer = GeneticAlgorithm(config,coordinates,  output,val_coordinates, val_output)
        optimizer.run()
    else:
        print("No optimizer found")


