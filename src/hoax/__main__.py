import numpy as np
import math
import random

from .NeuralNetwork import NeuralNetwork
from .GridSearch import GridSearch
from .RandomSearch import RandomSearch
from .SimulatedAnnealing import SimulatedAnnealing
from .GeneticAlgorithm import GeneticAlgorithm
from .ArgParser import ArgParser
from .JsonParser import JsonParser
from .Logger import Logger

from .DatabaseLoader import DatabaseLoader


def main():
    parser = ArgParser()
    jsonparser = JsonParser(parser.getConfigName())
    database = DatabaseLoader(parser.getDatabaseName(),jsonparser.getDatabase())
    logger = Logger(jsonparser.getLoggingFile(),jsonparser.getEpochs(),jsonparser.getEpochStep())


    if jsonparser.getOptimizer() == "grid_search":
        optimizer = GridSearch(jsonparser,database,logger)
        optimizer.run()

    elif  jsonparser.getOptimizer() == "random_search":
        optimizer = RandomSearch(jsonparser,database,logger)
        optimizer.run()

    elif  jsonparser.getOptimizer() == "simulated_annealing":
        optimizer = SimulatedAnnealing(jsonparser,database,logger)
        optimizer.run()

    elif  jsonparser.getOptimizer() ==  "genetic_algorithm":
        optimizer = GeneticAlgorithm(jsonparser,database,logger)
        optimizer.run()
    else:
        print("No optimizer found")



if __name__ == "__main__":
    main()