from .Optimizer import Optimizer
from .NeuralNetwork import NeuralNetwork
import math
import random





class GeneticAlgorithm(Optimizer):
    
    
    def __init__(self,jsonparser,database, logger):
        self.database = database
        self.jsonparser = jsonparser
        self.logger = logger
        self.nodesbounds = self.jsonparser.getConfig()['genetic_algorithm']["hiddenlayer_size"]
        self.layerbounds = self.jsonparser.getConfig()['genetic_algorithm']["hiddenlayer_number"]
        self.learningbounds = self.jsonparser.getConfig()['genetic_algorithm']["learning_rates"]
        self.batchsize = self.jsonparser.getConfig()['genetic_algorithm']['batch_size']

        self.nodes = [i for i in range(self.nodesbounds[0],self.nodesbounds[1],self.nodesbounds[2])]
        self.layers = [i for i in range(self.layerbounds[0],self.layerbounds[1],self.layerbounds[2])]
        self.iterations = self.jsonparser.getConfig()['genetic_algorithm']['iterations']
        self.searchlog = {}
        self.mutationrate = self.jsonparser.getConfig()['genetic_algorithm']["mutation_rate"]
        self.population_size = self.jsonparser.getConfig()['genetic_algorithm']["population_size"]

        nbound = len(self.nodes) -1
        lbound =len(self.layers) -1
        ebound = len(self.learningbounds) -1
        bbound = len(self.batchsize) -1
        nbitsize= len('{0:b}'.format(nbound-1))
        lbitsize= len('{0:b}'.format(lbound-1))
        ebitsize= len('{0:b}'.format(ebound-1))
        bbitsize= len('{0:b}'.format(bbound-1))

        self.bitsizes = [nbitsize,lbitsize,ebitsize,bbitsize]

        self.upperbounds = [nbound,lbound,ebound,bbound]
        self.lowerbounds = [0,0,0,0]
  



    def callCE(self,positionlist):
        if str(positionlist) in self.searchlog:
            error = self.searchlog[str(positionlist)]
            print("library")
        else:
            error = self.getNewNetwork(positionlist)
            self.searchlog[str(positionlist)] = error
        return error

    def getNewNetwork(self,positions):
        network = NeuralNetwork(self.jsonparser, self.database,self.logger,
                                                    hiddenlayer_size= self.nodes[positions[0]], hiddenlayer_number = self.layers[positions[1]], 
                                                    learning_rate=self.learningbounds[positions[2]],batch_size=self.batchsize[positions[3]]) 
        error = network.train()
        return error 
   

        
    def checkBoundary(self,x,bound):
        """Checks a binary string against the boundary and sets it to the boundary in case it goes out of bounds"""
        if int(x,2) < bound:
            return x
        else:
            return '{0:b}'.format(bound-1)


    def getOrder(self,population):
        populationscore = self.getScore(population)
        orderedpop = [pop for _, pop in sorted(zip(populationscore, population))]
        return orderedpop

    def getScore(self,population):
        print(f'Current population is: {population}')
        score = [self.callCE(pop) for pop in population]
        return score

    def change(self,i):
        if random.uniform(0, 1) < self.mutationrate:
            return i.replace('1', '2').replace('0', '1').replace('2', '0')
        else:
            return i

    def mutate(self,i,bitstring):
        output =""
        for c in bitstring:
            output+=self.change(c)
        
        return int(self.checkBoundary(output,i),2)
        
        
    def mutateBatch(self,offspring1,offspring2):
        return [self.mutate(self.upperbounds[i],x) for i,x in enumerate(offspring1)],[self.mutate(self.upperbounds[i],x) for i,x in enumerate(offspring2)]
        


    def newGeneration(self,orderedpop):
        parent1, parent2 = orderedpop[0],orderedpop[1]
        offspring1, offspring2 = self.mutateBatch(*self.getOffspring(parent1,parent2))
        
        return [parent1,parent2,orderedpop[2],orderedpop[3],offspring1,offspring2]


    def getOffspring(self,parent1,parent2):
        offspring1,offspring2 = ([],[])
        for index,(stat1,stat2) in enumerate(zip(parent1,parent2)):
            bit1 = self.bitsizes[index]
            bitparent1,bitparent2 = format(parent1[index],f'0{bit1}b'),format(parent2[index],f'0{bit1}b')
            cutoff = self.getCuttof(self.bitsizes[index])
            offspring1.append(bitparent1[:cutoff]+bitparent2[cutoff:])
            offspring2.append(bitparent2[:cutoff]+bitparent1[cutoff:])  
            print(f'these are bitwise: {offspring1} and {offspring2}')

            
        return offspring1,offspring2



    def getCuttof(self,x):
        return random.randrange(1,x)


    def run(self):
        population =[]
        for _ in range(self.population_size):
            population.append([random.randrange(self.upperbounds[0]),random.randrange(self.upperbounds[1]),random.randrange(self.upperbounds[2]),random.randrange(self.upperbounds[3])])

        for i in range(self.iterations):
            orderedpop=self.getOrder(population)
            print(f'this is the best: {orderedpop[0]}')
            population=self.newGeneration(orderedpop)
            




