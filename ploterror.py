#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:11:29 2020

@author: thie
"""


import pickle
import matplotlib.pyplot as plt
with open('printoutPyrazine6layerMLA2.txt', 'rb') as f:
    a = pickle.load(f)


def plotneuralerror(a,maxvalue):
    currentline = a[0][0]
    x =[a[0][1]]
    y =[min(a[0][2],maxvalue)]
    for result in a:
        print(result)
        if result[0] == currentline:
            x.append(result[1])
            y.append(min(result[2],maxvalue))
        else:
            plt.plot(x, y, label = str(currentline))
            currentline = result[0]
            x = [result[1]]
            y = [min(result[2],maxvalue)]
    plt.plot(x, y, label = str(currentline))
    plt.xlabel('Number of Epochs')
    plt.ylabel('Error in Hartree')
    plt.title('RMSE with epoch and hidden layer size')
    plt.legend()
    plt.show()
    
plotneuralerror(a,0.2)
                
