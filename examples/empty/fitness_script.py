import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

from numpy import *
from copy import deepcopy 
from numpy.random import uniform, seed,rand

from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing


import traceback

from matplotlib import pyplot
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

import os
import traceback
import sys

import subprocess
import threading, os
from time import gmtime, strftime
from copy import deepcopy    
from matplotlib.ticker import MaxNLocator
    
enableTracebacks = True
from numpy import * 


designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
#designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
#designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
# add extra for more spaces...


error_labels = {0:'Valid',1:'Overmap',3:'Inaccuracy',4:'Timing'}
#error_labels = {0:'Valid',3:'Inaccuracy'}
    

def name():
    return "empty"

## these are important for visualizations... set them depending on the fitness 
maxvalue = 0.0   
minVal = 0.0
maxVal = 300.0
worst_value = 0.0

def termCond(best):
    global optVal
    return best > optVal

rotate = True
def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    

### here you invoke benchmarks/builds
def fitnessFunc(particle, state):
        ## add the build code and return results
        ## ignore cost and state for the moment...
        cost =  array([1200])
        state = None
 
        buildBitsream()
        fitness, exit_code = evaluateBitstream()
        ##returns work as follow
        ###((array([value you want to sue for regression]), array([exit code]),array([0 if you want to add the value for regressions or 1 if you dont]), cost) , state)

        ### overmapping error, error code 1 and 
        if exit_code == 1:
            return ((array([maxvalue]), array([1]),array([1]), cost) , state) ## 20 minutes
            
        if exit_code == 2: ## example of an overmapping error, we return error code 3 but we add the result!
            return ((power, array([3]),array([0]), cost) , state)##!!!! zmien na 0.0 
        ### accuracy error
        
        ### ok values execution time, we return errror code 0 (valid) and 0 as a third result as we can add the value to regressions
        return ((array([fitness]), array([0]),array([0]), cost), state)



