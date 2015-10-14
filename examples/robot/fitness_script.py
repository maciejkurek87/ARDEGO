import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

from numpy import *
import numpy as np
from copy import deepcopy 
from numpy.random import uniform, seed,rand

from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold, LeaveOneOut
from sklearn import neighbors
import traceback

from matplotlib import pyplot
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import mixture
import pdb
import os
import traceback
import sys

import subprocess
import threading, os
from time import gmtime, strftime
from copy import deepcopy    
from matplotlib.ticker import MaxNLocator
from scipy import stats

enableTracebacks = True
from numpy import * 
    
platform = "Maia" #"Maia"
    
designSpace = []
accuracy_limit = 5.25
always_valid = [0.,40., 8096.]    
maxvalue = 100000.0
minVal = 0.0
maxVal = maxvalue
worst_value =maxvalue
        
optVal = {4.4:5457.0, 4.7:5276., 5.0:5237., 5.25:5237.}[accuracy_limit]



divisors = [1, 2, 4, 8, 16, 32]

designSpace.append( {"min":0.0,"max":5.0,"step":1.0, "type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) ## 
designSpace.append({"min":10.0,"max":40.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})
designSpace.append({"min":2048.0,"max":8096.0,"step":48.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"s"})


dist_map = {0:"log", 1:"log", 2:"log"}
dist_params_map = {0:(0.6803998452431661, 94.595933564461035, 106.33247051307779), 1:(0.84684668089725457, 195.60591958904155, 144.4860482636887), 2:(0.87210877144381505, 47.249766528482198, 498.77346263250172)}

    
error_labels = {0:'Valid',1:'par',2:'overmap',3:'Accuracy'  }
    
def get_x_axis_name():
    if doCores:
        return "Cores"
    else:
        return "$m_w$"
        
def get_y_axis_name():
    if doDf:
        return "$d_f$"
    else:
        return "$m_w$"

def name():
    return "fF_robot"
    
def termCond(best):
    global optVal
    print str(best) + " " + str(optVal)
    return best < optVal
    
rotate = True

global allData
allData=None

def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    
def fitnessFunc(particle, state, output = 0):
    global allData
    #print particle
    if not allData:
        allData = my_interpolate()
    # Dimensions dynamically rescalled
    ############Dimensions
     
    cores = 2 ** int(particle[0]) 
    mw = int(particle[1]) - 10
    np_down = int(floor((particle[2] - 2048.0) / 60.0))
    np_up = int(ceil((particle[2] - 2048.0) / 60.0))
    np_p = int(particle[2] - 2048.0)
    #pdb.set_trace()
    if cores > 4:
        pass ## we use those bitstreams then.. they are broken and are approximately what 32 would be
        execution_time = 1.0
        code = 1
        cores = 4
        accuracy = 1.0
    elif (mw  % 2 == 0): ## interpolation across np
        down_execution_time = allData[0][0][mw][np_down]
        down_accuracy       = allData[1][0][mw][np_down]

        up_execution_time = allData[0][0][mw][np_up] 
        up_accuracy       = allData[1][0][mw][np_up]
        
        delta = (np_p - np_down * 60.0) / (60.0) 
        execution_time = (1.0 - delta) * down_execution_time + delta * up_execution_time
        accuracy = (1.0 - delta) * down_accuracy + delta * up_accuracy
        
        code               = allData[3][0][cores][mw]
    else: ## interpolation across mw
        ## interpolatie lower
        down_execution_time = allData[0][0][mw-1][np_down]
        down_accuracy       = allData[1][0][mw-1][np_down]
        
        up_execution_time = allData[0][0][mw-1][np_up] 
        up_accuracy       = allData[1][0][mw-1][np_up]
        
        delta = (np_p - np_down * 60.0) / (60.0) 
        down_down_execution_time = (1.0 - delta) * down_execution_time + delta * up_execution_time
        down_down_accuracy = (1.0 - delta) * down_accuracy + delta * up_accuracy
        
        ##interoplate upper
        down_execution_time = allData[0][0][mw+1][np_down]
        down_accuracy       = allData[1][0][mw+1][np_down]
        
        up_execution_time = allData[0][0][mw+1][np_up] 
        up_accuracy       = allData[1][0][mw+1][np_up]
        
        delta = (np_p - np_down * 60.0) / (60.0) 
        up_up_execution_time = (1.0 - delta) * down_execution_time + delta * up_execution_time
        up_up_accuracy = (1.0 - delta) * down_accuracy + delta * up_accuracy
        
        delta = 0.5
        execution_time = (down_down_execution_time * delta  + delta * up_up_execution_time)
        accuracy = up_up_accuracy * delta  + delta * down_down_accuracy
        
        up_code           = allData[3][0][cores][mw-1]
        down_code           = allData[3][0][cores][mw+1]
        
        code = up_code
            
    execution_time = execution_time / (cores ) 
    
    cost, state = getCost( cores, mw, np, state) ## we need to cast to float
    
    if code == 0 and (accuracy > accuracy_limit):
        code = 3
    
    if output == 0:
        if code == 1: ## timing
            return ((array([maxvalue]), array([1]),array([1]), cost), state)
        elif code == 2: ## MPR failed
            return ((array([maxvalue]), array([2]),array([1]), cost), state)
        elif code == 3: ## innacurate
            return ((array([execution_time]), array([3]),array([0]), cost), state)
        elif code == 4: ## exit code 30
            return ((array([maxvalue]), array([4]),array([1]), cost), state)
        else: ## code 0, ok
            return ((array([execution_time]), array([0]),array([0]), cost), state)
    elif output == 1:
            return resource
    elif output == 2: #it is a linear model
        return accuracy
    else:
        print "KURWA COS NIE TAK"
        return None

## state, do we have bitstream in the reopi?
def getCost(cores, mw, np, bit_stream_repo):
    global allData
    bit_stream_repo_copy = deepcopy(bit_stream_repo) 
    if bit_stream_repo is None:
        bit_stream_repo_copy = {}
    software_cost  = array([100.0 + 200.0 * random.random()]) ## it is in the range of few minutes (experiments were repeated few hundred times, each taking up to 1~2 secnods)
    if bit_stream_repo_copy.has_key((cores, mw)): ## bit_stream evalauted
        return array([software_cost]), bit_stream_repo_copy
    else:
        bit_stream_repo_copy[(cores, mw)] = True
        cost = array([allData[2][0][cores][mw] + software_cost])
        if allData[2][0][cores][mw] == 0.0:  ## empty, then interpolate
            cost = array([allData[2][0][cores][mw-1]*0.5 + allData[2][0][cores][mw+1] * 0.5 + software_cost])
        return cost , bit_stream_repo_copy  ##software cost, very small fraction and linear
       
def has_hardware(running_que, running_que_cost, part):
    bit_stream_repo_copy = {}
    idx = 0 
    for pt in running_que:
        bit_stream_repo_copy[(pt[0],pt[1])] = running_que_cost[idx]
        idx = idx + 1
    if bit_stream_repo_copy.has_key((part[0], part[1])):
        return True, bit_stream_repo_copy[(part[0], part[1])]
    else:
        return False, 0.0
                
def predict_cost(error_code):
    dist = dist_map[error_code]
    params = dist_params_map[error_code] 
    if dist == "norm":
        return predict_normal(*params)
    elif dist == "uni":
        return predict_uniform(*params)
    elif dist == "log":
        return predict_lognormal(*params)
    else: 
        print "pierdol sie"
           
def transferValid(part):
    return True
           
def fit_lognormal(data):
    shape, loc, scale = stats.lognorm.fit(data, loc=0)
    return shape, loc, scale
    
def predict_lognormal(shape, loc, scale):    
    logsample = stats.lognorm.rvs(shape, scale=scale, loc=loc, size=1000) # logsample ~ N(mu=10, sigma=3)
    return logsample
    
def fit_normal(data):
    loc, scale = stats.norm.fit(data)
    return loc, scale
    
def predict_normal(loc, scale):    
    logsample = stats.norm.rvs(scale=scale, loc=loc, size=1) # logsample ~ N(mu=10, sigma=3)
    return logsample
    
def fit_uniform(data):
    high, low = stats.uniform.fit(data)
    return low, high
    
def predict_uniform(low, high):    
    unisample = stats.uniform.rvs(low, high, size=1) # logsample ~ N(mu=10, sigma=3)
    return unisample
           
def get_index(row, row_idx, desSp_idx):
    return int((int(row[row_idx])-designSpace[desSp_idx]["min"])/designSpace[desSp_idx]["step"])
        
def get_shape():
    return tuple([(int((d["max"] - d["min"]) / d["step"]) + 1) for d in designSpace])
           
def load_data(folder_path):    
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=' ', quotechar='"')
    execution_time = np.zeros(shape=(5,31,103))
    accuracy = np.zeros(shape=(5,31,103))
    count = np.zeros(shape=(5,31,103))
        
    for row in spamReader:
        if len(row) == 7:
            try:
                if not (float(row[6]) == 0.0 or float(row[5]) == 0.0):
                    execution_time[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60] = float(row[5]) + execution_time[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60]
                    accuracy[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60] = float(row[6]) + accuracy[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60]
                    count[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60] = count[int(row[0])-7][(int(row[3])-10)][(int(row[4])-2048)/60] + 1
            except:
                pdb.set_trace()
    return execution_time, accuracy, count
    
def load_cost(folder_path):
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=',', quotechar='"')
    cost = np.zeros(shape=(5,5,31))
    code = np.zeros(shape=(5,5,31))
        
    for row in spamReader:
        try:            
            for i in range(103):
                cost[int(row[0])-7][int(row[1])][(int(row[2])-10)] = float(row[5])
                code[int(row[0])-7][int(row[1])][(int(row[2])-10)] = float(row[3])
        except:
            pdb.set_trace()
    return cost, code
           
def my_interpolate():
    folder_path = "/homes/mk306/all_results_robot_all.txt"
    cost_folder_path = "/homes/mk306/MLO/examples/robot/all_results_robots_builds.csv"
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=' ', quotechar='"')
    execution_time, accuracy, count = load_data(folder_path)
    execution_time = execution_time / count
    accuracy = accuracy / count
    cost, code = load_cost(cost_folder_path)
    return execution_time, accuracy, cost, code
                
if __name__ == '__main__':
    all_results = my_interpolate()
    import itertools 
    maxEI = 100000000000.0
    maxEIcord = None
    space_def = []
    counter = 0
    '''
    xx = (2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 2.0)
    print fitnessFunc(xx,None)[0][0][0]
    '''
    for d in designSpace:
        space_def.append(arange(d["min"],d["max"]+1.0,d["step"]))
    #print space_def
    results = []
    cost = 0.0
    
    for z in itertools.product(*space_def):
        #print str(z)
        '''cost = fitnessFunc(z,{})[0][3] + cost + 5.0*(16*5.0*(0.8 + random.random()/5.0))
            max_cores = allData[11][z[1]][z[2]][0]
            z = list(z)
            z[0]=max_cores
            cost = cost + fitnessFunc(z,{})[0][3]
            #print fitnessFunc(z,{})[0][3]
            #print str(z)
        print cost
        '''
    #print fitnessFunc(array([12.,15.,6.]),{})[0]
    #print fitnessFunc(array([12.,16.,6.]),{})[0]
        
        counter = counter + 1
        #print fitnessFunc(z,{})
        EI = fitnessFunc(z,{})[0][0]
        code = fitnessFunc(z,{})[0][1]
        cost = fitnessFunc(z,{})[0][3]
        
        if code == 1: 
            print z
            print code
            print cost
            print EI
        #print str(z) + " " +  str(maxEIcord) + " " +  str(maxEI) + " " + str(code)
        
        if (maxEI > EI) and (code==0): ## no need for None checking
            maxEI = EI
            maxEIcord = z
    ### 
    print "DONE!"
    print maxEIcord
    print maxEI
    
    