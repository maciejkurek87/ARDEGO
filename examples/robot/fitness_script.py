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

always_valid = [1.,10., 2048.]    
maxvalue = 50.0
minVal = 50.0
maxVal = 160.0
worst_value = 50.0
optVal = 155.0 
designSpace.append({"min":1.0,"max":4.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})
designSpace.append({"min":10.0,"max":40.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})
designSpace.append({"min":2048.0,"max":8192.0,"step":60.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"s"})
dist_map = {0:"log", 1:"log", 2:"log"}
dist_params_map = {0:(0.6803998452431661, 94.595933564461035, 106.33247051307779), 1:(0.84684668089725457, 195.60591958904155, 144.4860482636887), 2:(0.87210877144381505, 47.249766528482198, 498.77346263250172)}

    
error_labels = {0:'Valid',1:'Overmap' }
    
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
    return "robot_"  + platform + "_" + str(maxError) + "_doCores" + str(doCores) + "_doDf" + str(doDf) + "_errorCorrection" + str(errorCorrection) 
    
def termCond(best):
    global optVal
    print str(best) + " " + str(optVal)
    return best > optVal
    
rotate = True

global allData
allData=None

def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    
def fitnessFunc(particle, state):
    global allData
    #print particle
    if not allData:
        allData = my_interpolate()
    # Dimensions dynamically rescalled
    ############Dimensions
     
    mw = int(particle[0]) - 4
    freq = int(particle[1]) - 80
    cores = int(particle[2]) - 1
    
    try:
        code = allData[1][cores][mw][freq]
        throughput =  array([allData[2][cores][mw][freq]])
    except:
        pdb.set_trace()
    
    cost, state = getCost( cores, mw, freq, code, state) ## we need to cast to float
    
    ##resource prediciton
    resource_util = max([allData[-1][cores][mw][freq],allData[-2][cores][mw][freq],allData[-3][cores][mw][freq]])
        
    if resource_util > 100.0 or ( mw > 41 and cores == 1) or ( mw > 29 and cores == 2) or ( mw > 14 and cores == 3) or ( mw > 8 and cores == 4) :
        cost = predict_uniform(70.0, 100.0)
        return ((array([maxvalue]), array([5]),array([1]), cost), state)
        
    #print allData[mw][df][cores], code, isValid    
    ### overmapping error
    #pdb.set_trace()
    if code == 1: ## timing
        return ((array([maxvalue]), array([1]),array([1]), cost), state)
    elif code == 2: ## MPR failed
        return ((array([maxvalue]), array([2]),array([1]), cost), state)
    elif code == 3: ## overmapping
        return ((array([maxvalue]), array([3]),array([1]), cost), state)
    elif code == 4: ## exit code 30
        return ((array([maxvalue]), array([4]),array([1]), cost), state)
    else: ## code 0, ok
        return ((throughput, array([0]),array([0]), cost), state)

## state, do we have bitstream in the reopi?
def getCost(cores, mw, freq, code, bit_stream_repo):
    global allData
    bit_stream_repo_copy = deepcopy(bit_stream_repo) 
    if bit_stream_repo is None:
        bit_stream_repo_copy = {}
    if bit_stream_repo_copy.has_key((cores, mw, freq)): ## bit_stream evalauted
        return array([0.0]), bit_stream_repo_copy
    else:
        bit_stream_repo_copy[(cores, mw, freq)] = True
        cost = array([predict_cost(code)])
        return cost , bit_stream_repo_copy  ##software cost, very small fraction and linear
                
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
           
def my_interpolate():
    folder_path = "/homes/mk306/all_results_robot_all.txt"
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=' ', quotechar='"')
    execution_time = zeros(shape=get_shape())
    accuracy = zeros(shape=get_shape())
    count = zeros(shape=get_shape())
    for row in spamReader:
        if len(row) == 7:
            if not (float(row[6]) == 0.0 or float(row[5]) == 0.0):
                try:
                    execution_time[0][(int(row[3])-10)/2][(int(row[4])-2048)/60] = float(row[5]) + execution_time[0][(int(row[3])-10)/2][(int(row[4])-2048)/60]
                    accuracy[0][(int(row[3])-10)/2][(int(row[4])-2048)/60] = float(row[6]) + accuracy[0][(int(row[3])-10)/2][(int(row[4])-2048)/60]
                    count[0][(int(row[3])-10)/2][(int(row[4])-2048)/60] = count[0][(int(row[3])-10)/2][(int(row[4])-2048)/60] + 1
                except:
                    pdb.set_trace()

    ## for both 0 and 2 gaussian mixture
    ## for 0 log tail
    ## 0 lognormal
    ## 1 normal
    ## 2 and 4 normal
    ## 3 uniofrm
    execution_time = execution_time / count
    accuracy = accuracy / count
    result_files=['all_results_robot_errors.txt']
    module_path = os.path.dirname(__file__)
    if module_path:
        module_path = module_path + "/"
    points = []
    values = [] 
    all_results = []
    for result_file in result_files:
        spamReader = csv.reader(open(module_path + result_file, 'rb'), delimiter=' ', quotechar='"')
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_4 = row[3:]
            points.append([row_0,row_1,row_2])
            values.append([float(value) for value in row_4])
    grid_x, grid_y, grid_z =  mgrid[designSpace[0]["min"]:designSpace[0]["max"]: get_shape()[0] * 1j, designSpace[1]["min"]:designSpace[1]["max"]: get_shape()[1] * 1j, designSpace[2]["min"]:designSpace[2]["max"]: get_shape()[2] * 1j]
    
    points_n = array(points)  ### all the points from which we can interpolate
    values_n = array(values)  ### the data corresponding to those points
    ###
    ###
    ### EXIT CODES
    ###
    ###
    
    training_labels = array(values)[:,0].reshape(len(values),1)
    
    def plot_hist(x):
        import numpy as np
        import matplotlib.mlab as mlab
        import matplotlib.pyplot as plt


        # example data
        mu = 100 # mean of distribution
        sigma = 15 # standard deviation of distribution

        num_bins = 50
        # the histogram of the data
        n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
        # add a 'best fit' line
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.xlabel('Smarts')
        plt.ylabel('Probability')
        plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        plt.show()
    
    for error_code in [0,1,2]:#;,1,2,3,4]:
        X =[] 
        Y =[]
        for point,value in zip(points,values):
            code = value[0]
            cost = value[2]
            if (code == error_code):# & (cost > 140):
                X.append(point)
                Y.append(cost)
        
    all_results.append(execution_time)
    all_results.append(accuracy)
                    
    for i in range(1,len(row_4)):
            data = griddata(points_n, values_n[:,i], (grid_x, grid_y, grid_z), method='linear')
            all_results.append(data)
                    
    return all_results
                
if __name__ == '__main__':
    my_interpolate()