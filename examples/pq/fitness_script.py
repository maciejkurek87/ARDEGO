import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

import itertools
from numpy import *
from copy import deepcopy 
from numpy.random import uniform, seed,rand, normal

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
initMin = -1
initMax = 1

import pyGP_OO
from pyGP_OO.Core import *
from pyGP_OO.Valid import valid

#aecc or execution time -- when error correction applied
#### [ 15.   6.]   75.3885409418 0.1  13.0, 15.0, 6.0
#### [ 20.  11.]   11.1996923818 0.05 
#### [ 19.  22.]   1.44387061283 0.01 12.   19.   22.0
#### [ 20.  32.]   0.469583069715 0.001

#### without error correction
#### [ 14.  11.   4.]   259.663179761 all
#### [ 13.  15.   7.]   46.8546137378 0.001

cost_maxVal = 15000.0
cost_minVal = 0.0
    
doCores = True
doMw = True
doDf = True
designSpace = []

designSpace.append({"min":4.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})
designSpace.append( {"min":80.0,"max":120.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"s"})
designSpace.append({"min":1.0,"max":4.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"h"})      

resource_class = {0:{"type":"logreg","name":"bram","lower_limit":0.0,"higher_limit":100.0},
                  1:{"type":"logreg","name":"lut","lower_limit":0.0,"higher_limit":100.0},
                  2:{"type":"logreg","name":"ff","lower_limit":0.0,"higher_limit":100.0},
                  3:{"type":"bin", "name":"timing"},
                  4:{"type":"bin", "name":"par"}}
        
dist_map = {0:"log", 1:"norm", 2:"norm", 3:"uni", 4:"norm"}
dist_params_map = {0:(0.82069062921168601, 143.6787370361115, 134.47330375752006), 1:(561.0, 0.0), 2:(1306.3009708737864, 493.22056026388299), 3:(45.0,206.0), 4:(1337.52, 473.07708631892098)}
## if  [ 14.  11.   4.]   0.000275081632653

error_labels = {0:'Valid',1:'Overmap' }
#error_labels = {0:'Valid',3:'Inaccuracy'}
    
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
    return "fF_pq"
    
always_valid = [4.,80.,1.]
#always_valid = [1.,1.]
    
maxvalue = 50.0
minVal = 50.0
maxVal = 160.0
worst_value = 50.0

#optVal = 153.5
#optVal = 151.0    #% 95%
optVal = 155.0  #% 97.5
#optVal = 159.24  %100

def termCond(best):
    global optVal
    print str(best) + " " + str(optVal)
    return best > optVal
    
rotate = True

global allData
allData=None

def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    
def fitnessFunc(particle, state, return_resource = False):
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
        code = 3
        cost = predict_uniform(70.0, 100.0)
        
    if return_resource :
        ##resource_class = {0:["logreg","bram","g"],1:["logreg","lut","g"],2:["logreg","ff","g"],3:["bin", "timing","a"],4:["bin", "par","a"]}
        if code == 0: ## timing
            return [array([allData[-1][cores][mw][freq]]), array([allData[-2][cores][mw][freq]]), array([allData[-3][cores][mw][freq]]), [0], [0]]
        else :
            return [[None], [None], [None], array([code == 1]), array([code == 2])]
    else:
            
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
                
def extrapolate_nans(x, y, v):
    '''  
    Extrapolate the NaNs or masked values in a grid INPLACE using nearest
    value.

    .. warning:: Replaces the NaN or masked values of the original array!

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.

    Returns:

    * v : 1D array
        The array with NaNs or masked values extrapolated.
    '''

    if ma.is_masked(v):
        nans = v.mask
    else:
        nans = isnan(v)
    notnans = logical_not(nans)
    v[nans] = griddata((x[notnans], y[notnans]), v[notnans],
        (x[nans], y[nans]), method='nearest').ravel()
    return v
           
def gp_predict(x,y, z):
    x=array(x[0:200])
    y=array(y[0:200])
    y=y.reshape(len(y),1)
    d = 3
    thetaL = 0.001
    thetaU = 10.0
    nugget = 3
    #k = cov.covMatern([1,1,3]) + cov.covNoise([-1])
    k = cov.covSEard([1]*(d+1)) + cov.covNoise([-1])
    l = lik.likGauss([log(0.3)])
    m = mean.meanZero()   
    conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
    #conf.min_threshold = 100
    o = opt.Minimize(conf)
    conf.likRange = [(0,0.2)]
    #conf.min_threshold = 20
    conf.max_trails = 10
    conf.covRange = [(thetaL,thetaU)]*(d+1) + [(3,3)]
    #conf.covRange = [(thetaL,thetaU)]*2 + [(3,3)] + [(-2,1)]
    conf.meanRange = [] 
    i = inf.infExact() 
    
    output_scaler = preprocessing.StandardScaler(with_std=False).fit(y)
    #output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(y))
    #adjusted_training_fitness = output_scaler.transform(log(y))
    adjusted_training_fitness = output_scaler.transform(y)
    
    input_scaler = preprocessing.StandardScaler().fit(x)
    scaled_training_set = input_scaler.transform(x)
            
    gp.train(i,m,k,l,scaled_training_set,adjusted_training_fitness,o)
    z = array(z)
    out = gp.predict(i,m,k,l,scaled_training_set,adjusted_training_fitness, input_scaler.transform(z))
    y_output = output_scaler.inverse_transform(out[2])
    #y_output = exp(output_scaler.inverse_transform(out[2]))
    return y_output
           
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
    logsample = stats.lognorm.rvs(shape, scale=scale, loc=loc, size=1) # logsample ~ N(mu=10, sigma=3)
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
           
def my_interpolate():
    ### interpolate into this space                                                 
    min_prec = 4
    max_prec = 53

    min_cores = 1
    max_cores = 4

    min_freq = 80
    max_freq = 120

    ### the input data has the following points

    prec = range(4,54) ## up to 53
    cores = range(1,5) ## up to 5
    freq = [80,85,90,95,100,105,110,115,120]

    result_files=['results1.csv','results2.csv','results3.csv','results4.csv',]

    module_path = os.path.dirname(__file__)
    if module_path:
        module_path = module_path + "/"
    points = []
    values = [] 
    all_results = []
    for result_file in result_files:
        spamReader = csv.reader(open(module_path + result_file, 'rb'), delimiter=';', quotechar='"')
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_4 = row[3:]
            points.append([row_0,row_1,row_2])
            values.append([float(value) for value in row_4])
    grid_x, grid_y, grid_z =  mgrid[min_cores:max_cores:4j, min_prec:max_prec:50j,min_freq:max_freq:41j]
    
    points_n = array(points)  ### all the points from which we can interpolate
    values_n = array(values)  ### the data corresponding to those points
    
    ###
    ###
    ### EXIT CODES
    ###
    ###
    
    training_labels = array(values)[:,0].reshape(len(values),1)
    #print "Learning... n=" + str(training_labels.shape[0])
    n_neighbors = 5
    inputScaler = preprocessing.StandardScaler().fit(array(points))
    scaledSvcTrainingSet = inputScaler.transform(array(points))
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(scaledSvcTrainingSet, training_labels.reshape(-1))
    #print "Done..."
    #pdb.set_trace()    

    ### EXIT CODE CORRECTION
    def predict_code(x_in):
        x = inputScaler.transform(x_in)
        #x = x_in
        k = x_in[0]
        i = x_in[1]
        overmapping_code = array([2.0])
        if k == 1:
            return clf.predict(x)
        elif k == 2:
            if i > 29 :
                return overmapping_code
            return clf.predict(x)
        elif k == 3:
            if i > 14 :
                return overmapping_code
            return clf.predict(x)
        elif k == 4:
            if i > 8 :
                return overmapping_code
            return clf.predict(x) 
        else:
            print "ERROR"
    
    ###
    ###
    ### COST
    ###
    ### 
    
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
    
    ## for both 0 and 2 gaussian mixture
    ## for 0 log tail
    ## 0 lognormal
    ## 1 normal
    ## 2 and 4 normal
    ## 3 uniofrm
    
    for error_code in [2]:#;,1,2,3,4]:
        X =[] 
        Y =[]
        for point,value in zip(points,values):
            code = value[0]
            cost = value[2]
            if (code == error_code):# & (cost > 140):
                X.append(point)
                Y.append(cost)
        #pdb.set_trace()
        #gmm = mixture.GMM(n_components=2, covariance_type='full')
        #gmm.fit(Y)
        #pdb.set_trace()
        #plot_hist(array(Y))
    
    
    points_n = array(points)  ### all the points from which we can interpolate
    values_n = array(values)  ### the data corresponding to those points
    all_results.insert(0,griddata(points_n, values_n[:,0], (grid_x, grid_y, grid_z), method='nearest'))
    ratio = {4:2.661*5, 5:1.865*5, 6:1.401*5, 7: 1.189*5, 8:1.086*5, 9:1.039*5, 10:1.014*5, 11:1.006*5, 12:1.003*5, 13:1.001*5, 14:1.001*5, 15:1.0*5}
    Np = 727902.0
    Nc = 20.0
    Loutput = 4.0
    upper = Np * (Nc + Loutput)
    throughputs = deepcopy(all_results[0]) * 0.0 ### just to get the same shape... yes I am lazy lol
    error_codes = deepcopy(all_results[0]) * 0.0 ### just to get the same shape... yes I am lazy lol
    for i in range(min_prec,max_prec+1):
        for j in range(min_freq,max_freq+1):
            for k in range(min_cores,max_cores+1):
                time = 0.0
                if i in ratio:
                        time = float(upper)/float(j*100000.0*k) + Np*Nc*0.0000000577386*ratio[i]
                else:
                        time = float(upper)/float(j*100000.0*k) + Np*Nc*0.0000000577386*5.0
                throughput = Np/time
                throughput = throughput/1000.0
                throughputs[k - min_cores][i - min_prec][j - min_freq] = throughput
                error_codes[k - min_cores][i - min_prec][j - min_freq] = predict_code([k,i,j])
    all_results.append(all_results[0])
    all_results.append(throughputs)
    for i in range(min_prec,max_prec+1):
        for j in range(min_freq,max_freq+1):
            for k in range(min_cores,max_cores+1):
                if k == 1:
                    if i > 41 :
                        all_results[1][k - min_cores][i - min_prec][j - min_freq] = array([2.0])
                if k == 2:
                    if i > 29 :
                        all_results[1][k - min_cores][i - min_prec][j - min_freq] = array([2.0])
                elif k == 3:
                    if i > 14 :
                        all_results[1][k - min_cores][i - min_prec][j - min_freq] = array([2.0])
                elif k == 4:
                    if i > 8 :
                        all_results[1][k - min_cores][i - min_prec][j - min_freq] = array([2.0])
                    
    for i in range(1,len(row_4)):
        data_l = griddata(points_n, values_n[:,i], (grid_x, grid_y, grid_z), method='linear')
        data_n = griddata(points_n, values_n[:,i], (grid_x, grid_y, grid_z), method='nearest')
        data = where(isnan(data_l),data_n,data_l)
        all_results.append(data + normal(0, data.mean()/100.0, data.shape))
                    
    
                    
    return all_results
                
if __name__ == '__main__':

    my_interpolate()
    '''
    import itertools 
    maxEI = 0.0
    maxEIcord = None
    space_def = []
    counter = 0
    alldata = getAllData()
    #xx = (2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 2.0)
    #print fitnessFunc(xx,None)[0][0][0]
    '''
    '''
    for d in designSpace:
        space_def.append(arange(d["min"],d["max"]+1.0,d["step"]))
    print space_def
    results = []
    for z in itertools.product(*space_def):
        #print str(z)
        counter = counter + 1
        EI = fitnessFunc(z,{})[0][0]
        #print "--" + str(EI)
        code = fitnessFunc(z,{})[0][1]
        print str(z) + " " +  str(EI)
        if (counter % 10000 == 0) :
            print str(counter) + " " +  str(maxEIcord) + " " +  str(maxEI)
        
        if (maxEI < EI) and (code==0): ## no need for None checking
            maxEI = EI
            maxEIcord = z
    ### 
    print "DONE!"
    print maxEIcord
    print maxEI
    print fitnessFunc([4,13,120],{})[0][0]
    print fitnessFunc([3,12,120],{})[0][0]
    print fitnessFunc([3,14,120],{})[0][0]
    print fitnessFunc([2,14,120],{})[0][0]
    print fitnessFunc([4,14,120],{})[0][0]
    print termCond(fitnessFunc([4,13,120],{})[0][0])
    
    '''
    
