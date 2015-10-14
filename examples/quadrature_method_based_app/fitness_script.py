<<<<<<< HEAD
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
import pdb 

enableTracebacks = True
from numpy import * 
initMin = -1
initMax = 1

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
    
platformMax3 = True
doCores = True
doMw = True
doDf = True
designSpace = []
maxError = 0.1
errorCorrection=True
fixMax4Bug = True
return_execution_time = True

if platformMax3:
    if doCores:
        designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
    ## always do mw
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})

    if doDf:
        designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"s"})
else: ## Maia
    if doCores:
        designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
    ## always do mw
    designSpace.append({"min":32.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})

    if doDf:
        designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"s"})
              
              
## if  [ 14.  11.   4.]   0.000275081632653

error_labels = {1:'Overmap',0:'Valid',2:'Inaccuracy'}
#error_labels = {0:'Valid',3:'Inaccuracy'}
    
def transferValid(part):
    if not platformMax3:  ##
        return part[1] >  31.0
    else: ## we are going to maia
        return True
    
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
    return "fF_ansonE" + str(maxError) + "Th" + str(return_execution_time)
    
if doCores:
    always_valid = [1.,53.,32.]
else:
    always_valid = [53.,32.]

    
if return_execution_time: ## Throughput case
    if platformMax3:
        maxvalue = 300.0
        minVal = 0.0390423230495
        
        worst_value = 0.0
        optVal = {0.1:75.37,0.01:1.44,0.05:11.19,0.001:0.469 }[maxError]
        maxVal = maxvalue
        def termCond(best):
            global optVal
            #print str(best) + " " + str(optVal)
            return best > optVal
        rotate = True
        def get_z_axis_name():
            return "Throughput ($\phi_{int}$)"
    else:
        maxvalue = 300.0
        minVal = 0.0390423230495
        
        worst_value = 0.0
        optVal = {0.1:43.,0.01:1.88,0.05:11.19,0.001:0.62 }[maxError]
        maxVal = maxvalue
        def termCond(best):
            global optVal
            #print str(best) + " " + str(optVal)
            return best > optVal
        rotate = True
        def get_z_axis_name():
            return "Throughput ($\phi_{int}$)"
            
else: ## Energy case
    if platformMax3:
        maxvalue = 375.3990152
        minVal = 0.05785136
        maxVal = 360.0
        worst_value = 360.0
        rotate = False
        optVal = {0.1:0.209,0.01:10.7,0.05:12.24277734,0.001:33.0 }[maxError]
    else:
        print "No data for Maia for power"
        raise
    
    def termCond(best):
        global optVal
        #print str(best) + " " + str(optVal)
        return best < optVal

    def get_z_axis_name():
        return "$\mathrm{W}$/$\phi_{int}$"

def alwaysCorrect():
    if doCores:
        return array([1.0,53.0,32.0])
    else :
        return array([53.0,32.0]) 
    
def fitnessFunc(particle, state):
    allData, power_data = getAllData()
    # Dimensions dynamically rescalled
    ############Dimensions
    if doCores:
        cores = int(particle[0])
        mw = int(particle[1])
    else:
        cores = 1.0
        mw = int(particle[0])
                
    if doDf:
        df = int(particle[-1])
        
        try:
            error = allData[11][mw][df][1]
        except:
            if (not platformMax3): ## maia does not build for some
                if fixMax4Bug:
                    cost, isValid, state_kupa = getCost(df, float(mw),float(cores), state)
                    delta = (mw - 45.0)  / 8.0 ## linear interpolation
                    error = (1.0  - delta) * allData[11][45][df][1]  + delta * allData[11][53][df][1]
                else:
                    cost, isValid, state = getCost(df, float(mw),float(cores), state)
                    return ((array([maxvalue]), array([3]),array([1]), cost) , state) ## maia does not build for some precisions.. a bug
        
        if not doCores:
            cores = allData[11][mw][df][0]
            
        
        #accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
                
        if(errorCorrection):
            ###error correction
            #for dff,acc in enumerate(accuracy):
            #    if ((32-dff) > df) and  (acc > error):
            #        error = acc 
                    
            for mww in range(mw,54):
                try:
                    accuracy = array([allData[11][mww][i+1][1] for i in xrange(32)])[::-1]
                    for dff,acc in enumerate(accuracy):
                        if ((32-dff) >= df) and  (acc > error):
                            error = acc 
                except: ## platformMax3 > 46
                    pass

            #print "mww ",mww," dff ",(32-dff)," acc ",acc," df ",df," mw ",mw," ",error
            #print " df ",df," mw ",mw," ",error

    else:
        accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
                
        error = accuracy[0]
        df = 0
        for dff,acc in enumerate(accuracy):
            if acc > maxError:
                break
            error = acc
            df = 32 - dff  
        
    
    #######################
    if return_execution_time:
        #cores = 1.0
        if (not platformMax3) and fixMax4Bug and (mw < 53 and mw > 45): ## maia does not build for some
            delta = (mw - 45.0)  / 8.0 ## linear interpolation
            executionTime = (1.0  - delta) * allData[11][45][df][2]  + delta * allData[11][53][df][2]
            executionTime = array([cores/executionTime])
            max_cores = int((1.0  - delta) * allData[11][45][df][0]  + delta * allData[11][53][df][0])
        else:
            executionTime = array([cores/allData[11][mw][df][2]])
            max_cores = allData[11][mw][df][0] 
        #executionTime = array([error])
        #print " df ",df," mw ",mw," ",error," ",allData[11][mw][df][2]
        cost, isValid, state = getCost(df, float(mw),float(cores), state) ## we need to cast to float
        ### overmapping error
        
        if max_cores < cores or isValid == 1:
            return ((array([maxvalue]), array([1]),array([1]), cost) , state)
        
        if error > maxError:
            return ((executionTime, array([2]),array([0]), cost) , state)##!!!! zmien na 0.0 
        ### accuracy error
        ### ok values execution time
        return ((executionTime, array([0]),array([0]), cost), state)
    else:
        #cores = 1.0
        try:
            power =  array([power_data[11][mw][cores][df]])
            execution_time = allData[11][mw][df][2]
            power = (execution_time * power) / float(cores)
        except Exception,e:
            #print str(e)
            power = maxvalue
            
        cost, isValid, state = getCost(df, float(mw),float(cores), state) ## we need to cast to float
            
        ### overmapping error
        if allData[11][mw][df][0] < cores or isValid == 1:
            return ((array([maxvalue]), array([1]),array([1]), array([1200])) , state) ## 20 minutes
            
        if error > maxError:
            return ((power, array([2]),array([0]), cost) , state)##!!!! zmien na 0.0 
        ### accuracy error
        
        ### ok values execution time
        return ((power, array([0]),array([0]), cost), state)
    

global allData, costModel, costModelInputScaler, costModelOutputScaler, power_data, cost_data
allData=None
costModel=None
costModelInputScaler=None
costModelOutputScaler=None
def getAllData():
    global allData, costModel, costModelInputScaler, costModelOutputScaler, power_data, cost_data
    if not allData:
        module_path = os.path.dirname(__file__)
        if module_path:
            module_path = module_path + "/"
        
        ## COST MODEL
        if platformMax3:
            spamReader = csv.reader(open(module_path + 'time_results.csv', 'rb'), delimiter=';', quotechar='"')
        else:
            spamReader = csv.reader(open(module_path + 'time_results2.csv', 'rb'), delimiter=',', quotechar='"') # TODO

        x = []
        y = []
        cost_data = {}
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_3 = float(row[3])
            
            try:
                try:
                    cost_data[row_0][row_1][row_2]= row_3
                except:
                    cost_data[row_0][row_1] = {row_2:row_3}
            except:
                cost_data[row_0] = {row_1:{row_2:row_3}}
        
        '''
            x.append([float(row[1]),float(row[2])])
            y.append([float(row[3]) + random.random()])
        x = array(x)
        y = array(y)
        
        input_scaler = preprocessing.StandardScaler().fit(x)
        scaled_training_set = input_scaler.transform(x)

                # Scale training data
        output_scaler = preprocessing.StandardScaler(with_std=False).fit(y)
        adjusted_training_fitness = output_scaler.transform(y)
        
        regr = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                         thetaL=1e-5, thetaU=3,
                         random_start=400)
        regr.fit(scaled_training_set, adjusted_training_fitness)
        costModel = regr
        costModelInputScaler = input_scaler
        costModelOutputScaler = output_scaler
        '''
        ## cores, accuracy, exeuction time
        if platformMax3:
            spamReader = csv.reader(open(module_path + 'AnsonCores.csv', 'rb'), delimiter=';', quotechar='"')
        else:
            spamReader = csv.reader(open(module_path + 'AnsonCores2.csv', 'rb'), delimiter=';', quotechar='"')
        cores = {11:{}}
        for row in spamReader:
            cores[11][int(row[1])] = int(row[0])

        maxcores = cores
        if platformMax3:
            spamReader = csv.reader(open(module_path + 'AnsonExec.csv', 'rb'), delimiter=';', quotechar='"')
        else:   
            spamReader = csv.reader(open(module_path + 'AnsonExec2.csv', 'rb'), delimiter=';', quotechar='"')
        
        allData = {}
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_3 = float(row[3])
            row_4 = float(row[4])
            data = [cores[row_0][row_1],row_3,row_4,{}]
            
            try:
                try:
                    allData[row_0][row_1][row_2] = data
                except:
                    allData[row_0][row_1] = {row_2:data}
            except:
                allData[row_0] = {row_1:{row_2:data}}
                
        
        power_data = {}
        if platformMax3:
            spamReader = csv.reader(open(module_path + 'AnsonModPower.csv', 'rb'), delimiter=';', quotechar='"')
            for row in spamReader:
                row_0 = int(row[0]) ##
                row_1 = int(row[1])
                row_2 = int(row[2]) 
                row_3 = int(row[3]) 
                row_4 = float(row[4]) ## for given cores
                try:
                    try:
                        try:
                            power_data[row_0][row_1][row_2][row_3] = row_4
                        except:
                            power_data[row_0][row_1][row_2]= {row_3:row_4}
                    except:
                        power_data[row_0][row_1] = {row_2:{row_3:row_4}}
                except:
                    power_data[row_0] = {row_1:{row_2:{row_3:row_4}}}
                                
        #spamReader.close()
    #print allData
    return allData, power_data
#print allData
#print cores
   
    
##add state saving, we use p and thread id 
def getCost(df, wF, cores, bit_stream_repo):
    global costModel, costModelInputScaler, costModelOutputScaler, cost_data
    bit_stream_repo_copy = deepcopy(bit_stream_repo) 
    if bit_stream_repo is None:
        bit_stream_repo_copy = {}
    isValid = 0
    if bit_stream_repo_copy.has_key((wF,cores)): ## bit_stream evalauted
        return array([df*5.0*(0.5 + random.random())]), isValid, bit_stream_repo_copy
    else:
        bit_stream_repo_copy[(wF,cores)] = True
        try:
            if (not platformMax3) and (wF < 53 and wF > 45): ## maia does not build for some
                delta = (wF - 45.0)  / 8.0 ## linear interpolation
                cost = (1.0  - delta) * cost_data[11][45][cores] + delta * cost_data[11][53][cores]
                cost = array([cost])
            else:
                cost = array([cost_data[11][wF][cores]])#+ df*5.0*(0.8 + random.random()/5.0)])
        except Exception, e:
            cost = array([1200.0+200.0*random.random()])
            isValid = 1
        return cost , isValid, bit_stream_repo_copy  ##software cost, very small fraction and linear
                
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
                
if __name__ == '__main__':

    import itertools 
    maxEI = 0.0
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
        code = fitnessFunc(z,{})[0][1]
        cost = fitnessFunc(z,{})[0][3]
        if cost < 200.0 and code == 0:
            pdb.set_trace()
            print "huj"
            print code
            print cost 
            print z
            print "pizda"
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
        EI = fitnessFunc(z,{})[0][0]
        
        #if (counter % 10 == 0) :
        #print str(z) + " " +  str(maxEIcord) + " " +  str(maxEI) + " " + str(code)
        
        if (maxEI < EI) and (code==0): ## no need for None checking
            maxEI = EI
            maxEIcord = z
    ### 
    print "DONE!"
    print maxEIcord
    print maxEI
    
=======
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
initMin = -1
initMax = 1

#aecc or execution time -- when error correction applied
#### [ 15.   6.]   75.3885409418 0.1  13.0, 15.0, 6.0
#### [ 20.  11.]   11.1996923818 0.05
#### [ 19.  22.]   1.44387061283 0.01
#### [ 20.  32.]   0.469583069715 0.001

#### without error correction
#### [ 14.  11.   4.]   259.663179761 all
#### [ 13.  15.   7.]   46.8546137378 0.001

cost_maxVal = 15000.0
cost_minVal = 0.0
    
doCores = True
doMw = True
doDf = True
rotate = True
designSpace = []
maxError = 0.1
errorCorrection=True

return_execution_time = True

minVal = 0.0
maxVal = 300.0
worst_value = 0.0
optVal = {0.1:75.37,0.01:1.44,0.05:11.19,0.001:0.469 }[maxError]

if doCores:
    designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
## always do mw
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"})

if doDf:
    designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0, "set":"s"})
              
maxvalue = 0.0
## if  [ 14.  11.   4.]   0.000275081632653

error_labels = {0:'Valid',1:'Overmap',3:'Inaccuracy'}
#error_labels = {0:'Valid',3:'Inaccuracy'}

def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    
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

def termCond(best):
    global optVal
    print str(best) + " " + str(optVal)
    return best > optVal

def name():
    return "anson_" + str(maxError) + "_doCores" + str(doCores) + "_doDf" + str(doDf) + "_errorCorrection" + str(errorCorrection) 
    
def alwaysCorrect():
    if doCores:
        return array([1.0,53.0,32.0])
    else :
        return array([53.0,32.0]) 
    
def fitnessFunc(particle, state):
    allData, power_data = getAllData()
    # Dimensions dynamically rescalled
    ############Dimensions
    if doCores:
        cores = int(particle[0])
        mw = int(particle[1])
    else:
        cores = 1.0
        mw = int(particle[0])
                
    if doDf:
        df = int(particle[-1])
        if not doCores:
            cores = allData[11][mw][df][0]
            
        error = allData[11][mw][df][1]
        #accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
                
        if(errorCorrection):
            ###error correction
            #for dff,acc in enumerate(accuracy):
            #    if ((32-dff) > df) and  (acc > error):
            #        error = acc 
                    
            for mww in range(mw,54):
                accuracy = array([allData[11][mww][i+1][1] for i in xrange(32)])[::-1]
                for dff,acc in enumerate(accuracy):
                    if ((32-dff) >= df) and  (acc > error):
                        error = acc 
            #print "mww ",mww," dff ",(32-dff)," acc ",acc," df ",df," mw ",mw," ",error
            #print " df ",df," mw ",mw," ",error
        
    else:
        accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
        error = accuracy[0]
        df = 0
        for dff,acc in enumerate(accuracy):
            if acc > maxError:
                break
            error = acc
            df = 32 - dff  
        
    frequency = 100
    #######################
    if return_execution_time:
        #cores = 1.0
        executionTime = array([cores/allData[11][mw][df][2]])
        #executionTime = array([error])
        #print " df ",df," mw ",mw," ",error," ",allData[11][mw][df][2]
        cost, state = getCost(df, float(mw),float(cores), state) ## we need to cast to float
        if error > maxError:
            return ((executionTime, array([3]),array([0]), cost) , state)##!!!! zmien na 0.0 
        ### accuracy error
            
        ### overmapping error
        if allData[11][mw][df][0] < cores :
            return ((array([maxvalue]), array([1]),array([1]), cost) , state)
        
        ### ok values execution time
        return ((executionTime, array([0]),array([0]), cost), state)
    else:
        #cores = 1.0
        power =  array([powerData[11][mw][df][cores]])
        #executionTime = array([error])
        #print " df ",df," mw ",mw," ",error," ",allData[11][mw][df][2]
        cost, state = getCost(df, float(mw),float(cores), state) ## we need to cast to float
        if error > maxError:
            return ((power, array([3]),array([0]), cost) , state)##!!!! zmien na 0.0 
        ### accuracy error
            
        ### overmapping error
        if allData[11][mw][df][0] < cores :
            return ((array([maxvalue]), array([1]),array([1]), cost) , state)
        
        ### ok values execution time
        return ((power, array([0]),array([0]), cost), state)
    
def calcMax():    
    if len(designSpace)==2:
        # make up data.
        x,y = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j]
        x=reshape(x,-1)
        y=reshape(y,-1)       
        z = array([[a,b] for (a,b) in zip(x,y)])
    else:
        x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
        x=reshape(x,-1)
        y=reshape(y,-1)
        v=reshape(v,-1)
        z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
        
    zReal = array([fitnessFunc(a)[0][0] for a in z])
    zRealClass = array([fitnessFunc(a)[1][0] for a in z])
    minn = argmin(zReal)
    filteredminn = []
    filteredZ=[]
    for i,zreal in enumerate(zReal):
        if zRealClass[i]==0.0:
            filteredminn.append(zreal)
            filteredZ.append(z[i])
            
    if filteredminn:
        doFor=5
        argsortedFiltered = argsort(filteredminn,axis=0)
        print "[returnMaxS2] ====================================="
        for kk in range(0,doFor):
            minn = argsortedFiltered[kk]
            maxx = argsortedFiltered[-(kk+1)]
            print "[returnMaxS2] Real min :",minn," ",filteredZ[minn]," ",filteredminn[minn]
            print "[returnMaxS2] Real max :",maxx," ",filteredZ[maxx]," ",filteredminn[maxx]

        print "[returnMaxS2] ====================================="

def calcMaxs():    
    global maxError
    for maxError in [0.001]:
        print "maxError: ",maxError
        calcMax()
        
def changeParams(params):
    global designSpace,optVal,errorCorrection,maxError,doCores,maxVal
    try:
        doCores = params["doCores"]
    except Exception,e:
        print "[changeParams] ",e
        pass
    
    try:
        maxError = params["maxError"]
    except Exception,e:
        print "[changeParams] ",e
        pass
        
    try:
        errorCorrection = params["errorCorrection"]
    except Exception,e:
        print "[changeParams] ",e
        pass

    optVal = {0.1:75.37,0.01:1.44,0.05:11.1,0.001:0.469 }[maxError]
    maxVal = {0.1:260.0,0.01:5.0,0.05:25.0,0.001:5.0 }[maxError]

    designSpace=[]
    if doCores:
        designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0})
    ## always do mw
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0})

    if doDf:
        designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0})
                  
def trialgen():
    params = {}
    global maxError,optVal,errorCorrection,doCores
    for maxError in [0.1]:
        for errorCorrection in [True]:
            for doCores in [False]:
                params["maxError"] = maxError
                params["errorCorrection"] = errorCorrection
                params["doCores"] = doCores
                changeParams(params)
                yield params
               
#python data dict creator
#
#AnsonExec.csv
#text delimiter " field delimiter  ,
#
#to get number of cores
#AnsonCores.csv


global allData, costModel, costModelInputScaler, costModelOutputScaler, power_data
allData=None
costModel=None
costModelInputScaler=None
costModelOutputScaler=None
def getAllData():
    global allData, costModel, costModelInputScaler, costModelOutputScaler, power_data
    if not allData or not costModel:
        module_path = os.path.dirname(__file__)
        ## COST MODEL
        spamReader = csv.reader(open(module_path + '/time_results.csv', 'rb'), delimiter=';', quotechar='"')

        x = []
        y = []
        for row in spamReader:
            x.append([float(row[1]),float(row[2])])
            y.append([float(row[3]) + random.random()])
        x = array(x)
        y = array(y)
        input_scaler = preprocessing.StandardScaler().fit(x)
        scaled_training_set = input_scaler.transform(x)

                # Scale training data
        output_scaler = preprocessing.StandardScaler(with_std=False).fit(y)
        adjusted_training_fitness = output_scaler.transform(y)
        
        regr = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                         thetaL=1e-5, thetaU=3,
                         random_start=400)
        regr.fit(scaled_training_set, adjusted_training_fitness)
        costModel = regr
        costModelInputScaler = input_scaler
        costModelOutputScaler = output_scaler
        
        ## cores, accuracy, exeuction time
        spamReader = csv.reader(open(module_path + '/AnsonCores.csv', 'rb'), delimiter=',', quotechar='"')
        cores = {11:{}}
        for row in spamReader:
            cores[11][int(row[1])] = int(row[0])

        maxcores = cores
        spamReader = csv.reader(open(module_path + '/AnsonExec.csv', 'rb'), delimiter=';', quotechar='"')

        allData = {}
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_3 = float(row[3])
            row_4 = float(row[4])
            data = [cores[row_0][row_1],row_3,row_4,{}]
            
            try:
                try:
                    allData[row_0][row_1][row_2] = data
                except:
                    allData[row_0][row_1] = {row_2:data}
            except:
                allData[row_0] = {row_1:{row_2:data}}
                
        power_data = {}
        if False:
            spamReader = csv.reader(open(module_path + '/AnsonPower.csv', 'rb'), delimiter=';', quotechar='"')
            for row in spamReader:
                row_0 = int(row[0]) ##
                row_1 = int(row[1])
                row_2 = int(row[2]) 
                row_3 = int(row[3]) 
                row_4 = float(row[4]) ## for given cores
                power_data[row_0][row_1][row_2][row_3] = row_4
        #spamReader.close()
    #print allData
    return allData, power_data
#print allData
#print cores
    
##add state saving, we use p and thread id 
def getCost(df, wF, cores, bit_stream_repo):
    global costModel, costModelInputScaler, costModelOutputScaler
    bit_stream_repo_copy = deepcopy(bit_stream_repo) 
    if bit_stream_repo is None:
        bit_stream_repo_copy = {}
       
    if bit_stream_repo_copy.has_key((wF,cores)): ## bit_stream evalauted
        return array([df*5.0*(0.5 + random.random())]), bit_stream_repo_copy
    else:
        bit_stream_repo_copy[(wF,cores)] = True
        return costModelOutputScaler.inverse_transform(costModel.predict(costModelInputScaler.transform(array([[wF,cores]])))) + (df*5.0*(0.8 + random.random()/5.0)), bit_stream_repo_copy  ##software cost, very small fraction and linear
                
if __name__ == '__main__':

    import itertools 
    maxEI = 0.0
    maxEIcord = None
    space_def = []
    counter = 0
    '''
    xx = (2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 2.0)
    print fitnessFunc(xx,None)[0][0][0]
    '''
    for d in designSpace:
        space_def.append(arange(d["min"],d["max"]+1.0,d["step"]))
    print space_def
    results = []
    for z in itertools.product(*space_def):
        counter = counter + 1
        EI = fitnessFunc(z,{})[0][0]
        code = fitnessFunc(z,{})[0][1]
        if (counter % 10000 == 0) :
            print str(counter) + " " +  str(maxEIcord) + " " +  str(maxEI)
        
        if (maxEI < EI) and (code==0): ## no need for None checking
            maxEI = EI
            maxEIcord = z
    ### 
    print "DONE!"
    print maxEIcord
    print maxEI
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
