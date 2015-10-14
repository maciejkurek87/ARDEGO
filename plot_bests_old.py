from matplotlib.pyplot import *
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
from scipy.stats import norm
import math 
import pdb
import os
from matplotlib.font_manager import FontProperties    
from sklearn import preprocessing
import csv

save = True
fontsize = 26
smallfontsize = 22
annotatefontsize = 18

import pyGP_OO
from pyGP_OO.Core import *
from pyGP_OO.Valid import valid

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

def load_data(folder_path):    
    files = os.listdir(folder_path)
    always_valid = True
    X = []
    Y = []
    for file in files:
        spamReader = csv.reader(open(folder_path + "/" + file, 'rb'), delimiter=';', quotechar='"')
        counter = -1
        for row in spamReader:
            if counter > 0:
                X.append(float(row[0]))
                Y.append(float(row[1]))
            else:
                LAST_X=float(row[0])
                LAST_Y=float(row[1])
            counter = counter + 1
            
    X.append(LAST_X)
    Y.append(LAST_Y)
    return X,Y
    
def gp_predict(x,y,max_x):
    d = 1
    thetaL = 6.0
    thetaU = 10.0
    nugget = 3
    
    m = mean.meanZero()
    k = cov.covSEard([1]*(d+1)) + cov.covNoise([-1])
    
    #k = cov.covMatern([1,1,3])
    l = lik.likGauss([np.log(0.3)])
       
    
    conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
    conf.max_trails = 20
    #conf.min_threshold = 100
    o = opt.Minimize(conf)
    conf.likRange = [(0,0.2)]
    #conf.min_threshold = 20
    conf.max_trails = 100
    conf.covRange = [(thetaL,thetaU)]*(d+1) + [(-2,1)]
    #conf.covRange = [(thetaL,thetaU)]*2 + [(3,3)] + [(-2,1)]
    conf.meanRange = [] 
    i = inf.infExact()
    
    output_scaler = preprocessing.StandardScaler(with_std=False).fit(np.log(y))
    adjusted_training_fitness = output_scaler.transform(np.log(y))
    
    input_scaler = preprocessing.StandardScaler().fit(x)
    scaled_training_set = input_scaler.transform(x)
            
    gp.train(i,m,k,l,scaled_training_set,adjusted_training_fitness,o)
    
    z = np.array([np.linspace(0.0,max_x,101)]).T
    out = gp.predict(i,m,k,l,scaled_training_set,adjusted_training_fitness, input_scaler.transform(z))
   
    return z,np.exp(output_scaler.inverse_transform(out[2])), np.exp(out[3])
    
def change_tick_font(ax, label2=False):
	if label2:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label2.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label2.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)
	else:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label1.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label1.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)

def change_tick_font(ax, label2=False):
	if label2:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label2.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label2.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)
	else:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label1.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label1.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)

from matplotlib.lines import Line2D
markers = []
for marker in Line2D.markers:
    try:
        if len(marker) == 1 and marker != ' ':
            markers.append(marker)
    except TypeError:
        pass
        
if 1: ## throughput
    fig = figure(figsize=(9,5))
    ax = fig.add_subplot(111)#
    symbols =['-o','-x','-^','-s','-D','-*']
    lines = []    
    ym = []
    lineNum = 0
    
    folder_paths = ["/homes/mk306/bests_dump_folder"]
    legend = ["test"]
    
    for i,folder_path in enumerate(folder_paths):
        X,Y = load_data(folder_path)
        X = np.array(X).reshape(-1,1)
        Y = np.array(Y).reshape(-1,1)
        print X
        print Y
        Xs,Ys,S2s = gp_predict(X,Y,np.max(X))
        Xs = np.array(Xs)
        Ys = np.array(Ys)
        S2s = np.array(S2s)
        ax.plot(X, Y, 'b+', linewidth = 3.0)
        ax.plot(Xs, Ys, 'r-', linewidth = 3.0, markersize = 10.0)
        
        xss  = np.reshape(Xs,(Xs.shape[0],))
        ymm  = np.reshape(Ys,(Ys.shape[0],))
        ys22 = np.reshape(Ys,(S2s.shape[0],))
        
        ax.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
        #lines.append(line)
        lineNum += 1
        ym = []
    #l = ax.legend((lines,legend),  title =  'Applications', loc="upper left",prop={'size':annotatefontsize-1}, frameon=False)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    #setp(l.get_title(), fontsize=smallfontsize)
    ax.set_xlabel('optimization time (seconds)', fontsize = fontsize)
    ax.set_ylabel('Throughput', fontsize = fontsize)
    ax.grid(False)
    change_tick_font(ax)
    if save:
        fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.21)
        fig.savefig("optimization_plot")
    else:
        show()
    close(fig)

