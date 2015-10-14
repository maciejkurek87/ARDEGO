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
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d, Axes3D

save = True
fontsize = 26
smallfontsize = 22
annotatefontsize = 22


rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
debug = False

stochastic = False

if stochastic:
    folder_path = "/homes/mk306/all_results_stochastic.txt"#"/homes/mk306/all_results_stochastic_100.txt"
    cost_folder_path = "/homes/mk306/MLO/examples/stochastic/all_results_stochastic_builds.csv"
else:
    folder_path = "/homes/mk306/all_results_robot_all.txt"
    cost_folder_path = "/homes/mk306/MLO/examples/robot/all_results_robots_builds.csv"
legend = "kurwaa"
title = "PQ Application Throughput Optimization"    
plot_name = "optimization_plot_pq"
ylabel = r"Particles/second"
maximization = True
unit =  "min" ## sec or min (data is displayed in hours, thats the data unit)
set_x_log = False


## bash command to copy data from a folder
##define folder variable
##counter=0; for f in `ls $folder`; do ((counter=$counter+1));echo $counter ; cp $folder$f/best_dump.csv best_dump_$counter.csv; done

def load_data(folder_path):    
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=' ', quotechar='"')
    if stochastic: 
        execution_time = np.zeros(shape=(5,16,67))
        accuracy = np.zeros(shape=(5,16,67))
        count = np.zeros(shape=(5,16,67))
    else:
        execution_time = np.zeros(shape=(5,16,103))
        accuracy = np.zeros(shape=(5,16,103))
        count = np.zeros(shape=(5,16,103))
    for row in spamReader:
        if len(row) == 7:
            try:
                if stochastic: 
                    if not (float(row[6]) == 0.0 or float(row[5]) == 0.0 or float(row[5]) > 12000.0):
                        execution_time[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60] = float(row[5]) + execution_time[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60]
                        accuracy[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60] = float(row[6]) + accuracy[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60]
                        count[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60] = count[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-96)/60] + 1
                else:
                    if not (float(row[6]) == 0.0 or float(row[5]) == 0.0):
                        execution_time[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60] = float(row[5]) + execution_time[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60]
                        accuracy[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60] = float(row[6]) + accuracy[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60]
                        count[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60] = count[int(row[0])-7][(int(row[3])-10)/2][(int(row[4])-2048)/60] + 1
            except:
                pdb.set_trace()
    return execution_time, accuracy, count
    
def load_cost(folder_path):
    spamReader = csv.reader(open(folder_path, 'rb'), delimiter=',', quotechar='"')
    if stochastic: 
        cost = np.zeros(shape=(30,5,16,67))
    else:
        cost = np.zeros(shape=(4,5,16,103))
    for row in spamReader:
        try:
            if stochastic: 
                for i in range(67):
                    cost[int(row[1])-1][int(row[0])-7][(int(row[2])-10)/2][i] = float(row[5]) 
            else:
                for i in range(103):
                    cost[int(row[1])-1][int(row[0])-7][(int(row[2])-10)/2][i] = float(row[5])
            
        except:
            pdb.set_trace()
    return cost
    
def lin_predict(x,y, min_XX, max_XX):
    def f_func(z):
        counter = 0
        if maximization:
            y_max = np.max(y)
        else:
            y_max = np.min(y)
        while counter < len(x) -1: 
            if (z >= x[counter]) & (z < x[counter+1]):
                return (y[counter+1]-y[counter])/(x[counter+1]-x[counter])*(z-x[counter]) + y[counter]
            counter = counter + 1
        return y_max

    z = np.array([np.linspace(min_XX,max_XX,101)]).T
    output = [f_func(zz) for zz in z]
    output[0] = y[0]
    return z,output
        
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
        
symbols =['--','-+','-.','-0','-D','-*']
lines = []    
ym = []
lineNum = 0    

execution_time, accuracy, count = load_data(folder_path)
cost = load_cost(cost_folder_path)
pdb.set_trace()
#import pdb
#pdb.set_trace()
execution_time = execution_time / count
accuracy = accuracy / count
if stochastic:
    y = np.arange(10, 40 + 2, 2)
    x = np.arange(96, 4100, 60)
else:
    y = np.arange(10, 40 + 2, 2)
    x = np.arange(2048, 8220, 60)

x, y = np.meshgrid(x, y)
x = np.reshape(x, -1)
y = np.reshape(y, -1)
z = np.array([[a, b] for (a, b) in zip(x,y)])

if stochastic:
    yi = np.linspace(10 - 0.01, 40 + 0.01, 100)
    xi = np.linspace(96 - 0.01, 4096 + 0.01, 100)
else:
    yi = np.linspace(10 - 0.01, 40 + 0.01, 100)
    xi = np.linspace(2400 - 0.01, 8160 + 0.01, 100)
    

fig = figure(figsize=(9,5))
ax = fig.add_subplot(111, projection='3d')#
X, Y = np.meshgrid(xi, yi)
zi = griddata((x, y), accuracy[0].reshape(-1,), (xi[None, :], yi[:, None]), method='linear')
surf = ax.plot_surface(X, Y, zi, rstride=1, cstride=1, cmap=cm.jet,linewidth=1)
    
if maximization:
    loc = "lower right"
else:
    loc = "upper right"

#l = ax.legend(legend, loc=loc,prop={'size':annotatefontsize-1}, frameon=False)
#setp(l.get_title(), fontsize=smallfontsize)
ax.set_xlabel('Optimization time (hours)', fontsize = fontsize)
ax.set_ylabel(ylabel, fontsize = fontsize)
ax.set_title(title, fontsize = fontsize, y=1.08)
#ax.grid(True)
change_tick_font(ax)
if save:
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.21)
    fig.savefig("robotcost" +  plot_name, bbox_inches='tight')
else:
    show()

close(fig)

