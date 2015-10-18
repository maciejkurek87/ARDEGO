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
from matplotlib.ticker import MaxNLocator

save = True
fontsize = 24
smallfontsize = 22
annotatefontsize = 22


rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
debug = False

stochastic = False

if stochastic:
    folder_path = "/media/sf_Dropbox/all_results_stochastic.txt"#"/homes/mk306/all_results_stochastic_100.txt"
else:
    folder_path = "/media/sf_Dropbox/all_results_robot_all.txt"
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
        

fig = figure(figsize=(9,5))
ax = fig.add_subplot(111, projection='3d')#
symbols =['--','-+','-.','-0','-D','-*']
lines = []    
ym = []
lineNum = 0    

execution_time, accuracy, count = load_data(folder_path)
#import pdb
#pdb.set_trace()
execution_time = execution_time / count
accuracy = accuracy / count
if stochastic:
    y = np.arange(10, 40 + 2, 2)
    x = np.arange(96, 4060, 60)
else:
    y = np.arange(10, 40 + 2, 2)
    x = np.arange(2048, 8220, 60)

x, y = np.meshgrid(x, y)
x = np.reshape(x, -1)
y = np.reshape(y, -1)
z = np.array([[a, b] for (a, b) in zip(x,y)])

if stochastic:
    yi = np.linspace(11 - 0.01, 39 + 0.01, 100)
    xi = np.linspace(100 - 0.01, 4050 + 0.01, 100)
else:
    yi = np.linspace(11 - 0.01, 39 + 0.01, 100)
    xi = np.linspace(2050 - 0.01, 8150 + 0.01, 100)
    

for plot_name, dataa in [("accuracy",accuracy),("execution_time",execution_time)]:
    X, Y = np.meshgrid(xi, yi)
    
    if stochastic:
        zi = griddata((x, y), dataa[1].reshape(-1,), (xi[None, :], yi[:, None]), method='linear')
    else:
        zi = griddata((x, y), dataa[0].reshape(-1,), (xi[None, :], yi[:, None]), method='linear')
        
    #np.place(zi, np.isnan(zi), np.nanmean(zi)*0.8)
    ax.set_ylabel("\n" + r'   Mantissa Width', fontsize = fontsize-2, rotation=37,linespacing=1.2  )
    #ax.xaxis.set_rotate_label(False)
    ax.set_xlabel("\n" + r'Number of Particles', fontsize = fontsize-2, rotation=-47,linespacing=1.2)
    #ax.zaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    if plot_name == "execution_time" :
        ax.set_zlabel("\n" + r'   Execution time ($\mu s$)', fontsize = fontsize-2, rotation=94, linespacing=-3.9 )
    else:
        ax.set_zlabel("\n" + r'Accuracy', fontsize = fontsize-2, rotation=94, linespacing=-1.8 )
    #pdb.set_trace()
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize - 6) 
    for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize- 6) 
    for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize- 6) 

    
    surf = ax.plot_surface(X, Y, zi, rstride=2, cstride=2, cmap=cm.jet,linewidth=0.5)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=29., azim=-116)
    ax.tick_params(direction='out', pad=100)
    if stochastic:
        ax.zaxis.set_major_locator( MaxNLocator(3) )
    else:
        ax.zaxis.set_major_locator( MaxNLocator(3) )
    ax.xaxis.set_major_locator( MaxNLocator(3) )
    ax.yaxis.set_major_locator( MaxNLocator(3) )
        
    
    #[t.set_va('center') for t in ax.get_zticklabels()]
    #[t.set_ha('left') for t in ax.get_zticklabels()]
    
    #ticks = ['','0.08','0.16','0.24']
    ticks = ['','5','6','7']
    ticks = [tick for tick in ticks]
    ax.set_zticklabels(ticks)
    
    
    show()
        
    if maximization:
        loc = "lower right"
    else:
        loc = "upper right"

    #l = ax.legend(legend, loc=loc,prop={'size':annotatefontsize-1}, frameon=False)
    #setp(l.get_title(), fontsize=smallfontsize)
    #ax.grid(True)
    change_tick_font(ax)
    if save:
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.26)
        fig.savefig(plot_name, bbox_inches='tight')
    else:
        show()
    close(fig)

