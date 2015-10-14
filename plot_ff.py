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
    
print count
X, Y = np.meshgrid(xi, yi)
zi = griddata((x, y), execution_time[4].reshape(-1,), (xi[None, :], yi[:, None]), method='linear')
surf = ax.plot_surface(X, Y, zi, rstride=1, cstride=1, cmap=cm.jet,linewidth=1)

show()
    
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
    fig.savefig(plot_name, bbox_inches='tight')
else:
    show()
close(fig)

