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
from scipy.interpolate import griddata 

    
if __name__ == '__main__':
    print getAllData()
                
    