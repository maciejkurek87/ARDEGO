from math import e, pow
from deap import benchmarks
from numpy import array

dimensions = 4  # Dimensionality of solution space
# Example definition of the design space
designSpace = []
for d in xrange(dimensions):
    designSpace.append({"min":-5.0,"max":5.0,"step":1.0,"type":"discrete", "set":"h"})
    #designSpace.append({'min': -2, 'max': 2, 'step': 0.5,
    #                    'type': 'continuous'})

# Min and max fitness values

minVal = 0.0
maxVal = 25.0
worst_value = 25.0
maxvalue = worst_value

# Defines names for classification codes
error_labels = {0: 'Valid', 1: 'Invalid'}
rotate = False


# Defines the problem to be maximization or minimization
def is_better(a, b):
    return a < b

worst_value = maxVal

cost_maxVal = 1.0
cost_minVal = 0.0

# Example fitness function for surrogate model testing
def fitnessFunc(part, state):
    code = 0 if is_valid(part) else 1
    return ((array(benchmarks.sphere(part)), array([0.0]), array([0.0]), array([1.0])), state) ## cost always 1
    
# Example function to define if a design is valid or invalid
def is_valid(part):
    return part[0] ** int(part[1]) / part[1] < e

def get_x_axis_name():
    return "$x$"
    
def get_y_axis_name():
    return "$y$"
    
def get_z_axis_name():
    return "$z$"

# Example Termination condition
def termCond(best):
    return best < 0.00001


# Name of the benchmark
def name():
    return 'sphere'

