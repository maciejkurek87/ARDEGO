import os
from numpy import arange, array
results_folder_path ="/homes/mk306/log/"
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False

goal = "max"
cost=True
max_eval = 5
### Basic setup
search = 'brute'
trials_count = 1
population_size = 30

hardware = False
limit_lambda_search = False
sampling_fancy = False
parall = 6
M = 10
trials_type ="P_ARDEGO_Trial"

if trials_type=="PSOTrial":
    max_fitness = 100 
    max_iter = max_fitness * M
elif trials_type=="Gradient_Trial":
    max_fitness = 1000
    max_iter = max_fitness
else:
    max_fitness = 150
    max_iter = max_fitness - 50


#old_run_dir = "/data/mk306/old/ardego_quad_software_latin_lambdaLimit_False_5000_True_True_False_True/tT_AARDEGO_corr_anisotropic_nsims_5000_parall_2_fF_ansonE0.001ThTrue/"
    
max_speed = 0.1
max_stdv = 0.1
min_stdv = 0.1
sample_on = "no"

F = 20  # The size of the initial training set
 
n_sims = 5000

### Trial-specific variables
surrogate_type ="bayes2"

weights_on = True

phi1 = 2.0
phi2 = 2.0

weight_mode = 'norm'
max_weight = 1.0
min_weight = 0.4
weight = 1.0

mode = 'exp'
exp = 2.0
admode = 'iter'  # Advancement mode, can be fitness

applyK = False
KK = 0.73

a="a1"
### Visualisation

vis_every_X_steps = 10000000000000000 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

### Regressor and classifier type
regressor = 'GaussianProcess4'
#regressor = 'R'
#regressor = 'KMeansGaussianProcessRegressor'
classifier = 'SupportVectorMachine'
#classifier = 'RelevanceVectorMachine2'
Kfolds=5

gamma = 10.0 * 1.25** arange(1., 20)
C = 1.5 ** arange(-20, 1)
local_gamma = 1.2 ** arange(-10, 10)
local_C = 10*1.25 ** arange(1, 10)

#gamma = 10.0 * 1.25** arange(-10., 20)
#C = 1.5 ** arange(-20, 20)
#local_gamma = 1.2 ** arange(-10, 10)
#local_C = 10*1.25 ** arange(-10, 10)

#gamma = 10.0 * 1.25** arange(-10., 30)
#C = 1.5 ** arange(-30, 30)
#local_gamma = 1.2 ** arange(-20, 20)
#local_C = 10*1.25 ** arange(-20, 20)

### GPR Regression settings
regr = 'linear'
corr2 = 'squared_exponential'
corr = "anisotropic"
theta0 = -10.0
thetaL = 0.001
thetaU = 10.0
nugget = 3
random_start = 10
always_valid=[1.,53.,32.]
