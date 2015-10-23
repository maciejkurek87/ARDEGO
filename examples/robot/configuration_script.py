#results_folder_path = '/mnt/data/cccad3/mk306/log'
import os
results_folder_path = '/homes/mk306/log'
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False

goal = "max"

max_eval = 5
### Basic setup
search = 'brute'
trials_count = 10
population_size = 30

max_fitness = 100.0
max_iter = 5000
max_speed = 0.1
max_stdv = 0.1
min_stdv = 0.1
sample_on="ei"
F = 20  # The size of the initial training set
M = 10  # How often to perturb the population, used in discrete problems


### Trial-specific variables
trials_type = 'PSOTrial'
surrogate_type = 'proper'  # Can be proper or dummy

#surrogate_type = 'proper'  # Can be proper or dummy
#trials_type = 'P_ARDEGO_Trial'

#trials_type = 'MonteCarloTrial'
parall = 1
#trials_type = 'PSOTrial_TimeAware'

#trials_type = 'Gradient_Trial'
#surrogate_type = 'bayes2'  # Can be proper or dummy

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

vis_every_X_steps = 1 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

### Regressor and classifier type
regressor = 'GaussianProcess4'
#regressor = 'R'
#regressor = 'KMeansGaussianProcessRegressor'
#classifier = 'RelevanceVectorMachine'
classifier = 'SupportVectorMachine'
Kfolds=5
### GPR Regression settings
regr = 'linear'
corr2 = 'squared_exponential'
corr = 'isotropic'
theta0 = -10.0
thetaL = 0.001
thetaU = 10.0
nugget = 3
random_start = 10
run_name = "pq_" + corr


