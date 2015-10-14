<<<<<<< HEAD
results_folder_path ="/homes/mk306/xinyu_rtm"
import os
from numpy import arange, array
=======
results_folder_path = '/homes/mk306/log'
import os
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False

goal = "min"

max_eval = 1
### Basic setup

trials_count = 1
<<<<<<< HEAD
population_size = 80
search = 'pso'

=======
population_size = 30

max_fitness = 200.0
max_iter = 5000
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
max_speed = 0.2
max_stdv = 0.1
min_stdv = 0.1

<<<<<<< HEAD
n_sims = 50
=======
surrogate_type = 'proper'  # Can be proper or dummy
F = 30  # The size of the initial training set
M = 10  # How often to perturb the population, used in discrete problems
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c

#old_run_dir = "/data/mk306/ardego_maiaUsingMax3_xinyu_rtm_software_latin_lambda_False_5000_True_True/tT_AARDEGO_corr_anisotropic_nsims_5000_parall_1_fF_rtm"
hardware = False
limit_lambda_search = False
weights_on = True
sampling_fancy = True
trials_type ="P_ARDEGO_Trial"
surrogate_type ="bayes2"
parall = 6

if parall==6:
	max_fitness = 300.0
else:
	max_fitness = 250.0

max_iter = max_fitness - 70 
F = 60  # The size of the initial training set
M = 10  # How often to perturb the population, used in discrete problems
sample_on='ei'

### Trial-specific variables

phi1 = 2.0
phi2 = 2.0

weight_mode = 'norm'
max_weight = 1.0
min_weight = 0.4
weight = 1.0

mode = 'exp'
exp = 2.0
admode = 'iter'  # Advancement mode, can be fitness
max_eval = 3
applyK = False
KK = 0.73

a="a1"
### Visualisation

vis_every_X_steps = 1 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

### Regressor and classifier type
#regressor = 'KMeansGaussianProcessRegressor'
regressor = 'GaussianProcess4'
#classifier = 'RelevanceVectorMachine'
classifier = 'SupportVectorMachine'
### GPR Regression settings
regr = 'linear'
corr2 = 'squared_exponential'
<<<<<<< HEAD
corr = "anisotropic"
theta0 = -4.
thetaL = -4.
thetaU = 2.0
nugget = 3
random_start = 10
run_name = "RTM_" + corr + "_" + surrogate_type + "_" + trials_type + "_par_" + str(parall)

gamma = 10.0 * 1.25** arange(1., 20)
C = 1.5 ** arange(-20, 1)
local_gamma = 1.2 ** arange(-10, 10)
local_C = 10*1.25 ** arange(1, 10)

# gamma = 10.0 * 1.25** arange(-10., 20)
# C = 1.5 ** arange(-20, 20)
# local_gamma = 1.2 ** arange(-10, 10)
# local_C = 10*1.25 ** arange(-10, 10)

# gamma = 10.0 * 1.25** arange(-10., 30)
# C = 1.5 ** arange(-30, 30)
# local_gamma = 1.2 ** arange(-20, 20)
# local_C = 10*1.25 ** arange(-20, 20)
=======
corr = 'matern3'
theta0 = 0.01
thetaL = 0.0001
thetaU = 3.0
nugget = 3
random_start = 10
run_name= corr + "_" + surrogate_type + "_" + trials_type
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
