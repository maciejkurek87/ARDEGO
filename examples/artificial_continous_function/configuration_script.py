<<<<<<< HEAD
#results_folder_path = '/mnt/data/cccad3/mk306/log'
import os
from numpy import arange, array
=======
import os
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
results_folder_path = '/homes/mk306/log'
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False

goal = "min"
<<<<<<< HEAD

max_eval = 5
=======
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
### Basic setup
search = 'brute'
trials_count = 1
<<<<<<< HEAD
population_size = 30

n_sims = 200000

hardware = False
limit_lambda_search = True
weights_on = True
sampling_fancy = False
=======
population_size = 40
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c

max_fitness = 300.0

<<<<<<< HEAD
max_speed = 0.1
max_stdv = 0.01
min_stdv = 0.01
sample_on ="s"
F = 20  # The size of the initial training set
M = 10  # How often to perturb the population, used in discrete problems
max_iter = 1000 * M
=======
surrogate_type = 'proper'  # Can be proper or dummy
F = (population_size/2)  # The size of the initial training set
M = 20  # How often to perturb the population, used in discrete problems
max_eval = 1
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c

### Trial-specific variables
trials_type = 'P_ARDEGO_Trial'
surrogate_type = 'bayes2'

parall = 1

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

<<<<<<< HEAD
vis_every_X_steps = 1 # How often to visualize
=======
vis_every_X_steps = 10 # How often to visualize
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

### Regressor and classifier type
regressor = 'GaussianProcess4'
#regressor = 'R'
#regressor = 'KMeansGaussianProcessRegressor'
#classifier = 'RelevanceVectorMachine2'
classifier = 'SupportVectorMachine'
Kfolds=5
### GPR Regression settings
regr = 'linear'
corr2 = 'squared_exponential'
corr = 'anisotropic'
<<<<<<< HEAD
theta0 = -10.0
thetaL = 0.001
thetaU = 10.0
nugget = 3
random_start = 10


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
theta0 = 0.01
thetaL = 0.0001
thetaU = 3.0
nugget = 5
random_start = 100
>>>>>>> 3af52321da6a5bfb3b3cc04df714eb04250e157c
