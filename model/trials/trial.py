import os
import sys
import logging

from threading import Thread
import time
from time import strftime
from datetime import datetime, timedelta
from copy import copy, deepcopy
import io
import pickle
import re
from numpy import multiply, array, ceil, floor, maximum, minimum, mean, min, max, argmax, argmin, ones, mgrid, log, where, isnan, take
from numpy.random import uniform, rand, randint 
from numpy.linalg import norm
import numpy
from scipy.stats import uniform as scipy_uniform
from scipy.optimize import minimize, anneal
import operator
import csv
from scipy.stats import spearmanr, pearsonr
import  scipy.stats as stats

from deap import base, creator, tools
toolbox = base.Toolbox()
##TODO - clean it up... only should be loaded when needed... 

from ..surrogatemodels.surrogatemodel import DummySurrogateModel, ProperSurrogateModel, LocalSurrogateModel, BayesClassSurrogateModel, BayesClassSurrogateModel2
from ..surrogatemodels.costmodel import DummyCostModel, ProperCostModel
from ..surrogatemodels.classifiers import Classifier, SupportVectorMachineClassifier, RelevanceVectorMachineClassifier, ResourceAwareClassifier
import lhs 
import pdb

from utils import numpy_array_index, load_script
import pyGP_OO
from pyGP_OO.Core import *
from pyGP_OO.Valid import valid
from sklearn import preprocessing

class Trial(Thread):

    def __init__(self, trial_no, my_run, fitness, configuration, controller,
                 run_results_folder_path):
        Thread.__init__(self)
        self.fitness = fitness
        self.controller = controller
        # True if the user has selected to pause the trial
        
        if configuration.surrogate_type == "dummy":
            self.surrogate_model = DummySurrogateModel(configuration,
                                                       self.controller,
                                                        self.fitness)
        elif configuration.surrogate_type == "local":
            self.surrogate_model = LocalSurrogateModel(configuration,
                                                       self.controller,
                                                        self.fitness)        
        elif configuration.surrogate_type == "bayes":
            self.surrogate_model = BayesClassSurrogateModel(configuration,
                                                       self.controller,
                                                        self.fitness)       
        elif configuration.surrogate_type == "bayes2":
            self.surrogate_model = BayesClassSurrogateModel2(configuration,
                                                       self.controller,
                                                        self.fitness)
        else:
            self.surrogate_model = ProperSurrogateModel(configuration,
                                                        self.controller,
                                                        self.fitness)
                                                        
        # Contains all the counter variables that may be used for visualization
        counter_dictionary = {}
        counter_dictionary['fit'] = 0 ## we always want to record fitness of the best configurations
        counter_dictionary['cost'] = 0.0 ## we always want to record fitness of the best configurations
        
        timer_dictionary = {}
        timer_dictionary['Running_Time'] = 0 ## we always want to record fitness of the best configurations
        timer_dictionary['Model Training Time'] = 0 
        timer_dictionary['Cost Model Training Time'] = 0 
        timer_dictionary['Model Predict Time'] = 0
        timer_dictionary['Cost Model Predict Time'] = 0
        self.configuration = configuration
        self.dump_time_file= None
        self.my_run = my_run
        run_name = self.my_run.get_name()
        
        self.state_dictionary = {
            'status' : 'Waiting',
            'retrain_model' : False,
            'model_failed' : False,
            'run_results_folder_path' : run_results_folder_path,
            'run_name' : run_name,
            'trial_type' : self.configuration.trials_type, 
            'trial_no' : trial_no,
            'name' : str(run_name) + '_' + str(trial_no),
            'all_particles_in_invalid_area' : False,
            'wait' : True,
            'generations_array' : [],
            'best_fitness_array' : [],
            'enable_traceback' : configuration.enable_traceback,
            'counter_dictionary' : counter_dictionary,
            'timer_dict' : timer_dictionary,
            'best' : None,
            'generate' : False,
            'fitness_state' : None,
            'fresh_run' : False,
            'terminating_condition' : False
            # True if the user has selected to pause the trial
        }
        self.set_start_time(datetime.now().strftime('%d-%m-%Y  %H:%M:%S'))
        self.kill = False
        self.total_time = timedelta(seconds=0)
        self.previous_time = datetime.now()
        self.controller.register_trial(self)
        self.view_update(visualize = False)
    ####################
    ## Helper Methods ##
    ####################
    
    ###check if between two calls to this functions any fitness functions have been evaluted, so that the models have to be retrained
    def training_set_updated(self):
        retrain_model_temp = self.get_retrain_model()
        self.set_retrain_model(False)
        return retrain_model_temp
    
    def is_better(self, a, b):
        try:
            if self.configuration.goal == "min":
                return a < b
            elif self.configuration.goal == "max":
                return a > b
            else:
                logging.info("configuration.goal attribute has an invalid value:" + str( self.configuration.goal))
        except:
            logging.info("configuration.goal has not been defined")
        logging.info("configuration.goal is assumed to be minimization")
        self.configuration.goal = "min"
        return a < b
    
    def train_surrogate_model(self):
        logging.info('Training surrogate model')
        start = datetime.now()
        self.set_model_failed(self.surrogate_model.train(self.hypercube()))
        diff = datetime.now() - start
        self.add_train_surrogate_model_time(diff)
        
    def predict_surrogate_model(self, population):
        logging.info('Using surrogate model for prediction')
        start = datetime.now()
        prediction = self.surrogate_model.predict(population)
        diff = datetime.now() - start
        self.add_predict_surrogate_model_time(diff)
        return prediction
        
    def train_cost_model(self):
        logging.info('Training cost model')
        start = datetime.now()
        self.set_model_failed(self.cost_model.train())
        diff = datetime.now() - start
        self.add_train_cost_model_time(diff)
        
    def predict_cost_model(self, population):
        start = datetime.now()
        prediction = self.cost_model.predict(population)
        diff = datetime.now() - start
        self.add_predict_cost_model_time(diff)
        return prediction
        
    def set_kill(self, kill):
        self.kill = kill
               
    def create_results_folder(self):
        """
        Creates folder structure used for storing results.
        Returns a results folder path or None if it could not be created.
        """
        path = str(self.get_run_results_folder_path()) + '/trial-' + str(self.get_trial_no())
        try:
            os.makedirs(str(self.get_run_results_folder_path()) + "/bests")
        except OSError, e:
            logging.info("bests already created")
        #self.hdlr = logging.FileHandler('/var/tmp/trial.log')
        #formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        #hdlr.setFormatter(formatter)
        #logger.addHandler(hdlr) 
        #logger.setLevel(logging.WARNING)
        try:
            os.makedirs(path)
            os.makedirs(path + "/images")
            os.makedirs(path + "/dump")
            return path, path + "/images", path + "/dump"
        except OSError, e:
            # Folder already exists
            logging.error('Could not create folder ' + str(e))
            return path, path + "/images", path + "/dump", 
        except Exception, e:
            logging.error('Could not create folder: ' + str(path) + ', aborting',
                          exc_info=sys.exc_info())
            return None, None, None
    
    ### check first if part is already within the training set
    def fitness_function(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        self.increment_counter('fit')
        #results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
        #self.set_fitness_state(state)
        #pdb.set_trace()
        try:
            results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            #pdb.set_trace()
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3]
        except:
            cost = array([1.0]) ## just keep it constant for all points
        
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost[0])
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.cost_model.add_training_instance(part, cost)
        self.set_retrain_model(True)
       
        logging.info(str(code) + " " + str(fitness))
        if code[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return array([self.fitness.worst_value]), code, cost
            
    # for MOPSOTrial
    def fitness_function1(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
        
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        self.increment_counter('fit')
        
        try:
            results, state = self.fitness.fitnessFunc1(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            logging.info(str(e))
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3][0]
        except:
            cost = 1.0 ## just keep it constant for all points
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost)
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.cost_model.add_training_instance(part, cost)
        self.set_retrain_model(True)
        
        if code[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return array([self.fitness.worst_value]), code, cost
            
    def fitness_function2(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
        
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        self.increment_counter('fit')
        try:
            results, state = self.fitness.fitnessFunc2(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            logging.info(str(e))
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3][0]
        except:
            cost = 1.0 ## just keep it constant for all points
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost)
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.cost_model.add_training_instance(part, cost)
        self.set_retrain_model(True)
        
        if code[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return array([self.fitness.worst_value]), code, cost
            
    ## indicator for the controller and viewer that the state has changed. 
    def view_update(self, visualize=False):
        self.current_time = datetime.now()
        diff = self.current_time - self.previous_time
        self.previous_time = self.current_time
        self.state_dictionary['timer_dict']['Running_Time'] = self.state_dictionary['timer_dict']['Running_Time'] + diff.seconds
        self.controller.view_update(trial=self, run=None, visualize=visualize) ##
#        self.controller.view_update(trial=self, run=self.my_run, visualize=False) ## update run state

    def increment_counter(self, counter):
        self.set_counter_dictionary(counter, self.get_counter_dictionary(counter) + 1)
        
    def exit(self):
        self.set_status('Finished')
        self.my_run.trial_notify(self)
        self.view_update(visualize = False)
        sys.exit(0)
        
    #######################
    ## Abstract Methods  ##
    #######################
        
    def initialise(self):
        raise NotImplementedError('Trial is an abstract class, '
                                   'this should not be called.')
    
    def run_initialize(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    ## main computation loop goes here
    def run(self):
        raise NotImplementedError('Trial is an abstract class, '
                                   'this should not be called.')
                                   
    def snapshot(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def save(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')                                  
    def load(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                       
    def get_time_stamp(self):
        return self.current_sim_time
    
    def update_system_time(self, time_passed,disp=True):
        self.current_sim_time = self.current_sim_time + time_passed
        if disp:
            logging.info("Moving simulation time forward, current time since start:" + str(self.current_sim_time))
        
                       
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_classifier(self):
        return self.get_surrogate_model().get_classifier() 
        
    def get_regressor(self):
        return self.get_surrogate_model().get_regressor()
    
    def get_surrogate_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        if self.configuration.surrogate_type == "dummy":
            model = DummySurrogateModel(self.get_configuration(),
                                                       self.controller,
                                                        self.fitness)
        elif self.configuration.surrogate_type == "local":
            model = LocalSurrogateModel(self.get_configuration(),
                                                       self.controller,
                                                        self.fitness)       
        elif self.configuration.surrogate_type == "bayes":
            model = BayesClassSurrogateModel(self.get_configuration(),
                                                       self.controller,
                                                        self.fitness)
        elif self.configuration.surrogate_type == "bayes2":
            model = BayesClassSurrogateModel2(self.get_configuration(),
                                                       self.controller,
                                                        self.fitness)
        else:     
            model = ProperSurrogateModel(self.get_configuration(),
                                                        self.controller,
                                                        self.fitness)
        
        model.set_state_dictionary(self.surrogate_model.get_state_dictionary())
        return model
        
    def get_run(self):
        return self.my_run 
        
    def get_fitness_state(self):
        return self.state_dictionary['fitness_state']
        
    def set_fitness_state(self, state):
        self.state_dictionary['fitness_state'] = state
        
    def add_train_surrogate_model_time(self, diff):
        self.state_dictionary['timer_dict']['Model Training Time'] = self.state_dictionary['timer_dict']['Model Training Time'] + diff.seconds
        
    def add_predict_surrogate_model_time(self, diff):
        self.state_dictionary['timer_dict']['Model Predict Time'] = self.state_dictionary['timer_dict']['Model Predict Time'] + diff.seconds
        
    def get_train_surrogate_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Model Training Time'])
        
    def get_predict_surrogate_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Model Predict Time'])
        
    def add_train_cost_model_time(self, diff):
        self.state_dictionary['timer_dict']['Cost Model Training Time'] = self.state_dictionary['timer_dict']['Cost Model Training Time'] + diff.seconds
        
    def add_predict_cost_model_time(self, diff):
        self.state_dictionary['timer_dict']['Cost Model Predict Time'] = self.state_dictionary['timer_dict']['Cost Model Predict Time'] + diff.seconds
        
    def get_train_cost_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Cost Model Training Time'])
        
    def get_predict_cost_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Cost Model Predict Time'])
        
    def get_running_time(self):
        return timedelta(seconds=self.state_dictionary['timer_dict']['Running_Time'])
    
    def get_main_counter_iterator(self):
        return range(0, self.get_counter_dictionary(self.get_main_counter_name()) + 1)
        
    def get_main_counter(self):
        return self.get_counter_dictionary(self.get_main_counter_name())
    
    def get_main_counter_name(self):
        return self.state_dictionary["main_counter_name"]
    
    def set_main_counter_name(self, name):
        self.state_dictionary["main_counter_name"] = name 
    
    def get_state_dictionary(self):
        return self.state_dictionary
    
    def set_state_dictionary(self, new_dictionary):
        self.state_dictionary = new_dictionary 
        
    def get_fitness(self):
        return self.fitness
        
    def get_best(self):
        return self.state_dictionary['best']
        
    def set_best(self, new_best):
        #if new_best > self.state_dictionary['best']:
        #     pdb.set_trace()
        try:
            logging.info('Old global best ' + str(self.get_best()) + ' fitness:'+ str(self.get_best().fitness.values))
        except:
            pass ## there was no best
            
        self.state_dictionary['best'] = new_best
        logging.info('New global best found' + str(new_best) + ' fitness:'+ str(new_best.fitness.values)) 
        try:
            dump_time_file = open(str(self.get_run_results_folder_path()) + '/bests/best_dump_' + str(self.get_trial_no()) + ".csv", "a")
            dump_file_writer = csv.writer(dump_time_file, delimiter=';', quotechar='"')
            try:
                dump_file_writer.writerow([self.get_counter_dictionary("cost")[0][0],self.get_best().fitness.values[0]])
            except:
                try:
                    dump_file_writer.writerow([self.get_counter_dictionary("cost")[0],self.get_best().fitness.values[0]])
                except:
                    dump_file_writer.writerow([self.get_counter_dictionary("cost"),self.get_best().fitness.values[0]])
            dump_time_file.close()
        except:
            logging.info("Cant dump best file... check if the algorithm is creating results folder correctly")
            pdb.set_trace()
        
    def get_counter_dictionary(self, counter):
        return self.state_dictionary['counter_dictionary'][counter]
        
    def set_counter_dictionary(self, counter, value):
        self.state_dictionary['counter_dictionary'][counter] = value
        
    ## should really be deep copy or something...
    ## we dont want it to be mutable outside
    def get_configuration(self):
        return self.configuration
        
    def set_start_time(self, start_time):
        self.state_dictionary['start_time'] = start_time

    def get_start_time(self):
        return self.state_dictionary['start_time']
        
    def set_surrogate_model(self, new_model):
        self.surrogate_model = new_model
        
    def set_waiting(self):
        self.set_status("Waiting")
        self.set_wait(False)
    
    def set_running(self):
        self.set_status("Running")
        self.set_wait(False)
    
    def set_paused(self):
        self.set_status("Paused")
        self.set_wait(True)
        
    def set_wait(self, new_wait): 
        self.state_dictionary["wait"] = new_wait
        
    def get_wait(self):
        return self.state_dictionary["wait"]

    def get_status(self):
        return self.state_dictionary["status"]
        
    def set_status(self, status):
        self.state_dictionary["status"] = status
        self.view_update()
        
    def set_retrain_model(self, status):
        self.state_dictionary["retrain_model"] = status

    def get_retrain_model(self):
        return self.state_dictionary["retrain_model"]
        
    def get_model_failed(self):
        return self.state_dictionary["model_failed"]
    
    def set_model_failed(self, state):
        self.state_dictionary["model_failed"] = state
    
    def get_run_results_folder_path(self):
        return self.state_dictionary["run_results_folder_path"]
        
    def get_trial_no(self):
        return self.state_dictionary["trial_no"]
    
    def get_terminating_condition(self):
        return self.state_dictionary["terminating_condition"]
    
    def set_terminating_condition(self, fitness):
        self.state_dictionary["terminating_condition"] = self.fitness.termCond(fitness[0]) or self.state_dictionary["terminating_condition"]
        
    def get_name(self):
        return self.state_dictionary["name"]
        
    def get_results_folder(self):
        return str(self.get_run_results_folder_path()) + '/trial-' + str(self.get_trial_no()) #self.state_dictionary['results_folder']
        
    def set_images_folder(self, folder):
        self.state_dictionary['images_folder'] = folder

    def get_images_folder(self):
        return self.get_results_folder() + "/images"
        
    def get_trial_type(self):
        return self.state_dictionary["trial_type"]
        
    def get_design_space(self):
        return self.fitness.designSpace
        
    def get_best_fitness_array(self):
        return self.state_dictionary['best_fitness_array']
        
    def get_generations_array(self):
        return self.state_dictionary['generations_array']
        
    def set_at_least_one_in_valid_region(self, state):
        if self.state_dictionary.has_key('at_least_one_in_valid_region'):
            self.state_dictionary['at_least_one_in_valid_region'] = state
        else:
            self.state_dictionary['at_least_one_in_valid_region'] = False

    def get_at_least_one_in_valid_region(self):
        return self.state_dictionary['at_least_one_in_valid_region']
        
class MonteCarloTrial(Trial):

    def naive_sample_plan(self, F, D, design_space):
        
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(F,D))
        max_bounds = [d["max"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples * max_bounds
        min_bounds = [d["min"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples + min_bounds
        for part in latin_hypercube_samples:
            part = self.toolbox.filter_particle(self.create_particle(part))
            part.fitness.values, code, cost = self.toolbox.evaluate(part)
            self.set_at_least_one_in_valid_region((code == 0) or self.get_at_least_one_in_valid_region())
            self.update_best(part)
    
    def get_random_valid_design(self):
        valid_designs = self.surrogate_model.get_valid_set()
        i = len(valid_designs)
        return valid_designs[randint(0,i,1)[0]] ## upperbound is e
    
    def classifier_aware_sample_plan(self, F, design_space):
        counter = 0
        counter2 = 0
        self.surrogate_model.classifier.train()
        while counter < F:
            part = self.generate(design_space)
            part = self.create_particle(part)
            part = self.filter_particle(part)
            label = self.get_classifier().predict([part]) 
            if counter2 % 10000 == 0.:
                logging.info('Iteration ' + str(counter2) + ' found so far:' + str(counter) + ' left:' + str(F - counter))
            if label == 1.:  ### its valid... we use different label for valid class
                logging.info("valid found" + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                part = self.create_particle(part)
                part.fitness.values , code, cost = self.toolbox.evaluate(part)
                self.set_at_least_one_in_valid_region((code == 0) or self.get_at_least_one_in_valid_region())
                #mport pdb
                #pdb.set_trace()
                self.update_best(part)
                self.surrogate_model.classifier.train()
               # if code == 0 or code == 3: ### its a hack... finish it
                counter = counter + 1
            else:
                if counter2 % 100000 == 0.: 
                    logging.info('Evaluation random perturbation of always_valid' + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                    part = self.create_particle(self.get_random_valid_design() + self.perturbation())
                    part = self.toolbox.filter_particle(part)
                    part.fitness.values , code, cost = self.toolbox.evaluate(part)
                    self.update_best(part)
                    #if code == 0:
                    counter = counter + 1
                    self.surrogate_model.classifier.train()
            counter2 = counter2 + 1
                
    def perturbation(self):
        d = [0.] * len(self.fitness.designSpace)
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i]= small_fraction = self.fitness.designSpace[i]["step"]
            elif self.fitness.designSpace[i]["type"] == "continuous":
                d[i] = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
        dimensions = len(self.fitness.designSpace)
        pertubation = multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
                
    def sample_plan(self):
        design_space = self.get_design_space()
        D = len(design_space)
        F = D * 10
        if False: ## fancy shmancy
            self.naive_sample_plan(F/2, D, design_space)
            logging.info("Naive sampling done")
            self.classifier_aware_sample_plan(F/2, design_space)
            logging.info("Sampling of valid space done")
        
        else: #self.configuration.sampling_plan == 'naive': 
            self.naive_sample_plan(F, D, design_space)
            
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        results_folder, images_folder, dump_folder = self.create_results_folder()
        #self.cost_model = ProperCostModel(self.configuration, self.controller, self.fitness)
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        self.state_dictionary['best'] = None
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("i")
        self.set_counter_dictionary("i",0)
        try:
            part = self.create_particle(self.fitness.always_valid)
            self.toolbox.filter_particle(part)
            part.fitness.values, part.code, cost = self.fitness_function(part)
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            self.set_at_least_one_in_valid_region(True)
            logging.info("Always valid configuration present, evaluated")
        except:
            logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
            self.set_at_least_one_in_valid_region(False)
        self.sample_plan()
        self.state_dictionary["fresh_run"] = True
                
        ### generate folders
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        return True
    
    def run_initialize(self):
        logging.info("Initialize MonteCarloTrial no:" + str(self.get_trial_no()))
        design_space = self.get_design_space()
        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space])
        self.toolbox = copy(base.Toolbox())
        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particle', self.filter_particle)
        self.toolbox.register('evaluate', self.fitness_function)
        self.new_best=False
                                  
    ## main computation loop goes here
    def run(self):
        self.state_dictionary['generate'] = True
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        self.state_dictionary["fresh_run"] = False
        while self.get_counter_dictionary('fit') < self.get_configuration().max_iter + 1:
            
            if self.training_set_updated():
                self.train_surrogate_model()
                self.train_cost_model()
            self.view_update(visualize = True)
            self.save()
            
            logging.info('[' + str(self.get_name()) + '] Iteration ' + str(self.get_counter_dictionary('i')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))
            logging.info('[' + str(self.get_best()) + '] Best')
            
            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            sample_x_times = 1
            budget = 50000.0 ## revise budget
            ## add proper copy
            #best_x = self.farsee(self.get_surrogate_model(), self.get_cost_model(), budget, sample_x_times)
            best_x = self.max_ei(self.surrogate_model)
            if best_x is None:
                #logging.info("max_ei couldn't find anything good... ")
                #best_x = self.surrogate_model.local_brute_search(self.fitness.designSpace, self.get_best())
                
                #if best_x is None:
                #    best_x = self.surrogate_model.local_brute_search(self.fitness.designSpace, self.get_best(), radius = 2.)
                #    logging.info("increasing radius")
                #    if best_x is None:
                #        logging.info("local max_ei with increased radius couldn't find anything good... " + str(perturbation))
                #        best_x = self.surrogate_model.local_brute_search(self.fitness.designSpace, self.get_best(), radius = 3.)
            #if best_x is None:
                logging.info("problem!!")
                perturbation = self.perturbation()
                best_x = perturbation + self.get_best()
            #else:
                logging.info("local perturbation... " + str(best_x))
            else:
                logging.info("Best predicted at: " + str(best_x))
                logging.info("Selected best action")
            #best_x = best_action_function(self.get_surrogate_model())
            best_x = self.toolbox.filter_particle(self.create_particle(best_x))
            best_x.fitness.values, code, cost = self.toolbox.evaluate(best_x)
            self.update_best(best_x)
            
            self.increment_main_counter()
        self.exit()
    #######################
    ### GET/SET METHODS ###
    #######################
    
    ### a hypercube that spans all the valid designs
    def hypercube(self):
        set_to_search = self.surrogate_model.get_valid_set()
        #find maximum
        max_diag = deepcopy(set_to_search[0])
        for part in set_to_search:
            max_diag = maximum(part,max_diag) 
        ###find minimum vectors
        min_diag = deepcopy(set_to_search[0])
        for part in set_to_search:
            min_diag = minimum(part,min_diag)
            
        ## we always ensure that the hypercube allows particles to maintain velocity components in all directions
        
        for i,dd in enumerate(max_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                max_diag[i] = minimum(dd + self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["max"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                max_diag[i] = minimum(dd + small_fraction, self.fitness.designSpace[i]["max"])
                
        for i,dd in enumerate(min_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                min_diag[i] = maximum(dd - self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["min"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                min_diag[i] = maximum(dd - small_fraction, self.fitness.designSpace[i]["min"])
        #logging.info("hypecube: " + str([max_diag,min_diag]))
        return [max_diag,min_diag]
    
    def generate(self, designSpace):
        '''
        [max_diag,min_diag] = self.hypercube()
        particle = [uniform(max_d,min_d)
                    for max_d,min_d
                    in zip(max_diag,min_diag)]
        
        '''
        particle = [uniform(dim_space["min"],dim_space["max"])
                    for dim_space
                    in designSpace]
        
        particle = self.create_particle(particle)
        #import pdb
        #pdb.set_trace()
        return particle
    
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        #model = ProperCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
                                  
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('i'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'goal': self.get_configuration().goal,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'propa_classifier': self.get_surrogate_model().propa_classifier,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness,
            'best': {"data":self.get_best()}
        }
        return return_dictionary

    def save(self):
        try:
            trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('i')) + '.txt'
            dict = self.state_dictionary
            surrogate_model_state_dict = self.surrogate_model.get_state_dictionary()
            dict['surrogate_model_state_dict'] = surrogate_model_state_dict
            cost_model_state_dict = self.cost_model.get_state_dictionary()
            dict['cost_model_state_dict'] = cost_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                #pickle.dump(dict, outfile)  
                if self.kill:
                    sys.exit(0)
        except Exception, e:
            logging.error(str(e))
            if self.kill:
                sys.exit(0)
            return False
            
    ## by default find the latest iteration
    def load(self, iteration = None):
        try:
            if iteration is None:
                # Figure out what the last iteration before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last iteration
                        iteration = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            iteration_file = str(iteration)
            trial_file = str(self.get_results_folder()) + '/' + str(iteration_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            self.surrogate_model.set_state_dictionary(dict['surrogate_model_state_dict'])
            self.cost_model.set_state_dictionary(dict['cost_model_state_dict'])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False
        
    ## actions and wrapper methods
        
    def all_actions(self):
        return {"max_ei": self.max_ei
                #"max_ei_cost": self.max_ei_cost
                }
        
    def max_ei(self, surrogate_model):
        hypercube = self.hypercube()
        particle = surrogate_model.max_ei(designSpace=self.get_design_space()) ## NOT SELF
        if particle is None:
            return None
        else:
            try:
                particle = self.toolbox.filter_particle(self.create_particle(particle))
            except:
                logging.info("KRUWAURUASFAKSDFA " + str(particle))
                return None
            return particle
        
    def max_ei_cost(self, surrogate_model):
        hypercube = self.hypercube()
        particle = surrogate_model.max_ei_cost(designSpace=self.get_design_space(), cost_func = self.predict_cost) ## NOT SELF
        particle = self.toolbox.filter_particle(self.create_particle(particle))
        return particle
        
    def keywithmaxval(self, d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]
     
    def keywithminval(self, d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(min(v))]
        
    def farsee(self, surrogate_model, cost_model, budget, sample_x_times):
        rewards = dict([(action, 0.0) for action in self.all_actions().keys()])
        for i in range(sample_x_times):
            #best_action, reward = self.reward_function(surrogate_model, cost_model, budget)
            best_action, reward = self.reward_function_v1(surrogate_model, cost_model, 0.0,0.0)
            rewards[best_action] += reward
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(rewards)
            return best_action, self.all_actions()[best_action]
        else:
            best_action = self.keywithminval(rewards)
            return best_action, self.all_actions()[best_action]
            
    ## finds the best action that offers best result within reaching budget
    ## might have to be adjusted... add a decay element? more then 3-4 samples away 
    ## is just pointless
    ## to evalaute the tree up to for example 50% of the budget simply change what you feed in to 10..20..50 or so %
    def reward_function(self, surrogate_model, cost_model, budget):
        reward = {}
        for action, action_function in self.all_actions().items():
            logging.info("Looking into future.. action:" + action)
            surrogate_model_copy = surrogate_model.get_copy()
            cost_model_copy = cost_model.get_copy()
            best_x = action_function(surrogate_model_copy)
            code, MU, S2, EI, P = surrogate_model_copy.predict([best_x])
            fitness = P[0] ## sampled from the distribution
            cost = cost_model_copy.predict([best_x])#[0]
            surrogate_model_copy.add_training_instance(best_x, code, fitness, [0.0]) ## always add
            if cost <= 0.0: ## for some of the regressors 
                logging.debug("detected negative cost... probably small value")
            cost_model_copy.add_training_instance(best_x, cost)
            #surrogate_model_copy.train(self.hypercube())
            #cost_model_copy.train()
            my_budget = budget - cost
            logging.info("Budget left: " + str(my_budget))
            if my_budget > 0.0:
                best_action, reward[action] = self.reward_function(surrogate_model_copy, cost_model_copy, my_budget)
                try:
                    if self.configuration.goal == "max":
                        reward[action] = max(fitness, reward[action])
                    else:
                        reward[action] = min(fitness, reward[action])
                except ValueError:
                    logging.info("kurwo..." + str(fitness) + " " + str(reward) + " " + str(reward[action]))
            else:
                reward[action] = fitness
                
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(reward)
            return best_action, reward[best_action]
        else:
            best_action = self.keywithminval(reward)
            return best_action, reward[best_action]
    
    ## favors fast convergence (what about local minimias?
    ## It cuts of the tree at one point and reports the budget used so far..
    ## the reward is calculated based on the mean best result / budget used value 
    ## obviously depends on whether we faces min/max problems
    def reward_function_v1(self, surrogate_model, cost_model, horizont, budget_used):
        reward = {}
        best_x
        logging.info("Investigating horizont level: " + str(horizont))
        for action, action_function in self.all_actions().items():
            surrogate_model_copy = surrogate_model.get_copy()
            best_x = action_function(surrogate_model_copy)
            logging.info("Best predicted at: " + str(best_x))
            code, MU, S2, EI, P = surrogate_model_copy.predict([best_x])
            fitness = P[0] ## sampled from the distribution
            surrogate_model_copy.add_training_instance(best_x, code, fitness, [0.0]) ## always add
            logging.info("sampling from cost")
            cost_model_copy = cost_model.get_copy()
            cost = cost_model_copy.predict([best_x])
            cost_model_copy.add_training_instance(best_x, cost)
            new_budget_used = cost + budget_used

            if horizont > 0.0:
                best_action, reward[action] = self.reward_function_v1(surrogate_model_copy, cost_model_copy, horizont-1.0, new_budget_used)
                if self.configuration.goal == "max":
                    reward[action] = max(fitness, reward[action])
                else:
                    reward[action] = min(fitness, reward[action])
            else:
                if self.configuration.goal == "max":
                    reward[action] = fitness / new_budget_used
                else:
                    reward[action] = fitness * new_budget_used
        import pdb
        pdb.set_trace()
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(reward)
            return best_action, reward[best_action]
        else:
            best_action = self.keywithminval(reward)
            return best_action, reward[best_action]
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())
        
    def filter_particle(self, p):#
        design_space = self.get_design_space()  
        pmin = [dimSetting['min'] for dimSetting in design_space]
        pmax = [dimSetting['max'] for dimSetting in design_space]
        for i, val in enumerate(p):
            #dithering
            if design_space[i]['type'] == 'discrete':
                p[i] = p[i]/design_space[i]['step']
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])
                p[i] = p[i] * design_space[i]['step']
            #dont allow particles to take the same value
            p[i] = minimum(pmax[i], p[i])
            p[i] = maximum(pmin[i], p[i])
        return p
            
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
            
    def predict_cost(self, particle):
        try:
            return self.cost_model.predict(particle)
        except Exception,e:
            logging.debug("Cost model is still not avaiable: " + str(e))
            "a" + 1 
            return 1.0 ## model has not been created yet

    def update_best(self, part):
        if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
            best = self.create_particle(part)
            best.fitness.values = part.fitness.values
            #pdb.set_trace()
            self.set_best(best)
            
### simulated... needs a bit of rework to be truly parallel.
class P_ARDEGO_Trial(Trial):

    def dump(self):
        '''
        dump_folder = self.dump_folder
        pdb.set_trace()
        list0 = dict(zip([str(x) for x in self.surrogate_model.get_dump()[0][0]],self.surrogate_model.get_dump()[0][1]))
        list1 = dict(zip([str(x) for x in self.surrogate_model.get_dump()[1][0]],self.surrogate_model.get_dump()[1][1]))
        output = []
        for k in list0.keys():
            try:
                output.append([k,list0[k],list1[k][0]])
            except:
                output.append([k,list0[k],''])
        with open(dump_folder + "/state.csv", 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            for o in output:
                spamwriter.writerow(o)
        logging.info("State dumped..")
        '''
        logging.info("State dump disabled")
                

    def fitness_function(self, part, add=False):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
            logging.info("Particle already evaluated " + str(part))
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return False
        try:
            results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            logging.info(str(e))
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3]
            try: ##checking if hardware is being built
                a, cost_b = self.fitness.has_hardware(self.running_que, self.running_que_cost, part)
                if a:
                    cost = cost_b + 1
            except:
                pass
        except:
            logging.info("This version makes no sense if you cannot get the cost... exiting")
            exit(0)
        if add:
            self.increment_counter('fit')
            self.currently_evaled = self.currently_evaled - 1
            logging.info("Finished Evaluating " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
            self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
            self.cost_model.add_training_instance(part, cost)     
            if code[0] == 0:
                part.fitness.values = fitness
                self.update_best(part)
                self.set_terminating_condition(fitness) 
                return True
            else:
                return True
        else:
            if not numpy_array_index(self.running_que,part)[0]:
                self.running_que.append(part)
                self.running_que_cost.append(cost)
                self.currently_evaled = self.currently_evaled + 1
                logging.info("Evaluating " + str(part))
                self.new_design_added = True
                return True
            else:
                logging.info("Design is already being evaluated")
                return False
                
    def knowledge_aware_sample_plan(self, F):
        design_space = self.get_design_space()
        ##retrieve the dump
        old_run_dir = self.get_configuration().old_run_dir ## run with old optimization results
        logging.info("using dir db: " + old_run_dir)
        val_db = []
        cval_db = []
        inval_db = []
        all_db = []
        fit2 = None
        try:
            fit2 = load_script(self.get_configuration().old_fit, 'fitness2') # its a hack which allows me to use some numbers which i forgot to store in the db...
            accuracy = []
        except:
            logging.info("full old db not supplied")
        
        switch = True
        try :
            self.get_configuration().map
        except:
            logging.info("switch true")
            switch = False
        
        with open(old_run_dir + '/trial-' + str(self.get_trial_no()) + "/dump/state.csv", 'r') as csvfile:
        #with open(old_run_dir + '/trial-3/dump/state.csv', 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                if row[2] == "":
                    try:
                        new_data = eval(row[0].replace('.',','))
                        try: ## map old designs onto new ones
                            new_data = self.get_configuration().map(new_data)
                            new_data = self.toolbox.filter_particle(self.create_particle(new_data))
                        except:
                            pass
                        inval_db.append([new_data, eval(row[1])])
                    except SyntaxError, e:
                        new_data = []
                        row[0] = row[0].replace(']',' ')
                        row[0] = row[0].replace('[',' ')
                        splitted = row[0].split(' ')
                        for elem in splitted: ## this appends only things which are numbers
                            try:
                                new_data.append(eval(elem))
                            except:
                                pass
                        try: ## map old designs onto new ones
                            new_data = self.get_configuration().map(new_data)
                            new_data = self.toolbox.filter_particle(self.create_particle(new_data))
                        except:
                            pass
                        inval_db.append([new_data, eval(row[1])])
                else:
                    try:
                        new_data = eval(row[0].replace('.',','))
                        old_data = copy(new_data)
                        try: ## map old designs onto new ones
                            new_data = self.get_configuration().map(new_data)
                            new_data = self.toolbox.filter_particle(self.create_particle(new_data))
                        except:
                            pass
                        if switch:
                            OK = self.fitness.transferValid(new_data)
                        else:
                            OK = self.fitness.transferValid(new_data) and row[1] == "1.0"
                        if (OK):
                            #val_db.append([new_data, eval(row[1]), eval(row[2]), fit2.fitnessFunc(old_data,{},2)])
                            val_db.append([new_data, eval(row[1]), eval(row[2])])
                        else:
                            cval_db.append([new_data, eval(row[1]), eval(row[2])])
                    except SyntaxError, e:
                        new_data = []
                        row[0] = row[0].replace(']',' ')
                        row[0] = row[0].replace('[',' ')
                        splitted = row[0].split(' ')
                        for elem in splitted: ## this appends only things which are numbers
                            try:
                                new_data.append(eval(elem))
                            except:
                                pass
                        try: ## map old designs onto new ones
                            old_data = copy(new_data)
                            new_data = self.get_configuration().map(new_data)
                            new_data = self.toolbox.filter_particle(self.create_particle(new_data))
                        except:
                            pass
                        if switch:
                            OK = self.fitness.transferValid(new_data)
                        else:
                            OK = self.fitness.transferValid(new_data) and row[1] == "1.0"
                            
                        if (OK):
                            #val_db.append([new_data, eval(row[1]), eval(row[2]), fit2.fitnessFunc(old_data,{},2)])
                            val_db.append([new_data, eval(row[1]), eval(row[2])])
                        else:
                            cval_db.append([new_data, eval(row[1]), eval(row[2])])
        valids = len(val_db)
        logging.info("State dumped.. valids:" +  str(valids) + " F: " + str(F))
        to_eval_valids = maximum(F, self.get_configuration().parall * 2 )
        ## eval half of F/2 using known valid designs
        eval_set = []
        set_to_eval = []
        fit_of_set_to_eval = []
        acc_of_set_to_eval = []
        done = False
        counter = 0 
        while(not done):
            idx = randint(len(val_db), size=1)
            if not idx in eval_set:
                eval_set.append(idx)
                set_to_eval.append(val_db[idx][0])
                fit_of_set_to_eval.append(val_db[idx][2])
                try:
                    acc_of_set_to_eval.append(val_db[idx][3])
                except:
                    pass
            counter = counter + 1
            done = (len(eval_set) == to_eval_valids)
            if counter > 1000:
                logging.info("possibly not enough valids which fit into the new space")
                break
        set_to_eval_copy = deepcopy(set_to_eval)
        logging.info("To evaluate previous designs number:" + str(len(set_to_eval_copy)))
        ## add extra designs to fill in all nodes
        if len(set_to_eval_copy) < self.get_configuration().parall:
            logging.info("not enough designs..")
            for kkk in range(len(set_to_eval_copy) - self.get_configuration().parall):
                part = self.generate(design_space)
                part = self.create_particle(part)
                part = self.filter_particle(part)
                set_to_eval.append(part)
            
        logging.info("To evaluate designs number:" + str(len(set_to_eval)))
        all_in = False
        while(not all_in):
            logging.info("First Stage Knowledge sampling plan has " + str(len(set_to_eval)) + " designs left for evaluation")
            des_to_eval = minimum(self.llambda-self.currently_evaled,len(set_to_eval)) 
            parts = set_to_eval[0:des_to_eval]
            set_to_eval = set_to_eval[des_to_eval:]
            for part in parts:
                part = self.create_particle(part)
                self.fitness_function(part)
            self.update_running_que()
            all_in = True
            for idx, st in enumerate(set_to_eval_copy):
                isin = False
                for ts in self.surrogate_model.classifier.training_set: ## put in extra into the que
                    if all(ts == array(st)):
                        isin = True
                        break
                all_in = isin and all_in
            print len(set_to_eval)
            if len(set_to_eval) == 0: 
                found = False
                self.surrogate_model.classifier_train()
                while ( not found ):
                    part = self.generate(design_space)
                    part = self.create_particle(part)
                    part = self.filter_particle(part)
                    label = self.surrogate_model.classifier.predict([part]) 
                    if label == 1.:
                        logging.info("found")
                        set_to_eval.append(part)
                        found = True
        try:
            self.surrogate_model.train()
        except:
            logging.info("not enough data in the db")
            self.classifier_aware_sample_plan(len(set_to_eval), design_space)
            return 
        
        
         #pdb.set_trace()
        #TODO hmmm
        #pdb.set_trace()
        #for ii in range(self.get_configuration().parall):
        #    self.update_running_que()
        ## find a fitness regression 
        old_fitnesses = []
        new_fitnesses = []
        old_accuracy = []
        new_accuracy = []
        print set_to_eval_copy
        print self.surrogate_model.regressor.training_set
        for idx1, ts in enumerate(self.surrogate_model.regressor.training_set):
            for idx, st in enumerate(set_to_eval_copy):
                if all(ts == array(st)):
                    old_fitnesses.append([fit_of_set_to_eval[idx]])
                    new_fitnesses.append([self.surrogate_model.regressor.training_fitness[idx1][0]])
                    try:
                        old_accuracy.append([acc_of_set_to_eval[idx]])
                        new_fitnesses.append([self.surrogate_model.regressor.training_fitness[idx1][0]])
                        new_accuracy.append([self.fitness.fitnessFunc(st,{},2)])
                    except:
                        pass
        old_accuracy = array(old_accuracy)
        new_accuracy = array(new_accuracy)
        old_fitnesses = array(old_fitnesses)
        new_fitnesses = array(new_fitnesses)
        logging.info("Old: " + str(old_fitnesses))
        logging.info("New: " + str(new_fitnesses))
        
        rho, p = pearsonr(old_fitnesses, new_fitnesses)
        logging.info("Pearson correlation:" + str(rho) + " " + str(p))       
        if (p < 0.1):
            logging.info("plugin using Pearson")
            output = stats.linregress(old_fitnesses.reshape((-1,)),new_fitnesses.reshape((-1,)))
        else:
            rho, p = spearmanr(old_fitnesses, new_fitnesses, axis=None)
            logging.info("Spearman correlation:" + str(rho) + " " + str(p))        
            if (p < 0.1):
                logging.info("using GP for maping")
                input_scaler = preprocessing.StandardScaler().fit(old_fitnesses)
                output_scaler = preprocessing.StandardScaler(with_std=False).fit(new_fitnesses)
                xx = input_scaler.transform(old_fitnesses)
                yy = output_scaler.transform(new_fitnesses)
                output = stats.linregress(xx.reshape((-1,)),yy.reshape((-1,)))
            
        logging.info("linregress: " + str(output))
        slope = output[0]
        intercept = output[1]
        
        def predict_fit(z):
            return intercept + slope * z
        #pdb.set_trace()
        #if fit2:
        #    pdb.set_trace()
        
        '''
        #linear model
        k = cov.covMatern([1,1,3]) 
        k = k + cov.covNoise([-1])
        m = mean.meanLinear([slope])
        l = lik.likGauss([log(0.1)])
        i = inf.infExact()
        conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
        conf.max_trails = 100
        conf.covRange = [(-2,4),(-2,4),(3,3),(-2,1)]
        conf.likRange = [(0,1.0)]
        o = opt.Minimize(conf)
        nlZ_trained = gp.train(i,m,k,l,xx,yy,o)
        logging.info("Neg Likelihood Transfer Knowledge: " + str(nlZ_trained))
        def predict_fit(z):
            zt = input_scaler.transform(z)
            out = gp.predict(i,m,k,l,xx,yy,zt)

            ym  = out[0]
            ys2 = out[1]
            mm  = output_scaler.inverse_transform(out[2])
            s2  = out[3]
            return mm
        ##retrain classifier 
        trained = self.train_surrogate_model()
        '''

        
        
        ##evaluate F/2 random designs over the valid region
        #gonna_eval = 0# max(F - to_eval_valids, 0)
        #logging.info("I am going to evaluate:" + str(gonna_eval))
        #self.classifier_aware_sample_plan(gonna_eval, design_space)
        
        
        #plug in all designs which are within valid region and have a valid regression (as valid)
        for idx in range(len(set_to_eval_copy)):
            if not (idx in eval_set):
                part = array(val_db[idx][0])
                val = array(val_db[idx][2])
                #code = self.surrogate_model.classifier.predict(array([part, part]))
                #if code[0] == 1.0: ## plug in the old designs
                logging.info("Plugging design")
                intput_array = array([[val], [val]])
                logging.info(intput_array)
                fit = predict_fit(intput_array)
                logging.info(str(fit[0]))
                self.surrogate_model.regressor.add_training_instance(part, fit[0])
                #if self.fitness.meetsResources(part):
                #    self.surrogate_model.classifier.add_training_instance(part, 1.0)
                #else:
                #    self.surrogate_model.classifier.add_training_instance(part, 2.0)
                #part.fitness.values = fit[0]
                #self.update_best(part) ## TODO - not sure about this
                #self.surrogate_model.add_training_instance(part, array([code[0]]), fit[0], array([0]))
                #else:
                #    logging.info("Bip bip invalid detected")
        #for idx in range(len(inval_db)): # hack for rtm
        #    part = array(inval_db[idx][0])
        #    if self.fitness.meetsResources(part):
        #        self.surrogate_model.classifier.add_training_instance(part, 1.0)
        #    else:
        #        self.surrogate_model.classifier.add_training_instance(part, 2.0)
        #some constraints failed, just add fitness.  self.create_particle(new_data)
        for idx in range(len(cval_db)):
            part = array(cval_db[idx][0])
            code = array(cval_db[idx][1])
            val = array(cval_db[idx][2])
            logging.info("Revaluating constraints and plugging design")
            intput_array = array([[val], [val]])
            logging.info(intput_array)
            fit = predict_fit(intput_array)
            #part.fitness.values = fit[0]
            #self.update_best(part) ## TODO - not sure about this
            if True:
                self.surrogate_model.regressor.add_training_instance(part, fit[0])
            #pdb.set_trace()
            if code == 2. and (not switch): #plugin failed constrains
                logging.info("plugging in accuracy code")
                self.surrogate_model.classifier.add_training_instance(part, code)
        logging.info("Knowledge sampling done")
                
    def naive_sample_plan(self, F, D, design_space):
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(int(F),int(D)))
        max_bounds = [d["max"] for d in design_space]
        min_bounds = [d["min"] for d in design_space]
        #for some reason nans appear in hypercube.. rpleace them with average value of the columns
        col_mean = stats.nanmean(latin_hypercube_samples,axis=0)
        print col_mean
        #Find indicies that you need to replace
        inds = where(isnan(latin_hypercube_samples))

        #Place column means in the indices. Align the arrays using take
        latin_hypercube_samples[inds]=take(col_mean,inds[1])

        steps = [d["step"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples * (array(max_bounds)-array(min_bounds))
        latin_hypercube_samples = latin_hypercube_samples + array(min_bounds) + array(steps) * 0.5
        while(len(latin_hypercube_samples) > 0):
            logging.info("Naive sampling plan has " + str(len(latin_hypercube_samples)) + " designs left for evaluation")
            des_to_eval = minimum(self.llambda-self.currently_evaled,len(latin_hypercube_samples)) 
            parts = latin_hypercube_samples[0:des_to_eval]
            latin_hypercube_samples = latin_hypercube_samples[des_to_eval:]
            for part in parts:
                part = self.toolbox.filter_particle(self.create_particle(part))
                self.fitness_function(part)
            self.update_running_que()
        
    def get_random_valid_design(self):
        valid_designs = self.surrogate_model.get_valid_set()
        i = len(valid_designs) 
        logging.info("huj i:" + str(i))
        if i == 0:
            logging.info("problem.. no valid designs")
        elif i == 1:
            logging.info("Only one valid design")
            return valid_designs[0]
        return valid_designs[randint(0,i,1)[0]] ## upperbound is e
    
    ##TODO - possibly disaple sample time... insignificant
    def classifier_aware_sample_plan( self, F, design_space):
        counter = 0
        counter2 = 0
        self.surrogate_model.classifier_train()
        while counter < F:
            if minimum(self.llambda-self.currently_evaled,F-counter):
                time_sample_rand = time.time()
                ### sample a design
                part = self.generate(design_space)
                part = self.create_particle(part)
                part = self.filter_particle(part)
                label = self.surrogate_model.classifier.predict([part]) 
                if counter2 % 10000 == 0.:
                    logging.info('Iteration ' + str(counter2) + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                if label == 1.:  ### its valid... we use different label for valid class
                    logging.info("valid found" + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                    time_sample_rand = time.time() - time_sample_rand  
                    self.update_running_que_by(time_sample_rand)
                    part = self.create_particle(part)
                    self.fitness_function(part)
                    self.surrogate_model.classifier_train()
                   # if code == 0 or code == 3: ### its a hack... finish it
                    counter = counter + 1
                else:
                    if counter2 % 100000 == 0.: 
                        logging.info('Evaluation random perturbation of always_valid' + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                        time_sample_rand = time.time() - time_sample_rand  
                        self.update_running_que_by(time_sample_rand)
                        part = self.create_particle(self.get_random_valid_design() + self.perturbation())
                        part = self.toolbox.filter_particle(part)
                        self.fitness_function(part)
                        #if code == 0:
                        counter = counter + 1
                        self.surrogate_model.classifier_train()
                    else:
                        time_sample_rand = time.time() - time_sample_rand  
                        self.update_running_que_by(time_sample_rand,disp=False)
                counter2 = counter2 + 1
            else:
                self.update_running_que()
        self.update_running_que()
        logging.info(str(self.get_time_stamp()))
        
    ##TODO - possibly disaple sample time... insignificant
    def classifier_aware_sample_plan_propa(self, F, design_space):
        counter = 0
        counter2 = 0
        self.surrogate_model.classifier_train()
        while counter < F:
            if minimum(self.llambda-self.currently_evaled,F-counter):
                time_sample_rand = time.time()
                ### sample a design
                part = self.generate(design_space)
                part = self.create_particle(part)
                part = self.filter_particle(part)
                label = self.surrogate_model.classifier.predict([part]) 
                if counter2 % 10000 == 0.:
                    logging.info('Iteration ' + str(counter2) + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                if label >= 0.5:  ### its valid... we use different label for valid class
                    logging.info("valid found" + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                    time_sample_rand = time.time() - time_sample_rand  
                    self.update_running_que_by(time_sample_rand)
                    part = self.create_particle(part)
                    self.fitness_function(part)
                    self.surrogate_model.classifier_train()
                   # if code == 0 or code == 3: ### its a hack... finish it
                    counter = counter + 1
                else:
                    if counter2 % 100000 == 0.: 
                        logging.info('Evaluation random perturbation of always_valid' + ' found so far:' + str(counter) + ' left:' + str(F - counter))
                        time_sample_rand = time.time() - time_sample_rand  
                        self.update_running_que_by(time_sample_rand)
                        part = self.create_particle(self.get_random_valid_design() + self.perturbation())
                        part = self.toolbox.filter_particle(part)
                        self.fitness_function(part)
                        #if code == 0:
                        counter = counter + 1
                        self.surrogate_model.classifier_train()
                    else:
                        time_sample_rand = time.time() - time_sample_rand  
                        self.update_running_que_by(time_sample_rand,disp=False)
                counter2 = counter2 + 1
            else:
                self.update_running_que()
        self.update_running_que()
        logging.info(str(self.get_time_stamp()))
                
    def perturbation(self):
        d = [0.] * len(self.fitness.designSpace)
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = self.fitness.designSpace[i]["step"]
            elif self.fitness.designSpace[i]["type"] == "continuous":
                d[i] = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
        dimensions = len(self.fitness.designSpace)
        pertubation = multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
                
    def sample_plan(self):
        design_space = self.get_design_space()
        D = len(design_space)
        F = D * 10
        allok = True
        try:
            self.get_configuration().old_run_dir ## gonna break if its not there
        except:
            allok = False
            
        if allok:
            logging.info("Previous optimization database supplied.. using knowledge aspects")
            self.knowledge_aware_sample_plan(F)
        else:
            try:
                part = self.create_particle(self.fitness.always_valid)
                self.toolbox.filter_particle(part)
                self.fitness_function(part)
                logging.info("Always valid configuration present, evaluated")
            except:
                pdb.set_trace()
                logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
                self.set_at_least_one_in_valid_region(False)
            self.update_running_que() ## important to do it here... so that we got at least one valid design built
            ## always valid
            try:
                part = self.create_particle(self.fitness.always_valid)
                self.toolbox.filter_particle(part)
                self.fitness_function(part)
                logging.info("Always valid configuration present, evaluated")
            except:
                logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
                self.set_at_least_one_in_valid_region(False)
            ## sampling
            logging.info("Previous optimization database not supplied.. ")
            if self.get_configuration().sampling_fancy: ## fancy shmancy
                
                #if self.surrogate_model.propa_classifier: ## this crap is added as RVMs are hard to integrate into this sampling plan
                #    store_classifier = self.surrogate_model.classifier
                #    self.surrogate_model.classifier = SupportVectorMachineClassifier()
                #    self.surrogate_model.classifier.training_set = store_classifier.training_set
                #    self.surrogate_model.classifier.training_labels = store_classifier.training_labels
                self.naive_sample_plan(2 * F/3, D, design_space)
                logging.info("Naive sampling done")
                if self.surrogate_model.propa_classifier:
                    self.classifier_aware_sample_plan_propa(F/3, design_space)
                else:
                    self.classifier_aware_sample_plan(F/3, design_space)
                logging.info("Sampling of valid space done")
                #if self.surrogate_model.propa_classifier:
                #    store_classifier.training_set = self.surrogate_model.classifier.training_set
                #    store_classifier.training_labels = self.surrogate_model.classifier.training_labels
                #    self.surrogate_model.classifier = store_classifier
            else: #self.configuration.sampling_plan == 'naive': 
                self.naive_sample_plan(F, D, design_space)
            
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        ### generate folders
        results_folder, images_folder, dump_folder = self.create_results_folder()
        self.dump_folder = dump_folder
        if not results_folder or not images_folder or not dump_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
            
        self.run_initialize()
        self.use_cost_model = False
        '''
        try:
            logging.info("Cost model will be used for prediction")
            self.configuration.cost
            self.use_cost_model = True
            self.cost_model = ProperCostModel(self.configuration, self.controller, self.fitness)
        except:
        '''
        logging.info("Cost model wont be used for prediction")
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        
        self.state_dictionary['best'] = None
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("i")
        self.set_counter_dictionary("i",0)
        
        self.current_sim_time = 0 ## in seconds
        
        self.running_que = []
        self.running_que_cost = []
        self.currently_evaled = 0
        self.new_design_added = False
        try:
            self.llambda = self.configuration.parall
        except:
            logging.info("Set the level of parallelizm parall in the config file")
            exit(0)
        self.sample_plan()
        self.state_dictionary["fresh_run"] = True
        return True
    
    def run_initialize(self):
        logging.info("Initialize MonteCarloTrial no:" + str(self.get_trial_no()))
        design_space = self.get_design_space()
        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space])
        self.toolbox = copy(base.Toolbox())
        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particle', self.filter_particle)
        self.new_best=False
                                  
    ## main computation loop goes here
    def run(self):
        self.state_dictionary['generate'] = True
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        self.state_dictionary["fresh_run"] = False
        #self.currently_evaled = 0
        #self.configuration.parall=1
        #while self.get_counter_dictionary('i') < self.get_configuration().max_iter + 1: ## HACK
        while self.get_counter_dictionary('fit') < self.get_configuration().max_fitness + 1: ## HACK
            logging.info('')
            logging.info('')
            logging.info('[' + str(self.get_name()) + '] Iteration ' + str(self.get_counter_dictionary('i')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))
            logging.info('[' + str(self.get_best()) + '] Best')
            #if self.get_counter_dictionary('fit') > 80.0:
            #    pdb.set_trace()
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            #if self.training_set_updated():
            time_train = time.time()
            try:
                trained = self.train_surrogate_model()
            except Exception, e:
                logging.info("Intiial training went wrong: " + str(e))
            trained_cost = self.train_cost_model()
            time_train = time.time() - time_train
            if len(self.fitness.designSpace) < 4: ## this is a simulator... our infill has a slow implementation and causes problems with design evaluation
                self.update_running_que_by(time_train)
            #if self.get_counter_dictionary('i') % 10 == 1.:
            #    pdb.set_trace()
            #TODO - with view_update it stalls... 
            self.view_update(visualize = True)
            self.save()
            try:
                self.surrogate_model.get_bests_limit()
            except:
                logging.info("Best wasnt set")
                design_space = self.get_design_space()
                D = len(design_space)
                self.classifier_aware_sample_plan(D, design_space)
                
            if self.get_model_failed():
                # termination condition - we put it here so that when the trial is reloaded
                # it wont run if the run has terminated already
                time_inf = time.time()
                logging.info("Currently evaluated: " + str(len(self.get_miu_set())) + " free nodes: " + str(self.get_no_to_eval()))
                time_ei_search = time.time()
                best_xs = self.max_ei_par(self.get_miu_set(), self.get_no_to_eval())
                time_ei_search = time.time() - time_ei_search
                logging.info("EI search took:" + str(time_ei_search))
                #pdb.set_trace()
                #if best_xs == []:
                #    self.surrogate_model.classifier.train(local_structure=True) ## we retrain the classifier to find local structures between the class boundaries
                #    best_xs = self.get_around_best(radius=3.,random=False, without_class=False)
                #    self.surrogate_model.classifier.train(local_structure=False)
                #    logging.info("EI didnt find anything.. evaluating " + str(best_xs))
                logging.info(str(self.surrogate_model.get_bests_limit()))
                
                if True: ## HACK
                    if (best_xs == []) or (best_xs is None):
                        best_xs = self.get_around_best(random=True, llambda=self.get_no_to_eval())
                        logging.info("Local EI didnt find anything.. evaluating random " + str(best_xs))
                    else:
                        logging.info("Local EIs predicted at: " + str(best_xs))
                
                # if best_xs == []:
                    # time_ei_search = time.time()
                    # best_xs = self.max_ei_par(self.get_miu_set(), self.get_no_to_eval(), local=True)
                    # time_ei_search = time.time() - time_ei_search
                    # logging.info("EI search took:" + str(time_ei_search))
                    # if best_xs == []:
                        # best_xs = self.get_around_best(random=True, llambda=self.get_no_to_eval())
                        # logging.info("Local EI didnt find anything.. evaluating random " + str(best_xs))
                    # else:
                        # logging.info("Local EIs predicted at: " + str(best_xs))
                # else:
                    # logging.info("Best EIs predicted at: " + str(best_xs))
                  
                if len(self.fitness.designSpace) < 4: ## this is a simulator... our infill has a slow implementation and causes problems with design evaluation
                    time_inf = time.time() - time_inf
                    self.update_running_que_by(time_inf)
                
                for best_x in best_xs:
                    logging.info("Preparing " + str(best_x))
                    if False: ## HACK
                        if not self.surrogate_model.contains_training_instance(best_x):
                            best_x = self.toolbox.filter_particle(self.create_particle(best_x))
                            self.fitness_function(best_x)
                    else:
                        logging.info("Trying " + str(best_x))
                        best_x = self.toolbox.filter_particle(self.create_particle(best_x))
                        logging.info("Filtered " + str(best_x))
                        already_evaled = self.fitness_function(best_x)
                        if not already_evaled:
                            logging.info("Will try perturbation")
                        if self.surrogate_model.contains_training_instance(best_x) or not already_evaled:
                            logging.info(str(best_x) + " already evaluated...")
                            best_x = self.get_around_best(random=True, llambda=1)[0]
                            logging.info("evaluating random " + str(best_x))
                            best_x = self.toolbox.filter_particle(self.create_particle(best_x))
                            self.fitness_function(best_x)
            else:
                logging.info("Training surrogate model failed.. evaluating random and retraining")
                best_x = self.get_around_best(random=True, llambda=self.get_no_to_eval())
                logging.info("Preparing " + str(best_x))
                best_x = self.toolbox.filter_particle(self.create_particle(best_x))
                self.fitness_function(best_x)
            #if self.was_new_design_added():
            self.update_running_que()
            self.increment_main_counter()
            
        ## dump data for next optimiaztion
        self.dump()
        self.exit()
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def was_new_design_added(self):
        temp = self.new_design_added
        self.new_design_added = False
        return temp
    
    def get_llambda(self):
        return  minimum(self.llambda,len(self.running_que)) 
    
    def update_running_que_by(self, time_passed, disp=True):
        running_que_cost = array([r.flatten() for r in self.running_que_cost])
        running_que = self.running_que
        
        llambda = self.get_llambda()
        running_que_cost[0:llambda] = array(running_que_cost[0:llambda]) - time_passed
        evaluated_list = [part for part, cost in zip(running_que,running_que_cost) if cost <= 0.0] ## desings which finished
        self.running_que = [part for part, cost in zip(running_que,running_que_cost) if cost > 0.0]
        self.running_que_cost = [cost for cost in running_que_cost if cost > 0.0]
        
        self.update_system_time(time_passed, disp=disp)
        self.set_counter_dictionary("cost", self.current_sim_time)
        for part in evaluated_list:
            self.fitness_function(part,add=True)
        #pdb.set_trace()
        
    def update_running_que(self):
        running_que_cost = array([r.flatten() for r in self.running_que_cost])
        if len(running_que_cost) > 0:
            if self.configuration.limit_lambda_search:
                if (self.get_no_to_eval() > 0): ## design can be added to evaluation que
                    pass
                else:
                    llambda = minimum(self.llambda,len(running_que_cost)) 
                    finishing = argmin(running_que_cost[0:llambda]) ## get the evaluations which are finishing now
                    time_update = running_que_cost[finishing][0]
                    self.update_running_que_by(time_update)
            else:
                llambda = minimum(self.llambda,len(running_que_cost)) 
                finishing = argmin(running_que_cost[0:llambda]) ## get the evaluations which are finishing now
                time_update = running_que_cost[finishing][0]
                self.update_running_que_by(time_update)
        else:
            logging.info("Running que emtpy... omitting, probably a duplicate is to be evaluated")
    
    def get_miu_set(self):
        return array(self.running_que)
        
    def get_no_to_eval(self):
        return self.llambda - self.currently_evaled
        
    ### a hypercube that spans all the valid designs
    def hypercube(self):
        set_to_search = self.surrogate_model.get_valid_set()
        #find maximum
        max_diag = deepcopy(set_to_search[0])
        for part in set_to_search:
            max_diag = maximum(part,max_diag) 
        ###find minimum vectors
        min_diag = deepcopy(set_to_search[0])
        for part in set_to_search:
            min_diag = minimum(part,min_diag)
            
        ## we always ensure that the hypercube allows particles to maintain velocity components in all directions
        
        for i,dd in enumerate(max_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                max_diag[i] = minimum(dd + self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["max"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                max_diag[i] = minimum(dd + small_fraction, self.fitness.designSpace[i]["max"])
                
        for i,dd in enumerate(min_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                min_diag[i] = maximum(dd - self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["min"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                min_diag[i] = maximum(dd - small_fraction, self.fitness.designSpace[i]["min"])
        #logging.info("hypecube: " + str([max_diag,min_diag]))
        return [max_diag,min_diag]
    
    def generate(self, designSpace):
        '''
        [max_diag,min_diag] = self.hypercube()
        particle = [uniform(max_d,min_d)
                    for max_d,min_d
                    in zip(max_diag,min_diag)]
        
        '''
        particle = [uniform(dim_space["min"],dim_space["max"])
                    for dim_space
                    in designSpace]
        
        particle = self.create_particle(particle)
        #import pdb
        #pdb.set_trace()
        return particle
    
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        if self.use_cost_model:
            model = ProperCostModel(self.get_configuration(), self.controller, self.fitness)
        else:
            model = DummyCostModel(self.get_configuration(), self.controller, self.fitness) 
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
                                  
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('i'))
        name = self.get_name()
        return_dictionary = {
            'goal': self.get_configuration().goal,
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'propa_classifier': self.get_surrogate_model().propa_classifier,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness,
            'best': {"data":self.get_best()}
        }
        return return_dictionary

    def save(self):
        try:
            trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('i')) + '.txt'
            dict = self.state_dictionary
            surrogate_model_state_dict = self.surrogate_model.get_state_dictionary()
            dict['surrogate_model_state_dict'] = surrogate_model_state_dict
            cost_model_state_dict = self.cost_model.get_state_dictionary()
            dict['cost_model_state_dict'] = cost_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                pickle.dump(dict, outfile)  
                if self.kill:
                    sys.exit(0)
        except Exception, e:
            logging.error(str(e))
            if self.kill:
                sys.exit(0)
            return False
            
    ## by default find the latest iteration
    def load(self, iteration = None):
        try:
            if iteration is None:
                # Figure out what the last iteration before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last iteration
                        iteration = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            iteration_file = str(iteration)
            trial_file = str(self.get_results_folder()) + '/' + str(iteration_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            self.surrogate_model.set_state_dictionary(dict['surrogate_model_state_dict'])
            self.cost_model.set_state_dictionary(dict['cost_model_state_dict'])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False

    def max_ei_par(self, miu_set, llambda, local=False):
        if llambda <1:
            pdb.set_trace()
        designSpace = self.get_design_space()  
        D = len(designSpace)
        use_hardware = self.configuration.hardware
        if self.configuration.limit_lambda_search:
            llambda = 1
            logging.info("Limiting llambda to 1")
        if self.use_cost_model:
            cost_model = self.cost_model
        else:
            cost_model = False
        if D < 5 and llambda == 1: ## speeds up everything
            npts = 0.01
            point = self.get_best()
            n_bins = npts*ones(D)
            steps = []
            for counter, d in enumerate(designSpace):
                if d["type"] == "discrete":
                    n_bins[counter] = int((d["max"] - d["min"])/ d["step"]) + 1.0
                    steps.append(d["step"])
                else:
                    n_bins[counter] = npts
                    steps.append(int((d["max"] - d["min"])/ npts))
            bounds = [(d["min"],d["max"]) for d in designSpace]
            result = mgrid[[slice(row[0], row[1], n*1.0j) for row,n in zip(bounds, n_bins)]]
            z = result.reshape(D,-1).T
            local_perturbation = [zz for zz in z if not (self.surrogate_model.contains_training_instance(zz) or numpy_array_index(self.running_que,zz)[0])]
            
            if use_hardware:
                particles = self.surrogate_model.max_ei_par(self.get_design_space(),  miu_set, 1., local, cost_model=cost_model)
            else:
                logging.info("no hardware acceleration turned on for EI")
                particles = self.surrogate_model.max_ei_par_soft(self.get_design_space(),  miu_set, 1., local, cost_model=cost_model)
        else:
            if use_hardware:
                particles = self.surrogate_model.max_ei_par(self.get_design_space(),  miu_set, llambda, local, cost_model=cost_model)
            else:
                logging.info("no hardware acceleration turned on for EI")
                particles = self.surrogate_model.max_ei_par_soft(self.get_design_space(),  miu_set, llambda, local, cost_model=cost_model)
        if particles is None:
            return []
        else:
            try:
                particles = particles.reshape(-1,len(self.get_design_space()))## NOT SELF
                return particles
            except:
                if particles is ():
                    return []
            
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())
        
    def get_around_best(self, llambda, radius = 1., random=True, without_class=False):
        designSpace = self.get_design_space()  
        D = len(designSpace)
        npts = 0.01
        
        n_bins = npts*ones(D)
        points = self.surrogate_model.get_bests_limit()
        ## define the space
        local_perturbation = []
        for point in points:
            steps = []
            for counter, d in enumerate(designSpace):
                if d["type"] == "discrete":
                    n_bins[counter] = d["step"]
                else:
                    n_bins[counter] = 1./npts
            bests = self.surrogate_model.get_bests_limit()
            bounds = [(maximum(point[i] - n_bins[i]* radius,d["min"]), minimum(point[i] + n_bins[i]*radius,d["max"])) for i,d in enumerate(designSpace)] 
            #current_max = -1.
            #current_max_cord = None 
            ### preapring search grid... used a lot of memory for large spaces
            result = mgrid[[slice(row[0], row[1], int((row[1]-row[0])/n_bins[i]+1)*1.0j) for i,row in enumerate(bounds)]]
            z = result.reshape(D,-1).T
            local_perturbation.extend([zz for zz in z if not (self.surrogate_model.contains_training_instance(zz) or numpy_array_index(self.running_que,zz)[0])])
        if random:
            length = len(local_perturbation)
            logging.info("Number of possible local perturbations is :" + str(length))
            if length > 0:
                rand_indicies = []
                #pdb.set_trace()
                while len(rand_indicies) < minimum(llambda,length):
                    rand_index = randint(0,length) 
                    rand_indicies.append(rand_index)
                chosen = [local_perturbation[rand_index] for rand_index in rand_indicies]
                logging.info("Perturbation index chosen :" + str(rand_indicies) + " desing: " + str(chosen))
                return chosen
            else: 
                logging.info("increasing perturbation radius... radius:" + str(radius))
                return self.get_around_best(llambda, radius=radius+1.)
        else:
            logging.info("Local exhaustive evaluation..")
            particles = self.surrogate_model.max_ei_par(self.get_design_space(),  self.get_miu_set(), 1., local_perturbation, without_class=without_class)
            if particles is None:
                return []
            else:
                return [particles]
            
    def filter_particle(self, p):#
        design_space = self.get_design_space()  
        pmin = [dimSetting['min'] for dimSetting in design_space]
        pmax = [dimSetting['max'] for dimSetting in design_space]
        for i, val in enumerate(p):
            #dithering
            if design_space[i]['type'] == 'discrete':
                p[i] = p[i]/design_space[i]['step']
                p[i] = floor(p[i])
                p[i] = p[i] * design_space[i]['step']
            #dont allow particles to take the same value
            p[i] = minimum(pmax[i], p[i])
            p[i] = maximum(pmin[i], p[i])
        return p
        
    def update_best(self, part):
        if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
            best = self.create_particle(part)
            best.fitness.values = part.fitness.values
            #pdb.set_trace()
            self.set_best(best)
            
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
            
    def predict_cost(self, particle):
        try:
            return self.cost_model.predict(particle)
        except Exception,e:
            logging.debug("Cost model is still not avaiable: " + str(e))
            "a" + 1 
            return 1.0 ## model has not been created yet
            
### simulated... needs a bit of rework to be truly parallel.
class Gradient_Trial(Trial):

    def sample_plan(self):
        try:
            part = self.create_particle(self.fitness.always_valid)
            self.toolbox.filter_particle(part)
            self.fitness_function(part)
            logging.info("Always valid configuration present, evaluated")
        except:
            pdb.set_trace()
            logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
            self.set_at_least_one_in_valid_region(False)
        self.update_running_que()
        design_space = self.get_design_space()
        D = len(design_space)
        F = self.configuration.parall#D * 10
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(F,D))
        max_bounds = [d["max"] for d in design_space]
        min_bounds = [d["min"] for d in design_space]
        steps = [d["step"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples * (array(max_bounds)-array(min_bounds))
        latin_hypercube_samples = latin_hypercube_samples + array(min_bounds) + array(steps) * 0.5
        while(len(latin_hypercube_samples) > 0):
             logging.info("Naive sampling plan has " + str(len(latin_hypercube_samples)) + " designs left for evaluation")
             des_to_eval = minimum(self.llambda-self.currently_evaled,len(latin_hypercube_samples)) 
             parts = latin_hypercube_samples[0:des_to_eval]
             latin_hypercube_samples = latin_hypercube_samples[des_to_eval:]
             for part in parts:
                part = self.toolbox.filter_particle(self.create_particle(part))
                self.fitness_function(part)
             self.update_running_que()
        logging.info("running que:" + str(len(self.running_que_cost)))
        #pdb.set_trace()
       

### check first if part is already within the training set

    def fitness_function(self, part, add=False):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
            logging.info("Particle already evaluated " + str(part))
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        try:
            results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            #logging.info(str(e))
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3]
            try: ##checking if hardware is being built
                a, cost_b = self.fitness.has_hardware(self.running_que, self.running_que_cost, part)
                if a:
                    cost = cost_b + 1
            except Exception, e:
                logging.info(str(e))
                pass
        except:
            logging.info("This version makes no sense if you cannot get the cost... exiting")
            exit(0)
        if add:
            self.increment_counter('fit')
            self.currently_evaled = self.currently_evaled - 1
            logging.info("Finished Evaluating " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
            self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
            self.cost_model.add_training_instance(part, cost)     
            if code[0] == 0:
                try:
                   part.fitness.values = fitness
                except: 
                   part = self.toolbox.filter_particle(self.create_particle(part))
                   part.fitness.values = fitness
                self.update_best(part)
                self.set_terminating_condition(fitness) 
                return fitness, code, cost
            else:
                return fitness, code, cost
        else:
            if not numpy_array_index(self.running_que,part)[0]:
                self.running_que.append(part)
                self.running_que_cost.append(cost)
                self.currently_evaled = self.currently_evaled + 1
                logging.info("Evaluating " + str(part))
                self.new_design_added = True
                return fitness, code, cost
            else:
                logging.info("Design is already being evaluated")
                return fitness, code, cost

    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        #self.cost_model = ProperCostModel(self.configuration, self.controller, self.fitness)
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        self.state_dictionary['best'] = None
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("i")
        self.set_counter_dictionary("i",0)
        self.state_dictionary["fresh_run"] = True

        self.current_sim_time = 0 ## in seconds
        
        self.running_que = []
        self.running_que_cost = []
        self.currently_evaled = 0
        self.new_design_added = False
        try:
            self.llambda = self.configuration.parall
        except:
            logging.info("Set the level of parallelizm parall in the config file")
            exit(0)
        
        ### generate folders
        results_folder, images_folder, dump_folder = self.create_results_folder()
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        return True
    
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
    
    def generate(self, designSpace):
        '''
        [max_diag,min_diag] = self.hypercube()
        particle = [uniform(max_d,min_d)
                    for max_d,min_d
                    in zip(max_diag,min_diag)]
        
        '''
        particle = [uniform(dim_space["min"],dim_space["max"])
                    for dim_space
                    in designSpace]
        
        particle = self.create_particle(particle)
        #import pdb
        #pdb.set_trace()
        return particle
        
    def filter_particle(self, p):#
        design_space = self.get_design_space()  
        pmin = [dimSetting['min'] for dimSetting in design_space]
        pmax = [dimSetting['max'] for dimSetting in design_space]
        for i, val in enumerate(p):
            #dithering
            if design_space[i]['type'] == 'discrete':
                p[i] = p[i]/design_space[i]['step']
                p[i] = floor(p[i])
                p[i] = p[i] * design_space[i]['step']
            #dont allow particles to take the same value
            p[i] = minimum(pmax[i], p[i])
            p[i] = maximum(pmin[i], p[i])
        return p
    
    def update_best(self, part):
        if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
            best = self.create_particle(part)
            best.fitness.values = part.fitness.values
            self.set_best(best)
    
    def run_initialize(self):
        logging.info("Initialize MonteCarloTrial no:" + str(self.get_trial_no()))
        design_space = self.get_design_space()
        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space])
        self.toolbox = copy(base.Toolbox())
        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particle', self.filter_particle)
        self.new_best=False
                                 

    def hill_climbing_optimizer(self, x):  
        designSpace = self.fitness.designSpace
        
        def value(part):
            return self.fitness_function(part)[0][0]
    
        def code(part):
            return self.fitness_function(part)[1]
     
        def sons(x, radius=1.0): ## radius +/-1
            npts = 1
            D = len(designSpace)
            n_bins = npts*ones(D)
            for counter, d in enumerate(designSpace):
                if d["type"] == "discrete":
                    n_bins[counter] = d["step"]
                else:
                    n_bins[counter] = 1./npts
            try:
                bounds = [(maximum(x[i] - n_bins[i]* radius,d["min"]), minimum(x[i] + n_bins[i]*radius,d["max"])) for i,d in enumerate(designSpace)] 
            except:
                pdb.set_trace()
            #current_max = -1.
            #current_max_cord = None 
            ### preapring search grid... used a lot of memory for large spaces
            result = mgrid[[slice(row[0], row[1], int((row[1]-row[0])/n_bins[i]+1)*1.0j) for i,row in enumerate(bounds)]]
            z = result.reshape(D,-1).T
            z = [zz for zz in z if not self.surrogate_model.contains_training_instance(zz)]
            return z

        def hill_climbing(x):
            """
            A represents a configuration of the problem.
            It has to be passed as argument of the function sons(A)
            ans value(A) that determines, respectively, the next
            configuration starting from A where Hill Climbing Algorithm
            needs to restart to evaluate and the heuristic function h.

            This function represents a template method.
            """
            best_val = self.fitness.worst_value
            best_conf = None
            for conf in sons(x):
                val = value(conf)
                logging.info("Lenght of running que: " + str(len(self.running_que)))
                logging.info("llambda: " + str(self.llambda))
                if (len(self.running_que)>=self.llambda):
                    self.update_running_que()
                    if self.configuration.goal == "min":
                        if best_val > val:
                            best_conf = conf
                            best_val = val
                            if self.get_terminating_condition() or (self.get_counter_dictionary('fit') > self.get_configuration().max_fitness):
                               return x
                    elif self.configuration.goal == "max":
                        if best_val < val:
                            best_conf = conf
                            best_val = val
                            if self.get_terminating_condition() or (self.get_counter_dictionary('fit') > self.get_configuration().max_fitness):
                                return x
        
            if best_conf is None:
                return x
            if self.configuration.goal == "min":
                if (best_val > value(x)) or self.get_terminating_condition() or (self.get_counter_dictionary('fit') > self.get_configuration().max_fitness):
                    return x
                return hill_climbing(best_conf)
            elif self.configuration.goal == "max":
                if (best_val < value(x)) or self.get_terminating_condition() or (self.get_counter_dictionary('fit') > self.get_configuration().max_fitness):
                    return x
                return hill_climbing(best_conf)
                
        import itertools
        import random
        from numpy import arange
        restarts = 10
        
        results = []
        results.append(hill_climbing(x))
        for r in xrange(restarts):
            if not self.get_terminating_condition():
                while (not self.get_terminating_condition()):
                    x =[]
                    for d in designSpace:
                        range = arange(d["min"],d["max"]+1.0,d["step"])
                        x.append(range[random.randint(0,len(range)-1)])      
                    if code(x) == 0:
                        results.append(hill_climbing(x))
                        break
                        
        logging.info("RESULTS: " + str(results))
                                  
    ## main computation loop goes here
    def run(self):
        self.state_dictionary['generate'] = True
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        self.state_dictionary["fresh_run"] = False
        self.sample_plan()
        x0 = self.get_best()
        logging.info("x0: " + str(x0))
        result = self.hill_climbing_optimizer(x0)
        logging.info("RESULT :" + str(result))
        '''
        design_space = self.get_design_space()
        
        
        def fitness_wrapper(part):
            return self.fitness_function(part)[0]
        lower = [int(dim_space["min"]) for dim_space in design_space]
        upper = [int(dim_space["max"]) for dim_space in design_space]
        #logging.info(str(lower))
        #logging.info(str(upper))
        res = anneal(fitness_wrapper, x0, lower=lower,upper=upper,disp=True)
        
        logging.info(str(res))
        ## set i counter
        ## set fit counter
        ## set best
        '''
        self.exit()
    
    
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        #model = ProperCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
                                  
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('i'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'goal': self.get_configuration().goal,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'propa_classifier': self.get_surrogate_model().propa_classifier,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness,
            'best': {"data":self.get_best()}
        }
        return return_dictionary

    def save(self):
        try:
            trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('i')) + '.txt'
            dict = self.state_dictionary
            surrogate_model_state_dict = self.surrogate_model.get_state_dictionary()
            dict['surrogate_model_state_dict'] = surrogate_model_state_dict
            cost_model_state_dict = self.cost_model.get_state_dictionary()
            dict['cost_model_state_dict'] = cost_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                pickle.dump(dict, outfile)  
                if self.kill:
                    sys.exit(0)
        except Exception, e:
            logging.error(str(e))
            if self.kill:
                sys.exit(0)
            return False
            
    ## by default find the latest iteration
    def load(self, iteration = None):
        try:
            if iteration is None:
                # Figure out what the last iteration before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last iteration
                        iteration = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            iteration_file = str(iteration)
            trial_file = str(self.get_results_folder()) + '/' + str(iteration_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            self.surrogate_model.set_state_dictionary(dict['surrogate_model_state_dict'])
            self.cost_model.set_state_dictionary(dict['cost_model_state_dict'])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False
 
    def get_llambda(self):
        return  minimum(self.llambda,len(self.running_que)) 
    
    def update_running_que_by(self, time_passed, disp=True):
        running_que_cost = array([r.flatten() for r in self.running_que_cost])
        running_que = self.running_que
        
        llambda = self.get_llambda()
        running_que_cost[0:llambda] = array(running_que_cost[0:llambda]) - time_passed
        evaluated_list = [part for part, cost in zip(running_que,running_que_cost) if cost <= 0.0] ## desings which finished
        self.running_que = [part for part, cost in zip(running_que,running_que_cost) if cost > 0.0]
        self.running_que_cost = [cost for cost in running_que_cost if cost > 0.0]
        
        self.update_system_time(time_passed, disp=disp)
        self.set_counter_dictionary("cost", self.current_sim_time)
        for part in evaluated_list:
            self.fitness_function(part,add=True)
        #pdb.set_trace()
        
    def update_running_que(self):
        running_que_cost = array([r.flatten() for r in self.running_que_cost])
        if len(running_que_cost) > 0:
            if self.configuration.limit_lambda_search:
                if (self.get_no_to_eval() > 0): ## design can be added to evaluation que
                    pass
                else:
                    llambda = minimum(self.llambda,len(running_que_cost)) 
                    finishing = argmin(running_que_cost[0:llambda]) ## get the evaluations which are finishing now
                    time_update = running_que_cost[finishing][0]
                    self.update_running_que_by(time_update)
            else:
                llambda = minimum(self.llambda,len(running_que_cost)) 
                finishing = argmin(running_que_cost[0:llambda]) ## get the evaluations which are finishing now
                time_update = running_que_cost[finishing][0]
                self.update_running_que_by(time_update)
        else:
            logging.info("Running que emtpy... omitting, probably a duplicate is to be evaluated")
    
    def get_miu_set(self):
        return array(self.running_que)
        
    def get_no_to_eval(self):
        return self.llambda - self.currently_evaled
 
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

            
class PSOTrial(Trial):

    #######################
    ## Abstract Methods  ##
    #######################
    
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        self.state_dictionary['best'] = None
        self.state_dictionary['pso_best'] = None
        self.state_dictionary['fitness_evaluated'] = False
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['new_best_over_iteration'] = False
        self.state_dictionary['population'] = None
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("g")
        self.set_counter_dictionary("g",0)
            
        results_folder, images_folder, dump_folder = self.create_results_folder()
        self.initialize_population()
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        
        return True
        
    def run_initialize(self):
        logging.info("Initialize PSOTrial no:" + str(self.get_trial_no()))
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        design_space = self.fitness.designSpace
        self.toolbox = copy(base.Toolbox())
        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           smin=self.smin, smax=self.smax,
                           speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                        self.smin)],
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space],
                           model=False, best=None, code=None)

        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particles', self.filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', self.filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', self.updateParticle, 
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)
        self.new_best=False
        
    def run(self):
        self.state_dictionary['generate'] = True
        
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        self.save() ## training might take a bit...
        # Initialise termination check
        
        self.check = False
        ## we do this not to retrain model twice during the first iteration. If we ommit
        ## this bit of code the first view_update wont have a model aviable.
        reevalute = False
        if self.state_dictionary["fresh_run"]: ## we need this as we only want to do it for initial generation because the view
            ## the problem is that we cannot
            self.train_surrogate_model()
            self.train_cost_model()
            #self.view_update(visualize = True)
            self.state_dictionary["fresh_run"] = False
            self.save()
            
        while self.get_counter_dictionary('g') < self.get_configuration().max_iter + 1:
            
            logging.info('[' + str(self.get_name()) + '] Generation ' + str(self.get_counter_dictionary('g')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))

            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already
        # see this
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            
            # Roll population
            first_pop = self.get_population().pop(0)
            self.get_population().append(first_pop)
            
            if self.get_counter_dictionary('fit') > self.get_configuration().max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break
            reevalute = False
            time_surr = time.time()
            # Train surrogate model
            if self.training_set_updated():
                self.train_surrogate_model()
                self.train_cost_model()
                reevalute = True
            #logging.info(str(self.get_population()))
            code, mu, variance, ei, p = self.predict_surrogate_model(self.get_population())
            time_surr = time.time() - time_surr
            self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + time_surr)
            reloop = False
            if (mu is None) or (variance is None):
                logging.info("Prediction Failed")
                self.set_model_failed(True)
            else:
                logging.info("mean S2 " + str(numpy.mean(variance)))
                logging.info("max S2  " + str(max(variance)))
                logging.info("min S2  " + str(min(variance)))
                logging.info("over 0.05  " + str(min(len([v for v in variance if v > 0.05]))))
                logging.info("over 0.01  " + str(min(len([v for v in variance if v > 0.01]))))
                logging.info("mean ei " + str(numpy.mean(ei)))
                logging.info("max ei  " + str(max(ei)))
                logging.info("min ei  " + str(min(ei)))
                reloop = self.post_model_filter(code, mu, variance)
            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                time_sample = time.time()
                self.sample_design_space() ## we want the local hypercube
                time_sample = time.time() - time_sample
                self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + time_sample)
            elif reloop:
                reevalute = True
                logging.info('Evaluated some particles, will try to retrain model')
            else:#
                if reevalute:
                    self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.get_counter_dictionary('g') % self.get_configuration().M == 0:# perturb
                    self.evaluate_best()
                self.new_best = False
            # Wait until the user unpauses the trial.
            while self.get_wait():
                time.sleep(0)
            if self.get_counter_dictionary('g') % 1000000000 == 0:
                self.view_update(visualize = True)
            self.increment_main_counter()
        self.exit()
        
    ### returns a snapshot of the trial state
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('g'))
        name = self.get_name()
        return_dictionary = {
            'goal': self.get_configuration().goal,
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'meta_plot': {"particles":{'marker':"o",'color':"white",'data':self.get_population()}},
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness
        }
        return return_dictionary
        
    def save(self):
        ### comented out
        if 0:
            try:
                trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('g')) + '.txt'
                dict = self.state_dictionary
                surrogate_model_state_dict = self.surrogate_model.get_state_dictionary()
                cost_model_state_dict = self.cost_model.get_state_dictionary()
                dict['cost_model_state_dict'] = cost_model_state_dict
                dict['surrogate_model_state_dict'] = surrogate_model_state_dict
                with io.open(trial_file, 'wb') as outfile:
                    pickle.dump(dict, outfile)  
                    if self.kill:
                        sys.exit(0)
            except Exception, e:
                logging.error(str(e))
                if self.kill:
                    sys.exit(0)
                return False
            
    ## by default find the latest generation
    def load(self, generation = None):
        try:
            if generation is None:
                # Figure out what the last generation before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last generation
                        generation = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            generation_file = str(generation)
            trial_file = str(self.get_results_folder()) + '/' + str(generation_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            self.surrogate_model.set_state_dictionary(dict['surrogate_model_state_dict'])
            self.cost_model.set_state_dictionary(dict['cost_model_state_dict'])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False
        
    ####################
    ## Helper Methods ##
    ####################
        
    def checkCollapse(self): #TODO
        ## this method checks if the particls
        ## a) collapsed onto a single point
        ## b) collapsed onto the edge of the search space
        ## if so it reintializes them.
        minimum_diverity = 0.95 ##if over 95 collapsed reseed
        if collapsed:
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
    
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
        
    def createUniformSpace(self, particles, designSpace):
        pointPerDimensions = 5
        valueGrid = mgrid[designSpace[0]['min']:designSpace[0]['max']:
                          complex(0, pointPerDimensions),
                          designSpace[1]['min']:designSpace[1]['max']:
                          complex(0, pointPerDimensions)]

        for i in [0, 1]:
            for j, part in enumerate(particles):
                part[i] = valueGrid[i].reshape(1, -1)[0][j]

    def filterParticles(self,  particles, designSpace):
        for particle in particles:
            self.filterParticle(particle, designSpace)
            
    def filterParticle(self, p, designSpace):
        p.pmin = [dimSetting['min'] for dimSetting in designSpace]
        p.pmax = [dimSetting['max'] for dimSetting in designSpace]

        for i, val in enumerate(p):
            #dithering
            if designSpace[i]['type'] == 'discrete':
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])

            #dont allow particles to take the same value
            p[i] = minimum(p.pmax[i], p[i])
            p[i] = maximum(p.pmin[i], p[i])

    def generate(self,  designSpace):
        particle = [uniform(dimSetting['min'], dimSetting['max'])
                    for dimSetting
                    in designSpace]
        particle = self.create_particle(particle)
        return particle

    # update the position of the particles
    # should change this part in order to change the leader election strategy
    def updateParticle(self,  part, generation, conf, designSpace):
        if conf.admode == 'fitness':
            fraction = self.fitness_counter / conf.max_fitness
        elif conf.admode == 'iter':
            fraction = generation / conf.max_iter
        else:
            raise('[updateParticle]: adjustment mode unknown.. ')

        u1 = [uniform(0, conf.phi1) for _ in range(len(part))]
        u2 = [uniform(0, conf.phi2) for _ in range(len(part))]
        
        ##########   this part particulately, leader election for every particle
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.get_pso_best(), part))
        weight = 1.0
        if conf.weight_mode == 'linear':
            weight = conf.max_weight - (conf.max_weight -
                                        conf.min_weight) * fraction
        elif conf.weight_mode == 'norm':
            weight = conf.weight
        else:
            raise('[updateParticle]: weight mode unknown.. ')
        weightVector = [weight] * len(part.speed)
        part.speed = map(operator.add,
                         map(operator.mul, part.speed, weightVector),
                         map(operator.add, v_u1, v_u2))

    # what's this mean?
        if conf.applyK is True:
            phi = array(u1) + array(u1)

            XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                            sqrt(pow(phi, 2.0) - 4.0 * phi))
            part.speed = map(operator.mul, part.speed, XVector)

    # what's the difference between these modes?
        if conf.mode == 'vp':
            for i, speed in enumerate(part.speed):
                speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
                if speed < -speedCoeff:
                    part.speed[i] = -speedCoeff
                elif speed > speedCoeff:
                    part.speed[i] = speedCoeff
                else:
                    part.speed[i] = speed
        elif conf.mode == 'norm':
            for i, speed in enumerate(part.speed):
                if speed < part.smin[i]:
                    part.speed[i] = part.smin[i]
                elif speed > part.smax[i]:
                    part.speed[i] = part.smax[i]
        elif conf.mode == 'exp':
            for i, speed in enumerate(part.speed):
                maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
                if speed < -maxVel:
                    part.speed[i] = -maxVel
                elif speed > maxVel:
                    part.speed[i] = maxVel
        elif conf.mode == 'no':
            pass
        else:
            raise('[updateParticle]: mode unknown.. ')
        part[:] = map(operator.add, part, part.speed)

    ### check first if part is already within the training set
    def fitness_function(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        self.increment_counter('fit')
        #results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
        #self.set_fitness_state(state)
        #pdb.set_trace()
        results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
        try:
            results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            pdb.set_trace()
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3]
        except:
            cost = array([1.0]) ## just keep it constant for all points
        
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost[0])
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.cost_model.add_training_instance(part, cost)
        self.set_retrain_model(True)
       
        logging.info(str(code) + " " + str(fitness))
        if code[0] == 0:
            part.fitness.values = fitness
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = fitness
                self.set_best(particle)
            if not self.get_pso_best() or self.is_better(part.fitness, self.get_pso_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = fitness
                self.set_pso_best(particle)
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return array([self.fitness.worst_value]), code, cost
        
    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        try:
            part = self.create_particle(self.fitness.always_valid)
            self.toolbox.filter_particle(part)
            part.fitness.values, part.code, cost = self.fitness_function(part)
            self.set_at_least_one_in_valid_region(True)
            logging.info("Always valid configuration present, evaluated")
        except Exception, e:
            #pdb.set_trace()
            logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
            self.set_at_least_one_in_valid_region(False)
        self.set_at_least_one_in_valid_region(True)
        logging.info(str(self.get_at_least_one_in_valid_region()))
        #F = copy(self.get_configuration().F)
        designSpace = self.fitness.designSpace
        D = len(designSpace)
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(self.get_configuration().population_size,D))
        max_bounds = array([d["max"] for d in designSpace])
        min_bounds = array([d["min"] for d in designSpace])
        latin_hypercube_samples = latin_hypercube_samples * (array(max_bounds)-array(min_bounds))
        latin_hypercube_samples = latin_hypercube_samples + min_bounds
        
        population = []
        f_counter = 0
        for part in latin_hypercube_samples:
            part = self.create_particle(part)
            self.toolbox.filter_particle(part)
            if f_counter < self.get_configuration().population_size/2.:
                part.fitness.values, part.code, cost = self.fitness_function(part)
                self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            population.append(part)
            f_counter = f_counter + 1
        self.set_population(population)
        while (not self.get_at_least_one_in_valid_region()):
            #logging.in
            exit(0)
            part = self.toolbox.particle() ## a random particle
            logging.info("All particles within invalid search space.. Evaluating extra examples: " + str(part))
            part.fitness.values, part.code, cost = self.toolbox.evaluate(part)
            self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
                        
        self.state_dictionary["fresh_run"] = True
        
        #what's this function do?
    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        ##while(self.get_at_least_one_in_valid_region()):
        ##    logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        logging.info("Meta Iteration")
        for part in self.get_population():
            if not part.best or self.is_better(part.fitness, part.best.fitness):
                part.best = self.create_particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.get_pso_best() or self.is_better(part.fitness, self.get_pso_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_pso_best(particle)
                self.new_best = True
                                
        #PSO
        for part in self.get_population():
            self.toolbox.update(part, self.get_counter_dictionary('g'))

    def filter_population(self):
        self.toolbox.filter_particles(self.get_population())
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function(self.get_pso_best())
            logging.info('New best was found after M :' + str(self.get_pso_best()))
        else:            
            ## TODO - clean it up...  messy
            perturbation = self.perturbation(radius = 100.0)                        
            logging.info('Best was already evalauted.. adding perturbation ' + str(perturbation))
            perturbed_particle = self.create_particle(self.get_pso_best())
            code, mean, variance, ei, p = self.predict_surrogate_model([perturbed_particle])
            if code is None:
                logging.debug("Code is none..watch out")
            if code[0] == 0:
                logging.info('Perturbation might be valid, evaluationg')
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            if self.surrogate_model.contains_training_instance(perturbed_particle):
                logging.info('Perturbation was already evaluated.. sampling')
                if not self.configuration.sample_on is None:
                    self.sample_design_space()
                else:
                    logging.info('sampling is turned off')
            else:
                fitness, code, cost = self.fitness_function(perturbed_particle) 
                perturbed_particle.fitness.values = fitness
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_pso_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

    # def sample_design_space(self):
        # particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace, hypercube = self.hypercube())
        # if particle is None:
            # logging.info("Local sampling has failed, probably all of the particles are within invalid region")
            # particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace)
            # particle = self.surrogate_model.max_s2(designSpace=self.fitness.designSpace)
            # if particle is None:
                # perturbation = self.perturbation(radius = 100.0)                        
                # logging.info('Evaluating random perturbation of real best ' + str(perturbation))
                # particle = self.create_particle(self.surrogate_model.get_best()[0])
                # for i,val in enumerate(perturbation):
                    # particle[i] = particle[i] + val       
                # logging.info("Global sampling failed as well.. Evaluating a random particle"  + str(particle))
        # particle = self.create_particle(particle)   
        # self.toolbox.filter_particle(particle)
        # particle.fitness.values, code, cost = self.fitness_function(particle) 
        
    def sample_design_space(self):
        logging.info('Evaluating best perturbation')
        perturbation = self.perturbation(radius = 100.0)           
        if self.configuration.sample_on == "no":
            logging.info("sampling turned off")
            particle = None
        else:
            logging.info("Using sampling setting: " + self.configuration.sample_on)
            particle = self.surrogate_model.max_uncertainty(designSpace=self.fitness.designSpace, hypercube = self.hypercube())
        if particle is None:
            logging.info('Evaluating a perturbation of a random particle')
            particle = self.create_particle(self.surrogate_model.get_best()[0])
            for i,val in enumerate(perturbation):
                particle[i] = particle[i] + val    
        particle = self.create_particle(particle)
        logging.info('Sampled particle:' + str(particle))
        self.toolbox.filter_particle(particle)
        logging.info('Sampled particle:' + str(particle))
        particle.fitness.values, code, cost = self.fitness_function(particle) 
            
     ## not used currently
    def get_dist(self):
        if best:
            distances = sqrt(sum(pow((self.surrogate.best),2),axis=1))  # TODO
            order_according_to_manhatan = argsort(distances)
            closest_array = [gpTrainingSet[index] for index in order_according_to_manhatan[0:conf.nClosest]]
        ###        
        ## limit to hypercube around the points
        #find maximum
        #print "[getDist] closestArray ",closestArray
        max_diag = deepcopy(closestArray[0])
        for part in closest_array:
            max_diag = maximum(part, max_diag)
        ###find minimum vectors
        min_diag = deepcopy(closest_array[0])
        for part in closest_array:
            min_diag = minimum(part, min_diag)
        return [max_diag, min_diag]
        
    ### a hypercube that contains all the particles
    def hypercube(self):
        #find maximum
        max_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            max_diag = maximum(part,max_diag) 
        ###find minimum vectors
        min_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            min_diag = minimum(part,min_diag)
            
        ## we always ensure that the hypercube allows particles to maintain velocity components in all directions
        
        for i,dd in enumerate(max_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                max_diag[i] = minimum(dd + self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["max"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                max_diag[i] = minimum(dd + small_fraction, self.fitness.designSpace[i]["max"])
                
        for i,dd in enumerate(min_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                min_diag[i] = maximum(dd - self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["min"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                min_diag[i] = maximum(dd - small_fraction, self.fitness.designSpace[i]["min"])
        logging.info("hypecube: " + str([max_diag,min_diag]))
        return [max_diag,min_diag]
        
    def perturbation(self, radius = 10.0):
        [max_diag,min_diag] = self.hypercube()
        d = (max_diag - min_diag)/radius
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = maximum(dd,self.fitness.designSpace[i]["step"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                d[i] = maximum(dd,small_fraction)
        dimensions = len(self.fitness.designSpace)
        pertubation =  multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
    
    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        eval_counter = 1
        self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
        if self.get_model_failed():
            return False
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            #### if all particles that have stdv > max been evalauted we have a prbolem and we shoudl do something...
            #### currently we sample design space again
            #### this can happen during first iteration...
            #all_evaled = True
            #counter = 0
            #for (p, c, m, v) in zip(self.get_population(), code, mean, variance):
            #    if ((v > self.get_configuration().max_stdv) and (c == 0)):
            #        all_evaled = all_evaled and (self.get_surrogate_model().contains_training_instance(p))
            #        counter = counter + 1
                    
            #if all_evaled and counter: ## randomize population
            #    logging.info("Houston... we got a problem.. this might happen at the beggining" + str(all_evaled) + " "  + str(counter) + " " + str(zip(self.get_population(), code, mean, variance)))
            #    self.sample_design_space()
            #    return True
                
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code, mean, variance)):
                if v > self.get_configuration().max_stdv and c == 0:
                    if eval_counter > self.get_configuration().max_eval:
                        logging.info("Evalauted more fitness functions per generation then max_eval")
                        return True
                    p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                    eval_counter = eval_counter + 1
                else:
                    try:
                        if c == 0:
                            p.fitness.values = m
                        else:
                            p.fitness.values = [self.fitness.worst_value]
                    except:
                        p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
        return False
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.get_population() if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.get_pso_best():
            bests_to_model.append(self.get_pso_best())
        if bests_to_model:
            logging.info("Reevaluating")
            code, bests_to_fitness, variance, ei, p = self.predict_surrogate_model(bests_to_model)
            if (code is None) or (bests_to_fitness is None) or (variance is None):
                logging.info("Prediction failed during reevaluation... omitting")
            else:
                for i,part in enumerate([p for p in self.get_population() if p.best]):
                    if code[i] == 0:
                        part.best.fitness.values = bests_to_fitness[i]
                    else:
                        part.best.fitness.values = [self.fitness.worst_value]
                if self.get_pso_best():
                    best = self.get_pso_best()
                    if code[-1] == 0:
                        best.fitness.values = bests_to_fitness[-1]
                    else:
                        best.fitness.values = [self.fitness.worst_value]
                    ## find best among the training set!!!
                    logging.info("Fixing best: " + str(best))
                    evaled_best, evaled_best_fitness = self.surrogate_model.get_best()
                    logging.info(str(evaled_best))
                    evaled_best = self.create_particle(evaled_best)   
                    self.toolbox.filter_particle(evaled_best)
                    evaled_best.fitness.values = evaled_best_fitness
                    if self.is_better(evaled_best_fitness, best.fitness.values):
                        logging.info("Real best better: " + str(evaled_best))
                        self.set_pso_best(evaled_best)
                    
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_pso_best(self):
        return self.state_dictionary['pso_best']
        
    def set_pso_best(self, new_best):
        self.state_dictionary['pso_best'] = new_best
    
    def get_predicted_time(self):
        predicted_time = self.state_dictionary['total_time'] * self.get_configuration().max_iter / (self.get_counter_dictionary('g') + 1.0)
        return str(timedelta(seconds=predicted_time))
    
    def set_population(self, population):
        self.state_dictionary["population"] = population
        
    def get_population(self):
        return self.state_dictionary["population"]

    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
    
class MOPSOTrial(Trial):

    #######################
    ## Abstract Methods  ##
    #######################
    
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        self.state_dictionary['best'] = None
        self.state_dictionary['fitness_evaluated'] = False
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['new_best_over_iteration'] = False
        self.state_dictionary['population'] = None
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("g")
        self.set_counter_dictionary("g",0)
        self.initialize_population()    
        results_folder, images_folder, dump_folder = self.create_results_folder()
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        
        return True
        
    def run_initialize(self):
        logging.info("Initialize Multi-Objective PSOTrial no:" + str(self.get_trial_no()))
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        design_space = self.fitness.designSpace
        self.toolbox = copy(base.Toolbox())
        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           smin=self.smin, smax=self.smax,
                           speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                        self.smin)],
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space],
                           model=False, best=None, code=None)

        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particles', self.filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', self.filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', self.updateParticle, 
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate1', self.fitness_function1)
        self.toolbox.register('evaluate2', self.fitness_function2)
        self.new_best=False
        
    def run(self):
        self.state_dictionary['generate'] = True
        
        logging.info(str(self.get_name()) + ' started')
        logging.info('Multi-Objective Trial prepared... executing')
        self.save() ## training might take a bit...
        # Initialise termination check
        
        self.check = False
        ## we do this not to retrain model twice during the first iteration. If we ommit
        ## this bit of code the first view_update wont have a model aviable.
        reevalute = False
        if self.state_dictionary["fresh_run"]: ## we need this as we only want to do it for initial generation because the view
            ## the problem is that we cannot
            self.train_surrogate_model()
            self.train_cost_model()
            self.view_update(visualize = True)
            self.state_dictionary["fresh_run"] = False
            self.save()
            
        while self.get_counter_dictionary('g') < self.get_configuration().max_iter + 1:
            
            logging.info('[' + str(self.get_name()) + '] Generation ' + str(self.get_counter_dictionary('g')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))

            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already

            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            
            # Roll population
            first_pop = self.get_population().pop(0)
            self.get_population().append(first_pop)
            
            if self.get_counter_dictionary('fit') > self.get_configuration().max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break
            reevalute = False
            # Train surrogate model
            if self.training_set_updated():
                self.train_surrogate_model()
                self.train_cost_model()
                reevalute = True
            ##print self.get_population()
            code, mu, variance, ei, p = self.predict_surrogate_model(self.get_population())
            reloop = False
            if (mu is None) or (variance is None):
                logging.info("Prediction Failed")
                self.set_model_failed(True)
            else:
                logging.info("mean S2 " + str(mean(variance)))
                logging.info("max S2  " + str(max(variance)))
                logging.info("min S2  " + str(min(variance)))
                logging.info("over 0.05  " + str(min(len([v for v in variance if v > 0.05]))))
                logging.info("over 0.01  " + str(min(len([v for v in variance if v > 0.01]))))
                reloop = self.post_model_filter(code, mu, variance)
            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                self.sample_design_space()
            elif reloop:
                reevalute = True
                logging.info('Evaluated some particles, will try to retrain model')
            else:#
                if reevalute:
                    self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.get_counter_dictionary('g') % self.get_configuration().M == 0:# perturb
                    self.evaluate_best()
                self.new_best = False
            # Wait until the user unpauses the trial.
            while self.get_wait():
                time.sleep(0)
            
            self.increment_main_counter()
            self.view_update(visualize = True)
        self.exit()
        
    ### returns a snapshot of the trial state
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('g'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'meta_plot': {"particles":{'marker':"o",'data':self.get_population()}},
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness
        }
        return return_dictionary
        
    ####################
    ## Helper Methods ##
    ####################
    
    def checkAllParticlesEvaled(self):
        all_evaled = True
        for part in self.get_population():
            all_evaled = self.get_surrogate_model().contains_training_instance(part) and all_evaled
        
        if all_evaled: ## randomize population
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
        
    def checkCollapse(self): #TODO
        ## this method checks if the particls
        ## a) collapsed onto a single point
        ## b) collapsed onto the edge of the search space
        ## if so it reintializes them.
        minimum_diverity = 0.95 ##if over 95 collapsed reseed
        
        
        if collapsed:
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
    
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
        
    
    def createUniformSpace(self, particles, designSpace):
        pointPerDimensions = 5
        valueGrid = mgrid[designSpace[0]['min']:designSpace[0]['max']:
                          complex(0, pointPerDimensions),
                          designSpace[1]['min']:designSpace[1]['max']:
                          complex(0, pointPerDimensions)]

        for i in [0, 1]:
            for j, part in enumerate(particles):
                part[i] = valueGrid[i].reshape(1, -1)[0][j]

    def filterParticles(self,  particles, designSpace):
        for particle in particles:
            self.filterParticle(particle, designSpace)
            
    def filterParticle(self, p, designSpace):
        p.pmin = [dimSetting['min'] for dimSetting in designSpace]
        p.pmax = [dimSetting['max'] for dimSetting in designSpace]

        for i, val in enumerate(p):
            #dithering
            if designSpace[i]['type'] == 'discrete':
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])

            #dont allow particles to take the same value
            p[i] = minimum(p.pmax[i], p[i])
            p[i] = maximum(p.pmin[i], p[i])

    def generate(self,  designSpace):
        particle = [uniform(dimSetting['min'], dimSetting['max'])
                    for dimSetting
                    in designSpace]
        particle = self.create_particle(particle)
        return particle

	# update the position of the particles
	# should change this part in order to change the leader election strategy
    def updateParticle(self,  part, generation, conf, designSpace):
        if conf.admode == 'fitness':
            fraction = self.fitness_counter / conf.max_fitness
        elif conf.admode == 'iter':
            fraction = generation / conf.max_iter
        else:
            raise('[updateParticle]: adjustment mode unknown.. ')

        u1 = [uniform(0, conf.phi1) for _ in range(len(part))]
        u2 = [uniform(0, conf.phi2) for _ in range(len(part))]
        
        ##########   this part particulately, leader election for every particle
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.get_best(), part))
        weight = 1.0
        if conf.weight_mode == 'linear':
            weight = conf.max_weight - (conf.max_weight -
                                        conf.min_weight) * fraction
        elif conf.weight_mode == 'norm':
            weight = conf.weight
        else:
            raise('[updateParticle]: weight mode unknown.. ')
        weightVector = [weight] * len(part.speed)
        part.speed = map(operator.add,
                         map(operator.mul, part.speed, weightVector),
                         map(operator.add, v_u1, v_u2))

	# what's this mean?
        if conf.applyK is True:
            phi = array(u1) + array(u1)

            XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                            sqrt(pow(phi, 2.0) - 4.0 * phi))
            part.speed = map(operator.mul, part.speed, XVector)

	# what's the difference between these modes?
        if conf.mode == 'vp':
            for i, speed in enumerate(part.speed):
                speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
                if speed < -speedCoeff:
                    part.speed[i] = -speedCoeff
                elif speed > speedCoeff:
                    part.speed[i] = speedCoeff
                else:
                    part.speed[i] = speed
        elif conf.mode == 'norm':
            for i, speed in enumerate(part.speed):
                if speed < part.smin[i]:
                    part.speed[i] = part.smin[i]
                elif speed > part.smax[i]:
                    part.speed[i] = part.smax[i]
        elif conf.mode == 'exp':
            for i, speed in enumerate(part.speed):
                maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
                if speed < -maxVel:
                    part.speed[i] = -maxVel
                elif speed > maxVel:
                    part.speed[i] = maxVel
        elif conf.mode == 'no':
            pass
        else:
            raise('[updateParticle]: mode unknown.. ')
        part[:] = map(operator.add, part, part.speed)

    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        
        self.set_at_least_one_in_valid_region(False)
        F = copy(self.get_configuration().F)
        while (not self.get_at_least_one_in_valid_region()):
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
            if self.get_configuration().eval_correct:
                self.get_population()[0] = self.create_particle(
                    self.fitness.alwaysCorrect())  # This will break
            for i, part in enumerate(self.get_population()):
                if i < F:
                    part.fitness.values, part.code, cost = self.toolbox.evaluate1(part)
                    part.fitness.values, part.code, cost = self.toolbox.evaluate1(part)
                    self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
                    if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                        particle = self.create_particle(part)
                        particle.fitness.values = part.fitness.values
                        self.set_best(particle)
                        
            ## add one example till we find something that works
            F = 1
            
        self.state_dictionary["fresh_run"] = True
        
        #what's this function do?
    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        ##while(self.get_at_least_one_in_valid_region()):
        ##    logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        logging.info("Meta Iteration")
        for part in self.get_population():
            if not part.best or self.is_better(part.fitness, part.best.fitness):
                part.best = self.create_particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
                self.new_best = True
                                
        #PSO
        for part in self.get_population():
            self.toolbox.update(part, self.get_counter_dictionary('g'))

    def filter_population(self):
        self.toolbox.filter_particles(self.get_population())
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function1(self.get_best())
            
            logging.info('New best was found after M :' + str(self.get_best()))
        else:
            ## TODO - clean it up...  messy
            perturbation = self.perturbation(radius = 100.0)                        
            logging.info('Best was already evalauted.. adding perturbation ' + str(perturbation))
            perturbed_particle = self.create_particle(self.get_best())
            code, mean, variance, ei ,p = self.predict_surrogate_model([perturbed_particle])
            if code is None:
                logging.debug("Code is none..watch out")
            if code is None or code[0] == 0:
                logging.info('Perturbation might be valid, evaluationg')
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            fitness, code, cost = self.fitness_function(perturbed_particle) 
            ##check if the value is not a new best
            perturbed_particle.fitness.values = fitness
            if not self.get_best() or self.is_better(perturbed_particle.fitness, self.get_best().fitness):
                self.set_best(perturbed_particle)
            else: ## why do we do this? 
                if code[0] != 0:
                    logging.info('Best is within the invalid area ' + str(code[0]) + ', sampling design space')
                    self.sample_design_space()
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

    def sample_design_space(self):
        logging.info('Evaluating best perturbation')
        perturbation = self.perturbation(radius = 10.0)                        
        hypercube = self.hypercube()
        if self.configuration.sample_on == "no":
            logging.info("sampling turned off")
            particle = None
        else:
            particle = self.surrogate_model.max_uncertainty(designSpace=self.fitness.designSpace, hypercube = hypercube)
        if particle is None:
            logging.info('Evaluating a perturbation of a random particle')
            particle = self.toolbox.particle()
        perturbedParticle = self.create_particle(particle)
        for i,val in enumerate(perturbation):
            perturbedParticle[i] = perturbedParticle[i] + val       
        self.toolbox.filter_particle(perturbedParticle)
        perturbedParticle.fitness.values, code, cost = self.fitness_function(perturbedParticle) 
        if not self.get_best() or self.is_better(perturbedParticle.fitness, self.get_best().fitness):
            self.set_best(perturbedParticle)
        
    ## not used currently
    # def get_perturbation_dist(self):
        # [max_diag, min_diag] = self.get_dist()
        # d = (max_diag - min_diag)/2.0
        # if best:
            # max_diag = best + d
            # for i,dd in enumerate(self.fitness.designSpace):
                # max_diag[i] = minimum(max_diag[i],dd["max"])
            # min_diag = best - d
            # for i,dd in enumerate(self.fitness.designSpace):
                # min_diag[i] = maximum(min_diag[i],dd["min"])
            # return [max_diag,min_diag]
        # else:
            # return getHypercube(pop)
            
     ## not used currently
    def get_dist(self):
        if best:
            distances = sqrt(sum(pow((self.surrogate.best),2),axis=1))  # TODO
            order_according_to_manhatan = argsort(distances)
            closest_array = [gpTrainingSet[index] for index in order_according_to_manhatan[0:conf.nClosest]]
        ###        
        ## limit to hypercube around the points
        #find maximum
        #print "[getDist] closestArray ",closestArray
        max_diag = deepcopy(closestArray[0])
        for part in closest_array:
            max_diag = maximum(part, max_diag)
        ###find minimum vectors
        min_diag = deepcopy(closest_array[0])
        for part in closest_array:
            min_diag = minimum(part, min_diag)
        return [max_diag, min_diag]
        
    def hypercube(self):
        #find maximum
        max_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            max_diag = maximum(part,max_diag)
        ###find minimum vectors
        min_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            min_diag = minimum(part,min_diag)
        return [max_diag,min_diag]
        
    def perturbation(self, radius = 10.0):
        [max_diag,min_diag] = self.hypercube()
        d = (max_diag - min_diag)/radius
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = maximum(dd,self.fitness.designSpace[i]["step"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                d[i] = maximum(dd,self.smax[i])
        dimensions = len(self.fitness.designSpace)
        pertubation =  multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
        
        ###check if between two calls to this functions any fitness functions have been evaluted, so that the models have to be retrained
    def training_set_updated(self):
        retrain_model_temp = self.get_retrain_model()
        self.set_retrain_model(False)
        return retrain_model_temp
    
    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        eval_counter = 1
        self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
        if self.get_model_failed():
            return False
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code,
                                                 mean, variance)):
                if v > self.get_configuration().max_stdv and c == 0:
                    if eval_counter > self.get_configuration().max_eval:
                        logging.info("Evalauted more fitness functions per generation then max_eval")
                        self.checkAllParticlesEvaled() ## if all the particles have been evalauted 
                        return True
                    p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                    eval_counter = eval_counter + 1
                else:
                    try:
                        if c == 0:
                            p.fitness.values = m
                        else:
                            p.fitness.values = [self.fitness.worst_value]
                    except:
                        p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                        logging.info("KURWA Start")
                        logging.info("KURWA End")
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
        return False
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.get_population() if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.get_best():
            bests_to_model.append(self.get_best())
        if bests_to_model:
            logging.info("Reevaluating")
            code, bests_to_fitness, variance, ei, p = self.predict_surrogate_model(bests_to_model)
            if (code is None) or (bests_to_fitness is None) or (variance is None):
                logging.info("Prediction failed during reevaluation... omitting")
            else:
                for i,part in enumerate([p for p in self.get_population() if p.best]):
                    if code[i] == 0:
                        part.best.fitness.values = bests_to_fitness[i]
                    else:
                        part.best.fitness.values = [self.fitness.worst_value]
                if self.get_best():
                    best = self.get_best()
                    if code[-1] == 0:
                        best.fitness.values = bests_to_fitness[-1]
                    else:
                        best.fitness.values = [self.fitness.worst_value]
                    self.set_best(best)
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_predicted_time(self):
        predicted_time = self.state_dictionary['total_time'] * self.get_configuration().max_iter / (self.get_counter_dictionary('g') + 1.0)
        return str(timedelta(seconds=predicted_time))
    
    def set_population(self, population):
        self.state_dictionary["population"] = population
        
    def get_population(self):
        return self.state_dictionary["population"]
        
    def get_best_fitness_array(self):
        return self.state_dictionary['best_fitness_array']
        
    def get_generations_array(self):
        return self.state_dictionary['generations_array']
        
    def set_at_least_one_in_valid_region(self, state):
        if self.state_dictionary.has_key('at_least_one_in_valid_region'):
            self.state_dictionary['at_least_one_in_valid_region'] = state
        else:
            self.state_dictionary['at_least_one_in_valid_region'] = False

    def get_at_least_one_in_valid_region(self):
        return self.state_dictionary['at_least_one_in_valid_region']

    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
