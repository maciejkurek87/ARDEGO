import logging

import sys
if sys.version_info > (2, 6):
    from classifiers import Classifier, SupportVectorMachineClassifier, ResourceAwareClassifier
else:
    from classifiers import Classifier, SupportVectorMachineClassifier, RelevanceVectorMachineClassifier, RelevanceVectorMachineClassifier2, ResourceAwareClassifier
    
from regressors import Regressor, GaussianProcessRegressor, GaussianProcessRegressor4

from utils import numpy_array_index
from scipy.interpolate import griddata
from scipy.stats import norm
from numpy import linspace, meshgrid, reshape, array, argmax, mgrid, ones, arange, place, maximum, minimum, zeros, concatenate, array_split, argmin, vstack , any
import itertools 
import pdb
from numpy.random import uniform, randint
from scipy.optimize import minimize
import ei_optimizers
import sys
import time
from time import strftime
from datetime import datetime, timedelta
from copy import copy,deepcopy
import os
import math
from operator import itemgetter
from ei_soft import ei_multi_max, ei_multi_min

class SurrogateModel(object):

    def __init__(self, configuration, controller, fitness):
        self.configuration = configuration
        self.fitness = fitness
        self.controller = controller
        self.was_trained = False
        
    def train(self, hypercube):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')
    def trained(self):
        return self.was_trained
                                  
    def predict(self, particles):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')

    def add_training_instance(self, part, code, fitness, addReturn):
        pass
        
    def contains_training_instance(self, part):
        pass    
        
    def get_training_instance(self, part):
        pass

    # def __getstate__(self):
        # Don't pickle fitness and configuration
        # d = dict(self.__dict__)
        # del d['configuration']
        # del d['fitness']
        # return d

    def contains_particle(self, part):
        pass
        
    def particle_value(self, part):
        pass
    
    def model_failed(self, part):
        pass
        
    def max_uncertainty(self, designSpace, hypercube=None, npts=200):
        pass

    def get_valid_set(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def get_state_dictionary(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def set_state_dictionary(self, dict):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')   

    def get_copy(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_regressor(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_classifier(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
class DummySurrogateModel(SurrogateModel):

    ## TODO - add dummy regressor/classifier
    def __init__(self, configuration, controller, fitness):
        super(DummySurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
        self.regressor = Regressor(controller, configuration, fitness)
        self.classifier = Classifier()

    def get_regressor(self):
        return self.regressor
                                  
    def get_classifier(self):
        return self.classifier
        
    def predict(self, particles):
        MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2

    def train(self, hypercube):
        self.was_trained = True
        return True

    def model_particle(self, particle):
        return 0, 0, 0
        
    def contains_training_instance(self, part):
        return False

    def model_failed(self, part):
        return False
        
    def get_state_dictionary(self):
        return {}
        
    def set_state_dictionary(self, dict):
        pass
        
    def get_copy(self):
        model_copy = DummySurrogateModel(self.configuration, self.controller)
        return model_copy
        
class ProperSurrogateModel(SurrogateModel):

    def __init__(self, configuration, controller, fitness):
        super(ProperSurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
                                                   
        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier(fitness, configuration)
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not found')
        self.regressor = self.regressor_constructor()
        
        try:
            if self.configuration.sample_on == "ei":
                self.max_uncertainty = self.max_ei
            elif self.configuration.sample_on == "s":
                self.max_uncertainty = self.max_s2
            elif self.configuration.sample_on == "m":
                self.max_uncertainty = self.max_m
        except:
            if self.max_uncertainty:
                pass
            else:
                logging.debug("Sampling scheme wasnt specified, using Expected Improvment")
                self.max_uncertainty = self.max_ei
        self.best = None
        self.best_fitness = None
        self.max_mem = 4 ## in GB
        self.fitness=fitness
        ### max cores
        
    def get_regressor(self):
        return self.regressor
                                  
    def get_classifier(self):
        return self.classifier
        
    def get_copy(self):
        model_copy = ProperSurrogateModel(self.configuration, self.controller, self.fitness)
        model_copy.set_state_dictionary(self.get_state_dictionary())
        return model_copy
            
    def predict(self, particles):
        try:
            #logging.debug("Using tranformation function for the regressor")
            trans_particles = particles
        except:
            trans_particles = [self.fitness.transformation_function(part) for part in particles]
        MU, S2, EI, P = self.regressor.predict(trans_particles)
        return self.classifier.predict(particles), MU, S2, EI, P

    def train(self, hypercube=None):
        self.was_trained = True
        if self.classifier.train() and self.regressor.train():
            logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
            
    def regressor_constructor(self):
        controller = self.controller
        configuration = self.configuration
        fitness = self.fitness
        if self.configuration.regressor == 'GaussianProcess':
            return GaussianProcessRegressor(controller, configuration, fitness)      
        elif self.configuration.regressor == 'GaussianProcess4':
            return GaussianProcessRegressor4(controller, configuration, fitness)     
        else:
            raise Exception('Regressor type ' + str(configuration.regressor) + '  not found')
        
    def add_training_instance(self, part, code, fitness, addReturn):
        self.classifier.add_training_instance(part, code)
        if addReturn[0] == 0: ## only update regressor if the fitness function produced a result
            try:
                trans_part = self.fitness.transformation_function(part)
                #logging.debug("Using tranformation function for the regressor")
            except:
                trans_part = part
            self.regressor.add_training_instance(trans_part, fitness)
            if (code == 0):
                logging.info("New possible real best")
                if self.best_fitness is None or (fitness > self.best_fitness and (self.configuration.goal == "max")):## maximization
                    self.best_fitness = fitness
                    self.best = part
                    self.regressor.set_y_best(fitness)
                if self.best_fitness is None or (fitness < self.best_fitness and (self.configuration.goal == "min")):## minimization
                    self.best_fitness = fitness
                    self.best = part
                    self.regressor.set_y_best(fitness)
                    
    def get_best(self):
        return self.best, self.best_fitness
        
    def contains_training_instance(self, part):
        try:
            trans_part = self.fitness.transformation_function(part)
            #logging.debug("Using tranformation function for the regressor")
        except:
            trans_part = part
        return self.regressor.contains_training_instance(trans_part) or self.classifier.contains_training_instance(part)  

    def get_training_instance(self, part):
        code = self.classifier.get_training_instance(part) 
        fitness = None
        if self.regressor.contains_training_instance(part):
            fitness = self.regressor.get_training_instance(part)            
        return code, fitness
        
    def model_failed(self, part):
        return False
        
    def max_ei(self, designSpace, hypercube=None, npts=10):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        grid = False
        if grid:
            if hypercube:
                result = mgrid[[slice(h_min, h_max, npts*1.0j) for h_max, h_min , n in zip(hypercube[0],hypercube[1], n_bins)]]
                z = result.reshape(D,-1).T
            else:
                bounds = [(d["min"],d["max"]) for d in designSpace]
                result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
                z = result.reshape(D,-1).T
                '''
                x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
                x=reshape(x,-1)
                y=reshape(y,-1)
                v=reshape(v,-1)
                z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
                '''
            try:             
                zClass, MU, S2, EI, P = self.predict(z)
                filteredEI=[]
                filteredZ=[]
                for i,ei in enumerate(EI):
                    if zClass[i]==0:
                        filteredEI.append(ei)
                        filteredZ.append(z[i])
                EI = array(filteredEI) 
                return filteredZ[argmax(EI)]
            except Exception,e:
                logging.error("Finding max S2 failed: " + str(e))
                return None
        else: ## more memory efficient yet slower
            maxEI = None
            maxEIcord = None
            maxEI2 = None
            maxEIcord2 = None
            space_def = []
            if hypercube:
                for counter, d in enumerate(designSpace):
                    if d["type"] == "discrete":
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter],((hypercube[0][counter]-hypercube[1][counter])/100.0)))
            else:
                for d in designSpace:
                    if d["type"] == "discrete":
                        space_def.append(arange(d["min"],d["max"]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                if not self.contains_training_instance(array(z)):
                    #pdb.set_trace()
                    zClass, MU, S2, EI, P = self.predict(array([z]))
                    cccounter = 0
                    while MU is None:
                        self.regressor.train()
                        zClass, MU, S2, EI, P = self.predict(array([z]))
                        logging.info("again this annoying error...")
                        if cccounter > 10:
                            return None
                        cccounter = cccounter + 1
                    #logging.info(str(z) + " " + str(zClass[0]) + " " + str(EI[0]))
                    if maxEI < EI[0] and zClass[0]==0: ## no need for None checking
                        maxEI = EI[0]
                        maxEIcord = z
                    if maxEI2 < EI[0]: ## no need for None checking
                        maxEI2 = EI[0]
                        maxEIcord2 = z
            logging.info("Maximum Expected Improvment is at:" + str(maxEIcord))
            logging.info("Maximum Expected Improvment without classifier is at:" + str(maxEIcord2))
            return maxEIcord
            
    def max_ei_cost(self, designSpace, hypercube=None, npts=10, cost_func = None):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        grid = False
        if grid:
            if hypercube:
                result = mgrid[[slice(h_min, h_max, npts*1.0j) for h_max, h_min , n in zip(hypercube[0],hypercube[1], n_bins)]]
                z = result.reshape(D,-1).T
            else:
                bounds = [(d["min"],d["max"]) for d in designSpace]
                result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
                z = result.reshape(D,-1).T
                '''
                x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
                x=reshape(x,-1)
                y=reshape(y,-1)
                v=reshape(v,-1)
                z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
                '''
            try:             
                zClass, MU, S2, EI, P = self.predict(z)
                filteredEI=[]
                filteredZ=[]
                for i,ei in enumerate(EI):
                    if zClass[i]==0:
                        filteredEI.append(ei)
                        filteredZ.append(z[i])
                EI = array(filteredEI) 
                return filteredZ[argmax(EI)]
            except Exception,e:
                logging.error("Finding max S2 failed: " + str(e))
                return None
        else: ## more memory efficient yet slower
            maxEI = None
            maxEIcord = None
            space_def = []
            for d in designSpace:
                if d["type"] == "discrete":
                    space_def.append(arange(d["min"],d["max"],d["step"]))
                else:
                    space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                zClass, MU, S2, EI, P = self.predict([z])
                EI_over_cost = EI / cost_func(z)
                if maxEI < EI: ## no need for None checking
                    maxEI = EI
                    maxEIcord = z
            return z
            
    def max_s2(self, designSpace, hypercube=None, npts=10):
        try:             
            maxS2 = None
            maxS2cord = None
            space_def = []
            if hypercube:
                for counter, d in enumerate(designSpace):
                    if d["type"] == "discrete":
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter],((hypercube[0][counter]-hypercube[1][counter])/100.0)))
            else:
                for d in designSpace:
                    if d["type"] == "discrete":
                        space_def.append(arange(d["min"],d["max"]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                if not self.contains_training_instance(array(z)):
                    #pdb.set_trace()
                    zClass, MU, S2, EI, P = self.predict(array([z]))
                    cccounter = 0
                    while S2 is None:
                        self.regressor.train()
                        zClass, MU, S2, EI, P = self.predict(array([z]))
                        logging.info("again this annoying error...")
                        if cccounter > 10:
                            return None
                        cccounter = cccounter + 1
                    #logging.info(str(z) + " " + str(zClass[0]) + " " + str(EI[0]))
                    if maxS2 < S2[0] and zClass[0]==0: ## no need for None checking
                        maxS2 = S2[0]
                        maxS2cord = z
            return maxS2cord
        except Exception,e:
            logging.error("Finding max S2 failed: " + str(e))
            return None
            
    def max_m(self, designSpace, hypercube=None, npts=10):
        try:             
            maxMU = None
            maxMUcord = None
            space_def = []
            if hypercube:
                for counter, d in enumerate(designSpace):
                    if d["type"] == "discrete":
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter],((hypercube[0][counter]-hypercube[1][counter])/100.0)))
            else:
                for d in designSpace:
                    if d["type"] == "discrete":
                        space_def.append(arange(d["min"],d["max"]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                if not self.contains_training_instance(array(z)):
                    #pdb.set_trace()
                    zClass, MU, S2, EI, P = self.predict(array([z]))
                    if self.configuration.goal == "min": 
                        MU = MU * -1.0
                    cccounter = 0
                    while S2 is None:
                        self.regressor.train()
                        zClass, MU, S2, EI, P = self.predict(array([z]))
                        logging.info("again this annoying error...")
                        if cccounter > 10:
                            return None
                        cccounter = cccounter + 1
                    #logging.info(str(z) + " " + str(zClass[0]) + " " + str(EI[0]))
                    if maxMU < MU[0] and zClass[0]==0: ## no need for None checking
                        maxMU = MU[0]
                        maxMUcord = z
            return maxMUcord
        except Exception,e:
            logging.error("Finding max MU failed: " + str(e))
            return None
            
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])
        
class LocalSurrogateModel(ProperSurrogateModel):

    def __init__(self, configuration, controller, fitness):
        super(LocalSurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
        D = len(fitness.designSpace)
        self.max_r = D*10
        self.regressor = self.regressor_constructor()
        self.local_regressor = self.regressor_constructor()
        
        try:
            if self.configuration.sample_on == "ei":
                self.max_uncertainty = self.max_ei
            elif self.configuration.sample_on == "s":
                self.max_uncertainty = self.max_s2
        except:
            if self.max_uncertainty:
                pass
            else:
                logging.debug("Sampling scheme wasnt specified, using Expected Improvment")
                self.max_uncertainty = self.max_ei
                
    def max_ei(self, designSpace, hypercube=None, npts=10):
        logging.info("IN")
        if hypercube: ## train the regressor globally and use it for sampling
            logging.info("Training global regressor for sampling")
            self.train_global()
            return super(LocalSurrogateModel, self).max_ei(designSpace=designSpace, hypercube=hypercube, npts=npts)
        else: ## use the local regressor, in order to keep the max_ei method intact we need to swap regressors for a moment
            temp_regressor = self.regressor
            self.regressor = self.local_regressor
            logging.info("Using local regressor for sampling")
            results = super(LocalSurrogateModel, self).max_ei(designSpace=designSpace, hypercube=hypercube, npts=npts)
            self.regressor = temp_regressor
            return results 
            
    def get_regressor(self):
        return self.local_regressor
                                  
    def get_classifier(self):
        return self.classifier
            
    def train_global(self): 
        super(LocalSurrogateModel, self).train() ## train global...
                
    def predict(self, particles):
        try:
            #logging.debug("Using tranformation function for the regressor")
            trans_particles = particles
        except:
            trans_particles = [self.fitness.transformation_function(part) for part in particles]
        MU, S2, EI, P = self.local_regressor.predict(trans_particles)
        return self.classifier.predict(particles), MU, S2, EI, P
                
    def train_local(self, hypercube):
        self.local_regressor = self.regressor_constructor()
        regressor_training_fitness = self.regressor.get_training_fitness()
        regressor_training_set = self.regressor.get_training_set()
        [maxDiag,minDiag] = hypercube
        logging.info(str(hypercube))
        for k, part in enumerate(regressor_training_set):
            if all(part <= maxDiag):
                if all(part >= minDiag):
                    self.local_regressor.add_training_instance(part, regressor_training_fitness[k])
        ## add most recent
        valid_examples = min(self.max_r,len(regressor_training_set))
        for k,part in enumerate(regressor_training_set[-self.max_r:]):
            if not self.local_regressor.contains_training_instance(part):
                self.local_regressor.add_training_instance(part, regressor_training_fitness[k-valid_examples])
        return self.local_regressor.train()
        
    def train(self, hypercube):
        self.was_trained = True
        if self.classifier.train() and self.train_local(hypercube):
            logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
                        
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary(), 
            "local_regressor_state_dict":self.local_regressor.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.local_regressor.set_state_dictionary(dict["local_regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])

class BayesClassSurrogateModel(SurrogateModel):

    def __init__(self, configuration, controller, fitness):
        super(BayesClassSurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
        self.propa_classifier = False
        if configuration.classifier == 'RelevanceVectorMachine':
            self.classifier = RelevanceVectorMachineClassifier(fitness)
            self.propa_classifier = True
        elif configuration.classifier == 'RelevanceVectorMachine2':
            self.classifier = RelevanceVectorMachineClassifier2(fitness)
        elif configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier(fitness)
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not supported')
        self.regressor = self.regressor_constructor()
        
        if self.configuration.sample_on == "ei":
            self.max_uncertainty = self.max_ei
        self.best = None
        self.best_counter = 0
        self.best_fitness = None
        self.max_mem = 4 ## in GB
        self.retrain_regressor = True
        ### max cores
        
    def get_regressor(self):
        return self.regressor
                                  
    def get_classifier(self):
        return self.classifier
        
    def get_copy(self):
        model_copy = BayesClassSurrogateModel(self.configuration, self.controller, self.fitness)
        model_copy.set_state_dictionary(self.get_state_dictionary())
        return model_copy
            
    def predict(self, particles, with_EI=True, raw=False):
        try:
            
            trans_particles = [self.fitness.transformation_function(part) for part in particles]
            #logging.debug("Using tranformation function for the regressor")
        except:
            trans_particles = particles
            
        MU, S2, EI, L = self.regressor.predict(trans_particles, with_EI=with_EI, raw=raw)
        ''' MIGHJT CAUSE PROBLEMS... annoying to add transformation 
        if with_EI:
            for i,zz in enumerate(trans_particles):
                if self.contains_training_instance(zz):
                    S2[i]=0.0
                    MU[i]=self.get_training_instance(zz)[0]
                    EI[i]=0.0
        else:
            try:
                for i,zz in enumerate(trans_particles):
                    try:
                        if self.contains_training_instance(zz):
                            S2[i]=0.0
                            MU[i]=self.get_training_instance(zz)[0]
                    except Exception,e:
                        logging.info(str(e)+ " B")
                        pdb.set_trace()
            except Exception,e:
                logging.info(str(e)+ " A")
                pdb.set_trace()
        '''
        return self.classifier.predict(particles), MU, S2, EI, L

    def train(self, hypercube=None, proba_regr=1.0):
        self.was_trained = True
        training_ok = True
        if self.retrain_regressor:
            training_ok = self.regressor.train() and training_ok
            self.retrain_regressor = False
        else:
            logging.info("Regressor training set was not updated... only classifier will be retrained")
        if self.classifier.train() and training_ok:
            logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
            
    def regressor_constructor(self):
        controller = self.controller
        configuration = self.configuration
        return GaussianProcessRegressor4(controller, configuration, self.fitness)     
        
    def add_training_instance(self, part, code, fitness, addReturn):
        if addReturn[0] == 0: ## only update regressor if the fitness function produced a result
            try:
                trans_part = part
                #logging.debug("Using tranformation function for the regressor")
            except:
                trans_part = self.fitness.transformation_function(part)
                
            self.regressor.add_training_instance(trans_part, fitness)
            self.retrain_regressor = True
            if (code == 0):
                logging.info("New possible real best")
                if self.best_fitness is None or (fitness > self.best_fitness and (self.configuration.goal == "max")):## maximization
                    self.best_fitness = fitness
                    self.best = part
                    try:
                        self.best_counter = len(self.classifier.training_set)
                    except:
                        self.best_counter = 0
                if self.best_fitness is None or (fitness < self.best_fitness and (self.configuration.goal == "min")):## minimization
                    self.best_fitness = fitness
                    self.best = part
                    try:
                        self.best_counter = len(self.classifier.training_set)
                    except:
                        self.best_counter = 0
                self.regressor.set_y_best(fitness)
                self.classifier.add_training_instance(part, 1.)
            else:
                self.classifier.add_training_instance(part, -1.)
        else:
            self.classifier.add_training_instance(part, -1.)
                    
    def get_best(self):
        return self.best, self.best_fitness
        
    def contains_training_instance(self, part):
        try:
            trans_part = self.fitness.transformation_function(part)
            #logging.debug("Using tranformation function for the regressor")
        except:
            trans_part = part
        return self.regressor.contains_training_instance(trans_part) or self.classifier.contains_training_instance(part)  

    def get_training_instance(self, part):
        code = self.classifier.get_training_instance(part) 
        fitness = None
        if self.regressor.contains_training_instance(part):
            fitness = self.regressor.get_training_instance(part)            
        return code, fitness
        
    def model_failed(self, part):
        return False
        
    def max_ei(self, designSpace, npts=10):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        num_points = 1 
        mb_per_point = 1
        space_def = []
        steps = []
        for counter, d in enumerate(designSpace):
            if d["type"] == "discrete":
                num_points = num_points * int((d["max"] - d["min"])/ d["step"])
                n_bins[counter] = int((d["max"] - d["min"])/ d["step"]) + 1.0
                steps.append(d["step"])
            else:
                num_points = num_points * npts
                n_bins[counter] = npts
                steps.append(int((d["max"] - d["min"])/ npts))
        current_max = -1.
        current_max_cord = None
        for ii in [1]:
            bounds = [(d["min"],d["max"]) for d in designSpace]
            result = mgrid[[slice(row[0], row[1], n*1.0j) for row,n in zip(bounds, n_bins)]]
            z = result.reshape(D,-1).T
            
            if self.propa_classifier: 
                class_prob, MU, S2, EI, P = self.predict(z)
                corr_EI = EI * class_prob 
            else:
                labels, MU, S2, EI, P = self.predict(z)
                labels = labels.reshape(EI.shape[0],1)
                place(labels,labels == -1.,[0.0]) 
                corr_EI = EI * labels
                #import pdb
                #pdb.set_trace()
            temp_max_cord = argmax(corr_EI)
          
            if corr_EI[temp_max_cord] > current_max:
                current_max = corr_EI[temp_max_cord]
                current_max_cord = z[temp_max_cord]
        return current_max_cord
            
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])
        
class BayesClassSurrogateModel2(BayesClassSurrogateModel):

    def get_dump(self):
        return [(self.classifier.training_set, self.classifier.training_labels),(self.regressor.training_set, self.regressor.training_fitness)]

    def __init__(self, configuration, controller, fitness):
        #super(BayesClassSurrogateModel2, self).__init__(configuration,
        #                                           controller,
        #                                           fitness)
        self.controller = controller
        self.fitness = fitness
        self.configuration = configuration
        self.propa_classifier = False
        if configuration.classifier == 'RelevanceVectorMachine':
            self.classifier = RelevanceVectorMachineClassifier(fitness, configuration)
            self.propa_classifier = True
        elif configuration.classifier == 'RelevanceVectorMachine2':
            self.classifier = RelevanceVectorMachineClassifier2(fitness, configuration)
        elif configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier(fitness, configuration)
        elif configuration.classifier == 'ResourceAwareClassifier':
            self.classifier = ResourceAwareClassifier(fitness, configuration, controller)
            self.propa_classifier = True
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not supported')
        self.retrain_regressor = True
        self.regressor = self.regressor_constructor()
        if self.configuration.sample_on == "ei":
            self.max_uncertainty = self.max_ei
        self.best = None
        self.bests_counter = []
        self.bests = []
        self.best_fitness = None
        self.best_parts = []
        self.max_mem = 4 ## in GB
        ### max cores
       
    def classifier_train(self):
        self.classifier.train(bests=self.gets_bests())
        
    def train(self, hypercube=None):
        self.was_trained = True
        training_ok = True
        if self.retrain_regressor:
            if len(self.regressor.training_set) > 100:
                logging.info("Regressor training set reached substantial size:" + str(len(self.regressor.training_set)) + " random retrain inbound..")
                faith_roll = uniform(0.0,1.0)
                if faith_roll < 0.1:
                    logging.info("Faith rolled " + str(faith_roll) + " ... retraining")
                    training_ok = self.regressor.train() and training_ok
                else:
                    logging.info("Using old regressor hyperparameter estimation...")
                    training_ok = True
                self.retrain_regressor = False    
            else:    
                training_ok = self.regressor.train() and training_ok
                self.retrain_regressor = False
        else:
            logging.info("Regressor training set was not updated... only classifier will be retrained")
        if self.classifier.train(bests=self.get_bests_index()) and training_ok:
            logging.info("Trained Surrogate Model, error up to:" + str(self.regressor.error_tolerance()))
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
        
    def get_copy(self):
        model_copy = BayesClassSurrogateModel2(self.configuration, self.controller, self.fitness)
        model_copy.set_state_dictionary(self.get_state_dictionary())
        return model_copy
        
    def update_bests(self, part, fitness):
        try:
            best_counter = len(self.classifier.training_set)
        except:
            best_counter = 0                
        self.bests.append(fitness)
        self.best_parts.append(part)
        self.bests_counter.append(best_counter)
        
    def get_bests_limit(self):
        #return [self.best]
        limit = self.regressor.error_tolerance()
        data = zip(self.best_parts,self.bests)
        #return [part for (part,fitness) in data if math.fabs((fitness-self.best_fitness)/self.best_fitness)<limit] ## rel error
        output = [part for (part,fitness) in data if math.fabs(fitness-self.best_fitness)<limit] ## abs error
        if (output is []) or (output is None) or (len(output) == 0): ## ffs..
            logging.info("There was either something wrong with the regressor, or only one valdi design exists")
            return [self.best]
        else:
            return output
        
    def get_bests_index(self):
        limit = self.regressor.error_tolerance()
        data = zip(self.bests_counter,self.bests)
        #return [(counter,fitness) for (counter,fitness) in data if math.fabs((fitness-self.best_fitness)/self.best_fitness)<limit] ## rel error
        return [(counter,fitness) for (counter,fitness) in data if math.fabs(fitness-self.best_fitness)<limit] ## abs error
    
    def gets_bests(self):
        data = zip(self.bests_counter,self.bests)
        sorted_data = sorted(data, key=lambda tup: tup[1])
        if self.configuration.goal == "max":
            sorted_data.reverse()
        return sorted_data[0:min(5,len(sorted_data))]
            
    def get_valid_set(self):
        valid_set = []
        for i,d in enumerate(self.classifier.training_labels):
            if d == 1.:
                valid_set.append(self.classifier.training_set[i])
        return valid_set
        
    def add_training_instance(self, part, code, fitness, addReturn):
        #self.propa_classifier = False
        code = code[0]
        if code == 0:
            code = 1.
        elif code == 1:
            code = 0.
            
        if addReturn[0] == 0: ## only update regressor if the fitness function produced a result
            try:
                trans_part = self.fitness.transformation_function(part)
                #logging.debug("Using tranformation function for the regressor")
            except:
                trans_part = part
            self.regressor.add_training_instance(trans_part, fitness)
            self.retrain_regressor = True
            if (code == 1.):
                logging.info("New possible real best")
                if self.best_fitness is None or (fitness > self.best_fitness and (self.configuration.goal == "max")):## maximization
                    self.best_fitness = fitness
                    self.best = part
                    self.regressor.set_y_best(fitness)
                if self.best_fitness is None or (fitness < self.best_fitness and (self.configuration.goal == "min")):## minimization
                    self.best_fitness = fitness
                    self.best = part
                    self.regressor.set_y_best(fitness)
                self.update_bests(part,fitness)
            self.classifier.add_training_instance(part, code)
        else:
            self.classifier.add_training_instance(part, code)
                    
    def max_ei_par(self, designSpace, miu_set, llambda, local=False, without_class=False, local_classifier=False, cost_model=False): 
        if local:
            logging.info("Training local classifier")
            self.classifier.train(bests=self.gets_bests(), local_structure = True)
        if llambda == 1. and len(miu_set) < 1:
            logging.info("Only one worker aviable, using standard ei procedure..")
            return self.max_ei(designSpace)
        current_w_dir = os.getcwd()
        os.chdir('libs/ei_fpga') ### this is not ideal...
        stupid_predict_bug = False
        #bitstream configuration
        max_lambda = 2
        max_miu = 6
        cores = 2
        c_slowing = 50
        loop_unroll = 5
        mius = len(miu_set)
        n_sims = self.configuration.n_sims
        
        ####prepare miu template
        random_ints = (3 * max_miu +  3 * max_lambda) * loop_unroll
        if self.configuration.goal == "min":
            miu_mean = ones(max_miu)*10000000000.0
        elif self.configuration.goal == "max":
            miu_mean = ones(max_miu)*-10000000000.0
        miu_s2 = zeros(max_miu)
        
        ####prepare lambda template
        if self.configuration.goal == "min":
            lambdas_mean = ones((max_lambda, c_slowing))*10000000000.0
        elif self.configuration.goal == "max":
            lambdas_mean = ones((max_lambda, c_slowing))*-10000000000.0
        lambdas_s2 = zeros((max_lambda, c_slowing))
        lambdas_template = concatenate([lambdas_s2,lambdas_mean],axis=1)
        
        ### prepare mius
        mius_done = False
        mius_done_counter = 0
        if miu_set.size>0:
            while (not mius_done) and (mius_done_counter < 10):
                try:
                    labels_miu, miu_means, miu_s2s, empty, empty  = self.predict(array(miu_set), with_EI=False, raw=True)
                    if not self.propa_classifier: ## for propabilistic classifiers we act a bit differently
                        labels_miu = labels_miu.reshape(-1,1)
                        labels_miu_copy = copy(labels_miu)
                        place(labels_miu_copy,labels_miu_copy != 1.,[0.0]) 
                        if without_class: ## makes all valid
                            labels_miu_copy = ones(labels_miu_copy.shape)
                        miu_s2[0:miu_set.shape[0]] = (miu_s2s*labels_miu_copy).reshape(-1,)
                        if self.configuration.goal == "min":
                            place(miu_means,labels_miu != 1.,[10000000000.0]) 
                        elif self.configuration.goal == "max":
                            place(miu_means,labels_miu != 1.,[-10000000000.0]) 
                    #miu_s2[0:miu_set.shape[0]] = (miu_s2s*labels_miu_copy).reshape(-1,)
                    miu_mean[0:miu_set.shape[0]] = miu_means.reshape(-1,)
                    mius_done = True
                except Exception, e:
                    logging.info("This stupid predict error... redoing Mius")
                    self.regressor.train()
                    mius_done = False
                mius_done_counter = mius_done_counter + 1
        y_best = self.regressor.get_y_best()        
        
        from CpuStream import CpuStream
        def g(z_in, n_sims=n_sims):
            try:
                labels, lambda_mean, lambda_s2, empty, L = self.predict(array(z_in).reshape(-1,len(designSpace)), with_EI=False, raw=True)
                labels = labels.reshape(-1,1)
                if not self.propa_classifier:
                    place(labels,labels != 1.,[0.0]) 
            except:
                logging.info("This stupid predict error...")
                stupid_predict_bug = True
                return None
            if without_class: ## makes all valid
                labels = ones(labels.shape)
            adjusted_s2 = lambda_s2*labels
            if not self.propa_classifier:
                if self.configuration.goal == "min":
                    place(lambda_mean,labels != 1.,[10000000000.0]) 
                elif self.configuration.goal == "max":
                    place(lambda_mean,labels != 1.,[-10000000000.0])             
            ####
            lambdass = []
            for i in range(cores):
                lambdas = copy(lambdas_template)
                for ii in range(c_slowing):
                    indexx = i*c_slowing + ii*llambda
                    lambdas[0:llambda,ii] = adjusted_s2[indexx:indexx+llambda].reshape(-1,)
                    lambdas[0:llambda,ii+c_slowing] = lambda_mean[indexx:indexx+llambda].reshape(-1,)
                lambdass.append(lambdas)
            lambdass = concatenate(lambdass)
            
            #time_train = time.time()
            if self.configuration.goal == "min":
                EI = concatenate(CpuStream(n_sims, 0, y_best, miu_mean, miu_s2,randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),*lambdass))
            elif self.configuration.goal == "max":   
                EI = concatenate(CpuStream(n_sims, 1, y_best, miu_mean, miu_s2,randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),randint(0,sys.maxint,random_ints),*lambdass))
            
            if self.propa_classifier:
                EI = EI * labels

            return EI.reshape(-1,1)
        
        if local is False:
            best = self.get_best()[0]*int(llambda)
            results, ei_val = ei_optimizers.optimize(g, designSpace*int(llambda), GEN=c_slowing*llambda*len(designSpace), surrogate=self, set_best=self.get_best()[0]*int(llambda))
            improvment_l = math.fabs(ei_val[0]/(loop_unroll*n_sims))
            improvment = (math.exp(improvment_l)-1.)*100.0
            if improvment < 0.0:
                os.chdir(current_w_dir)
                return None
            logging.info("Predicted improvment " + str(improvment) + "%")
            if stupid_predict_bug:
                logging.info("Stupid predict error... retraining regressor and rerunning optimizer (!MIGHT LOOP!)")
                pdb.set_trace()
                self.regressor.train()
                return self.max_ei_par(designSpace, miu_set, llambda)
            else: 
                os.chdir(current_w_dir)
                return results
        else:  ## LOCAL exhaustive search
            batch_size = 50
            max_counter = len(local)
            counter = 0 
            best_counter = 0
            best_ei = [0.0]
            best = None
            while counter < max_counter:
                eval = min(batch_size,max_counter-counter)
                #pdb.set_trace()
                if eval<batch_size:
                    empty = [zeros(len(designSpace))] * (batch_size-eval)
                    local = local + empty
                EI = g(local[counter:counter+batch_size]) ## append zeros
                if EI is None:
                    pdb.set_trace()
                else:
                    best_in_batch = argmin(EI[0:eval])
                    if EI[best_in_batch] < best_ei:
                        best = local[counter:counter+batch_size][best_in_batch]
                        best_ei = EI[best_in_batch]
                        best_counter = counter
                    counter = counter + eval
            #pdb.set_trace()
            if stupid_predict_bug:
                logging.info("Stupid predict error... retraining regressor and rerunning optimizer (!MIGHT LOOP!)")
                pdb.set_trace()
                self.regressor.train()
                return self.max_ei_par(designSpace, miu_set, llambda, local)
            else: 
                os.chdir(current_w_dir)
                improvment_l = math.fabs(best_ei[0]/(loop_unroll*n_sims))
                improvment = (math.exp(improvment_l)-1)*100.0#/self.regressor.get_y_best(raw=True)
                logging.info("Predicted improvment " + str(improvment) + "% " + str(improvment_l))
                if improvment < 0.0:
                    logging.info("Improvment under 1%, returning None :" + str(improvment) + "%")
                    return None
                return best
                   
    def max_ei_par_soft(self, designSpace, miu_set, llambda, local=False, without_class=False, local_classifier=False, cost_model=False):
        if local:
            logging.info("Training local classifier")
            self.classifier.train(bests=self.gets_bests(), local_structure = True)
            
        #gdefintion
        stupid_predict_bug = False
        n_sims = self.configuration.n_sims
        D = len(self.fitness.designSpace)
        if len(miu_set) > 0:
            if self.propa_classifier: ## probabilsitic classifier
                try:
                    labels_miu, miu_means, miu_s2s, empty, L  = self.predict(array(miu_set), with_EI=False, raw=True)
                    miu_set = miu_set[labels_miu > 0.0]
                    labels_miu = labels_miu[labels_miu > 0.0]
                    miu_s2s = miu_s2s[labels_miu > 0.0]
                except: ## this
                    logging.info("something has gone wrong with the mius, trying to retrain the model") #From the looks of it is the 
                    try:
                        self.train()
                        labels_miu, miu_means, miu_s2s, empty, L  = self.predict(array(miu_set), with_EI=False, raw=True)
                        miu_set = miu_set[labels_miu > 0.0]
                        labels_miu = labels_miu[labels_miu > 0.0]
                        miu_s2s = miu_s2s[labels_miu > 0.0]
                    except:
                        logging.info("Didnt work...")
                        return None
                if (len(miu_set) == 0 ):
                    logging.info("All mius predicted to be invalid, wont use any")
            else:
                try:
                    labels_miu, miu_means, miu_s2s, empty, L  = self.predict(array(miu_set), with_EI=False, raw=True)
                    pdb.set_trace()
                    miu_set = miu_set[labels_miu==1]
                    labels_miu = labels_miu[labels_miu==1]
                    miu_s2s = miu_s2s[labels_miu==1]
                except: ## this
                    logging.info("something has gone wrong with the mius, trying to retrain the model") #From the looks of it is the 
                    try:
                        self.train()
                        labels_miu, miu_means, miu_s2s, empty, L  = self.predict(array(miu_set), with_EI=False, raw=True)
                        miu_set = miu_set[labels_miu==1]
                        labels_miu = labels_miu[labels_miu==1]
                        miu_s2s = miu_s2s[labels_miu==1]
                    except:
                        logging.info("Didnt work...")
                        return None
                if (len(miu_set) == 0 ):
                    logging.info("All mius predicted to be invalid, wont use any")
        else:
            miu_means = array([[]])
            miu_s2s = array([[]])
        if int(llambda) == 1 and len(miu_set) == 0:
            logging.info("Only one worker aviable, using standard ei procedure..")
            return self.max_ei(designSpace)
        #g definition
        
        def g(zz, n_sims=n_sims, miu_set=miu_set):
            EI = []
            for z in zz:
                z = array(z).reshape((-1, D))
                #if (len(z) == 1):
                #   z = z[0] #cant be asked to fix it
                #labels, lambda_mean, lambda_s2, empty, empty = self.predict([z], with_EI=False, raw=True) ### needs to be raw!
                #if labels is None:
                for zzz in z:
                     #if (numpy_array_index(miu_set,zzz)[0]):
                     #  # logging.info("Design being evaled")
                     #   EI.append(array([.0])) ## if any is predicted to be invalid we prevent expensive ei calculation
                     #   continue
                     if (self.contains_training_instance(zzz)):
                       # logging.info("Design already evaluated")
                        EI.append(array([.0])) ## if any is predicted to be invalid we prevent expensive ei calculation
                        continue
                try:
                    labels = self.classifier.predict(z)
                    if self.propa_classifier:
                        if (any(labels==0.)):
                            EI.append(array([.0])) ## if any is predicted to be invalid we prevent expensive ei calculation
                            continue
                        if len(labels) == 1:
                            labels, MU, S2, ei, P = self.predict(z)
                            labels = labels.reshape(ei.shape[0],1)
                            corr_EI = ei * labels
                            EI.append(-1.0*corr_EI.reshape(1,1))
                            continue
                    else:
                        if (any(labels!=1.)):
                            EI.append(array([.0])) ## if any is predicted to be invalid we prevent expensive ei calculation
                            continue
                        if len(labels) == 1:
                            labels, MU, S2, ei, P = self.predict(z)
                            place(labels,labels != 1.,[0.0]) 
                            labels = labels.reshape(ei.shape[0],1)
                            corr_EI = ei * labels
                            EI.append(-1.0*corr_EI.reshape(1,1))
                            continue

                    zz = vstack((miu_set,z))
                    labels, mean_predict, s2_predict, empty, L = self.predict(zz, with_EI=False, raw=True) ### needs to be raw!
                    
                    labels_miu = labels[0:len(miu_set)]
                    miu_means = mean_predict[0:len(miu_set)]
                    miu_s2s = s2_predict[0:len(miu_set)]
                    
                    labels = labels[len(miu_set):]
                    lambda_mean = mean_predict[len(miu_set):]
                    lambda_s2 = s2_predict[len(miu_set):]
                    
                    if not (miu_means is None):
                        miu_means.reshape((-1,))
                    
                    if self.configuration.goal == "min":
                        result = -1.*ei_multi_min(miu_means.reshape((-1,)), [], L.reshape((-1,)), lambda_mean.reshape((-1,)), self.regressor.get_y_best()[0], n_sims) / n_sims
                        if cost_model:
                            cost = cost_model.predict(z)
                            EI.append(array([result/cost])) ## take into acount design time
                        else:
                            EI.append(array([result]))
                    elif self.configuration.goal == "max":
                        result = -1.*ei_multi_max(miu_means.reshape((-1,)), [], L.reshape((-1,)), lambda_mean.reshape((-1,)), self.regressor.get_y_best()[0], n_sims) / n_sims
                        if cost_model:
                            cost = cost_model.predict(z)
                            EI.append(array([result/z])) ## take into acount design time
                        else:
                            EI.append(array([result]))
                                
                except Exception, e:
                        #pdb.set_trace()
                        logging.info(str(e))
                        logging.info("This stupid predict error...")
                        stupid_predict_bug = True
                        return None
            return EI
        #gdefinition end
        if local is False:
            best = self.get_best()[0]*int(llambda)
            results, ei_val = ei_optimizers.optimize(g, designSpace*int(llambda), GEN=500, surrogate=self, set_best=self.get_best()[0]*int(llambda))
            improvment_l = math.fabs(ei_val[0])
            improvment = (math.exp(improvment_l)-1.)*100.0
            if improvment < 0.0:
                return None
            logging.info("Predicted improvment " + str(improvment) + "%")
            #improvment_l = math.fabs(ei_val[0]/n_sims)
            #improvment = (math.exp(improvment_l)-1.)*100.0
            #if improvment < 0.0:
            #    return None
            #logging.info("Predicted improvment " + str(improvment) + "%")
            if stupid_predict_bug:
                logging.info("Stupid predict error... retraining regressor and rerunning optimizer (!MIGHT LOOP!)")
                #pdb.set_trace()
                self.regressor.train()
                return self.max_ei_par_soft(designSpace, miu_set, llambda, cost_model=cost_model)
            else:
                return results
        else:  ## LOCAL exhaustive search
            batch_size = 50
            max_counter = len(local)
            counter = 0
            best_counter = 0
            best_ei = [0.0]
            best = None
            while counter < max_counter:
                eval = min(batch_size,max_counter-counter)
                #pdb.set_trace()
                if eval<batch_size:
                    empty = [zeros(len(designSpace))] * (batch_size-eval)
                    local = local + empty
                EI = g(local[counter:counter+batch_size]) ## append zeros
                if EI is None:
                    pdb.set_trace()
                else:
                    best_in_batch = argmin(EI[0:eval])
                    if EI[best_in_batch] < best_ei:
                        best = local[counter:counter+batch_size][best_in_batch]
                        best_ei = EI[best_in_batch]
                        best_counter = counter
                    counter = counter + eval
            if stupid_predict_bug:
                logging.info("Stupid predict error... retraining regressor and rerunning optimizer (!MIGHT LOOP!)")
                pdb.set_trace()
                self.regressor.train()
                return self.max_ei_par_soft(designSpace, miu_set, llambda, local, cost_model=cost_model)
            else:
                improvment_l = math.fabs(best_ei[0]/(loop_unroll*n_sims))
                improvment = (math.exp(improvment_l)-1)*100.0
                logging.info("Predicted improvment " + str(improvment) + "% " + str(improvment_l))
                if improvment < 0.0:
                    logging.info("Improvment under 1%, returning None :" + str(improvment) + "%")
                    return None
                return best

               
    def max_ei(self, designSpace):

        #result = self.brute_search(designSpace)
        #logging.info("brute force predicts : " + self.brute_search(designSpace))
            
        if self.configuration.search=='brute': # brute search
            return self.brute_search(designSpace)
        elif False: ## gradient descent, exhasstive is too expensive...
            return self.gradient_minimizer(designSpace)
        else: ### pso search
            return self.pso_search(designSpace)
            
    def local_brute_search(self, designSpace, point, radius=1., npts=10):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        ## define the space
        
        steps = []
        for counter, d in enumerate(designSpace):
            if d["type"] == "discrete":
                n_bins[counter] = d["step"]
            else:
                n_bins[counter] = 1./npts
        bounds = [(maximum(point[i] - n_bins[i]* radius,d["min"]), minimum(point[i] + n_bins[i]*radius,d["max"])) for i,d in enumerate(designSpace)] 
        
        #current_max = -1.
        #current_max_cord = None
        ### preapring search grid... used a lot of memory for large spaces
        result = mgrid[[slice(row[0], row[1], int((row[1]-row[0])/n_bins[i]+1)*1.0j) for i,row in enumerate(bounds)]]
        z = result.reshape(D,-1).T
        ### perform prediction
        labels, MU, S2, EI, P = self.predict(z)
        if sum(EI) == 0.0:
            logging.info("again didnt find shitTTT...")
            return None
            
        temp_max_cord = argmax(EI)
        current_max = EI[temp_max_cord]
        current_max_cord = z[temp_max_cord]
        return current_max_cord
            
    def brute_search(self, designSpace, npts=10, hypercube = None):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        ## define the space
        
        num_points = 1 
        mb_per_point = 1
        space_def = []
        steps = []
        for counter, d in enumerate(designSpace):
            if d["type"] == "discrete":
                num_points = num_points * int((d["max"] - d["min"])/ d["step"])
                n_bins[counter] = int((d["max"] - d["min"])/ d["step"]) + 1.0
                steps.append(d["step"])
            else:
                num_points = num_points * npts
                n_bins[counter] = npts
                steps.append(int((d["max"] - d["min"])/ npts))
            bounds = [(d["min"],d["max"]) for d in designSpace] 
        #current_max = 0.
        #current_max_cord = None
        
        ### preapring search grid... used a lot of memory for large spaces
        
        result = mgrid[[slice(row[0], row[1], n*1.0j) for row,n in zip(bounds, n_bins)]]
        z = result.reshape(D,-1).T
        z = [zz for zz in z if not self.contains_training_instance(zz)]
        ### perform prediction
        labels, MU, S2, EI, P = self.predict(z)
        labels = labels.reshape(EI.shape[0],1)
        if not self.propa_classifier:
            place(labels,labels != 1.,[0.0]) 
        corr_EI = EI * labels
        
        ### pos processing
        
        if sum(corr_EI) == 0.0:
            logging.info("didnt find shit...")
            return None
       
        temp_max_cord = argmax(corr_EI)
        
        #if corr_EI[temp_max_cord] > current_max:
        current_max = corr_EI[temp_max_cord]
        logging.info("Maximum expected improvment: " + str(current_max) + " y_best: "  + str(self.regressor.get_y_best()))
        #if math.fabs(current_max/self.regressor.get_y_best()) < 0.01:
        #    return None
        current_max_cord = z[temp_max_cord]
        return current_max_cord
    
    def pso_search(self, designSpace):
        def g(z):
            labels, MU, S2, EI, P = self.predict(array(z), with_EI=True)
            labels = labels.reshape(EI.shape[0],1)
            if not self.propa_classifier:
                place(labels,labels != 1.,[0.0]) 
            result = -1.*(EI * labels)
            return result
        try:
            return array(ei_optimizers.optimize(g, designSpace, GEN=1000, surrogate=self, set_best=self.get_best()[0])[0])
        except:
            logging.info("Probably only one example q")
            return None 
        
    def gradient_minimizer(self, designSpace):
        bounds = [(d["min"],d["max"]) for d in designSpace] 
        def f(z):
            labels, MU, S2, EI, P = self.predict(array([z]), with_EI=True)
            labels = labels.reshape(EI.shape[0],1)
            if not self.propa_classifier:
                place(labels,labels != 1.,[0.0])      
            return (EI * labels)[0]
            
        ### find valid seeds
        valid_seeds = []
        for i in xrange(0,10000):
            part = [round(uniform(d["min"],d["max"])) for d in designSpace]
            
            if f(part) > 0.:
                valid_seeds.append(part)
            
        logging.info('Finished finding valid seeds')
        
        def g(z):
            labels, MU, S2, EI, P = self.predict(array([z]), with_EI=True)
            labels = labels.reshape(EI.shape[0],1)
            if not self.propa_classifier:
                place(labels,labels == -1.,[0.0]) 
            result = -1.*(EI * labels)[0]
            return result
        
        best_so_far = None
        best_so_far_val = 0.0
        res = []
        if len(valid_seeds) == 0:
            logging.info("Valid seeds is empty:")
            import pdb
            pdb.set_trace()
        for x0 in valid_seeds:
            res = minimize(g, x0, method='L-BFGS-B', bounds = bounds,  options={'gtol': 1e-6, 'disp': False})
            if res.fun > best_so_far_val:
                best_so_far = res.x
        return best_so_far
            
            
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary(), "propa_classifier": self.propa_classifier}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])
        self.propa_classifier = dict["propa_classifier"]
    