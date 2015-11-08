
import logging
import traceback

from numpy import unique, asarray, bincount, array, append, arange, place, ones, maximum, minimum, mgrid, exp, sqrt, logical_and, logical_or, square, zeros
from numpy.random import uniform, rand, randint 
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold, LeaveOneOut
from copy import deepcopy
from pyGP_OO.Valid import valid
from scipy.stats import norm
from regressors import * 

from utils import numpy_array_index

from pickle import dumps
import pdb

import sys
if sys.version_info < (2, 7):
    from rvm import *
else:
    logging.info("cannot use rvms with python versions other than 2.6")

#TODO - should this be an abstract class instead?
class Classifier(object):

    def __init__(self, fitness=None, configuration=None):
        self.fitness = fitness
        self.training_set = None
        self.training_labels = None
        self.clf = None
        self.oneclass = False
        self.conf = configuration

    def train(self, local_structure = False):
        return True

    def predict(self, z):
        output = []
        for input_vector in z:
            output.append(0)
        output = array(output)
        return output

    ## TODO - check if element is in the array... just for the sake of it
    def add_training_instance(self, part, label):
        if self.training_set is None:
            self.training_set = array([part])
            self.training_labels = array([label])
        else:
            contains = self.contains_training_instance(self.training_set)
            if contains:
                logging.info('A particle duplicate is being added.. check your code!!')
            else:
                self.training_set = append(self.training_set, [part], axis=0)
                self.training_labels = append(self.training_labels, [label],
                                              axis=0)
                                          
    def contains_training_instance(self, part):
        contains, index = numpy_array_index(self.training_set, part)
        return contains
            
    def get_training_instance(self, part):
        contains, index = numpy_array_index(self.training_set, part)
        if self.training_set is None:
            logging.error('cannot call get_training_instance if training_set is empty')
            return False
        elif contains:
            return self.training_labels[index]
        else :
            logging.error('cannot call get_training_instance if training_set does not contain the particle')
            return False

    ###############
    ### GET/SET ###
    ###############
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels}
        return dict
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']

    def get_parameter_string(self):
        return "Not implemented"
        
class SupportVectorMachineClassifier(Classifier):

    def find_frequency(self, array):
        output = {}
        for elem in self.training_labels:
            try:
                output[elem] = output[elem] + 1.
            except:
                output[elem] = 1.
                
        for key, value in output.iteritems():
            output[key] = 1.#/value
        return output
        
    def train(self, local_structure=False, bests=None):
        try:
            inputScaler = preprocessing.StandardScaler().fit(self.training_set)
            scaledSvcTrainingSet = inputScaler.transform(self.training_set)
            all_labels = unique(asarray(self.training_labels))
            class_weight = dict([(i,1.0)for i in all_labels])
            #class_weights[0] = 2.0
            #pdb.set_trace()
            if len(unique(asarray(self.training_labels))) < 2:
                logging.info('Only one class encountered, we do not need to use a classifier')
                #self.clf = svm.OneClassSVM()
                #self.clf.fit(scaledSvcTrainingSet)
                self.oneclass = True
            else:
                self.oneclass = False
                #class_weight = self.find_frequency(self.training_labels)
                sample_weight = ones(len(self.training_labels))
                if self.conf.weights_on :
                    if not (bests is None):  
                    #logging.info("Sample weight for: " + str(self.training_set[best]) + " adjusted.. " + str(best))
                        for i,(index,fitness) in enumerate(bests):
                            sample_weight[index] = 1.5#0 + (len(bests)-i)*0.25
                #sample_weight[best]=10.
                if local_structure:
                    logging.info("Using local structures for classifier")
                    param_grid = {
                        'gamma' : self.conf.gamma,
                        'C' : self.conf.C
                        #'gamma': 10.0 * 1.25** arange(1., 20),
                        #'C':     1.5 ** arange(-20, 1)
                        }
                    #class_weight[1.] = class_weight[1.] #* 1.5
                else:
                    param_grid = {
                        'gamma' : self.conf.local_gamma,
                        'C' : self.conf.local_C
                        #'gamma': 1.2 ** arange(-10, 10),
                        #'C':     10*1.25 ** arange(1, 10)
                        }
                    #class_weight[1.] = class_weight[1.] #* 1.5
                logging.info(str(class_weight))
                try:
                    try:
                        self.type = 2
                        self.clf = GridSearchCV(svm.SVC(class_weight=class_weight), param_grid=param_grid, 
                                        cv=LeaveOneOut(n=len(self.training_labels.reshape(-1))))
                        self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1))
                        self.clf = self.clf.best_estimator_
                    except Exception,e: ##in case when we cannot construct equal proportion folds
                        self.type = 1
                        logging.debug('Using KFold cross validation for classifier training: ' + str(e))
                        self.clf = GridSearchCV(svm.SVC(class_weight=class_weight), param_grid=param_grid,
                                                cv=KFold(n=self.training_labels.shape[0],n_folds=self.training_labels.shape[0]))
                        self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1))
                        self.clf = self.clf.best_estimator_ ## gridsearch cant be pickled...
                        #logging.info(str(self.training_labels.shape[0])))
                except Exception, e:## in case for example when we have single element of a single class, cant construct two folds
                    self.type = 0
                    logging.debug('One of the classes has only one element, cant use cross validation:' + str(e))
                    if local_structure:
                        self.clf = svm.SVC(kernel='rbf', gamma=100.0, C = 10., class_weight=class_weight)
                    else:
                        self.clf = svm.SVC(kernel='rbf', gamma=0.05, C = 10., class_weight=class_weight)
                self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1), sample_weight=sample_weight) ## we refit using all data
                logging.info('Classifier training successful - C:' + str(self.clf.C) + " gama:" +  str(self.clf.gamma))
            return True
        except Exception, e:
            logging.error('Classifier training failed.. ' + str(e))
            return False

    def predict(self, z):
        try:
            if self.oneclass:
                ## TODO - rewrite it not to use a stupid loop...
                return array([self.training_labels[0]] * len(z))
            else:
                # Scale inputs and particles
                inputScaler = preprocessing.StandardScaler().fit(self.training_set)
                scaledz = inputScaler.transform(z)
                zClass = self.clf.predict(scaledz)
                for i,zz in enumerate(z):
                    if self.contains_training_instance(zz):
                        zClass[i]=self.get_training_instance(zz)
                return zClass
        except Exception, e:
            logging.error('Prediction failed... ' + str(e))
            return None
            
    def get_parameter_string(self):
        try:
            return str(self.clf.gamma) + "_" + str(self.clf.C)
        except:
            return "N\A"
            
    ## TODO - come up with a smart way of storing these...
    def get_state_dictionary(self):
        '''
        if self.clf is None:
            clf = None
            self.type = None
        else:
            #try:
                if self.type == 0 :
                    clf = deepcopy(self.clf.get_params(deep=True))
                    logging.info(str(clf))
                else:
                    clf = deepcopy(self.clf.best_estimator_.get_params(deep=True))
                    logging.info(str(clf))
            #except:
            #    self.type = None
        ''' 
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels,
                'oneclass': self.oneclass,
                'clf': deepcopy(self.clf)}

        return dict
        
    ###
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']
        self.oneclass = dict['oneclass']
        self.clf = dict['clf']
        '''
        self.type = dict['type']

        #try:
        self.clf = svm.SVC()
        self.clf.set_params(**dict['clf'])
        #except:
        #    self.clf = None
        '''
        
class RelevanceVectorMachineClassifier(Classifier):

    def translate_to_neg_pos(self, labels):
        #place(labels, labels == 1., [-1.0]) 
        place(labels, labels != 1., [-1.0])
        return labels 
        
    def translate_from_prob(self, result):
        place(result, result < 0.5,[0.0]) 
        place(result, result > 0.1,[1.0]) 
        return result 
                  
    def train_no_cross(self):
        n_folds = 5
        samples = deepcopy(self.training_set)
        labels = deepcopy(self.training_labels)
        labels = self.translate_to_neg_pos(labels)
        normalizer = VectorNormalizer()
        normalizer.train(samples)
        for i in xrange(len(samples)):
            samples[i] = normalizer(samples[i])
        trainer = Trainer(RadialBasisKernel(self.best_gamma), 0.000001)
        self.pfn = NormalizedFunction(trainer.trainProbabilistic(samples, labels, n_folds), normalizer)
        return True
                  
    def train(self, bests=None, local_structure=False):
        logging.info("Using local structure :" + str(local_structure))
        self.bests = bests
        bubble = self.get_around_best(bests)
        #try:
        samples = deepcopy(self.training_set)
        n = samples.shape[1]
        for p in bubble:
            samples = append(samples,p.reshape(1,n),axis=0)
        labels = deepcopy(self.training_labels)
        for p in bubble:
            labels = append(labels,1)
        labels = self.translate_to_neg_pos(labels)
                
        normalizer = VectorNormalizer()
        normalizer.train(samples)
        for i in xrange(len(samples)):
            samples[i] = normalizer(samples[i])
        self.normalizer = normalizer
        
        inputScaler = preprocessing.StandardScaler().fit(self.training_set)
        samples = inputScaler.transform(self.training_set)
        
        ## swap position of best
        if self.bests: 
            normalized_training_set = self.normalize(self.training_set)
            self.best_distance = []
            self.normalized_best = []
            for i in xrange(len(self.bests)):
                best_index = bests[i][0]
                distance, closest = self.get_distance_to_closest_wrong(normalized_training_set[best_index], normalized_training_set)
                self.best_distance.append(closest)
                self.normalized_best.append(normalized_training_set[best_index])
            self.normalized_best = array(self.normalized_best)
            self.best_distance = array(self.best_distance)
            logging.info(str(self.best_distance) + " " + str(self.tiny_bit()))
        best_gamma = 0.0
        best_cross = -10.0
        #gamma = 2.0 / rvm_binding.compute_mean_squared_distance(VectorSample(samples))
        #gamma_range = (2.0 ** arange(-10.0, 10.0)) / rvm_binding.compute_mean_squared_distance(VectorSample(samples))
        
        if local_structure:
            gamma = 0.001
            gamma_limit = 1000.0
        else:
            gamma = 1.2 ** - 10
            gamma_limit = 1.2 ** 10
        n_folds = 5
        if False:
            while gamma <= gamma_limit: 
                trainer = Trainer(RadialBasisKernel(gamma), 0.00001)
                cross = trainer.crossValidate(samples, labels, n_folds)
                if best_cross < cross:
                    best_cross = cross
                    best_gamma = gamma
                gamma = gamma + 0.005
            logging.info("finished cross")
            if False:
                samples = deepcopy(self.training_set)
                labels = deepcopy(self.training_labels)
                n = samples.shape[1]
                for p in bubble:
                    labels = append(labels,1)
                normalizer = VectorNormalizer()
                normalizer.train(samples)
                for i in xrange(len(samples)):
                    samples[i] = normalizer(samples[i])
                labels = self.translate_to_neg_pos(labels)
                normalizer = VectorNormalizer()
                normalizer.train(samples)
        else:
            best_gamma = 2.0 / rvm_binding.compute_mean_squared_distance(VectorSample(samples))
        trainer = Trainer(RadialBasisKernel(best_gamma), 0.00001)
        self.best_gamma = best_gamma
        self.pfn = trainer.trainProbabilistic(samples, labels, n_folds)
        
        #print "KURWAAAA" + str()
        self.limit = 0.35
        #for best in self.bests:
        #    self.limit = min(self.pfn(samples[best[0]]), self.limit)
        #max_prob = 0.0
        #for i in range(len(samples)):
        #    if self.training_labels[i] != 1.:
        #        max_prob = max(self.pfn(samples[i]), max_prob)
        #self.limit = max(self.limit, max_prob)
        #logging.info("RVM valid limit:" + str(self.limit) + " " + str(max_prob))
        
        return True
        # except Exception, e:
            # logging.error('Classifier training failed..' + str(e))
            # pdb.set_trace()
            # return False

    def predict(self, z):
        zClass = []
        within_dist = []
        inputScaler = preprocessing.StandardScaler().fit(self.training_set)
        z = inputScaler.transform(z)
        for zz in z:
            zClass.append([self.pfn(zz)])
            within = False
            normalized_zz = self.normalize(zz)
            for i in xrange(len(self.bests)):
                #pdb.set_trace()
                #within = within or (self.get_distance(self.normalized_best[i], normalized_zz) < self.best_distance[i])
                #pdb.set_trace()
                within = within or (abs(self.normalized_best[i] - normalized_zz) < abs(self.normalized_best[i] - self.best_distance[i])).all()
            within_dist.append([within])
            #try:
            #    within_dist.append([self.get_distance(self.normalized_best, zz)] < self.best_distance)
            #except:
            #    within_dist.append([False])
        #import pdb
        result = array(zClass)
        within_dist = array(within_dist)
        place(result, ~ logical_or(result > self.limit, within_dist),[0.0])
        #place(result, within_dist,[1.0])  
        #place(result, ~within_dist,[0.0])          
        #result = 1/(1+exp(-result*60+30)) ## sigmoid function
        #result = result * result * result * result * result * result 
        return result
            
    def tiny_bit(self):
        try:
            self.bit
        except:
            self.normalize(self.training_set[0])
            step_array = []
            for i,dd in enumerate(self.fitness.designSpace):
                step_array.append((dd["max"] - dd["min"]) / dd["step"])
            step_array = array(step_array)
            steps = 1./step_array
            origin = zeros(self.normal_array.shape)
            self.bit = self.get_distance(origin, steps)
        return self.bit / 2
        
    def normalize(self, z):
        try:
            self.normal_array
        except:
            normalizing_array = []
            for i,dd in enumerate(self.fitness.designSpace):
                normalizing_array.append(dd["max"] - dd["min"])
            self.normal_array = array(normalizing_array)
        return z / self.normal_array
            
    def get_parameter_string(self):
        return "RVM"
            
    ## TODO - come up with a smart way of storing these...
    def get_state_dictionary(self):
        try:
            self.bests
        except:
            self.bests = None
        try:
            self.normalizer
        except:
            self.normalizer = None
        try:
            self.best_distance
        except:
            self.best_distance = None
        try:
            self.normalized_best
        except:
            self.normalized_best = None    
        try: 
            self.best_gamma
        except:
            self.best_gamma = None
            
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels, 
                'bests': self.bests,
                'best_gamma' : self.best_gamma,
                'normalizer' : self.normalizer,
                'normalized_best' : self.normalized_best,
                'best_distance' : self.best_distance}

        return dict
        
    ###
    def set_state_dictionary(self, dict):
        self.bests = dict['bests']
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']
        self.best_gamma = dict['best_gamma']
        self.normalizer = dict['normalizer']
        self.normalized_best = dict['normalized_best']
        self.best_distance = dict['best_distance']
        if self.bests:
            self.train(self.bests)
        else:
            self.train_no_cross()
        
    def get_around_best(self, bests, radius = 1.):
        designSpace = self.fitness.designSpace  
        D = len(designSpace)
        npts = 0.01
        n_bins = npts*ones(D)
        ## define the space
        local_perturbation = []
        
        try:
            chosen = []
            for index in bests:
                point = self.training_set[int(index[0])]
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
                local_perturbation.extend([zz for zz in z if not (self.contains_training_instance(zz))])
                rand_indicies = []
                #pdb.set_trace()
                length = len(local_perturbation)
                while len(rand_indicies) < len(self.training_set)/2: # just a way of weighting best
                    rand_index = randint(0,length) 
                    rand_indicies.append(rand_index)
                chosen.extend([local_perturbation[rand_index] for rand_index in rand_indicies])
            return []
        except:
            return []
    
    def get_distance_to_closest_wrong(self, best, normalized_set):
        invalid_samples = []
        for i in xrange(len(normalized_set)):
            if self.training_labels[i] != 1.:
                invalid_samples.append(normalized_set[i])
        min_distance = 1000000.0
        closest = None
        for i in xrange(len(invalid_samples)):
            dist = self.get_distance(best, invalid_samples[i])
            if min_distance > dist :
                min_distance = dist
                closest = invalid_samples[i]
        return min_distance, closest
                
    def get_distance(self, a, b):
        a = array(a)
        b = array(b)
        diff = a - b
        squared_diffs = square(diff)
        sum_squared_diffs = sum(squared_diffs)
        dist = sqrt(sum_squared_diffs)
        return dist
            
class RelevanceVectorMachineClassifier2(Classifier):

    def translate_to_neg_pos(self, labels):
        #place(labels, labels == 1., [-1.0]) 
        place(labels, labels != 1., [-1.0])
        return labels 
        
    def train(self, bests=None, local_structure=False):
        logging.info("Using local structure :" + str(local_structure))
        self.bests = bests
        bubble = self.get_around_best(bests)
        #try:
        samples = deepcopy(self.training_set)
        n = samples.shape[1]
        for p in bubble:
            samples = append(samples,p.reshape(1,n),axis=0)
        labels = deepcopy(self.training_labels)
        for p in bubble:
            labels = append(labels,1)
        labels = self.translate_to_neg_pos(labels)
                
        normalizer = VectorNormalizer()
        normalizer.train(samples)
        for i in xrange(len(samples)):
            samples[i] = normalizer(samples[i])
        self.normalizer = normalizer
        
        inputScaler = preprocessing.StandardScaler().fit(self.training_set)
        samples = inputScaler.transform(self.training_set)
        
        ## swap position of best
        if self.bests: 
            normalized_training_set = self.normalize(self.training_set)
            self.best_distance = []
            self.normalized_best = []
            for i in xrange(len(self.bests)):
                best_index = bests[i][0]
                distance, closest = self.get_distance_to_closest_wrong(normalized_training_set[best_index], normalized_training_set)
                self.best_distance.append(closest)
                self.normalized_best.append(normalized_training_set[best_index])
            self.normalized_best = array(self.normalized_best)
            self.best_distance = array(self.best_distance)
            logging.info(str(self.best_distance) + " " + str(self.tiny_bit()))
        best_gamma = 2.0 / rvm_binding.compute_mean_squared_distance(VectorSample(samples))
        trainer = Trainer(RadialBasisKernel(best_gamma), 0.00001)
        self.best_gamma = best_gamma
        self.pfn = trainer.train(samples, labels)        
        return True
        # except Exception, e:
            # logging.error('Classifier training failed..' + str(e))
            # pdb.set_trace()
            # return False

    def predict(self, z):
        zClass = []
        within_dist = []
        inputScaler = preprocessing.StandardScaler().fit(self.training_set)
        z = inputScaler.transform(z)
        for zz in z:
            zClass.append([self.pfn(zz)])
        result = array(zClass)
        #within_dist = array(within_dist)
        #place(result, ~ logical_or(result > self.limit, within_dist),[0.0])
        place(result, result > 0.0,[1.0])
        place(result, result < 0.0,[0.0])        
        #place(result, ~within_dist,[0.0])          
        #result = 1/(1+exp(-result*60+30)) ## sigmoid function
        #result = result * result * result * result * result * result 
        return result
            
    def tiny_bit(self):
        try:
            self.bit
        except:
            self.normalize(self.training_set[0])
            step_array = []
            for i,dd in enumerate(self.fitness.designSpace):
                step_array.append((dd["max"] - dd["min"]) / dd["step"])
            step_array = array(step_array)
            steps = 1./step_array
            origin = zeros(self.normal_array.shape)
            self.bit = self.get_distance(origin, steps)
        return self.bit / 2
        
    def normalize(self, z):
        try:
            self.normal_array
        except:
            normalizing_array = []
            for i,dd in enumerate(self.fitness.designSpace):
                normalizing_array.append(dd["max"] - dd["min"])
            self.normal_array = array(normalizing_array)
        return z / self.normal_array
            
    def get_parameter_string(self):
        return "RVM"
            
    ## TODO - come up with a smart way of storing these...
    def get_state_dictionary(self):
        try:
            self.bests
        except:
            self.bests = None
        try:
            self.normalizer
        except:
            self.normalizer = None
        try:
            self.best_distance
        except:
            self.best_distance = None
        try:
            self.normalized_best
        except:
            self.normalized_best = None            
            
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels, 
                'bests': self.bests,
                'best_gamma' : self.best_gamma,
                'normalizer' : self.normalizer,
                'normalized_best' : self.normalized_best,
                'best_distance' : self.best_distance}

        return dict
        
    ###
    def set_state_dictionary(self, dict):
        self.bests = dict['bests']
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']
        self.best_gamma = dict['best_gamma']
        self.normalizer = dict['normalizer']
        self.normalized_best = dict['normalized_best']
        self.best_distance = dict['best_distance']
        if self.bests:
            self.train(self.bests)
        else:
            self.train_no_cross()
        
    def get_around_best(self, bests, radius = 1.):
        designSpace = self.fitness.designSpace  
        D = len(designSpace)
        npts = 0.01
        n_bins = npts*ones(D)
        ## define the space
        local_perturbation = []
        
        try:
            chosen = []
            for index in bests:
                point = self.training_set[int(index[0])]
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
                local_perturbation.extend([zz for zz in z if not (self.contains_training_instance(zz))])
                rand_indicies = []
                #pdb.set_trace()
                length = len(local_perturbation)
                while len(rand_indicies) < len(self.training_set)/2: # just a way of weighting best
                    rand_index = randint(0,length) 
                    rand_indicies.append(rand_index)
                chosen.extend([local_perturbation[rand_index] for rand_index in rand_indicies])
            return []
        except:
            return []
    
    def get_distance_to_closest_wrong(self, best, normalized_set):
        invalid_samples = []
        for i in xrange(len(normalized_set)):
            if self.training_labels[i] != 1.:
                invalid_samples.append(normalized_set[i])
        min_distance = 1000000.0
        closest = None
        for i in xrange(len(invalid_samples)):
            dist = self.get_distance(best, invalid_samples[i])
            if min_distance > dist :
                min_distance = dist
                closest = invalid_samples[i]
        return min_distance, closest
                
    def get_distance(self, a, b):
        a = array(a)
        b = array(b)
        diff = a - b
        squared_diffs = square(diff)
        sum_squared_diffs = sum(squared_diffs)
        dist = sqrt(sum_squared_diffs)
        return dist
            
class ResourceAwareClassifier(Classifier):

    def __init__(self, fitness, conf, controller):
        super(ResourceAwareClassifier, self).__init__(fitness, conf)
                
        self.posresourceregressors = []
        self.resourceregressors = []
        self.binaryresourceclassifiers = []
        
        for key, rc in self.fitness.resource_class.iteritems():
            if rc["type"] == "logreg":
                regre = GaussianProcessRegressor4(controller, conf, fitness)
                regre.forceLog = True
                self.posresourceregressors.append((regre,key))
            if rc["type"] == "reg":
                self.resourceregressors.append((GaussianProcessRegressor4(controller, conf, fitness),key))
            if rc["type"] == "bin":
                self.binaryresourceclassifiers.append((SupportVectorMachineClassifier(fitness, conf),key))
        
    def add_training_instance(self, part, label):      
        super(ResourceAwareClassifier, self).add_training_instance(part, label)
        re_output = self.fitness.fitnessFunc(part, {}, return_resource = True)
        for reg, key in self.posresourceregressors:
            if not (re_output[key] is None):
                reg.add_training_instance(part, re_output[key])
        for reg, key in self.resourceregressors:
            if not (re_output[key] is None):
                reg.add_training_instance(part, re_output[key])
        for clas, key in self.binaryresourceclassifiers:
            #pdb.set_trace()
            if label == 1:
                clas.add_training_instance(part, 1)
            elif re_output[key]:
                clas.add_training_instance(part, 0)
                
                
                
    def train(self, local_structure=False, bests=None):
        try:
            regressors_trained = True
            for reg, key in self.posresourceregressors:
                regressors_trained = regressors_trained and reg.train()
            logging.info("Resource regression training done" + str(regressors_trained))
            for reg, key in self.resourceregressors:
                regressors_trained = regressors_trained and reg.train()
            logging.info("Other data regression classification training done" + str(regressors_trained))
            for clas, key in self.binaryresourceclassifiers:
                regressors_trained = regressors_trained and clas.train()
            logging.info("Binary classifier training done " + str(regressors_trained))
            return regressors_trained
        except Exception, e:
            logging.error('Classifier training failed.. ' + str(e))
            return False

    def predict(self, z):
        output = 1.0
        
        for reg, key in self.posresourceregressors:
            mu, s2, bla, bla2 = reg.predict(z, False, False)
            low = norm.cdf(self.fitness.resource_class[key]["lower_limit"], loc=mu, scale=s2)
            high = norm.cdf(self.fitness.resource_class[key]["higher_limit"], loc=mu, scale=s2)
            output = output * (high - low)
            pdb.set_trace()
        for reg, key in self.resourceregressors:
            mu, s2, bla, bla2 = reg.predict(z, False, False)
            low = norm.cdf(self.fitness.resource_class[key]["lower_limit"], loc=mu, scale=s2)
            high = norm.cdf(self.fitness.resource_class[key]["higher_limit"], loc=mu, scale=s2)
            output = output * (high - low)
        for clas, key in self.binaryresourceclassifiers: 
            output = output * clas.predict(z)
        return output
        #try:
        #    
        #except Exception, e:
        #    logging.error('Prediction failed... ' + str(e))
        #    return None
            
    def get_parameter_string(self): # not used really
        return "N\A"
            
    ## TODO - come up with a smart way of storing these...
    def get_state_dictionary(self):
        dict = {'posresourceregressors' : self.posresourceregressors,
                'resourceregressors': self.resourceregressors,
                'binaryresourceclassifiers': self.binaryresourceclassifiers,
                'training_set' : self.training_set,
                'training_labels': self.training_labels}
        return dict
        
    ###
    def set_state_dictionary(self, dict):
        self.posresourceregressors = dict['posresourceregressors']
        self.resourceregressors = dict['resourceregressors']
        self.binaryresourceclassifiers = dict['binaryresourceclassifiers']
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']
        