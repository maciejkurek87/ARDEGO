import logging
from multiprocessing import Process, Pipe
import traceback
import os
from random import randrange
import shutil
import time

from numpy import unique, asarray, bincount, array, append, sqrt, log, sort, exp, isinf, all, sum, place, delete, maximum, minimum, linalg, eye, dot
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.cross_validation import ShuffleSplit
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from numpy.random import uniform, shuffle, permutation
#from rpy2.robjects.packages import importr
#from rpy2.robjects import r
#from rpy2.robjects.vectors import FloatVector
import gc 

from utils import numpy_array_index
from copy import deepcopy
import pdb

## pyGPR
#from UTIL.utils import hyperParameters
#from Tools.min_wrapper import min_wrapper
#from GPR.gp import gp
###old pyXGPR
##from GPR import gpr

import pyGP_OO
from pyGP_OO.Core import *
from pyGP_OO.Valid import valid

#TODO - abstract class
class Regressor(object):

    def __init__(self, controller, conf, fitness):
        self.training_set = None
        self.training_fitness = None
        self.regr = None
        self.controller = controller
        self.fitness = fitness
        self.conf = conf
        self.y_best = None

    def train(self):
        return True

    def predict(self, z):
        output = []
        output2 = []
        for input_vector in z:
            output.append([0.0])
            output2.append([100.0])
        return output, output2

    def shuffle(self):
        p = permutation(len(self.training_fitness))
        self.training_fitness=self.training_fitness[p]
        self.training_set=self.training_set[p]

    def add_training_instance(self, part, fitness):
        try:
            part = delete(part, self.fitness.exclude_from_regression, 0) 
        except:
            pass
        if self.training_set is None:
            self.training_set = array([part])
            self.training_fitness = array([fitness])
        else:
            contains = self.contains_training_instance(self.training_set)
            if contains:
                logging.debug('A particle duplicate is being added.. check your code!!')
            else:
                self.training_set = append(self.training_set, [part], axis=0)
                self.training_fitness = append(self.training_fitness, [fitness], axis=0)

    def contains_training_instance(self, part):
        try:
            part = delete(part, self.fitness.exclude_from_regression, 0) 
        except:
            pass
        contains, index = numpy_array_index(self.training_set, part)
        return contains
            
    def get_training_instance(self, part):
        try:
            part = delete(part, self.fitness.exclude_from_regression, 0) 
        except:
            pass
        contains, index = numpy_array_index(self.training_set, part)
        if self.training_set is None:
            logging.error('cannot call get_training_instance if training_set is empty')
            return False
        elif contains:
            return self.training_fitness[index]
        else :
            logging.error('cannot call get_training_instance if training_set does not contain the particle')
            return False
                                           
    def get_training_set(self):
        return self.training_set

    
    def e_impr(self, s2, y_mean):
        y_best = self.get_y_best()
        if self.conf.goal == "min":
            y_diff_vector = y_best-y_mean
        elif self.conf.goal == "max":
            y_diff_vector = y_mean - y_best
        y_diff_vector_over_s2 = y_diff_vector / s2
        result = (y_diff_vector) * norm.cdf(y_diff_vector_over_s2) + s2 * norm.pdf(y_diff_vector_over_s2)
        place(result,isinf(result),[0.0]) 
        return result
        
    ## TODO - do it so it would be done in a numpy-ish way
    def e_multi(self, miu_s2, miu_mean, lambda_s2, lambda_mean, n_sims = 100):
        y_best = self.get_y_best()
        sum_ei = 0.0
        if self.conf.goal == "min":
            for i in range(0,n_sims):
                mius = norm.rvs(loc=miu_mean, scale=miu_s2)
                mins = minimum(y_best, min(mius))
                lambdas = norm.rvs(loc=lambda_mean, scale=lambda_s2)
                e_i = maximum(0.0, mins - min(lambdas))
                sum_ei = e_i + sum_ei
        else:
            for i in range(0,n_sims):
                mius = norm.rvs(loc=miu_mean, scale=miu_s2)
                maxs = maximum(y_best, max(mius))
                lambdas = norm.rvs(loc=lambda_mean, scale=lambda_s2)
                e_i = maximum(0.0, max(lambdas) - maxs)
                sum_ei = e_i + sum_ei
        return sum_ei/n_sims
        
    def get_y_best(self, raw=False):
        if raw:
            y_best = self.y_best
        else:
            if self.transLog:
                y_best = self.output_scaler.transform(log(self.y_best - self.shift_by()))
            else:
                y_best = self.output_scaler.transform(self.y_best)  
        return y_best
        
    def e_multi_fpga(self, miu_s2, miu_mean, lambda_s2, lambda_mean, y_best, n_sims = 100):
        return None
        
    def prob(self, s2, y_mean, y_best):
        if self.conf.goal == "min":
            return norm.cdf(y_best, y_mean, s2)
        elif self.conf.goal == "max":
            return 1.0-norm.cdf(y_best, y_mean, s2)
           
    def sample(self, s2, y_mean, y_best):
        return norm.rvs(y_mean, s2)
            
    def get_training_fitness(self):
        return self.training_fitness
    # def __getstate__(self):
        # Don't pickle controller
        # d = dict(self.__dict__)
        # del d['controller']
        # return d
        
    def get_nlml(self):
        return None
        
    def training_set_empty(self):
        return (self.training_set is None)
        
    def get_parameter_string(self):
        return "Not Implemented"
        
    def set_y_best(self, y_best):
        self.y_best = y_best
        
    def error_tolerance(self):
        return 0.01
        
    ###############
    ### GET/SET ###
    ###############
        
    def get_state_dictionary(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def set_state_dictionary(self, dict):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
class GaussianProcessRegressor(Regressor):

    def __init__(self, controller, conf):
        super(GaussianProcessRegressor, self).__init__(controller, conf, fitness)
        self.input_scaler = None
        self.output_scaler = None
        self.conf = conf
        self.gp = None

    def regressor_countructor(self):
        conf = self.conf
        dimensions = len(self.training_set[0])
        if conf.nugget == 0:
            gp = GaussianProcess(regr=conf.regr, corr=conf.corr2,
                                 theta0=array([conf.theta0] * dimensions),
                                 thetaL=array([conf.thetaL] * dimensions),
                                 thetaU=array([conf.thetaU] * dimensions),
                                 random_start=conf.random_start)
        else:
            gp = GaussianProcess(regr=conf.regr, corr=conf.corr2,
                                 theta0=array([conf.theta0] * dimensions),
                                 thetaL=array([conf.thetaL] * dimensions),
                                 thetaU=array([conf.thetaU] * dimensions),
                                 random_start=conf.random_start, nugget=conf.nugget)
        return gp

    def train(self):
        conf = self.conf
        if len(self.training_set) == 0:
            return True
        try:
            # Scale inputs and particles?
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            scaled_training_set = self.input_scaler.transform(
                self.training_set)

            # Scale training data
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness)
            adjusted_training_fitness = self.output_scaler.transform(
                self.training_fitness)
            gp = self.regressor_countructor()
            # Start a new process to fit the data to the gp, because gp.fit is
            # not thread-safe
            parent_end, child_end = Pipe()

            self.controller.acquire_training_sema()
            logging.info('Training regressor')
            p = Process(target=self.fit_data, args=(gp, scaled_training_set,
                                                    adjusted_training_fitness,
                                                    child_end))
            p.start()
            self.regr = parent_end.recv()
            if self.regr is None:
                raise Exception("Something went wrong with the regressor")
            else:
                logging.info('Regressor training successful')
                self.controller.release_training_sema()
                self.gp = gp
                return True
        except Exception, e:
            logging.info('Regressor training failed.. retraining.. ' + str(e))
            return False

    def predict(self, z):
        z = delete(z,self.fitness.exclude_from_regression,1)
        try:
            #logging.info("z " + str(z))
            #logging.info("z.shape " + str(z.shape))
            # Scale inputs. it allows us to realod the regressor not retraining the model
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness) 
                
            #logging.debug(z)
            MU, S2 = self.regr.predict(self.input_scaler.transform(array(z)), eval_MSE=True)
            #logging.debug(MU)
            S2 = sqrt(S2.reshape(-1, 1))
            MU = MU.reshape(-1, 1)
            
            y_best = self.output_scaler.transform(self.get_y_best())
            EI = self.e_impr(S2, MU, y_best)
            P = self.output_scaler.inverse_transform(self.sample(S2, MU, y_best))
            
            MU = self.output_scaler.inverse_transform(MU)
            
            return MU, S2, EI, P
        except Exception, e:
            logging.error('Prediction failed.... ' + str(e))
            return None, None, None, None

    def fit_data(self, gp, scaled_training_set, adjusted_training_fitness,
                 child_end):
        try:
            gp.fit(scaled_training_set, adjusted_training_fitness)
        except:
            gp = None
        child_end.send(gp)    
        
    def get_state_dictionary(self):
        try:
            theta_ = self.gp.theta_
        except:
            theta_ = None
            
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'gp_theta': theta_}
        return dict
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.gp = self.regressor_countructor()
        try:
            self.gp.theta_ = dict['gp_theta']
        except:
            pass
        
## Different implementation of GPR regression, based on pyGPR00
class GaussianProcessRegressor4(Regressor):
        
    def __init__(self, controller, conf, fitness):
        super(GaussianProcessRegressor4, self).__init__(controller, conf, fitness)
        self.input_scaler = None
        self.output_scaler = None
        self.l = None
        self.m = None
        self.k = None
        self.i = None
        self.transLog = True
        self.forceLog = False
        
    def error_tolerance(self):
        return exp(self.output_scaler.inverse_transform(self.k.gethyp())[-1] + self.shift_by())
        
    def get_optimizer(self, max_trails=1):
        
        d = len(self.training_set[0])
        l = lik.likGauss([log(0.1)])
        
        #m = mean.meanLinear([1]*d)
        m = mean.meanZero()
        #meanRange = [(-5,5)]*d
        meanRange = []
        conf = self.conf
        if conf.corr == "isotropic":
            k = cov.covSEiso([-1,1])
            covRange = [(conf.thetaL,conf.thetaU)]*2
        elif conf.corr == "anisotropic":
            k = cov.covSEard([1]*(d+1))
            covRange = [(conf.thetaL,conf.thetaU)]*(d+1)       
        elif conf.corr == "linear":
            k = cov.covLINard([1]*(d))
            covRange = [(conf.thetaL,conf.thetaU)]*(d)
        elif conf.corr == "matern3":
            k = cov.covMatern([1,1,3])
            covRange = [(conf.thetaL,conf.thetaU)]*2 + [(3,3)]
        elif conf.corr == "matern5":
            k = cov.covMatern([1,1,5])
            covRange = [(conf.thetaL,conf.thetaU)]*2 + [(5,5)]      
        k = k + cov.covNoise([-1])
        covRange = covRange + [(-2,1)]
        
        conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
        conf.likRange = [(0,0.2)]
        #conf.min_threshold = 20
        conf.max_trails = max_trails
        conf.covRange = covRange 
        conf.meanRange = meanRange 
        i = inf.infExact()
        
        return k,m,l,i,opt.Minimize(conf)
        
    def train(self):
        #return self.train_cross()
        if self.training_set is None:
            logging.info("Training set is empty, cant train...")
            return True
        else:
            return self.train_nlml()
            
    def train_nlml(self):
        input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        scaled_training_set = input_scaler.transform(self.training_set)
        self.input_scaler = input_scaler
        
        ### log
        log_output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(self.training_fitness - self.shift_by()))
        log_adjusted_training_fitness = log_output_scaler.transform(log(self.training_fitness - self.shift_by()))
        
        conf = self.conf
        log_k,log_m,log_l,log_i,log_o = self.get_optimizer(max_trails=conf.random_start)    

        log_error = False
        log_nlm = 10000000000000.0
        try:
            log_nlm = gp.train(log_i,log_m,log_k,log_l,scaled_training_set,log_adjusted_training_fitness,log_o)
            logging.info('Regressor training (log) successful: ' + str(log_nlm))
        except Exception,e:
            log_error = True
            logging.debug("Regressor training (log) Failed: " + str(e))
        
        ### standard
        output_scaler = preprocessing.StandardScaler(with_std=False).fit(self.training_fitness)
        adjusted_training_fitness = output_scaler.transform(self.training_fitness)
        conf = self.conf
        k,m,l,i,o = self.get_optimizer(max_trails=conf.random_start)    

        s_error = False
        nlm = 10000000000000.0 
        try:
            nlm = gp.train(i,m,k,l,scaled_training_set,adjusted_training_fitness,o)
            logging.info('Regressor training successful: ' + str(nlm))
        except Exception,e:
            s_error = True
            logging.debug("Regressor training Failed: " + str(e))
            
            
        ## results
        if (log_error or s_error):
            logging.info('Training Regressors with/without log failed')
            return False
            
        if (log_nlm < nlm or self.forceLog): 
            logging.info('Using log transform')
            self.k=log_k
            self.m=log_m
            self.l=log_l
            self.i=log_i
            self.transLog = True
            self.output_scaler = log_output_scaler
            adjusted_training_fitness = log_adjusted_training_fitness
        else:
            self.k=k
            self.m=m
            self.l=l
            self.i=i
            self.transLog = False
            self.output_scaler = output_scaler
        
        self.sW = gp.analyze(self.i,self.m,self.k,self.l,scaled_training_set,adjusted_training_fitness, der=False)[1].sW[0]
        M = self.k.proceed(scaled_training_set) + eye(len(scaled_training_set))/(self.sW*self.sW)
        try:
            self.LL = linalg.cholesky(M) ### used by par ego
        except:
            logging.info("Trying to construct cholesky with added nugget")
            try:
                ### sued by par ego
                #pdb.set_trace()
                w, v = linalg.eig(M)
                self.LL = linalg.cholesky(M + eye(len(scaked_training_set)) * -2. * w.min()) 
            except:
                return False
        return True
    
    def train_cross(self):
        press_best = None
        for x_train, x_test, y_train, y_test in valid.k_fold_validation(self.training_set,self.training_fitness,self.conf.Kfolds):
             # Scale inputs 
            input_scaler = preprocessing.StandardScaler().fit(x_train)
            scaled_training_set = input_scaler.transform(x_train)
            scaled_testing_set = input_scaler.transform(x_test)
            if self.transLog:
                output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(y_train - self.shift_by()))
                adjusted_training_fitness = output_scaler.transform(log(y_train - self.shift_by()))
                adjusted_testing_fitness = output_scaler.transform(log(y_test - self.shift_by()))
            else:
                output_scaler = preprocessing.StandardScaler(with_std=False).fit(y_train)
                adjusted_training_fitness = output_scaler.transform(y_train)
                adjusted_testing_fitness = output_scaler.transform(y_test)
            k,m,l,i,o = self.get_optimizer(max_trails=1)
            gp.train(i,m,k,l,scaled_training_set,adjusted_training_fitness,o)
            vargout = gp.predict(i,m,k,l,scaled_training_set,adjusted_training_fitness,scaled_testing_set)
            predicted_fitness = vargout[2]
            press = self.calc_press(predicted_fitness, adjusted_testing_fitness)
            if (((not press_best) or (press < press_best))):
                self.m = m
                self.l = l
                self.k = k
                self.o = o
                self.i = i  
                self.output_scaler = output_scaler
                self.input_scaler = input_scaler
        if  press_best:    
            logging.info('Regressor training successful')
            return True        
        else:
            logging.debug("Regressor training Failed")
            return False
            
    def calc_press(self, y, yhat):
        return sum((y - yhat)*(y - yhat))
            
    def calc__press(self, y, yhat):
        return sum((y - yhat)*(y - yhat))
            
    def get_parameter_string(self):
        try:
            return str(self.nlml) + "_".join([str(round(i,3)) for i in self.hyp.cov])
        except Exception, e:
            try:
                return str(self.press) + "_".join([str(round(i,3)) for i in self.hyp.cov])
            except Exception, e:
                logging.debug(str(e))
                return "Not Trained"
            
    def shift_by(self): ## we need to due this due to log transformation of the training data
        nugget = 0.000000001
        if min(self.training_fitness) <= 0.0:
            return min(self.training_fitness) - nugget
        else:
            return 0.0
        
    def predict(self, z, with_EI=True, raw=False, L=False): ## TODO - broken for one element z
        single_input = len(z)==1
        if single_input:
            try:
                z = [z[0],z[0]*1.1]
            except:
                z = [z[0],z[0]]
        try:
            trans_z = delete(z,self.fitness.exclude_from_regression,1)
        except:
            trans_z = array(z)
        if self.k is None:
            logging.error('Train GP before using it!!')
            return None, None, None, None
        if self.transLog:
            adjusted_training_fitness = self.output_scaler.transform(log(self.training_fitness - self.shift_by()))
        else:
            adjusted_training_fitness = self.output_scaler.transform(self.training_fitness)

        scaled_training_set = self.input_scaler.transform(self.training_set)
        ## do predictions
        try:            
            trans_zz = self.input_scaler.transform(trans_z)
            vargout = gp.predict(self.i,self.m,self.k,self.l,scaled_training_set ,adjusted_training_fitness, trans_zz)
        except Exception,e:
            logging.error(str(e))
            return None, None, None, None
        
        S2 = vargout[1]
        place(S2,S2 < 0.0,0.0) ##get rid of negative variance... refer to some papers (there is a lot of it out there)
        place(S2,S2 < 0.0,0.0)
        
        if raw: 
            MU = vargout[0]
            if single_input:
                return array([MU[0]]), array([S2[0]]), None, None
            else:
                
                if (len(self.LL) != len(scaled_training_set)): ## this happens when regressor was not retrained but someone needs decomposed covariance matrix
                    logging.info("need to calculate LL in predict")
                    self.sW = gp.analyze(self.i,self.m,self.k,self.l,scaled_training_set,adjusted_training_fitness, der=False)[1].sW[0]
                    M = self.k.proceed(scaled_training_set) + eye(len(scaled_training_set))/(self.sW*self.sW)
                    self.LL = linalg.cholesky(M) 
                
                ## calculate the covariance matrix and its L decomposition betweeen the samples
                kk = self.k.proceed(scaled_training_set,trans_zz)
                v =  linalg.solve(self.LL, kk) 
                first = self.k.proceed(trans_zz) - v.T.dot(v)
                #pdb.set_trace()
                try:
                    L = linalg.cholesky(maximum(first,0.0))
                except:
                    logging.info("Couldnt construct cholesky")
                    try:                        
                        K = linalg.inv(self.k.proceed(scaled_training_set))
                        first = self.k.proceed(trans_zz) - kk.T.dot(K).dot(kk)
                        L = linalg.cholesky(first)
                    except:
                        logging.info("Couldnt construct inverse")
                return MU, S2, None, L
        else:
            if self.transLog:
                MU = exp(self.output_scaler.inverse_transform(vargout[0]) + self.shift_by()) 
                S2 = exp(self.output_scaler.inverse_transform(S2) + self.shift_by()) 
            else:
                MU = self.output_scaler.inverse_transform(vargout[0])
                S2 = self.output_scaler.inverse_transform(S2)
            if with_EI:
                EI = self.e_impr(vargout[1], vargout[0])
                place(EI,EI < 0.000001,0.0) ## we dont want those crappy little numbers...
            else:
                EI = []
            
            if single_input:
                if with_EI:
                    return array([MU[0]]), array([S2[0]]), array([EI[0]]), None
                else:
                    return array([MU[0]]), array([S2[0]]), None, None
            else:
                return MU, S2, EI, None
        #except Exception, e:
        #    logging.error('Prediction failed... ' + str(e))
        #    return None, None
        
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'input_scaler': self.input_scaler,
                'output_scaler': self.output_scaler,
                'y_best': self.y_best,
                'm': self.m,
                'l': self.l,
                'k': self.k,
                'i': self.i}
        return deepcopy(dict)
    
    def set_state_dictionary(self, dict):
        dict = deepcopy(dict)
        self.training_set = dict['training_set']
        self.input_scaler = dict['input_scaler']
        self.training_fitness = dict['training_fitness']
        self.output_scaler = dict['output_scaler']
        self.y_best = dict['y_best']
        self.m = dict['m']
        self.l = dict['l']
        self.k = dict['k']
        self.i = dict['i']

     
