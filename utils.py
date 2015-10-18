from imp import load_source
import logging
import sys
from numpy import array_equal

#LOG_FORMAT = '%(asctime)s|[%(process)d_%(thread)d] - [%(module)s][%(funcName)s][%(lineno)d] %(levelname)s: %(message)s'
LOG_FORMAT = '%(asctime)s|[%(process)d_%(thread)d] - [%(module)s][%(lineno)d] %(levelname)s: %(message)s'
LOG_LEVEL = logging.DEBUG

def load_script(filename, script_type):
    """
    Loads a fitness or configuration script. script_type is either
    'fitness' or 'configuration'.
    """
    try:
        return load_source(script_type, filename)
    except Exception,e:
        logging.error(str(script_type.capitalize()) + ' file (' + str(filename) + ') could not be loaded ' + str(e))
        return None

def numpy_array_index(multi_array, array):
    #TODO - check if multi_array is non empty and if they match size.. throw appropariate warnings
    if not multi_array is None:
        try:
            for i,trainp in enumerate(multi_array):
                if array_equal(trainp,array):
                    return True, i
        except:
            pass ## should probabnly give debug messages
    return False, 0

##returns class constructor     

def get_trial_dict():
    from model.trials.trial import PSOTrial, MOPSOTrial, MonteCarloTrial, P_ARDEGO_Trial, Gradient_Trial
    return {"PSOTrial" : PSOTrial, 
            "MOPSOTrial" : MOPSOTrial,
            "MonteCarloTrial" : MonteCarloTrial,
            "P_ARDEGO_Trial" : P_ARDEGO_Trial,
            "Gradient_Trial" : Gradient_Trial,
            "Blank" : None} 

def get_trial_constructor(str_name):
    return get_trial_dict()[str_name]

def get_possible_trial_type():
    return get_trial_dict().keys()
    
def get_trial_type_visualizer(trial_name):
    from views.visualizers.plot import MLOImageViewer, MLOTimeAware_ImageViewer, MonteCarlo_ImageViewer, P_ARDEGO_Trial_ImageViewer
    return {"PSOTrial" : {"MLOImageViewer" : MLOImageViewer, "default" : MLOImageViewer}, 
            "MonteCarloTrial" : {"MonteCarlo_ImageViewer" : MonteCarlo_ImageViewer, "default" : MonteCarlo_ImageViewer},
            "P_ARDEGO_Trial" : {"P_ARDEGO_Trial_ImageViewer" : P_ARDEGO_Trial_ImageViewer, "default" : MonteCarlo_ImageViewer}, ## we should finish the new visualizer
            "Gradient_Trial" : {"P_ARDEGO_Trial_ImageViewer" : P_ARDEGO_Trial_ImageViewer, "default" : MonteCarlo_ImageViewer}, ## we should finish the new visualizer
            "Blank" : {"Blank" : None, "default" : None}}[trial_name]

def get_run_type_visualizer(trial_name):
    from views.visualizers.plot import MLORunReportViewer,MLORegressionReportViewer
    return {"PSOTrial" : {"MLOReportViewer" : MLORunReportViewer,"regressions": MLORegressionReportViewer, "default" : MLORunReportViewer}, 
            "MonteCarloTrial" : {"MLOReportViewer" : MLORunReportViewer,"regressions": MLORegressionReportViewer, "default" : MLORunReportViewer} ,
            "P_ARDEGO_Trial" : {"MLOReportViewer" : MLORunReportViewer,"regressions": MLORegressionReportViewer, "default" : MLORunReportViewer} ,
            "Gradient_Trial" : {"MLOReportViewer" : MLORunReportViewer,"regressions": MLORegressionReportViewer, "default" : MLORunReportViewer} ,
            "Blank" : {"Blank" : None, "default" : None}}[trial_name] 
            
            

    


    
     
