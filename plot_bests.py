from matplotlib.pyplot import *
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
from scipy.stats import norm
import math 
import pdb
import os
from matplotlib.font_manager import FontProperties    
from sklearn import preprocessing
import csv
from plotSpecification import * 
import matplotlib as mpl
from scipy import stats
save = True
fontsize = 26
smallfontsize = 22
annotatefontsize = 22

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
debug = False

def splitter(folder):
    tags_list = folder.split("_")
    tuples = [tuple(tags_list[i:i + 2]) for i in range(0, len(tags_list), 2)] #split tags/data into pairs
    folder_tags = dict(tuples)
    return folder_tags

def aggregation_engine(results_paths, filterDict, off_keys=[], add_list=[]):
    
    filtered_folders = []
    legend = []
    for j, result_path in enumerate(results_paths):
        folders = os.listdir(result_path)
        for folder in sorted(folders):
            #print "folder: " + str(folder)
            folder_tags = splitter(folder)
            # filter
            is_in = True
            for k,v in filterDict.iteritems():
                try:
                    is_in = is_in and (folder_tags[k] in v)
                except:
                    pass
            if not is_in:
                continue
            filtered_folders.append(result_path + "/" + folder + "/bests")
            print filtered_folders
            if add_list:
                l, ylabel, unit, set_y_log, maximization, best_val, title = legendFunction(folder_tags, off_keys, add_list[j])
            else:
                l, ylabel, unit, set_y_log, maximization, best_val, title = legendFunction(folder_tags, off_keys)
            legend.append(l)
            print title
    return (legend, filtered_folders, ylabel, unit, set_y_log, maximization, best_val, title)
     
     
#############
#############
calc_stats = False
#############
#############
set_x_log = False
add_list = []
plot_name = "kurwa"
ncol = 1
off_keys = ["fF"]
#dir = "/data/Dropbox/pso_results/"
dir = "/home/maciej/pso_results/"
table_data = {}
table_data2 = {}
table_data2_time = 150
table_data2_time2 = 300

###############
#### QUAD #####
############
error = "0.1"
###
error2 = error[2:]
des = "quad"
ncol = 1
if True:
    #title = "Energy Efficiency"
    on = "False"
    tit = "en"
else:
    #title = "Throughput optimization"
    on = "True"
    tit = "th"
filterDict = {"fF" : "ansonE" + error + "Th" + on}
# title = title + ", Quadrature Apllication Optimization, Error " + error

# results_paths = [dir + "quad/standard", dir + "quad/hc"]
# ncol = 1
# plot_name = "kernels_stdv_no_" + error2 + "_" + des + "_" + tit
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "corr"
# tagx = "stddv"

# results_paths = [dir + "quad/standard", dir + "quad/m_s"]
# plot_name = "quad_kernels_ms_no_" + error2 + "_" + des + "_" + tit
# filterDict["corr"] = ["anisotropic"]
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "sampleon"
# tagx = "stddv"


# results_paths = [dir + "quad/standard", dir + "quad/class1/", dir + "quad/class2/"]
# plot_name = "kernels_class_stdv_no_" + error2 + "_" + des + "_" + tit
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# filterDict["corr"] = ["anisotropic"]
# tagy = "class"
# tagx = "stddv"

# results_paths = [dir + "quad/standard", dir + "quad/ms/100",   dir + "quad/ms/50"]
# plot_name = "kernels_Ms_stdv_no_" + error2 + "_" + des + "_" + tit
# add_list = ["M=100", "M=50", "M=10"]
# filterDict["corr"] = ["anisotropic"]
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "m"
# tagx = "stddv"

# results_paths = [dir + "quad/ms/50", dir + "quad/ms/100", dir + "quad/m_s",dir + "quad/class1", dir + "quad/class2", dir + "quad/standard", dir + "quad/hc"]
# ncol = 1
# plot_name = "kernels_stdv_no_" + error2 + "_" + des + "_" + tit
# filterDict["stddv"] = ["0.05", "0.01", "0.01"]
#############
#### PQ #####
#############
#best_val = 151.0    #% 95%
#best_val = 155.0  #% 97.5
#best_val = 159.24  %100

# results_paths = [dir + "pq/standard", dir + "pq/hc"] #, dir + "pq/hc"
# filterDict = {}
# plot_name = "pq_standard"
# ncol = 1
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "corr"
# tagx = "stddv"

# results_paths = [dir + "pq/ms", dir + "pq/standard"]#, dir + "pq/standard", dir + "pq/hc"]
# filterDict = {}
# title = "PQ, Anisotropic kernel function"
# plot_name = "pq_ms"
# ncol = 2
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# filterDict["corr"] = ["anisotropic"]
# tagy = "sampleon"
# tagx = "stddv"

# results_paths = [dir + "pq/standard", dir + "pq/class1", dir + "pq/class2"]
# filterDict = {}
# title = "PQ, Anisotropic kernel function"
# plot_name = "pq_class"
# ncol = 2
# filterDict["corr"] = ["anisotropic"]
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "class"
# tagx = "stddv"

# results_paths = [dir + "pq/standard", dir + "pq/m_s/50", dir + "pq/m_s/100"]
# filterDict = {}
# title = "PQ, Anisotropic kernel function"
# plot_name = "pq_m_s"
# filterDict["corr"] = ["anisotropic"]
# filterDict["stddv"] = ["0.01", "0.05", "0.1"]
# tagy = "m"
# tagx = "stddv"

# results_paths = [dir + "pq/ms", dir + "pq/standard", dir + "pq/m_s/50", dir + "pq/m_s/100", dir + "pq/class1", dir + "pq/class2", dir + "pq/hc"]
# filterDict = {}
# title = "PQ, Anisotropic kernel function"
# plot_name = "pq_m_s"
#results_paths = ["/data/tmp/mk306/ardego3_quad_software_latin_lambdaLimit_True_50000"]


#############
#### ARDEGO QUAD #####
#############
#dir2 = "/data/data/mk306/"
#dir3 = "/data/Dropbox/hc_rtm/"
dir2 = "/home/maciej/ardego/"

# results_paths = [dir + "quad/hc", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_False", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_False_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_True_False"]
# plot_name = "ardego" + error2 + "_" + des + "_" + tit
# add_list = ["Z","A", "B", "C", "D"]

plot_name = "ardego" + error2 + "_" + des + "_" + tit
#results_paths = [dir2 + "ardego_quad_software_latin_lambdaLimit_False_500000_True_False"]#, dir2 + "ardego2_quad_software_latin_lambdaLimit_False_500_True_True", dir2 + "ardego2_quad_software_latin_lambdaLimit_False_5000_True_True"]
#results_paths = [dir2 + "ardego_quad_software_latin_lambdaLimit_False_5000_True_False",dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_True_False",dir2 + "ardego_quad_software_latin_lambdaLimit_False_500000_True_False"]#, dir + "quad/hc", dir + "quad/m_s"]
#results_paths = [dir2 + "ardego_quad_software_latin_lambdaLimit_False_5_True_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50_True_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_500_True_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_5000_True_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_True_True", dir2 + "ardego_quad_software_latin_lambdaLimit_False_500000_True_True"]#, dir3 + "tT_Hill\ Climbing_fF_rtm", dir + "quad/m_s"]

#results_paths = [dir2 + "hacked/ardego_quad_software_latin_lambdaLimit_False_500_True_True"]
#results_paths = [dir2 + "hacked/ardego_pq_software_latin_lambda_False_500_True_True"]
#results_paths = [dir2 + "hacked/ardegohack_xinyu_rtm_software_latin_lambda_False_5000_True_True"]
#results_paths = [dir2 + "hacked/ardego_xinyu_software_latin_lambda_False_500_True_True"]

#results_paths = [dir2 + "ardego_quad_software_latin_lambdaLimit_False_5000_True_False",dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000_True_False",dir2 + "ardego_quad_software_latin_lambdaLimit_False_500000_True_False"]#, dir + "quad/hc", dir + "quad/m_s"]
#filterDict["corr"] = ["anisotropic"]
#filterDict["stddv"] = ["0.01"]
#filterDict["parall"] = ["6"]
#filterDict["nsims"] = ["500"]
tagx = "nsims"
tagy = "parall"

# results_paths = [dir2 + "ardego_quad_software_latin_lambdaLimit_False_50000",dir2 + "ardego_quad_software_latin_lambdaLimit_False_500000"]
# plot_name = "ardego" + error2 + "_" + des + "_" + tit

#results_paths = [dir2 + "ardego_pq_software_latin_lambda_False_500000", dir2 + "ardego_pq_software_latin_lambda_False_50000", dir2 + "ardego_pq_software_latin_lambda_False_5000", dir2 + "ardego_pq_software_latin_lambda_False_500_True_True"]
#plot_name = "ardego" + error2 + "_" + des + "_" + tit
#filterDict = {}

# results_paths = [dir2 + "ardego_pq_software_latin_lambda_False_500000", dir2 + "ardego_pq_software_latin_lambda_False_50000", dir2 + "ardego_pq_software_latin_lambda_False_5000", dir + "pq/hc", dir + "pq/ms"]
# filterDict = {}
# filterDict["corr"] = ["anisotropic"]
# filterDict["stddv"] = ["0.01"]
# filterDict["sampleon"] = ["m"]
# filterDict["nsims"] = ["50000"]
# plot_name = "pq"
# plot_name = "ardego" + error2 + "_" + des + "_" + tit
# filterDict = {}

# results_paths = [dir2 + "ardego_pq_software_latin_lambda_False_5_True_True",dir2 + "ardego_pq_software_latin_lambda_False_50_True_True",dir2 + "ardego_pq_software_latin_lambda_False_500_True_True",dir2 + "ardego_pq_software_latin_lambda_False_5000_True_False",dir2 + "ardego_pq_software_latin_lambda_False_50000_True_False",dir2 + "ardego_pq_software_latin_lambda_False_500000_True_False"]
# plot_name = "ardego" + error2 + "_" + des + "_" + tit
# filterDict = {}

# results_paths = [dir2 + "ardego_pq_software_latin_lambda_False_5000_True_False",dir2 + "ardego_pq_software_latin_lambda_False_50000_True_False",dir2 + "ardego_pq_software_latin_lambda_False_500000_True_False"]
# plot_name = "ardego" + error2 + "_" + des + "_" + tit
# filterDict = {}

# results_paths = [dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000", dir2 + "hc"]
# plot_name = "ardegortm"
# filterDict = {}
# filterDict["nsims"] = ["500000"]

#results_paths = [dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_5_True_True", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50_True_True",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500_True_True",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_5000", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500000"]
results_paths = [dir2 + "ardego_pq_software_latin_lambda_False_50_True_True",dir2 + "ardego_pq_software_latin_lambda_False_500_True_True",dir2 + "ardego_pq_software_latin_lambda_False_5000", dir2 + "ardego_pq_software_latin_lambda_False_50000", dir2 + "ardego_pq_software_latin_lambda_False_500000", dir + "pq/ms", dir + "pq/hc"]
plot_name = "pq"
#filterDict = {}
filterDict["fF"] = ["pq"]
filterDict["corr"] = ["anisotropic"]
filterDict["stddv"] = ["0.01"]
filterDict["sampleon"] = ["m"]
filterDict["nsims"] = ["50000"]
off_keys.append("nsims")
off_keys.append("corr")
off_keys.append("stddv")
off_keys.append("sampleon")


#results_paths = [dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500000_True_False",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000_True_False",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_5000_True_False"]# ,dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500_True_False",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_5000_True_False", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000_True_False", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500000_True_False"]
#results_paths = [dir2 + "ardego11_quad_software_latin_lambdaLimit_False_5000_True_False"]
#plot_name = "ardego" + error2 + "_" + des + "_" + tit
#filterDict = {}

#results_paths = [dir2 + "ardego_xinyu_pq_software_latin_lambda_False_500000_True_False",dir2 + "ardego_xinyu_pq_software_latin_lambda_False_50000_True_False",dir2 + "ardego_xinyu_pq_software_latin_lambda_False_5000_True_False"]# ,dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500_True_False",dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_5000_True_False", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000_True_False", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_500000_True_False"]
#plot_name = "ardego" + error2 + "_" + des + "_" + tit
#filterDict = {}


# results_paths = [dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000", dir2 + "ardego_xinyu_rtm_software_latin_lambda_False_50000_False"]
# plot_name = "ardego" + error2 + "_" + des + "_" + tit
# filterDict = {}

tagx = "nsims"
tagy = "parall"
set_x_log = True

#########
print "MENDOOO"
legend, folder_paths, ylabel, unit, set_y_log, maximization, best_val, title = aggregation_engine(results_paths, filterDict, off_keys, add_list)
if unit == "sec":
    unit_divisor = 3600
elif unit == "min":
    unit_divisor = 60
## bash command to copy data from a folder
##define folder variable
##counter=0; for f in `ls $folder`; do ((counter=$counter+1));echo $counter ; cp $folder$f/best_dump.csv best_dump_$counter.csv; done

def load_data(folder_path):    
    files = os.listdir(folder_path)
    if debug :
        always_valid = True
    X = []
    Y = []
    n = 0
    over = 0
    under = 0
    total = 0
    total2 = 0
    print "folder_path " + folder_path
    for file in files:
        #pdb.set_trace()
        tags = splitter(folder_path.split("/")[-2])
        spamReader = csv.reader(open(folder_path + "/" + file, 'rb'), delimiter=';', quotechar='"')
        counter = -1
        XX = []
        YY = []
        for row in spamReader:
            XX.append(float(row[0]))
            YY.append(float(row[1]))
            #print row
        if float(row[1]) >= best_val:
            over = over + 1
        if float(row[1]) <= best_val:
            under = under + 1
        total = total + float(row[0])
        total2 = total2 + float(row[1])
        X.append(XX)
        Y.append(YY)
        n = n + 1
        #pdb.set_trace()
    opt_time = str(int(float(total)/(unit_divisor*n)))
    if maximization:
        avg_perf = str(int(100 * float(total2)/(n * best_val)))
    else:
        avg_perf = str(int(100 * (best_val* n)/(float(total2))))
        
    print "over " + str(float(over)/n)
    print "under " + str(float(under)/n)
    print "avg time " + opt_time
    print "avg perf " + avg_perf
    try:
        try:
            table_data[tags[tagx]][tags[tagy]] = [opt_time,avg_perf]
        except:
            table_data[tags[tagx]] = {tags[tagy]:[opt_time,avg_perf]}
    except:
        print "something went wrong... probs hill climbing"
    return X,Y, n
    
def lin_predict(x,y, min_XX, max_XX):
    def f_func(z):
        counter = 0
        if maximization:
            y_max = np.max(y)
        else:
            y_max = np.min(y)
        while counter < len(x) -1: 
            if (z >= x[counter]) & (z < x[counter+1]):
                return (y[counter+1]-y[counter])/(x[counter+1]-x[counter])*(z-x[counter]) + y[counter]
            counter = counter + 1
        return y_max

    z = np.array([np.linspace(min_XX,max_XX,101)]).T
    output = [f_func(zz) for zz in z]
    output[0] = y[0]
    return z,output
    
def prev_predict(x,y, min_XX, max_XX):
    def f_func(z):
        # if z < x[0]: 
            # return 
        counter = 0
        if maximization:
            y_max = np.max(y)
        else:
            y_max = np.min(y)
        #pdb.set_trace()
        while counter < len(x) -1: 
            if (z < x[counter]):
                return y[0]
            if (z >= x[counter]) & (z < x[counter+1]):
                return y[counter]
            counter = counter + 1
        return y_max

    z = np.array([np.linspace(min_XX,max_XX,101)]).T
    output = [f_func(zz) for zz in z]
    #pdb.set_trace()
    output[0] = y[0]
    return z,output
    
def t_test(y_t, id_1, id_2, id_3, true):
    return stats.ttest_ind(np.swapaxes(np.swapaxes(np.array(y_t),0,2),1,2)[id_3][id_1],np.swapaxes(np.swapaxes(np.array(y_t),0,2),1,2)[id_3][id_2])
    
def gp_predict(x,y, min_XX, max_XX):
    d = 1
    thetaL = 1.0
    thetaU = 10.0
    nugget = 3
    
    m = mean.meanZero()
    k = cov.covSEard([1]*(d+1)) + cov.covNoise([-1])
    #k = cov.covMatern([1,1,3]) + cov.covNoise([-1])
    
    l = lik.likGauss([np.log(0.3)])
       
    
    conf = pyGP_OO.Optimization.conf.random_init_conf(m,k,l)
    conf.max_trails = 20
    #conf.min_threshold = 100
    o = opt.Minimize(conf)
    conf.likRange = [(0,0.2)]
    #conf.min_threshold = 20
    conf.max_trails = 10
    conf.covRange = [(thetaL,thetaU)]*(d+1) + [(-2,1)]
    #conf.covRange = [(thetaL,thetaU)]*2 + [(3,3)] + [(-2,1)]
    conf.meanRange = [] 
    i = inf.infExact()
    
    output_scaler = preprocessing.StandardScaler(with_std=False).fit(np.log(y))
    adjusted_training_fitness = output_scaler.transform(np.log(y))
    
    input_scaler = preprocessing.StandardScaler().fit(x)
    scaled_training_set = input_scaler.transform(x)
            
    gp.train(i,m,k,l,scaled_training_set,adjusted_training_fitness,o)
    
    z = np.array([np.linspace(min_XX,max_XX,101)]).T
    out = gp.predict(i,m,k,l,scaled_training_set,adjusted_training_fitness, input_scaler.transform(z))
    y_output = np.exp(output_scaler.inverse_transform(out[2]))
    if maximization:
        np.place(y_output, z > np.max(x),np.max(y))
    else:
        np.place(y_output, z > np.max(x),np.min(y))
    return z,y_output
    
def change_tick_font(ax, label2=False):
	if label2:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label2.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label2.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)
	else:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label1.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label1.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)

def change_tick_font(ax, label2=False):
	if label2:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label2.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label2.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)
	else:
		for tick in ax.xaxis.get_major_ticks():
		        tick.label1.set_fontsize(smallfontsize) 
		for tick in ax.yaxis.get_major_ticks():
               	 	tick.label1.set_fontsize(smallfontsize) 
		ax.yaxis.get_major_ticks()[0].set_visible(False)
		#ax.yaxis.get_major_ticks()[-1].set_visible(False)

from matplotlib.lines import Line2D
markers = []
for marker in Line2D.markers:
    try:
        if len(marker) == 1 and marker != ' ':
            markers.append(marker)
    except TypeError:
        pass

        
if 1: ## throughput
    fig = figure(figsize=(12,5))
    ax = fig.add_subplot(111)#
    symbols =['--','-.','-D','-*','->','-d','-|','-1','-2','-3','-4','-+','-<','-.','-D','-*','--','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*','--','-+','-.','-0','-D','-*']
    lines = []    
    ym = []
    lineNum = 0
    y_t = []    
    for i,folder_path in enumerate(folder_paths):
        print str(i) + " " + folder_path
        XX,YY, n = load_data(folder_path)
        try:
            if calc_stats:
                max_XX = 200 * 3600
            else:
                max_XX = np.max([np.max(XXX) for XXX in XX]) ## nasty
        except:
            pdb.set_trace()
        min_XX = np.min([np.min(XXX) for XXX in XX]) ## nasty
        Xss = 0
        Yss = 0
        y_tt = []
        for X,Y in zip(XX,YY):
            X = np.array(X).reshape(-1,1)
            Y = np.array(Y).reshape(-1,1)
            #print X
            #print Y
            Xs,Ys = prev_predict(X,Y, min_XX, max_XX)
            Xss = np.array(Xs) 
            Yss = np.array(Ys) + Yss
            y_tt.append(Ys)
        print "N ",n
        
        y_t.append(y_tt)
        #Xss = np.array(Xs) 
        print "max_XX ",str(max_XX /unit_divisor)
        
        Yss = np.array(Yss) / n 
        Xss = Xss / unit_divisor
        #ax.plot(X, Y, 'b+', linewidth = 3.0)
        
        try:
            tags = splitter(folder_path.split("/")[-2])
            idx = (np.abs(Xss-table_data2_time)).argmin()
            idx2 = (np.abs(Xss-table_data2_time2)).argmin()
            if maximization:
                avg_perf1 = str(int(100 * Yss[idx][0]/(best_val)))
                avg_perf2 = str(int(100 * Yss[idx2][0]/(best_val)))
            else:
                avg_perf1 = str(int(100 * best_val/(Yss[idx][0])))
                avg_perf2 = str(int(100 * best_val/(Yss[idx2][0])))
            try:
                table_data2[tags[tagx]][tags[tagy]] = [avg_perf1, avg_perf2]
            except:
                table_data2[tags[tagx]] = {tags[tagy]:[avg_perf1, avg_perf2]}
        except:
            #pdb.set_trace()
            print "something went wrong... probs hill climbing"
        
        colormap = mpl.pyplot.get_cmap("gist_ncar")
        #fig.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, 20)])
        line, = ax.plot(Xss, Yss, symbols[i], linewidth = 3.0, markersize = 6.0)
        xss  = np.reshape(Xss,(Xss.shape[0],))
        ymm  = np.reshape(Yss,(Yss.shape[0],))
        
        #ax.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
        lines.append(line)
        lineNum += 1
        ym = []
    
    if maximization:
        #loc = "upper left"
        loc = "lower right"
    else:
        #loc = "upper right"
        loc = "lower left"
    l = ax.legend(legend, loc=loc,prop={'size':annotatefontsize-6}, ncol=ncol)
    if set_y_log:
        ax.set_yscale('log')
    if set_x_log:
        ax.set_xscale('log')
   
    # quad en
    # axvline(x=24*1,color="gray", linestyle="--")
    # axvline(x=24*7,color="gray", linestyle="--")
    # axhline(y=best_val,color="gray", linestyle="--")
    # ax.annotate('global optimal performance', xy=(670, 162),  xycoords='figure pixels', fontsize = annotatefontsize-4)
    # ax.annotate('1 day', xy=(607, 418),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
    # ax.annotate('7 days', xy=(925, 418),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
   
    #for quad th 0.001
    # axvline(x=24*1,color="gray", linestyle="--")
    # axvline(x=24*7,color="gray", linestyle="--")
    # axhline(y=best_val,color="gray", linestyle="--")
    # ax.annotate('global optimal performance', xy=(230, 418),  xycoords='figure pixels', fontsize = annotatefontsize-4)
    # ax.annotate('1 day', xy=(622, 290),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
    # ax.annotate('7 days', xy=(942, 290),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)

    #for rtm
    # axvline(x=24*1,color="gray", linestyle="--")
    # axvline(x=24*7,color="gray", linestyle="--")
    # axhline(y=best_val,color="gray", linestyle="--")
    # ax.annotate('global optimal performance', xy=(870, 165),  xycoords='figure pixels', fontsize = annotatefontsize-4)
    # ax.annotate('1 day', xy=(504, 435),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
    # ax.annotate('7 days', xy=(737, 435),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
   
    ## for pq
    axvline(x=24*7,color="gray", linestyle="--")
    axvline(x=24*14,color="gray", linestyle="--")
    axhline(y=best_val*0.95,color="gray", linestyle="--")
    ax.annotate('95% of globally optimal performance', xy=(120, 410),  xycoords='figure pixels', fontsize = annotatefontsize-4)
    ax.annotate('7 days', xy=(734, 160),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)
    ax.annotate('14 days', xy=(818, 166),  xycoords='figure pixels', fontsize = annotatefontsize-4, rotation=90)


    setp(l.get_title(), fontsize=smallfontsize)
    ax.set_xlabel('Optimization time (hours)', fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title, fontsize = fontsize, y=1.08)
    #ax.grid(True)
    change_tick_font(ax)
    if save:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.21)
        fig.savefig("images/" + plot_name, bbox_inches='tight')
    else:
        show()
    close(fig)
    if calc_stats:
        nnn = len(folder_paths)
        times = [5,10,20,30,40,50,60,70,80,90,95,99]
        for t in times:
            print ""
            print t
            print ""
            for i in range(nnn):
                for j in range(i + 1,nnn):
                    print str(i) + " " + str(j) + " :" + str(t_test(y_t,i,j,t, False))
#pdb.set_trace()
print ""
for key in sorted(table_data.keys()):
    text = key
    #print sorted(table_data[key].keys())
    for key2 in sorted(table_data[key].keys()):
        text = text + " & " + str(table_data[key][key2][0]) + " (" + str(table_data[key][key2][1]) + "\%)"
    print text + " \\\\\hline"
    
print ""
for key in sorted(table_data2.keys()):
    text = key
    for key2 in sorted(table_data2[key].keys()):
        text = text + " & " + table_data2[key][key2][0] + "\% /" + table_data2[key][key2][1] + "\%"
    print text + " \\\\\hline"