import os

def kernelTagToName(tag):
    if tag == "isotropic":
        return "$\mathrm{Isotropic}$ $\mathrm{S. E}$"
    elif tag == "anisotropic":
        return "$\mathrm{S. E.}$ $\mathrm{with}$ $\mathrm{ARD}$"
    elif tag == "matern3":
        return r'$\mathrm{Mat\'{e}rn}$ $\nu=3/2$'
    else:
        raise("wrong kernel function tag" + str(tag))

def stdvTagToName(tag):
    return r'$min_{\sigma}$ = ' + str(tag)
        
def sampleonTagToName(tag):
    if tag == "ei":
        return "EGO sampling"
    elif tag == "s":
        return r'Var$[f(\mathbf{x}_*)]$ infill'
    elif tag == "m":
        return r'$\bar{f}(\mathbf{x}_*)$ infill'
    elif tag == "no":
        return "No infill"
    else:
        raise("wrong sample function tag")
        
def trialTypeToName(tag):
    if tag == "MLO":
        return "MLO"
    elif tag == "Hill Climbing":
        return "Hill Climbing"
    elif tag == "AARDEGO":
        return "ARDEGO"
    elif tag == "AUTO":
        return "AUTO-REUSE"
    else:
        raise("wrong trail type tag")
        
######## MAIN FUNCTION ##############
        
def functionTagToName(tag):
    print tag
    if tag == "pq" or tag == "ansonE":
        best_val = 159.0
        return ("PQ Application Throughput Optimization", r"Throughput ( Particles / Second )", "min", False, True, best_val)
    if tag == "rtm":
        best_val = 0.027594929146944471#0.066666667
        return ("RTM Application Execution Time Optimization", r"Execution time ( Seconds )", "sec", True, False, best_val)
    elif tag == "robot":
        best_val = 5237.0
        return ("Robot SMC Execution Time Optimization", r"Execution time ( $\mu$s )", "sec", True, False, best_val)
    if tag == "stochastic":
        best_val = 286.0
        return ("Stochastic SMC Application Execution Time Optimization", r"Execution time ( $\mu$s )", "sec", True, False, best_val)
    elif (tag[:5] == "anson") and (tag[-4:] == "True"): 
        error = tag[6:-6]
        best_val = {'0.1':75.37,'0.01':1.44,'0.05':11.19,'0.001':0.469 }[error]
        print best_val
        return (r"Throughput Optimization ($\epsilon_{rms}=" + error + "$)", "Throughput ( Integrals / Second )", "sec", False, True, best_val)
    elif (tag[:5] == "anson") and (tag[-5:] == "False"): 
        error = tag[6:-7]
        best_val = {'0.1':0.20281596923076922,'0.01':10.7,'0.05':1.35,'0.001':33.0 }[error]
        return (r"Energy Efficiency Optimzation ($\epsilon_{rms}=" + error + "$)", "Energy Efficiency ( ($\mathrm{W}$ x Seconds) / Integrals )", "sec", True, False, best_val)
    else:
        raise "Wrong function name"
        
def legendFunction(folder_tags, off_keys =[], add=""):
    legend = []
    trialTypename = ""
    for key, value in folder_tags.iteritems():
        if (key == "corr") and not ( key in off_keys):
            legend.append(kernelTagToName(value))
        elif (key == "tT") and not ( key in off_keys):
            trialTypename = trialTypeToName(value)
        elif (key == "stddv") and not ( key in off_keys):
            legend.append(stdvTagToName(value))
        elif (key == "nsims") and not ( key in off_keys):
            legend.append(value)
        elif (key == "m") and not ( key in off_keys):
            legend.append("m = " + str(value))
        elif (key == "parall") and not ( key in off_keys):
            legend.append("P = " + str(value))
        elif (key == "extra"):
            legend.append(str(value))
        elif (key == "fF"):
            title, ylabel, unit, set_x_log, maximization, best_val = functionTagToName(value)
            if not ( key in off_keys):
                legend.append(value)
        elif (key == "sampleon") and not ( key in off_keys):
            legend.append(sampleonTagToName(value))
        else:
            pass
    legend.insert(0,trialTypename)
    if add:
        return ", ".join(legend)  + " ," + add , ylabel, unit, set_x_log, maximization, best_val, title
    else:
        return ", ".join(legend), ylabel, unit, set_x_log, maximization, best_val, title
