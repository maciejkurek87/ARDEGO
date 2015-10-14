import os

def kernelTagToName(tag):
    if tag == "isotropic":
        return "$\mathrm{Isotropic}$ $\mathrm{S. E}$"
    elif tag == "anisotropic":
        return "$\mathrm{Anisotropic}$ $\mathrm{S. E.}$"
    elif tag == "matern3":
        return r'$\mathrm{Mat\'{e}rn}$ $\nu=3/2$'
    else:
        raise("wrong kernel function tag" + str(tag))

def stdvTagToName(tag):
    return "Stdv limit = " + str(tag)
        
def sampleonTagToName(tag):
    if tag == "ei":
        return "EGO sampling"
    elif tag == "s":
        return "S2 sampling"
    elif tag == "m":
        return "M sampling"
    elif tag == "no":
        return "No sampling"
    else:
        raise("wrong sample function tag")
        
def trialTypeToName(tag):
    if tag == "MLO":
        return "MLO"
    elif tag == "Hill Climbing":
        return "Hill Climbing"
    elif tag == "AARDEGO":
        return "ARDEGO"
    else:
        raise("wrong trail type tag")
        
######## MAIN FUNCTION ##############
        
def functionTagToName(tag):
    print tag
    if tag == "pq" or tag == "ansonE":
        best_val = 159.0
        return ("PQ Application Throughput Optimization", r"Particles/Second", "min", False, True, best_val)
    if tag == "rtm":
        best_val = 0.066666667
        return ("RTM Application Throughput Optimization", r"Particles/second", "sec", True, False, best_val)
    elif (tag[:5] == "anson") and (tag[-4:] == "True"): 
        error = tag[6:-6]
        best_val = {'0.1':75.37,'0.01':1.44,'0.05':11.19,'0.001':0.469 }[error]
        print best_val
        return (r"Throughput Optimization ($\epsilon_{rms}=" + error + "$)", "$\phi_{int}$", "sec", False, True, best_val)
    elif (tag[:5] == "anson") and (tag[-5:] == "False"): 
        error = tag[6:-7]
        best_val = {'0.1':0.20281596923076922,'0.01':10.7,'0.05':1.35,'0.001':33.0 }[error]
        return (r"Energy Efficiency Optimzation ($\epsilon_{rms}=" + error + "$)", "$\mathrm{W}$/$\phi_{int}$", "sec", True, False, best_val)
    else:
        raise "Wrong function name"
        
def legendFunction(folder_tags, off_keys =[], add=""):
    legend = []
    for key, value in folder_tags.iteritems():
        if (key == "corr") and not ( key in off_keys):
            legend.append(kernelTagToName(value))
        elif (key == "tT") and not ( key in off_keys):
            legend.append(trialTypeToName(value))
        elif (key == "stddv") and not ( key in off_keys):
            legend.append(stdvTagToName(value))
        elif (key == "nsims") and not ( key in off_keys):
            legend.append(value)
        elif (key == "parall") and not ( key in off_keys):
            legend.append("P = " + str(value))
        elif (key == "fF"):
            title, ylabel, unit, set_x_log, maximization, best_val = functionTagToName(value)
            if not ( key in off_keys):
                legend.append(value)
        elif (key == "sampleon") and not ( key in off_keys):
            legend.append(sampleonTagToName(value))
        else:
            pass
            
    if add:
        return ", ".join(legend)  + " ," + add , ylabel, unit, set_x_log, maximization, best_val, title
    else:
        return ", ".join(legend), ylabel, unit, set_x_log, maximization, best_val, title
