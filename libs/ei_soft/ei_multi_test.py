import numpy as np
import ei_soft
import pdb
from scipy.stats import norm


if __name__ == '__main__':
    n_sims = 10000
    
    def run(nn_sims):
        y_best = 1.0
        L = [2.1,0.,2.1,.112312]
        means = [16.38, 16.38]
        mins = 0.0
        sum_ei = 0.0
        for j in range(0,nn_sims):
            mins = y_best
            lambdas = norm.rvs(loc=0, scale=1.0,size=2) 
            e_i = np.maximum(0.0, (np.array(L).reshape((2,2)).dot(lambdas) + np.array(means).reshape((2,1)).T).max() - mins)
            sum_ei = e_i + sum_ei
        print sum_ei / nn_sims
        print ei_soft.ei_multi_max([],[],L,means,y_best,n_sims)/n_sims
    
    run(10)
    print ""
    run(100)
    print ""
    run(1000)
    print ""
    run(10000)
    print ""
    run(100000)
    pdb.set_trace()

    
	



