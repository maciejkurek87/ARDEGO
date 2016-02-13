#!/bin/bash
#export PYTHONPATH=libs/pyXGPR/src:libs/HTML.py-0.04
#export PYTHONPATH=libs/pyGPR/src:libs/pyGPR/src/GPR:libs/HTML.py-0.04
export PYTHONPATH=/homes/mk306/MLO/libs/pyGPOO
#export PYTHONPATH=libs/pyGPR/src:libs/HTML.py-0.04 $@ | grep -v "Segmentation fault"
python fitness_script.py $@
#python2.7 -m cProfile optimizer.py $@
