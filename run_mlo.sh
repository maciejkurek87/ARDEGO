#!/bin/bash
#export PYTHONPATH=libs/pyXGPR/src:libs/HTML.py-0.04
#export PYTHONPATH=libs/pyGPR/src:libs/pyGPR/src/GPR:libs/HTML.py-0.04
export PYTHONPATH=libs/pyGPOO/:libs/HTML.py-0.04:libs/dlib-18.3:libs/ei_soft
#export PYTHONPATH=libs/pyGPR/src:libs/HTML.py-0.04 $@ | grep -v "Segmentation fault"
python optimizer.py $@
#python2.7 -m cProfile optimizer.py $@
