#!/bin/bash
#export PYTHONPATH=libs/pyXGPR/src:libs/HTML.py-0.04
export PYTHONPATH=libs/pyGPR/src:libs/pyGPR/src/GPR:libs/HTML.py-0.04
#export PYTHONPATH=libs/pyGPR/src:libs/HTML.py-0.04
python2.7 -m pdb optimizer.py $@ | grep -v "Segmentation fault"
