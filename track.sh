#!/bin/bash


run=$1
export PATH=/export/Anaconda3-2019.07/bin:$PATH
cd /home/mbezdek/scripts
source activate tracking
##cd /scratch/mbezdek/pysot
##python setup.py build_ext --inplace


python merge_tracking.py ${run}