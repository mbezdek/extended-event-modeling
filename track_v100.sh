#!/bin/bash

#PBS -N tan_tracking_V100
#PBS -l nodes=1:ppn=8:gpus=1:V100_32,mem=8gbs,walltime=4:00:00
run=$1
tag=$2
cd /scratch/tantan132/extended-event-modeling/
source activate sem-pysot-37-new
python tracking/tracking_to_correct_label.py -c configs/config_tracking_to_correct_label.ini --run $run --tag $tag 2>&1 | tee "logs/$run$tag.log"
