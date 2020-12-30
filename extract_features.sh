#!/bin/bash

#PBS -N dec_28_features
#PBS -l nodes=1:ppn=16:gpus=1,mem=16gbs,walltime=24:00:00

cd /scratch/tantan132/extended-event-modeling/
source activate tf
#python vidfeatures/appear_feature.py -c configs/config_appear.ini
#python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini
#python skelfeatures/skelfeatures.py -c configs/config_skelfeatures.ini
#python vidfeatures/vidfeatures.py -c configs/config_vidfeatures.ini

python run_sem_with_features.py -c configs/config_run_sem.ini
