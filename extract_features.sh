#!/bin/bash

#PBS -N extract_features_and_run_sem
#PBS -l nodes=1:ppn=16:gpus=1,mem=16gbs,walltime=24:00:00

cd /scratch/tantan132/extended-event-modeling/
source activate tf
rm sem_complete.txt sem_error.txt appear_complete.txt objhand_complete.txt skel_complete.txt vid_complete.txt results_appear_features.json results_objhand.json results_skel_features.json results_vid_features.json results_sem_run.json

#python vidfeatures/appear_feature.py -c configs/config_appear.ini
#python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini
#python skelfeatures/skelfeatures.py -c configs/config_skelfeatures.ini
#python vidfeatures/vidfeatures.py -c configs/config_vidfeatures.ini

python run_sem_with_features.py -c configs/config_run_sem.ini
