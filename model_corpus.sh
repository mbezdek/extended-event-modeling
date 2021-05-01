#!/bin/bash

#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 128:00:00


cd /scratch/n.tan/extended-event-modeling/
source activate pt-tf-37

#python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini
#python job_split.py 1

#python run_sem_pretrain.py -c configs/config_run_sem.ini --run $1
python run_sem_pretrain.py -c configs/config_run_sem.ini --train $1 --valid $2
# sbatch model_corpus.sh chapter_1_sorted.txt
