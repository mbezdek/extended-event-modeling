#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 8G
#SBATCH --time 72:00:00


cd /scratch/n.tan/extended-event-modeling/
source activate tf

python run_sem_pretrain.py -c configs/config_run_sem.ini --run $1
# sbatch model_corpus.sh chapter_1_sorted.txt
