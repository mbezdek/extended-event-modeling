#!/bin/bash

#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 6G
#SBATCH --time 60:00:00
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --output=logs/%j.%x.out


cd /scratch/n.tan/extended-event-modeling/ || echo "cd failed.."
source activate sem-viz-jupyter
export PYTHONPATH=${PYTHONPATH}:/scratch/n.tan/SEM2
export SEED=$8

#python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini
#python job_split.py 1

#python run_sem_pretrain.py -c configs/config_run_sem.ini --run $1
# make sure to double check the order of arguments in grid_search.py (which calls this script)
echo "Execute: python src/train_eval_inference/run_sem_pretrain.py -c configs/config_run_sem.ini --train ${1}
 --valid ${2} --alfa ${3} --lmda ${4} --sem_tag ${5} --trigger ${6} --threshold ${7}"
python src/train_eval_inference/run_sem_pretrain.py -c configs/config_run_sem.ini --train $1 --valid $2 \
--alfa $3 --lmda $4 --sem_tag $5 --trigger $6 --threshold $7
#python run_sem_pretrain.py -c configs/config_run_sem.ini --train $1 --valid $2
# sbatch model_corpus.sh chapter_1_sorted.txt
