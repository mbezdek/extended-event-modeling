#!/bin/bash

#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 12G
#SBATCH --time 72:00:00
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --output=logs/%j.%x.out
#SBATCH --partition=tier2_cpu
#SBATCH --account=jeffrey_zacks


cd /scratch/n.tan/extended-event-modeling/ || echo "cd failed.."
source activate sem-viz-jupyter
export PYTHONPATH=${PYTHONPATH}:/scratch/n.tan/SEM2
export SEED=$8

#python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini
#python job_split.py 1

#python run_sem_pretrain.py -c configs/config_run_sem.ini --run $1
# MAKE SURE TO DOUBLE CHECK THE ORDER OF ARGUMENTS IN GRID_SEARCH.PY (WHICH CALLS THIS SCRIPT)
echo "Execute: python src/train_eval_inference/run_sem_pretrain.py -c configs/config_run_sem.ini --train ${1}
 --valid ${2} --alfa ${3} --lmda ${4} --sem_tag ${5} --trigger ${6} --threshold ${7} --lr ${9} --equal_sigma ${10}"
python src/train_eval_inference/run_sem_pretrain.py -c configs/config_run_sem.ini --train $1 --valid $2 \
--alfa $3 --lmda $4 --sem_tag $5 --trigger $6 --threshold $7 --lr $9 --equal_sigma ${10}
# if --equal_sigma $10, it'll become equal_sigma="$1"+"0"
#python run_sem_pretrain.py -c configs/config_run_sem.ini --train $1 --valid $2
# sbatch model_corpus.sh chapter_1_sorted.txt
