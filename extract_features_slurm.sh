#!/bin/bash

#SBATCH --array 1-10
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 4G
#SBATCH --time 24:00:00


cd /scratch/n.tan/extended-event-modeling/
source activate tf

#rm intersect_features.txt appear_complete.txt objhand_complete.txt skel_complete.txt vid_complete.txt
#rm output/run_sem/* output/vid/* output/objhand/* output/appear/* output/skel/*
#
#srun --exclusive --ntasks 1 python vidfeatures/appear_feature.py -c configs/config_appear.ini &
#srun --exclusive --ntasks 1 python vidfeatures/object_hand_features.py -c configs/config_objhand_features.ini &
#srun --exclusive --ntasks 1 python skelfeatures/skelfeatures.py -c configs/config_skelfeatures.ini &
#srun --exclusive --ntasks 1 python vidfeatures/vidfeatures.py -c configs/config_vidfeatures.ini &
#wait

#python merge_and_split.py 10
input=`head -n $SLURM_ARRAY_TASK_ID job_input.txt| tail -n 1`
#srun python -c 'import time; time.sleep(10)' & echo $input
srun python run_sem_with_features.py -c configs/config_run_sem.ini --run $input
#for i in {1..10}; do
#  run="intersect_features_$i.txt"
#  echo $run
#  srun --nodes 1 --ntasks 1 python run_sem_with_features.py -c configs/config_run_sem.ini --run $run &
#done
#wait
