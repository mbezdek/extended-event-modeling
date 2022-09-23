#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --mem-per-cpu 8G
#SBATCH --cpus-per-task 1
#SBATCH --time 3:00:00
#SBATCH --output=%j.%x.out


cd /scratch/n.tan/extended-event-modeling/ || echo "cd failed.."
source activate sem-viz-jupyter

input=`head -n $SLURM_ARRAY_TASK_ID $1 | tail -n 1`
echo $input
export OMP_NUM_THREADS=1
srun --nodes 1 --ntasks 1 python preprocess_features.py -c configs/config_preprocess.ini --run $input
# sbatch --array 1-148 --job-name feature_indv parallel_preprocess_indv_run.sh objhand_complete.txt
