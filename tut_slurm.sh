#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --array 1-10
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

# srun inherits sbatch ntasks and nodes, so we need to specify for each step
# if there is no --ntasks 1, each step below will be ran 10 times (10 tasks)
# if there is no --nodes 1, each step will be assigned to many nodes leading to warning
# Warning: can't run 1 processes on 5 nodes, setting nnodes to 1
# for & to run in parallel, the number of tasks in SBATCH must be equal or larger than number of steps here
# otherwise, srun: Job 6515 step creation temporarily disabled, retrying (Requested nodes are busy)
#srun --nodes 1 --ntasks 1 python -c 'import time; print(5); time.sleep(1)' &
#srun --nodes 1 --ntasks 1 python -c 'import time; print(5); time.sleep(1)' &
#srun --nodes 1 --ntasks 1 python -c 'import time; print(5); time.sleep(1)' &
#srun --nodes 1 --ntasks 1 python -c 'import time; print(5); time.sleep(1)' &
#wait
python -c 'import numpy as np; print(np.random.randint(5))'
input=`head -n $SLURM_ARRAY_TASK_ID job_input.txt| tail -n 1`
srun python -c 'import time; time.sleep(10)' & echo $input

#for i in {1..10}; do
#  srun --ntasks 1 python -c 'import time; print(10); time.sleep(1)' &
#done
#wait
