#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem-per-cpu 1G
#SBATCH --cpus-per-task 32
#SBATCH --time 24:00:00
#SBATCH --output=%j.%x.out


cd /scratch/n.tan/extended-event-modeling/ || echo "cd failed.."
source activate sem-viz-jupyter
export PYTHONPATH=${PYTHONPATH}:/scratch/n.tan/SEM2

#srun --exclusive --nodes 1 --ntasks 1 python individual_features/appear_feature.py -c configs/config_appear.ini &
#srun --exclusive --nodes 1 --ntasks 1 python individual_features/object_hand_features.py -c configs/config_objhand_features.ini &
#srun --exclusive --nodes 1 --ntasks 1 python individual_features/skelfeatures.py -c configs/config_skelfeatures.ini &
#srun --exclusive --nodes 1 --ntasks 1 python individual_features/vidfeatures.py -c configs/config_vidfeatures.ini &
#wait
srun --output=%j.%x.out --time=4:00:00 -c 16 --nodes 1 --ntasks 1 python individual_features/appear_feature.py -c configs/config_appear.ini
srun --output=%j.%x.out --time=4:00:00 -c 16 --nodes 1 --ntasks 1 python individual_features/object_hand_features.py -c configs/config_objhand_features.ini
srun --output=%j.%x.out --time=4:00:00 -c 16 --nodes 1 --ntasks 1 python individual_features/skelfeatures.py -c configs/config_skelfeatures.ini
srun --output=%j.%x.out --time=4:00:00 -c 16 --nodes 1 --ntasks 1 python individual_features/vidfeatures.py -c configs/config_vidfeatures.ini

