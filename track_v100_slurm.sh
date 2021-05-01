#!/bin/bash

#SBATCH --export=run=$run,tag=$tag
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --mincpus=16

echo $run
echo $tag
#echo "track_v100.sh"
cd /scratch/n.tan/extended-event-modeling/
source activate sem-pysot-37-new
python tracking/tracking_to_correct_label.py -c configs/config_tracking_to_correct_label.ini --run $run --tag $tag 2>&1 | tee "logs/$run$tag.log"
