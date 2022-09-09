#!/bin/bash

#for run in 1.1.1_C1 1.1.2_C1 1.1.3_C1 1.1.4_C1 1.1.5_C1 1.1.6_C1 1.1.7_C1 1.1.8_C1 1.1.9_C1 1.1.10_C1 1.2.1_C1 1.2.2_C1 1.2.3_C1 1.2.4_C1 1.2.5_C1 1.2.6_C1 1.2.7_C1 1.2.8_C1 1.2.9_C1 1.2.10_C1 1.3.1_C1 1.3.2_C1 1.3.3_C1 1.3.4_C1 1.3.5_C1 1.3.6_C1 1.3.7_C1 1.3.8_C1 1.3.9_C1 1.3.10_C1 2.2.1_C1 2.2.2_C1 2.2.3_C1 2.2.4_C1 2.2.5_C1 2.2.6_C1 2.2.7_C1 2.2.8_C1 2.2.9_C1 2.2.10_C1 2.3.1_C1 2.3.2_C1 2.3.3_C1 2.3.4_C1 2.3.5_C1 2.3.6_C1 2.3.7_C1 2.3.8_C1 2.3.9_C1 2.3.10_C1 2.4.1_C1 2.4.2_C1 2.4.3_C1 2.4.4_C1 2.4.5_C1 2.4.6_C1 2.4.7_C1 2.4.8_C1 2.4.9_C1 2.4.10_C1 3.1.1_C1 3.1.2_C1 3.1.3_C1 3.1.4_C1 3.1.5_C1 3.1.6_C1 3.1.7_C1 3.1.8_C1 3.1.9_C1 3.1.10_C1 3.3.1_C1 3.3.2_C1 3.3.3_C1 3.3.4_C1 3.3.5_C1 3.3.6_C1 3.3.7_C1 3.3.8_C1 3.3.9_C1 3.3.10_C1 3.4.1_C1 3.4.2_C1 3.4.3_C1 3.4.4_C1 3.4.5_C1 3.4.6_C1 3.4.7_C1 3.4.8_C1 3.4.9_C1 3.4.10_C1 3.1.1_C1 3.1.2_C1 3.1.3_C1 3.1.4_C1 3.1.5_C1 3.1.6_C1 3.1.7_C1 3.1.8_C1 3.1.9_C1 3.1.10_C1 3.3.1_C1 3.3.2_C1 3.3.3_C1 3.3.4_C1 3.3.5_C1 3.3.6_C1 3.3.7_C1 3.3.8_C1 3.3.9_C1 3.3.10_C1 3.4.1_C1 3.4.2_C1 3.4.3_C1 3.4.4_C1 3.4.5_C1 3.4.6_C1 3.4.7_C1 3.4.8_C1 3.4.9_C1 3.4.10_C1 4.3.1_C1 4.3.2_C1 4.3.3_C1 4.3.4_C1 4.3.5_C1 4.3.6_C1 4.3.7_C1 4.3.8_C1 4.3.9_C1 4.3.10_C1 4.4.1_C1 4.4.2_C1 4.4.3_C1 4.4.4_C1 4.4.5_C1 4.4.6_C1 4.4.7_C1 4.4.8_C1 4.4.9_C1 4.4.10_C1 6.1.1_C1 6.1.2_C1 6.1.3_C1 6.1.4_C1 6.1.5_C1 6.1.6_C1 6.1.7_C1 6.1.8_C1 6.1.9_C1 6.1.10_C1 6.2.1_C1 6.2.2_C1 6.2.3_C1 6.2.4_C1 6.2.5_C1 6.2.6_C1 6.2.7_C1 6.2.8_C1 6.2.9_C1 6.2.10_C1 6.3.1_C1 6.3.2_C1 6.3.3_C1 6.3.4_C1 6.3.5_C1 6.3.6_C1 6.3.7_C1 6.3.8_C1 6.3.9_C1 6.3.10_C1 1.1.1_C2 1.1.2_C2 1.1.3_C2 1.1.4_C2 1.1.5_C2 1.1.6_C2 1.1.7_C2 1.1.8_C2 1.1.9_C2 1.1.10_C2 1.2.1_C2 1.2.2_C2 1.2.3_C2 1.2.4_C2 1.2.5_C2 1.2.6_C2 1.2.7_C2 1.2.8_C2 1.2.9_C2 1.2.10_C2 1.3.1_C2 1.3.2_C2 1.3.3_C2 1.3.4_C2 1.3.5_C2 1.3.6_C2 1.3.7_C2 1.3.8_C2 1.3.9_C2 1.3.10_C2 2.2.1_C2 2.2.2_C2 2.2.3_C2 2.2.4_C2 2.2.5_C2 2.2.6_C2 2.2.7_C2 2.2.8_C2 2.2.9_C2 2.2.10_C2 2.3.1_C2 2.3.2_C2 2.3.3_C2 2.3.4_C2 2.3.5_C2 2.3.6_C2 2.3.7_C2 2.3.8_C2 2.3.9_C2 2.3.10_C2 2.4.1_C2 2.4.2_C2 2.4.3_C2 2.4.4_C2 2.4.5_C2 2.4.6_C2 2.4.7_C2 2.4.8_C2 2.4.9_C2 2.4.10_C2 3.1.1_C2 3.1.2_C2 3.1.3_C2 3.1.4_C2 3.1.5_C2 3.1.6_C2 3.1.7_C2 3.1.8_C2 3.1.9_C2 3.1.10_C2 3.3.1_C2 3.3.2_C2 3.3.3_C2 3.3.4_C2 3.3.5_C2 3.3.6_C2 3.3.7_C2 3.3.8_C2 3.3.9_C2 3.3.10_C2 3.4.1_C2 3.4.2_C2 3.4.3_C2 3.4.4_C2 3.4.5_C2 3.4.6_C2 3.4.7_C2 3.4.8_C2 3.4.9_C2 3.4.10_C2 4.1.1_C2 4.1.2_C2 4.1.3_C2 4.1.4_C2 4.1.5_C2 4.1.6_C2 4.1.7_C2 4.1.8_C2 4.1.9_C2 4.1.10_C2 4.3.1_C2 4.3.2_C2 4.3.3_C2 4.3.4_C2 4.3.5_C2 4.3.6_C2 4.3.7_C2 4.3.8_C2 4.3.9_C2 4.3.10_C2 4.4.1_C2 4.4.2_C2 4.4.3_C2 4.4.4_C2 4.4.5_C2 4.4.6_C2 4.4.7_C2 4.4.8_C2 4.4.9_C2 4.4.10_C2 6.1.1_C2 6.1.2_C2 6.1.3_C2 6.1.4_C2 6.1.5_C2 6.1.6_C2 6.1.7_C2 6.1.8_C2 6.1.9_C2 6.1.10_C2 6.2.1_C2 6.2.2_C2 6.2.3_C2 6.2.4_C2 6.2.5_C2 6.2.6_C2 6.2.7_C2 6.2.8_C2 6.2.9_C2 6.2.10_C2 6.3.1_C2 6.3.2_C2 6.3.3_C2 6.3.4_C2 6.3.5_C2 6.3.6_C2 6.3.7_C2 6.3.8_C2 6.3.9_C2 6.3.10_C2; do

#for run in 1.1.1_C1 1.1.2_C1 1.1.3_C1 1.1.4_C1 1.1.5_C1 1.1.6_C1 1.1.7_C1 1.1.8_C1 1.1.9_C1 1.1.10_C1; do
#for run in 1.2.1_C1 1.2.2_C1 1.2.3_C1 1.2.4_C1 1.2.5_C1 1.2.6_C1 1.2.7_C1 1.2.8_C1 1.2.9_C1 1.2.10_C1; do
#for run in 1.3.1_C1 1.3.2_C1 1.3.3_C1 1.3.4_C1 1.3.5_C1 1.3.6_C1 1.3.7_C1 1.3.8_C1 1.3.9_C1 1.3.10_C1; do
#for run in 2.2.1_C1 2.2.2_C1 2.2.3_C1 2.2.4_C1 2.2.5_C1 2.2.6_C1 2.2.7_C1 2.2.8_C1 2.2.9_C1 2.2.10_C1 2.3.1_C1 2.3.2_C1 2.3.3_C1 2.3.4_C1 2.3.5_C1 2.3.6_C1 2.3.7_C1 2.3.8_C1 2.3.9_C1 2.3.10_C1; do
#for run in 2.4.1_C1 2.4.2_C1 2.4.3_C1 2.4.4_C1 2.4.5_C1 2.4.6_C1 2.4.7_C1 2.4.8_C1 2.4.9_C1 2.4.10_C1; do
#for run in 3.1.1_C1 3.1.2_C1 3.1.3_C1 3.1.4_C1 3.1.5_C1 3.1.6_C1 3.1.7_C1 3.1.8_C1 3.1.9_C1 3.1.10_C1 3.3.1_C1 3.3.2_C1 3.3.3_C1 3.3.4_C1 3.3.5_C1 3.3.6_C1 3.3.7_C1 3.3.8_C1 3.3.9_C1 3.3.10_C1 3.4.1_C1 3.4.2_C1 3.4.3_C1 3.4.4_C1 3.4.5_C1 3.4.6_C1 3.4.7_C1 3.4.8_C1 3.4.9_C1 3.4.10_C1; do
#for run in 4.1.1_C1 4.1.2_C1 4.1.3_C1 4.1.4_C1 4.1.5_C1 4.1.6_C1 4.1.7_C1 4.1.8_C1 4.1.9_C1 4.1.10_C1; do
#for run in 4.4.1_C1 4.4.2_C1 4.4.3_C1 4.4.4_C1 4.4.5_C1 4.4.6_C1 4.4.7_C1 4.4.8_C1 4.4.9_C1 4.4.10_C1; do
#for run in 6.2.1_C1 6.2.2_C1 6.2.3_C1 6.2.4_C1 6.2.5_C1 6.2.6_C1 6.2.7_C1 6.2.8_C1 6.2.9_C1 6.2.10_C1; do

#for run in 1.1.1_kinect 1.1.2_kinect 1.1.3_kinect 1.2.4_kinect 1.2.5_kinect 1.2.6_kinect 1.3.7_kinect 1.3.8_kinect 1.3.9_kinect 1.3.10_kinect; do
#for run in 2.2.1_kinect 2.2.2_kinect 2.2.3_kinect 2.3.4_kinect 2.3.5_kinect 2.3.6_kinect 2.4.7_kinect 2.4.8_kinect 2.4.9_kinect 2.4.10_kinect; do
#for run in 3.1.1_kinect 3.1.2_kinect 3.1.3_kinect 3.3.4_kinect 3.3.5_kinect 3.3.6_kinect 3.4.7_kinect 3.4.8_kinect 3.4.9_kinect 3.4.10_kinect; do
#for run in 4.3.1_kinect 4.3.2_kinect 4.3.3_kinect 4.4.4_kinect 4.4.5_kinect 4.4.6_kinect 6.1.7_kinect 6.1.8_kinect 6.1.9_kinect 6.1.10_kinect; do
#for run in 6.2.1_kinect 6.2.2_kinect 6.2.3_kinect 6.3.4_kinect 6.3.5_kinect 6.3.6_kinect; do
#for run in 1.1.1_C1 1.1.2_C1 1.1.3_C1 1.2.4_C1 1.2.5_C1 1.2.6_C1 1.3.7_C1 1.3.8_C1 1.3.9_C1 1.3.10_C1; do
#for run in 2.2.1_C1 2.2.2_C1 2.2.3_C1 2.3.4_C1 2.3.5_C1 2.3.6_C1 2.4.7_C1 2.4.8_C1 2.4.9_C1 2.4.10_C1; do
#for run in 3.1.1_C1 3.1.2_C1 3.1.3_C1 3.3.4_C1 3.3.5_C1 3.3.6_C1 3.4.7_C1 3.4.8_C1 3.4.9_C1 3.4.10_C1; do
#for run in 4.3.1_C1 4.3.2_C1 4.3.3_C1 4.4.4_C1 4.4.5_C1 4.4.6_C1 6.1.7_C1 6.1.8_C1 6.1.9_C1 6.1.10_C1; do
#for run in 6.2.1_C1 6.2.2_C1 6.2.3_C1 6.3.4_C1 6.3.5_C1 6.3.6_C1; do
tag=" _nov_16"
while read run; do
  echo $run
  export run=$run
  export tag=$tag
  sbatch --export=ALL track_v100_slurm.sh -J ${run}${tag}
done < next_8.txt
