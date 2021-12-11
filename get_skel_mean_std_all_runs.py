import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import glob
from joblib import Parallel, delayed
import pandas as pd
import sys
import argparse
from utils import contain_substr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_sample", default=200, type=int)
args = parser.parse_args()

sample = args.n_sample

# load all skel and re-sample
# input_paths = glob.glob('output/skel/*.csv')
import os
skel_complete = open('skel_complete.txt', 'rt').readlines()
skel_complete = [os.path.join('output/skel', s.strip() + '_skel_features.csv') for s in skel_complete]
input_paths = skel_complete
print(f'Total runs: {len(input_paths)}')


def load_and_sample(path, sample):
    input_df = pd.read_csv(path)
    return input_df.sample(n=sample)


input_dfs = Parallel(n_jobs=16)(delayed(load_and_sample)(path, sample=sample) for path in input_paths)
combined_runs = pd.concat(input_dfs, axis=0)
print(f'Total data points to get Mean and Std: {len(combined_runs)}')
combined_runs.drop(['J1_dist_from_J1', 'J1_3D_rel_X', 'J1_3D_rel_Y', 'J1_3D_rel_Z'], axis=1, inplace=True)
keeps = ['accel', 'speed', 'dist', 'interhand', 'rel']
for c in combined_runs.columns:
    if contain_substr(c, keeps):
        continue
    else:
        combined_runs.drop([c], axis=1, inplace=True)
# (accel + speed + dist + interhand) + rel = 77 + 72 = 149
assert len(combined_runs.columns) == 149, f"len(combined_runs.columns)={len(combined_runs.columns)} != 149"
combined_runs.to_csv('sampled_skel_features_dec_6.csv', index_label=False)
# get statistics and save
# for some reason, J*_acceleration and inf value
pd.set_option('use_inf_as_na', True)
combined_runs.dropna(axis=0, inplace=True)
# for acceleration dimensions, some values are too high or too low, affecting mean and std.
combined_runs_q = combined_runs[(combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))]
