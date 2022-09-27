import os
import pandas as pd
from src.utils import contain_substr
from joblib import Parallel, delayed


# load all skel and re-sample

def load_and_sample(path, sample):
    input_df = pd.read_csv(path)
    return input_df.sample(n=sample)


def pool_features(complete_skel_path='output/skel_complete.txt', output_stats_path='sampled_skel_features_sep_09.csv',
                  tag='sep_09', sample=200):
    skel_complete = open(complete_skel_path, 'rt').readlines()
    skel_complete = [os.path.join('output/skel', s.strip() + f'_{tag}_skel_features.csv') for s in skel_complete]
    input_paths = skel_complete
    print(f'Total runs: {len(input_paths)}')

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
    combined_runs.to_csv(output_stats_path, index_label=False)
    print(f"Saved {output_stats_path}!")
