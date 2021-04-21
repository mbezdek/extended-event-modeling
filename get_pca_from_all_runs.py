import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import glob
from joblib import Parallel, delayed
import pandas as pd
import sys

if len(sys.argv) > 1:
    tag = sys.argv[1]
else:
    tag = 'mar_20_individual_depth_scene'
if len(sys.argv) > 2:
    sample = int(sys.argv[2])
else:
    sample = 500

print(f'Tag is: {tag}')
print(f'Sample for each run is: {sample}')
# load all inputdf and re-construct combined_df
input_paths = glob.glob(f'output/run_sem/{tag}/*inputdf_1.pkl')
print(f'Total runs: {len(input_paths)}')


def load_and_sample(path, sample):
    input_df = pkl.load(open(path, 'rb'))
    data_frames = [input_df.optical_post, input_df.skel_post, input_df.objhand_post]
    if 'scene_motion' in tag:
        data_frames.append(input_df.objspeed_post)
    input_df = pd.concat(data_frames, axis=1)
    return input_df.sample(n=sample)


input_dfs = Parallel(n_jobs=16)(delayed(load_and_sample)(path, sample=sample) for path in input_paths)
combined_runs = pd.concat(input_dfs, axis=0)
print(f'Total data points to PCA: {len(combined_runs)}')
# run pca and save pca pickle
pca = PCA(n_components=30, whiten=True)

pca.fit(combined_runs)

with open(f'{tag}_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)
