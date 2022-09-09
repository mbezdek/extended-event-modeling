import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import glob
from joblib import Parallel, delayed
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache_tag", default='may_20_alfa0_appear_cont_individual')
parser.add_argument("--n_sample", default=200, type=int)
parser.add_argument("--n_components", default=30, type=int)
# TODO: cross-check with run_sem_pretrain.py to make sure features' positions are correct
parser.add_argument("--appear_components", default=2, type=int)
parser.add_argument("--optical_components", default=1, type=int)
parser.add_argument("--skel_components", default=14, type=int)
parser.add_argument("--emb_components", default=13, type=int)
parser.add_argument("--pca_tag", default='all', type=str)
args = parser.parse_args()

tag = args.cache_tag
sample = args.n_sample
pca_tag = args.pca_tag
appear_components = args.appear_components
optical_components = args.optical_components
skel_components = args.skel_components
emb_components = args.emb_components
n_components = appear_components + optical_components + skel_components + emb_components

print(f'Cache Tag is: {tag}')
print(f'Sample for each run is: {sample}')
print(f'PCA tag is : {pca_tag}')
print(f'# components: {n_components}')
# load all inputdf and re-construct combined_df
input_paths = glob.glob(f'output/run_sem/{tag}/*inputdf_1.pkl')
print(f'Total runs: {len(input_paths)}')


def load_and_sample(path, sample):
    input_df = pkl.load(open(path, 'rb'))
    if pca_tag == '' or pca_tag == 'all':
        # use scene as a default, since sep 22
        data_frames = [input_df.appear_post, input_df.optical_post, input_df.skel_post, input_df.objhand_post,
                       input_df.scene_post]
    elif pca_tag == 'objhand_only':
        data_frames = [input_df.objhand_post]
    elif pca_tag == 'skel_only':
        data_frames = [input_df.skel_post]
    elif pca_tag == 'skel_objhand_only' or pca_tag == 'objhand_skel_only':
        data_frames = [input_df.skel_post, input_df.objhand_post]
    else:
        raise Exception(f'Unclear which features to include!!!')
    if 'motion' in tag:
        data_frames.append(input_df.objspeed_post)
    input_df = pd.concat(data_frames, axis=1)
    try:
        res = input_df.sample(n=sample)
        return res
    except Exception as e:
        print(f'Failed: Path={path}, len={len(input_df)}')
        return pd.DataFrame()
    # return input_df.sample(n=sample)


input_dfs = Parallel(n_jobs=16)(delayed(load_and_sample)(path, sample=sample) for path in input_paths)
combined_runs = pd.concat(input_dfs, axis=0)
print(f'Total data points to PCA: {len(combined_runs)}')
# run pca and save pca pickles
pca = PCA(n_components=n_components, whiten=True)

pca.fit(combined_runs)

print(f"pca.components_.shape={pca.components_.shape}")
print(f'Saving {tag}_{pca_tag}_{n_components}_pca.pkl')
with open(f'{tag}_{pca_tag}_{n_components}_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)

pca = PCA(n_components=appear_components, whiten=True)
pca.fit(combined_runs.iloc[:, :2])
print(f"pca.components_.shape={pca.components_.shape}")
print(f'Saving {tag}_{pca_tag}_{n_components}_appear_pca.pkl')
with open(f'{tag}_{pca_tag}_{n_components}_appear_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)

pca = PCA(n_components=optical_components, whiten=True)
pca.fit(combined_runs.iloc[:, 2:4])
print(f"pca.components_.shape={pca.components_.shape}")
print(f'Saving {tag}_{pca_tag}_{n_components}_optical_pca.pkl')
with open(f'{tag}_{pca_tag}_{n_components}_optical_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)

pca = PCA(n_components=skel_components, whiten=True)
pca.fit(combined_runs.iloc[:, 4:-100])
print(f"pca.components_.shape={pca.components_.shape}")
print(f'Saving {tag}_{pca_tag}_{n_components}_skel_pca.pkl')
with open(f'{tag}_{pca_tag}_{n_components}_skel_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)

pca = PCA(n_components=emb_components, whiten=True)
pca.fit(combined_runs.iloc[:, -100:])
print(f"pca.components_.shape={pca.components_.shape}")
print(f'Saving {tag}_{pca_tag}_{n_components}_emb_pca.pkl')
with open(f'{tag}_{pca_tag}_{n_components}_emb_pca.pkl', 'wb') as f:
    pkl.dump(pca, f)
print('Done!')
