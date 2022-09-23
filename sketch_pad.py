import numpy as np
import matplotlib.pyplot as plt

# Plot boundaries for different updating
e_hat = np.argmax(sem_model.results.post, axis=1)
frame_boundaries = np.concatenate([[0], e_hat[1:] != e_hat[:-1]])
e_hat = np.argmax(sem_model.results.post_pe, axis=1)
frame_boundaries_pe = np.concatenate([[0], e_hat[1:] != e_hat[:-1]])
set(np.where(frame_boundaries)[0]).difference(set(np.where(frame_boundaries_pe)[0]))
set(np.where(frame_boundaries_pe)[0]).difference(set(np.where(frame_boundaries)[0]))
plt.vlines(list(
    set(np.where(frame_boundaries_pe)[0]).difference(set(np.where(frame_boundaries)[0]))),
    ymin=0, ymax=1, color='r', alpha=0.5, label='lik_next only', linestyles='dotted')
plt.vlines(list(
    set(np.where(frame_boundaries)[0]).difference(set(np.where(frame_boundaries_pe)[0]))),
    ymin=0, ymax=1, color='g', alpha=0.5, label='lik_next + lik_restart', linestyles='dotted')
plt.vlines(list(
    set(np.where(frame_boundaries)[0]).intersection(set(np.where(frame_boundaries_pe)[0]))),
    ymin=0, ymax=1, color='b', alpha=0.5, label='Shared', linestyles='dotted')
plt.vlines(list(np.where(sem_model.results.boundaries)[0]),
           ymin=0, ymax=1, color='b', alpha=0.5, label='Event model boundaries',
           linestyles='dotted')
plt.legend()
plt.title('dishes')
plt.savefig('output/run_sem/dishes_diff.png')
plt.show()

with open('sem_readouts.pkl', 'rb') as f:
    sem_readouts = pkl.load(f)

colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}
sem_readouts.frame_dynamics['old_lik'] = list(map(np.max, sem_readouts.frame_dynamics['old_lik']))
sem_readouts.frame_dynamics['old_prior'] = list(map(np.max, sem_readouts.frame_dynamics['old_prior']))
df = pd.DataFrame(sem_readouts.frame_dynamics)
df['new_post'] = df.filter(regex='new_').sum(axis=1)
df['old_post'] = df.filter(regex='old_').sum(axis=1)
df['repeat_post'] = df.filter(regex='repeat_').sum(axis=1)
df['restart_post'] = df.filter(regex='restart_').sum(axis=1)
df['switch'] = df.filter(regex='_post').idxmax(axis=1)
plt.vlines(df[df['switch'] == 'new_post'].index, ymin=0, ymax=1, alpha=0.5, label='Switch to New Event', color=colors['new'],
           linestyles='dotted')
plt.vlines(df[df['switch'] == 'old_post'].index, ymin=0, ymax=1, alpha=0.5, label='Switch to Old Event', color=colors['old'],
           linestyles='dotted')
# plt.vlines(df[df['switch'] == 'repeat_post'].index, ymin=0, ymax=1, alpha=0.5, label='Repeat Event', color=colors['repeat'],
#            linestyles='dotted')
# plt.vlines(df[df['switch'] == 'restart_post'].index, ymin=0, ymax=1, alpha=0.5, label='Restart Event', color=colors['restart'],
#            linestyles='dotted')
plt.legend()
plt.show()
# Plot numerical values to debug SEM
sem_readouts.frame_dynamics['old_lik'] = list(map(np.max, sem_readouts.frame_dynamics['old_lik']))
plt.plot(sem_readouts.frame_dynamics['new_lik'], alpha=0.4, label='new_lik')
plt.plot(sem_readouts.frame_dynamics['repeat_lik'], alpha=0.4, label='repeat_lik')
plt.plot(sem_readouts.frame_dynamics['restart_lik'], alpha=0.4, label='restart_lik')
plt.plot(sem_readouts.frame_dynamics['old_lik'], alpha=0.4, label='old_lik')
plt.ylim([-5 * 1, 5 * 1])
plt.legend()
plt.title('Likelihood')
plt.show()

sem_readouts.frame_dynamics['old_prior'] = list(map(np.max, sem_readouts.frame_dynamics['old_prior']))
plt.plot(sem_readouts.frame_dynamics['new_prior'], alpha=0.4, label='new_prior')
plt.plot(sem_readouts.frame_dynamics['repeat_prior'], alpha=0.4, label='current_repeat_prior')
plt.plot(sem_readouts.frame_dynamics['restart_prior'], alpha=0.4, label='current_restart_prior')
plt.plot(sem_readouts.frame_dynamics['old_prior'], alpha=0.4, label='old_prior')
plt.ylim([-1 * 1, 1 * 1])
plt.legend()
plt.title('Prior')
plt.show()

# Testing BERT
from pytorch_transformers import *
import torch

model_class, tokenizer_class, pretrained_weights = (
    BertModel, BertTokenizer, 'bert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
input_ids = torch.tensor([tokenizer.encode("dumbbell")])
with torch.no_grad():
    all_hidden_states_db, _ = model(input_ids)[-2:]
input_ids = torch.tensor([tokenizer.encode("kitchen")])
with torch.no_grad():
    all_hidden_states_kc, _ = model(input_ids)[-2:]
cos = torch.nn.CosineSimilarity(dim=0)
cos(all_hidden_states_db[-2][0][0], all_hidden_states_db[-2][0][1])
cos(all_hidden_states_kc[-2][0][0], all_hidden_states_db[-2][0][0])
cos(all_hidden_states_kc[-2][0][0], all_hidden_states_db[-2][0][1])

# Testing gensim
import glob
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import gensim.downloader
import matplotlib.pyplot as plt

# glove_vectors = gensim.downloader.load('glove-twitter-100')
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
# glove_vectors = gensim.downloader.load('word2vec-google-news-300')
glove_vectors['dumbbell']

csv_files = glob.glob('data/ground_truth_labels/*.csv')
all_categories = set()
for path in csv_files:
    df = pd.read_csv(path)
    categories = df['class']
    all_categories.update(set(categories))
all_categories = list(all_categories)
M = np.zeros(shape=(0, 100))
labels = []
for c in all_categories:
    r = np.zeros(shape=(1, 100))
    try:
        r += glove_vectors[c]
    except Exception as e:
        print(f'category {c} does not exist')
        try:
            c = c.split(' ')
            for w in c:
                w = w.replace('(', '').replace(')', '')
                labels.append(w)
                r += glove_vectors[w]
            r /= len(c)
            M = np.vstack([M, r])
        except Exception as e:
            print(repr(e))
cos_m = cosine_similarity(M)

fig, ax = plt.subplots()
cax = ax.matshow(cos_m, interpolation='nearest')
ax.grid(True)
plt.title('San Francisco Similarity matrix')

plt.xticks(range(113), labels, rotation=90);
plt.yticks(range(113), labels);
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
plt.show()


def remove_number(string):
    for i in range(100):
        string = string.replace(str(i), '')
    return string


all_categories = list(map(remove_number, all_categories))

for x in ['fasttext-wiki-news-subwords-300',
          'conceptnet-numberbatch-17-06-300',
          'word2vec-ruscorpora-300',
          'word2vec-google-news-300',
          'glove-wiki-gigaword-50',
          'glove-wiki-gigaword-100',
          'glove-wiki-gigaword-200',
          'glove-wiki-gigaword-300',
          'glove-twitter-25',
          'glove-twitter-50',
          'glove-twitter-100',
          'glove-twitter-200',
          '__testing_word2vec-matrix-synopsis']:
    gensim.downloader.load(x)

# List of successful tracking runs
import glob
import os
import collections

tracks = glob.glob('output/tracking/*dec*.csv')
tag = [os.path.basename(t).split('_')[0] for t in tracks]
dup = [item for item, count in collections.Counter(tag).items() if count > 1]
for x in dup:
    os.remove('output/tracking/' + x + '_C1_dec_21.csv')

for t in tracks:
    os.replace(t, t.replace('dec_21', 'r50').replace('dec_22', 'r50'))

videos = glob.glob('output/tracking/*.avi')
for video in videos:
    if '_bw' in video or '_fw' in video:
        os.remove(video)

import os
import glob

with open('track_complete.txt', 'r') as f:
    tracks = f.readlines()
    track_tags = [t.strip() for t in tracks if 'kinect' in t]
skels = glob.glob('data/projected_skeletons/*.csv')
skel_tags = [os.path.basename(skel).replace('_skel.csv', '') for skel in skels]
runs = list(set(skel_tags).intersection(set(track_tags)))
with open('feasible_runs.txt', 'w') as f:
    runs = sorted(runs)
    f.writelines('\n'.join(runs))

# Intersection between successfull feature runs
with open('appear_complete.txt', 'r') as f:
    appears = f.readlines()

with open('vid_complete.txt', 'r') as f:
    vids = f.readlines()

with open('skel_complete.txt', 'r') as f:
    skels = f.readlines()

with open('objhand_complete.txt', 'r') as f:
    objhands = f.readlines()

sem_runs = set(appears).intersection(set(skels)).intersection(set(vids)).intersection(
    set(objhands))
with open('intersect_features.txt', 'w') as f:
    f.writelines(sem_runs)

import json

# Average metric for all runs
res = json.load(open('results_sem_run.json', 'r'))
bicorrs = []
pers = []
for xy, v in res.items():
    if v['bicorr'] is not None:
        bicorrs.append(v['bicorr'])
        pers.append(v['percentile'])

print(sum(bicorrs) / len(bicorrs))
print(sum(pers) / len(pers))
sorted(res.items(), key=lambda item: item[1]['bicorr'])

# check projected skeletons
import glob
import pandas as pd

ps = glob.glob('data/projected_skeletons/*.csv')
exceed = []
keeps = ['2D', '3D']


def check_keep_feature(column: str, keeps):
    for k in keeps:
        if k in column:
            return 1
    return 0


for f in ps:
    df = pd.read_csv(f)
    for c in df.columns:
        if not check_keep_feature(c, keeps):
            df.drop(c, axis=1, inplace=True)
    if (df > 2000).sum().sum():
        exceed.append(f)

# archive results
import glob
import os

tag = 'jan_04_use_scene'
sem_runs = glob.glob(f'output/run_sem/*{tag}*.png')
for r in sem_runs:
    dir = f'tmp/{tag}/run_sem/'
    os.makedirs(dir, exist_ok=True)
    os.rename(r, f'tmp/{tag}/run_sem/{os.path.basename(r)}')
txt = glob.glob('*complete.txt')
for t in txt:
    if 'track' in t:
        continue
    dir = f'tmp/{tag}/'
    os.makedirs(dir, exist_ok=True)
    os.rename(t, os.path.join(dir, os.path.basename(t)))
jsons = glob.glob('results*.json')
for j in jsons:
    dir = f'tmp/{tag}/'
    os.makedirs(dir, exist_ok=True)
    os.rename(j, os.path.join(dir, os.path.basename(j)))

import shutil

shutil.move('output/appear', f'tmp/{tag}/appear')
shutil.move('output/objhand', f'tmp/{tag}/objhand')
shutil.move('output/vid', f'tmp/{tag}/vid')
shutil.move('output/skel', f'tmp/{tag}/skel')

# post process SEM runs
import json

res = json.load(open('tmp/jan_04_3sec/results_sem_run.json', 'r'))
agg = dict(actor=dict(), chapter=dict())
for xy, v in res.items():
    if xy[0] == 't':
        continue
    if xy[0] not in agg['actor']:
        agg['actor'][xy[0]] = []
    else:
        agg['actor'][xy[0]].append((xy, v['percentile'], v['bicorr']))
    if xy[2] not in agg['chapter']:
        agg['chapter'][xy[2]] = []
    else:
        agg['chapter'][xy[2]].append((xy, v['percentile'], v['bicorr']))

# plot results
colors = {'chapter 1': 'red', 'chapter 2': 'green', 'chapter 3': 'blue', 'chapter 4': 'purple'}
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), sharey=True)
for xy, v in agg['chapter'].items():
    ax[0].scatter(np.mean([x[1] for x in v]) / 100, np.mean([x[2] for x in v]),
                  label=f'chapter {xy}', c=colors[f'chapter {xy}'])
ax[0].set_xlabel('Percentile')
ax[0].set_ylabel('Biserial Correlation')
ax[0].set_title('Aggregate metrics for each Chapter')
ax[0].legend()

for xy, v in res.items():
    ax[1].scatter(v['percentile'], v['bicorr'], c=colors[f'chapter {xy[2]}'])
    ax[1].annotate(xy[:5], (v['percentile'], v['bicorr']))
ax[1].set_xlabel('Percentile')
ax[1].set_ylabel('Biserial Correlation')
ax[1].set_title('Metrics for each Run')
plt.savefig('jan_05_36runs_kinect.png')
plt.show()
fig, ax = plt.subplots()
ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.percentile / 100, group.bicorr, marker='o', linestyle='', ms=12, label=name)
    ax.legend()
    ax.set_xlabel('Biserial Correlation')
    ax.set_ylabel('Percentile')
    plt.show()

import pandas as pd

df = pd.read_csv('results_sem_run.csv')
df[df['grain'] == 'fine']['percentile'].mean()

with open('output/run_sem/1.3.9_kinect_trimjan_07_1000ms_gtfreqs.pkl', 'rb') as f:
    gt_freqs = pkl.load(f)
with open('output/run_sem/1.3.9_kinect_trimjan_07_1000ms_diagnostic.pkl', 'rb') as f:
    sem_readouts = pkl.load(f)

    ax.plot(gt_freqs, label='Subject Boundaries')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Boundary Probability')
    ax.set_title(title)
    colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}
    sem_readouts.frame_dynamics['old_lik'] = list(map(np.max, sem_readouts.frame_dynamics['old_lik']))
    sem_readouts.frame_dynamics['old_prior'] = list(map(np.max, sem_readouts.frame_dynamics['old_prior']))
    df = pd.DataFrame(sem_readouts.frame_dynamics)
    df['new_post'] = df.filter(regex='new_').sum(axis=1)
    df['old_post'] = df.filter(regex='old_').sum(axis=1)
    df['repeat_post'] = df.filter(regex='repeat_').sum(axis=1)
    df['restart_post'] = df.filter(regex='restart_').sum(axis=1)
    df['switch'] = df.filter(regex='_post').idxmax(axis=1)
    ax.vlines(df[df['switch'] == 'new_post'].index / frame_interval + offset, ymin=0, ymax=1, alpha=0.5, label='Switch to New '
                                                                                                               'Event',
              color=colors['new'], linestyles='dotted')
    ax.vlines(df[df['switch'] == 'old_post'].index / frame_interval + offset, ymin=0, ymax=1, alpha=0.5, label='Switch to Old '
                                                                                                               'Event',
              color=colors['old'], linestyles='dotted')
    # ax.vlines(df[df['switch'] == 'repeat_post'].index, ymin=0, ymax=1, alpha=0.5, label='Repeat Event', color=colors['repeat'],
    #            linestyles='dotted')
    # ax.vlines(df[df['switch'] == 'restart_post'].index, ymin=0, ymax=1, alpha=0.5, label='Restart Event', color=colors['restart'],
    #            linestyles='dotted')
    ax.legend()
    ax.set_ylim([0, 1.0])

# save frames
import glob
import os
import cv2
import pickle as pkl
import joblib

tag = 'mar_20_individual_depth_scene'
kinects = glob.glob(f'output/run_sem/{tag}/*inputdf_0.pkl')
kinects = list(set([os.path.basename(k).split('_')[0] for k in kinects]))
print(len(kinects))
for run_select in kinects:
    if os.path.exists(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_frames.joblib'):
        print(f'Already generated {run_select}')
        continue
    with open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_inputdf_0.pkl', 'rb') as f:
        inputdfs = pkl.load(f)
    vidfile = f'data/small_videos/{run_select}_kinect_trim.mp4'

    appear = inputdfs[0]
    frames = appear.index
    video_capture = cv2.VideoCapture()
    cached_videos = dict()
    if video_capture.open(vidfile):
        frame_id = 0
        while video_capture.isOpened():
            frame_id += 1
            ret, frame = video_capture.read()
            if not ret:
                print('End of video stream, ret is False!')
                break
            if frame_id in frames:
                cached_videos[frame_id] = cv2.resize(frame, None, fx=0.5, fy=0.5)
        if not os.path.exists(f'output/run_sem/{tag}/'):
            os.makedirs(f'output/run_sem/{tag}/')
        joblib.dump(cached_videos, f'output/run_sem/frames/{run_select}_kinect_trim{tag}_frames.joblib',
                    compress=True)
        # with open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_frames.pkl', 'wb') as f:
        #         pkl.dump(cached_videos, f)

import pickle as pkl
import glob


class DiagnosticResults:
    def __init__(self):
        pass


diagnostics = glob.glob('output/run_sem/*diagnostic.pkl')
for d in diagnostics:
    with open(f'{d}', 'rb') as f:
        sem_readouts = pkl.load(f)
    with open(f'{d}', 'wb') as f:
        inputdf = DiagnosticResults()
        inputdf.__dict__ = sem_readouts.__dict__
        pkl.dump(inputdf, f)

for all_lik, new_lik, repeat_lik in zip(sem_readouts['frame_dynamics']['old_lik'], sem_readouts['frame_dynamics']['new_lik'],
                                        sem_readouts['frame_dynamics']['repeat_lik']):
    print(all_lik, new_lik, repeat_lik)
    break

sem_readouts['frame_dynamics']['old_lik'] = [
    [l for l in all_lik if not (np.isclose(l, new_lik, rtol=1e-2) or np.isclose(l, repeat_lik, rtol=1e-2))]
    for all_lik, new_lik, repeat_lik in zip(sem_readouts['frame_dynamics']['old_lik'], sem_readouts['frame_dynamics']['new_lik'],
                                            sem_readouts['frame_dynamics']['repeat_lik'])]
sem_readouts['frame_dynamics']['old_lik'] = [l if len(l) else [-5000] for l in sem_readouts['frame_dynamics']['old_lik']]

# draw all interested metrics against epoch, not necessary anymore because we have scatter matrix.
import pandas as pd

df = pd.read_csv('output/run_sem/results_sem_run.csv')
grouped = df.groupby('tag')
agg = grouped[['bicorr', 'percentile']].describe()
grouped[['bicorr', 'percentile']].describe()[
    [('bicorr', 'mean'), ('bicorr', 'std'), ('percentile', 'mean'), ('percentile', 'std')]]

import matplotlib.pyplot as plt

y_interesteds = ['bicorr', 'percentile', 'model_boundaries', 'n_event_models', 'mean_pe', 'std_pe']
for y_interested in y_interesteds:

    tags = ['mar_01_like_21', 'mar_01_like_21_alfa100_lmda1e6', 'mar_01_like_21_gru']
    fig, ax = plt.subplots()
    for tag in tags:
        li = []
        for i in range(30):
            sr = df[(df['epoch'] == i) & (df['tag'] == f'{tag}')]
            if sr.shape[0] == 148:
                li.append(sr.mean())
        dump_df = pd.concat(li, axis=1).T
        dump_df.plot(kind='line', x='epoch', y=f'{y_interested}', ax=ax)
    ax.legend(tags)
    if y_interested == 'bicorr':
        ax.set_ylim((-0.1, 0.3))
    ax.set_ylabel(f'{y_interested}')
    plt.savefig(f'compare_{y_interested}.png')
    plt.close(fig)

#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# load data frame
df = pd.read_csv('output/run_sem/results_sem_run_pearson.csv')
df['chapter'] = df['run'].apply(lambda x: int(x[2]))
# indices_sorted = df[df['chapter'] == 2]['bicorr'].sort_values().index
# df.drop(indices_sorted[:30], inplace=True)
# Sometimes all runs haven't finished, filter epochs having all runs
# li = []
# for i in range(30):
#     sr = df[(df['epoch'] == i)]
#     if sr.shape[0] == 148:
#         li.append(sr.mean())
# dump_df = pd.concat(li, axis=1).T
# df = df[df['epoch'] < dump_df.index[-1]]
# Define interested metrics

numerics = ['mean_pe', 'pearson_r', 'epoch', 'n_event_models', 'number_boundaries']
# compare between tags
interested_tags = ['april_29_no_sm_same_seed_random_sequence_1', 'april_29_no_sm_same_seed_random_sequence_2',
                   'april_29_no_sm_same_seed_random_sequence_3', 'april_29_no_sm_same_seed_random_sequence_4',
                   'april_29_no_sm_same_seed_random_sequence_5']
df_select = df[df['tag'].isin(interested_tags)]
df_select = df[(df['tag'].isin(interested_tags)) & (df['is_train'] == False)]
# df_select['nh'] = np.select([df_select['tag'].str.contains('nh16'), ~df_select['tag'].str.contains('nh16')], [16, 32])
sns.pairplot(df_select[numerics + ['tag']], hue='tag', palette='bright',
             kind='reg', plot_kws={'scatter_kws': {'alpha': 0.3}})
# compare between chapters within a tag
interested_tags = ['april_12_scene_motion']
df_select = df[df['tag'].isin(interested_tags)]
sns.pairplot(df_select[numerics + ['chapter']], hue='chapter', palette='bright',
             kind='reg', plot_kws={'scatter_kws': {'alpha': 0.3}})
plt.savefig('scatter_matrix_chapter.png')
plt.show()
sns.pairplot(df[numerics], hue='epoch', palette='viridis', vars=numerics)
plt.savefig('scatter_matrix_epoch.png')
plt.show()
# Run regression model, accounting for chapter
chapter = pd.get_dummies(df['chapter'], prefix='chapter')
df_std = (df - df.mean()) / df.std()
df_std = pd.concat([df_std, chapter], axis=1)
# df = sm.add_constant(df)
# pe_model = sm.OLS(df_std['mean_pe'], df_std[['n_event_models', 'model_boundaries', 'epoch']]).fit()
b_model = sm.OLS(df_std['bicorr'], df_std[['n_event_models', 'epoch', 'chapter_1', 'chapter_2', 'chapter_3', 'chapter_4']]).fit()
b_model.summary()
# Run regression model, accounting for each run.
runs = pd.get_dummies(df['run'])
df_std = (df - df.mean()) / df.std()
df_std = (df_select - df_select.mean()) / df_select.std()
df_std = pd.concat([df_std, runs], axis=1)
# drop columns that I don't want to include in the model
df_run = df_std.drop(['chapter', 'mean_pe', 'std_pe', 'n_event_models', 'percentile'], axis=1)
b_model = sm.OLS(df_run['bicorr'], df_run.drop(['bicorr', 'model_boundaries'], axis=1)).fit()
b_model.summary()
# add pearson_r column
# THIS IS ONE OF THE EXAMPLES THAT FOR LONG-TERM USED METRICS, I SHOULDN'T BOTHER TO DO AD-HOC
import pickle as pkl
import pandas as np
import numpy as np
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from utils import get_point_biserial, get_binned_prediction


def get_pearson_r(run, tag, epoch):
    sem_readouts = pkl.load(
        open(f"output/run_sem/{run}_trim{tag}_diagnostic_{epoch}.pkl", 'rb'))
    inputdf = pkl.load(
        open(f"output/run_sem/{run}_trim{tag}_inputdf_{epoch}.pkl", 'rb'))
    # offset to align prediction boundaries with exact video timepoint
    first_frame = inputdf[0].index[0]
    fps = 25
    second_interval = 1
    pred_boundaries = get_binned_prediction(sem_readouts['post'], second_interval=second_interval,
                                            sample_per_second=3)
    pred_boundaries = np.hstack([[0] * round(first_frame / fps / second_interval), pred_boundaries])
    gt_freqs = pkl.load(open(f"output/run_sem/{run}_trim{tag}_gtfreqs.pkl", 'rb'))
    last = min(len(pred_boundaries), len(gt_freqs))
    bicorr = get_point_biserial(pred_boundaries[:last].astype(int), gt_freqs[:last])
    pred_boundaries_gaussed = gaussian_filter1d(pred_boundaries.astype(float), 1)
    r, p = stats.pearsonr(pred_boundaries_gaussed[:last], gt_freqs[:last])
    return r


df['pearson_r'] = df.apply(lambda x: get_pearson_r(x.run, x.feature_tag, x.current_epoch), axis=1)
# Showing that conditioning on n_event_models is a bad idea
agg = df.groupby('chapter').mean()
fig, axs = plt.subplots(ncols=2)
sns.scatterplot(x=df['epoch'], y=df['n_event_models'], hue=df['chapter'], palette='bright', ax=axs[0])
sns.scatterplot(x=agg.index, y=agg['bicorr'], hue=agg.index, palette='bright', ax=axs[1])
plt.show()
# just a handy line
df[df['tag'] == 'mar_04_individual_3'][select_columns].sort_values('pearson_r')

### FIXING mismatch tracking
# re-scale videos
# ffmpeg -i data\small_videos\1.3.4_kinect_trim.mp4 -s 960x540 -c:a copy 1.3.4_kinect_trim_960_540.mp4
# ffmpeg -i data\small_videos\6.2.1_kinect_trim.mp4 -s 960x540 -c:a copy 6.2.1_kinect_trim_960_540.mp4

# re-scale label files
import pandas as pd
import cv2
from shutil import copyfile

modified_runs = open('scaled_down_runs.txt', 'rt').readlines()
modified_runs = [r.strip() for r in modified_runs]
for run in modified_runs:
    capture = cv2.VideoCapture(f'data/small_videos/{run}_trim.mp4')
    height, width = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    if height != 540 or width != 960:
        print(f'Video dimension is {width}x{height}, not 960x540, skip!')
        continue
    df = pd.read_csv(f'data/ground_truth_labels/{run}_labels.csv')
    print(f"Label dimension is {df['width'].unique()}x{df['height'].unique()}, scaling to 960x540!")
    print(f'Copy label file to data/ground_truth_labels/{run}_labels_copy.csv for backup...')
    copyfile(f'data/ground_truth_labels/{run}_labels.csv', f'data/ground_truth_labels/{run}_labels_copy.csv')
    scale = 540 / df['height'].iloc[0]
    df['height'] = 540
    df['width'] = 960
    df['xmin'] = (df['xmin'] * scale).astype(int)
    df['xmax'] = (df['xmax'] * scale).astype(int)
    df['ymin'] = (df['ymin'] * scale).astype(int)
    df['ymax'] = (df['ymax'] * scale).astype(int)

    df.to_csv(f'data/ground_truth_labels/{run}_labels.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import plotly.express as px
import datetime


def my_parse(timestamp):
    h, m, s, ms = timestamp.split(':')
    return 3600 * int(h) + 60 * int(m) + int(s) + int(ms) / 1000


df = pd.read_excel('timing_start_stop_clap.xlsx', sheet_name=1)
df = df.rename(columns={"Scene\n(actor, chapter, run)": "Scene"})
df_select = pd.concat([df.filter(like='Scene'), df.filter(like='Camera'), df.filter(like="event")], axis=1)
interested_runs = ['1.2.3', '2.2.7', '6.2.2']
df_select = df_select[(df_select['Scene'].isin(interested_runs)) & (df_select['Camera'] == 'kinect')]
events = []
for i, test_series in df_select.iterrows():
    for e in range(1, 8):
        events.append(dict(Run=test_series.filter(like='Scene')[0],
                           Start=pd.to_datetime(test_series.filter(like=f'event{e}Start')[0][:-3]),
                           Finish=pd.to_datetime(test_series.filter(like=f'event{e}End')[0][:-3]),
                           # Start=my_parse(test_series.filter(like=f'event{e}Start')[0]),
                           # Finish=my_parse(test_series.filter(like=f'event{e}End')[0]),
                           Event=test_series[f'event{e}']))
df_con = pd.DataFrame(events)
fig = px.timeline(df_con, x_start="Start", x_end="Finish", y="Run", color="Event",
                  color_discrete_sequence=px.colors.qualitative.Dark24
                  )
midnight = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
offset = midnight + datetime.timedelta(0, 100)
fig.add_vline(x=b)
fig.update_yaxes(autorange="reversed")
fig.write_image('gantt.png')

import os
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import glob

df_pe = pd.DataFrame(columns=['run', 'epoch', 'pe', 'pe_w', 'pe_w2', 'pe_w3', 'pe_york'])
for i in range(20):
    files = glob.glob(f'output/run_sem/oct_11_multi_worlds_grid_lr1E-03_alfa1E-02_lmda1E+06/*diagnostic_{i}.pkl')
    if len(files):
        for f in files:
            sem_readouts = pkl.load(open(f, 'rb'))
            run = os.path.basename(f).split('_')[0]
            epoch = f.split('_')[-1].split('.')[0]
            df_pe.loc[len(df_pe)] = [run, epoch, sem_readouts['pe'].mean(), sem_readouts['pe_w'].mean(),
                                     sem_readouts['pe_w2'].mean(), sem_readouts['pe_w3'].mean(), sem_readouts['pe_york'].mean()]
    else:
        print(f'no files for epoch {i}')

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(4 * 5, 4), squeeze=False, sharex=True, sharey=True)
i = 0
df_pe['epoch'] = df_pe['epoch'].astype(int)
for r in df_pe.columns:
    if 'pe' in r:
        sns.regplot(x='epoch', y=r, data=df_pe, ax=axes[0][i], lowess=True)
        axes[0][i].set_title(f'{r}')
        i += 1
plt.savefig('multi_worlds.png')

# scree plots
import pandas as pd
import pickle as pkl
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

tag = 'dec_6_rotated_skel'
files = glob.glob(f'output/run_sem/{tag}/*.pkl')
pca_tag = 'all'
sample = 500


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
    print(f'Path={path}, len={len(input_df)}')
    return input_df.sample(n=sample)


input_dfs = Parallel(n_jobs=4)(delayed(load_and_sample)(path, sample=sample) for path in files)


def fit_and_plot(df, components, title=''):
    pca = PCA(n_components=len(df.columns), whiten=True).fit(df)
    sns.lineplot(data=pca.explained_variance_ratio_)
    plt.title(f'{title} \n' + f'Ori={len(df.columns)}, Cut={components}, ' +
              f'Var_explained={pca.explained_variance_ratio_[:components].sum():.2f}')
    plt.savefig(f'{title}.png')
    plt.close()
    # plt.show()


combined_runs = pd.concat(input_dfs, axis=0)
appear_df = pd.concat([input_df.iloc[:, :2] for input_df in input_dfs], axis=0)
optical_df = pd.concat([input_df.iloc[:, 2:4] for input_df in input_dfs], axis=0)
skel_df = pd.concat([input_df.iloc[:, 4:-100] for input_df in input_dfs], axis=0)
emb_df = pd.concat([input_df.iloc[:, -100:] for input_df in input_dfs], axis=0)
fit_and_plot(combined_runs, 30, title='PCA for Concatenated Features')
fit_and_plot(skel_df, 14, title='PCA for Skel Features')
fit_and_plot(emb_df, 13, title='PCA for ObjHand+AllObj Features')
fit_and_plot(appear_df, 2, title='PCA for Appear')
fit_and_plot(optical_df, 1, title='PCA for Optical')

# get categories
import glob
import os
import pandas as pd

files = glob.glob('output/objhand/*.csv')
run_to_categories = dict()
for f in files:
    df = pd.read_csv(f)
    cat = df.filter(regex='dist').columns
    run_to_categories[os.path.basename(f).split('_')[0] + '_kinect'] = set([''.join(c for c in x.split('_')[0] if not c.isdigit())
                                                                            for x in cat])

import pickle as pkl
pkl.dump(run_to_categories, open('visualization/scene_categories.pkl', 'wb'))

## Compare PE across tags among different components
# define tag and pca tag
import pandas as pd
import glob
import pickle as pkl
import numpy as np



# tag = 'jan_25_lr_normal_grid_lr1E-03_alfa1E-01_lmda1E+05'
pca_tag = 'dec_6_rotated_skel_all_30'

# load pca to get n_components
pca_appear = pkl.load(open(f'{pca_tag}_appear_pca.pkl', 'rb'))
pca_optical = pkl.load(open(f'{pca_tag}_optical_pca.pkl', 'rb'))
pca_skel = pkl.load(open(f'{pca_tag}_skel_pca.pkl', 'rb'))
pca_emb = pkl.load(open(f'{pca_tag}_emb_pca.pkl', 'rb'))
indices = [pca_appear.n_components,
           pca_appear.n_components + pca_optical.n_components,
           pca_appear.n_components + pca_optical.n_components + pca_skel.n_components,
           pca_appear.n_components + pca_optical.n_components + pca_skel.n_components + pca_emb.n_components]

# create a dataframe to cache
pe_types = ['pe', 'pe_w', 'pe_w2']
df_pe = pd.DataFrame(columns=['run', 'tag', 'epoch'] +
                             sum([[f'{pe_type}_appear', f'{pe_type}_optical', f'{pe_type}_skel', f'{pe_type}_emb', f'{pe_type}']
                                 for pe_type in pe_types], []))

tag = 'jan_10_no_skel_grid_lr1E-03_alfa1E-01_lmda1E+05'
# for each epoch, glob all existed runs
for e in range(30):
    files = glob.glob(f'output/run_sem/{tag}/*{tag}*inputdf_{e}.pkl')
    # manipulate each file
    for f in files:
        print(f'Processing file {f}')
        inputdf = pkl.load(open(f'{f}', 'rb'))
        diag = pkl.load(open(f"{f.replace('inputdf', 'diagnostic')}", 'rb'))
        run = f.split('/')[-1].split('_')[0]

        # split pe into typed pe
        def split_pe(name='pe'):
            pe = (diag[f"x_hat{name.replace('pe', '')}"] * np.sqrt(inputdf.x_train_pca.shape[1])) - inputdf.x_train_pca
            pe_array = pe.apply(np.linalg.norm, axis=0).to_numpy()
            pe_appear = pe_array[:indices[0]]
            pe_optical = pe_array[indices[0]:indices[1]]
            pe_skel = pe_array[indices[1]:indices[2]]
            pe_emb = pe_array[indices[2]:]
            return np.average(pe_appear), np.average(pe_optical), np.average(pe_skel), np.average(pe_emb), np.average(pe_array)
        concat_pe = []
        # compute typed pe and add to a dataframe
        for name in pe_types:
            pe_appear, pe_optical, pe_skel, pe_emb, pe_all = split_pe(name)
            concat_pe.extend([pe_appear, pe_optical, pe_skel, pe_emb, pe_all])
        row = [f'{run}', f'{tag}', f'{e}'] + concat_pe
        # print(f'appending {row}')
        df_pe.loc[len(df_pe.index)] = row
df_pe.to_csv(f'df_pe_{tag}.csv', index=False)

# plotting
df_new = pd.read_csv(f'df_pe_{tag}.csv', index_col=None)
pe1 = 'pe'
pe2 = 'pe_w'
df_new.plot(
    kind='scatter',
    x='epoch',
    y=[f'{pe1}', f'{pe2}'],
    backend='plotly',
    width=500,
    trendline="lowess",
).write_image(f'{pe1}_and_{pe2}.png')


import pandas as pd
import os
import plotly.express as px
import cv2
import numpy as np
from utils import SegmentationVideo, get_point_biserial

grain = 'coarse'

df = pd.read_csv('output/run_sem/results_purity_coverage.csv')
df = df[~df['tag'].isna()]
df = df[~df['bicorr'].isna()]
df = df.dropna(axis=0)
df = df[(df.feature_tag == 'feb_11_cleaned_segmentation_grid_lr1E-03_alfa1E-01_lmda1E+05') & (df.grain == grain)]

if not os.path.exists('output/low_correlation_runs'):
    os.makedirs('output/low_correlation_runs')

data_frame = pd.read_csv('seg_data_analysis_clean.csv')
df_compare = pd.DataFrame(columns=['bicorr', 'type', 'run'])
for e in range(50, 52):
    # df_e = df[(df['percentile'] < 5) & (df['epoch'] == e)]
    df_e = df[(df['epoch'] == e)]
    for run in list(set(df_e.run)):
        video_path = run + '_trim.mp4'
        seg_video = SegmentationVideo(data_frame=data_frame, video_path=video_path)
        seg_video.get_human_segments(n_annotators=100, condition=grain, second_interval=1)
        # this function aggregate subject boundaries, apply a gaussian kernel and calculate correlations for subjects
        capture = cv2.VideoCapture(os.path.join('data/small_videos/', f'{video_path}'))
        end_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        last = int(end_frame / fps)
        number_boundaries = int(df_e[df_e.run == run]['number_boundaries'])
        biserials = seg_video.get_biserial_subjects(second_interval=1, end_second=last)
        for b in biserials:
            df_compare.loc[len(df_compare), :] = [b, 'human', run]
            random_boundaries = np.zeros(shape=(last,), dtype=bool)
            random_indices = np.random.choice(range(last), size=number_boundaries, replace=False)
            random_boundaries[random_indices] = 1
            b = get_point_biserial(random_boundaries, seg_video.gt_freqs)
            df_compare.loc[len(df_compare), :] = [b, 'random', run]
        df_compare.loc[len(df_compare), :] = [float(df_e[df_e.run == run].bicorr), 'sem', run]

# df = df[(df['epoch'] <= 51) & df.epoch >= 31]
fig = px.strip(df_compare, y='bicorr', x='type', color='run')

import glob
track_complete = open('track_complete.txt', 'r').readlines()
track_complete = [x for x in track_complete if "Stats" not in x]
track_complete = [x.strip() for x in track_complete]
all_tracks = open('all_runs.txt', 'r').readlines()
all_tracks = [x.strip() for x in all_tracks if "C1" in x]
next_8 = list(set(all_tracks).difference(set(track_complete)))[:8]
open('next_8.txt', 'w').writelines('\n'.join(next_8))

import cv2
import pandas as pd
import numpy as np
from utils import CV2VideoReader
from joblib import Parallel, delayed
movies = ['6.3.9_C1_trim.mp4', '2.4.1_C1_trim.mp4', '3.1.3_C1_trim.mp4', '1.2.3_C1_trim.mp4']
# for m in movies:
def gen_stats(m):
    df = pd.DataFrame(columns=['time(s)', 'pixel_change_mean', 'pixel_change_var', 'luminance_mean', 'luminance_var'])
    cv2_video_reader = CV2VideoReader('data/small_videos/' + m)
    fps = cv2_video_reader.fps
    frame_id = 0
    prev_frame = None
    while cv2_video_reader.capture.isOpened():
        frame_id += 1
        # if frame_id > 400:
        #     break
        ret, frame = cv2_video_reader.read_frame()
        if not ret:
            print('End of video stream, ret is False!')
            break
        if frame_id > 1:
            lum = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
            pixel_change = frame - prev_frame
            df.loc[len(df)] = [frame_id / fps, np.mean(pixel_change), np.var(pixel_change), np.mean(lum), np.var(lum)]
        prev_frame = frame
    df.to_csv(f'{m[:-4]}.csv', index=False)
    print(f'Done {m}!')
    cv2_video_reader.capture.release()
Parallel(n_jobs=4)(delayed(gen_stats)(m) for m in movies)

import glob
import os

features = ['vid', 'skel', 'appear', 'objhand']
for f in features:
    files = glob.glob(f'output/{f}/*.csv')
    for x in files:
        if 'kinect_objhand' in x or 'kinect_skel' in x or 'kinect_video' in x or 'kinect_appear' in x:
            os.rename(x, x[:x.find('t_')+1] + '_july_18' + x[x.find('t_')+1:])

# upload files
import subprocess
file_names = glob.glob('output/run_sem/frames/*frames.joblib')
# file_names = file_names[:4]


def upload(name):
    print(f'Uploading {name}...')
    subprocess.run(['scp', f'{name}', 'n.tan@login3-02.chpc.wustl.edu:/scratch/n.tan/extended-event-modeling/output/run_sem/frames/'])

    print(f'Done {name}!')


Parallel(n_jobs=8)(delayed(upload)(file_name) for file_name in file_names)
