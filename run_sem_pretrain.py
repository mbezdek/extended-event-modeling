import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import math
import os
import json
import sys
import traceback
import csv
import pickle as pkl
import re
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats

sys.path.append('../pysot')
sys.path.append('../SEM2')
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore
from sem.event_models import GRUEvent, LinearEvent, LSTMEvent
from sem import SEM
from utils import SegmentationVideo, get_binned_prediction, get_point_biserial, \
    logger, parse_config, contain_substr
from joblib import Parallel, delayed
import gensim.downloader
import random
from typing import List, Dict

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
with open('gen_sim_glove_50.pkl', 'rb') as f:
    glove_vectors = pkl.load(f)
# glove_vectors = gensim.downloader.load('word2vec-ruscorpora-300')
emb_dim = glove_vectors['apple'].size


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(int).astype(int)
    return appear_df


def preprocess_optical(vid_csv, standardize=True):
    vid_df = pd.read_csv(vid_csv, index_col='frame')
    # vid_df.drop(['pixel_correlation'], axis=1, inplace=True)
    for c in vid_df.columns:
        if not standardize:
            vid_df.loc[:, c] = (vid_df[c] - min(vid_df[c].dropna())) / (
                    max(vid_df[c].dropna()) - min(vid_df[c].dropna()))
        else:
            vid_df.loc[:, c] = (vid_df[c] - vid_df[c].dropna().mean()) / vid_df[
                c].dropna().std()
    return vid_df


def preprocess_skel(skel_csv, use_position=False, standardize=True):
    skel_df = pd.read_csv(skel_csv, index_col='frame')
    skel_df.drop(['sync_time', 'raw_time', 'body', 'J1_dist_from_J1'], axis=1, inplace=True)
    if use_position:
        keeps = ['accel', 'speed', 'dist', 'interhand', '3D', '2D']
    else:
        keeps = ['accel', 'speed', 'dist', 'interhand']
    for c in skel_df.columns:
        if contain_substr(c, keeps):
            if not standardize:
                skel_df.loc[:, c] = (skel_df[c] - min(skel_df[c].dropna())) / (
                        max(skel_df[c].dropna()) - min(skel_df[c].dropna()))
            else:
                skel_df.loc[:, c] = (skel_df[c] - skel_df[c].dropna().mean()) / skel_df[
                    c].dropna().std()
        else:
            skel_df.drop([c], axis=1, inplace=True)

    return skel_df


def remove_number(string):
    for i in range(100):
        string = string.replace(str(i), '')
    return string


def get_emb_category(category_distances, emb_dim=100):
    # Add 1 to avoid 3 objects with 0 distances (rare but might happen), then calculate inverted weights
    category_distances = category_distances + 1
    # Add 1 to avoid cases there is only one object
    category_weights = 1 - category_distances / (category_distances.sum() + 1)
    if category_weights.sum() == 0:
        logger.error('Sum of probabilities is zero')
    average = np.zeros(shape=(1, emb_dim))
    for category, prob in category_weights.iteritems():
        r = np.zeros(shape=(1, emb_dim))
        try:
            r += glove_vectors[category]
        except Exception as e:
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        average += r * prob
    return average / category_weights.sum()


def get_embeddings(objhand_df: pd.DataFrame, emb_dim=100, num_objects=3):
    scene_embs = np.zeros(shape=(0, emb_dim))
    obj_handling_embs = np.zeros(shape=(0, emb_dim))
    categories = pd.DataFrame()

    for index, row in objhand_df.iterrows():
        all_categories = list(row.index[row.notna()])
        if len(all_categories):
            # averaging all objects
            # scene_embedding = np.zeros(shape=(0, emb_dim))
            # for category in all_categories:
            #     cat_emb = get_emb_category([category], emb_dim)
            #     scene_embedding = np.vstack([scene_embedding, cat_emb])
            # scene_embedding = np.mean(scene_embedding, axis=0).reshape(1, emb_dim)
            scene_embedding = get_emb_category(row.dropna().sort_values()[:7], emb_dim)

            # pick the nearest object
            nearest = row.argmin()
            assert nearest != -1
            # obj_handling_emb = get_emb_category(row.index[nearest], emb_dim)
            new_row = pd.Series(data=row.dropna().sort_values().index[:num_objects], name=row.name)
            categories = categories.append(new_row)
            obj_handling_emb = get_emb_category(row.dropna().sort_values()[:num_objects], emb_dim)
        else:
            scene_embedding = np.full(shape=(1, emb_dim), fill_value=np.nan)
            obj_handling_emb = np.full(shape=(1, emb_dim), fill_value=np.nan)
        scene_embs = np.vstack([scene_embs, scene_embedding])
        obj_handling_embs = np.vstack([obj_handling_embs, obj_handling_emb])
    return scene_embs, obj_handling_embs, categories


def preprocess_objhand(objhand_csv, standardize=True, use_depth=False, num_objects=3):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')
    # Drop non-distance columns
    for c in objhand_df.columns:
        if 'dist' not in c:
            objhand_df.drop([c], axis=1, inplace=True)
    if use_depth:
        objhand_df = objhand_df.filter(regex=f'_dist_z$')
    else:
        objhand_df = objhand_df.filter(regex=f'_dist$')
    # remove track number
    objhand_df.rename(remove_number, axis=1, inplace=True)
    all_categories = set(objhand_df.columns.values)
    # get neareast distance for the same category at each row
    for category in all_categories:
        if isinstance(objhand_df[category], pd.Series):
            if use_depth:
                objhand_df[category.replace('_dist_z', '')] = objhand_df[category]
            else:
                objhand_df[category.replace('_dist', '')] = objhand_df[category]
        else:
            if use_depth:
                objhand_df[category.replace('_dist_z', '')] = objhand_df[category].apply(lambda x: np.min(x), axis=1)
            else:
                # For some strange reason, min(89, NaN) could be 89 or NaN
                objhand_df[category.replace('_dist', '')] = objhand_df[category].apply(lambda x: np.min(x), axis=1)
            # objhand_df.loc[:, c] = (objhand_df[c] - min(objhand_df[c].dropna())) / (
            #         max(objhand_df[c].dropna()) - min(objhand_df[c].dropna()))
    # remove redundant columns
    for c in set(objhand_df.columns):
        if 'dist' in c:
            objhand_df.drop([c], axis=1, inplace=True)
    # normalize
    # TODO: currently, many values exceed 5000, need debug in projected_skeletons
    # mean = np.nanmean(objhand_df.values)
    # std = np.nanstd(objhand_df.values)
    # objhand_df = (objhand_df - mean) / std
    scene_embs, obj_handling_embs, categories = get_embeddings(objhand_df, emb_dim=emb_dim, num_objects=num_objects)
    if standardize:
        scene_embs = (scene_embs - np.nanmean(scene_embs, axis=0)) / np.nanstd(scene_embs,
                                                                               axis=0)
        obj_handling_embs = (obj_handling_embs - np.nanmean(obj_handling_embs,
                                                            axis=0)) / np.nanstd(
            obj_handling_embs, axis=0)
    scene_embs = pd.DataFrame(scene_embs, index=objhand_df.index,
                              columns=list(map(lambda x: f'scene_{x}', range(emb_dim))))
    obj_handling_embs = pd.DataFrame(obj_handling_embs, index=objhand_df.index,
                                     columns=list(
                                         map(lambda x: f'objhand_{x}', range(emb_dim))))
    return scene_embs, obj_handling_embs, categories


def interpolate_frame(dataframe: pd.DataFrame):
    first_frame = dataframe.index[0]
    last_frame = dataframe.index[-1]
    dummy_frame = pd.DataFrame(np.NaN, index=range(first_frame, last_frame),
                               columns=dataframe.columns)
    dummy_frame = dummy_frame.combine_first(dataframe).interpolate(limit_area='inside')
    return dummy_frame


def pca_dataframe(dataframe: pd.DataFrame):
    dataframe.dropna(axis=0, inplace=True)
    pca = PCA(args.pca_explained, whiten=True)
    dummy_array = pca.fit_transform(dataframe.values)

    return pd.DataFrame(dummy_array)


def combine_dataframes(data_frames, rate='40ms', fps=30):
    # Some features such as optical flow are calculated not for all frames, interpolate first
    data_frames = [interpolate_frame(df) for df in data_frames]
    combine_df = pd.concat(data_frames, axis=1)
    # After dropping null values, variances are not unit anymore, some are around 0.8.
    combine_df.dropna(axis=0, inplace=True)
    first_frame = combine_df.index[0]
    combine_df['frame'] = combine_df.index
    # After resampling, some variances drop to 0.3 or 0.4
    combine_df = resample_df(combine_df, rate=rate, fps=fps)
    # because resample use mean, need to adjust categorical variables
    combine_df['appear'] = combine_df['appear'].apply(math.ceil).astype(float)
    combine_df['disappear'] = combine_df['disappear'].apply(math.ceil).astype(float)
    # Add readout to visualize
    data_frames = [combine_df[df.columns] for df in data_frames]
    for df in data_frames:
        df.index = combine_df['frame'].apply(round)

    assert combine_df.isna().sum().sum() == 0
    combine_df.drop(['sync_time', 'frame'], axis=1, inplace=True, errors='ignore')
    return combine_df, first_frame, data_frames


def plot_subject_model_boundaries(gt_freqs, pred_boundaries, title='', save_fig=True,
                                  show=True, bicorr=0.0, percentile=0.0):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.title(title)
    b = np.arange(len(pred_boundaries))[pred_boundaries][0]
    plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75, color='b')
    for b in np.arange(len(pred_boundaries))[pred_boundaries][1:]:
        plt.plot([b, b], [0, 1], 'k:', alpha=0.75, color='b')

    plt.text(0.1, 0.3, f'bicorr={bicorr:.3f}, perc={percentile:.3f}', fontsize=14)
    plt.legend(loc='upper left')
    plt.ylim([0, 0.4])
    sns.despine()
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()
    plt.close()


def plot_diagnostic_readouts(gt_freqs, sem_readouts, frame_interval=3.0, offset=0.0, title='', show=False, save_fig=True,
                             bicorr=0.0, percentile=0.0, pearson_r=0.0):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.ylim([0, 0.4])
    plt.title(title)
    plt.text(0.1, 0.3, f'bicorr={bicorr:.3f}, perc={percentile:.3f}, pearson={pearson_r:.3f}', fontsize=14)
    colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}

    cm = plt.get_cmap('gist_rainbow')
    post = sem_readouts.e_hat
    boundaries = sem_readouts.boundaries
    # NUM_COLORS = post.max()
    # Hard-code 40 events for rainbow to be able to compare across events
    NUM_COLORS = 30
    """
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
    """
    for i, (b, e) in enumerate(zip(boundaries, post)):
        if b != 0:
            second = i / frame_interval + offset
            if b == 1:
                plt.axvline(second, linestyle=(0, (5, 10)), alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Old Event')
            elif b == 2:
                plt.axvline(second, linestyle='solid', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='New Event')
            elif b == 3:
                plt.axvline(second, linestyle='dotted', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Restart Event')
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(vmin=0, vmax=NUM_COLORS, clip=False)),
                 orientation='horizontal')
    from matplotlib.lines import Line2D
    linestyles = ['dashed', 'solid', 'dotted']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=ls) for ls in linestyles]
    labels = ['Old Event', 'New Event', 'Restart Event']
    plt.legend(lines, labels)
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()


def plot_pe(sem_readouts, frame_interval, offset, title):
    fig, ax = plt.subplots()
    df = pd.DataFrame({'prediction_error': sem_readouts.pe}, index=range(len(sem_readouts.pe)))
    df['second'] = df.index / frame_interval + offset
    df.plot(kind='line', x='second', y='prediction_error', alpha=1.00, ax=ax)
    plt.savefig('output/run_sem/' + title + '.png')


def resample_df(objhand_df, rate='40ms', fps=30):
    # fps matter hear, we need feature vector at anchor timepoints to correspond to segments
    outdf = objhand_df.set_index(pd.to_datetime(objhand_df.index / fps, unit='s'), drop=False,
                                 verify_integrity=True)
    resample_index = pd.date_range(start=outdf.index[0], end=outdf.index[-1], freq=rate)
    dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=outdf.columns)
    outdf = outdf.combine_first(dummy_frame).interpolate(method='time', limit_area='inside').resample(rate).mean()
    return outdf


def merge_feature_lists(txt_out):
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
    with open(txt_out, 'w') as f:
        f.writelines(sem_runs)


class SEMContext:
    """
    This class maintain global variables for SEM training and inference
    """

    def __init__(self, sem_model=None, run_kwargs=None, tag='', configs=None):
        self.sem_model = sem_model
        self.run_kwargs = run_kwargs
        self.tag = tag
        if not os.path.exists(f'output/run_sem/{self.tag}'):
            os.makedirs(f'output/run_sem/{self.tag}')
        self.configs = configs
        self.epochs = int(self.configs.epochs)
        self.train_stratified = int(self.configs.train_stratified)
        self.second_interval = 1
        self.sample_per_second = 1000 / float(self.configs.rate.replace('ms', ''))

        self.current_epoch = None
        # These variables will be set based on each run
        self.run = None
        self.movie = None
        self.title = None
        self.grain = None
        # FPS is used to pad prediction boundaries, should be inferred from run
        self.fps = None
        self.is_train = None  # this variable is set depending on mode
        # These variables will be set by read_train_valid_list
        self.train_list = None
        self.train_dataset = None
        self.valid_dataset = None
        # These variables will bet set after processing features
        self.first_frame = None
        self.last_frame = None
        self.end_second = None
        self.data_frames = None
        self.combine_df = None
        self.categories = None
        self.categories_z = None

    def iterate(self, is_eval=True):
        for e in range(self.epochs):
            self.current_epoch = e
            logger.info('Training')
            if self.train_stratified:
                self.train_dataset = self.train_list[e % 8]
            else:
                self.train_dataset = self.train_list
            self.training()
            if is_eval:
                logger.info('Evaluating')
                self.evaluating()

    def training(self):
        # Randomize order of video
        random.shuffle(self.train_dataset)
        for index, run in enumerate(self.train_dataset):
            self.is_train = True
            logger.info(f'Training video {run}')
            self.set_epoch_variables(run)
            self.infer_on_video(store_dataframes=True)

    def evaluating(self):
        # Randomize order of video
        random.shuffle(self.valid_dataset)
        for index, run in enumerate(self.valid_dataset):
            self.is_train = False
            logger.info(f'Evaluating video {run}')
            self.set_epoch_variables(run)
            self.infer_on_video(store_dataframes=True)

    def parse_input(self, token, is_stratified=0) -> List:
        # Whether the train.txt or 4.4.4_kinect
        if '.txt' in token:
            with open(token, 'r') as f:
                list_input = f.readlines()
                list_input = [x.strip() for x in list_input]
            if is_stratified:
                assert '.txt' in list_input[0], f"Attempting to stratify individual {list_input}"
                return [self.parse_input(i) for i in list_input]
            else:
                # sum(list, []) to remove nested list
                return sum([self.parse_input(i) for i in list_input], [])
        else:
            # degenerate case, e.g. 4.4.4_kinect
            return [token]

    def read_train_valid_list(self):
        self.valid_dataset = self.parse_input(self.configs.valid)
        self.train_list = self.parse_input(self.configs.train, is_stratified=self.train_stratified)

    def set_epoch_variables(self, run):
        self.run = run
        self.movie = run + '_trim.mp4'
        self.title = os.path.join(self.tag, os.path.basename(self.movie[:-4]) + self.tag)
        self.grain = 'coarse'
        # FPS is used to pad prediction boundaries, should be inferred from run
        if 'kinect' in run:
            self.fps = 25
        else:
            self.fps = 30

    def infer_on_video(self, store_dataframes=True):
        try:
            self.process_features()

            if store_dataframes:
                # Infer coordinates from nearest categories and add both to data_frame for visualization
                objhand_csv = os.path.join(self.configs.objhand_csv, self.run + '_objhand.csv')
                objhand_df = pd.read_csv(objhand_csv)
                objhand_df = objhand_df.loc[self.data_frames[0].index, :]

                def add_category_and_coordinates(categories: pd.DataFrame, use_depth=False):
                    # Readout to visualize object-hand features
                    # Re-index: some frames there are no objects near hand (which is not possible, this bug is due to min(89, NaN)=NaN
                    # categories = categories.reindex(range(categories.index[-1])).ffill()
                    categories = categories.ffill()
                    categories = categories.loc[self.data_frames[0].index, :]
                    self.data_frames.append(categories)
                    # coordinates variable is determined by categories variable, thus having only 3 objects
                    coordinates = pd.DataFrame()
                    for index, r in categories.iterrows():
                        frame_series = pd.Series(dtype=float)
                        # There could be a case where there are two spray bottles near the hand: 6.3.6
                        # When num_objects is large, there are nan in categories -> filter
                        # TODO: filter seen instances
                        all_categories = set(r.dropna().values)
                        for c in list(all_categories):
                            # Filter by category name and select distance
                            # Note: paper towel and towel causes duplicated columns in series,
                            # Need anchor ^ to distinguish towel and paper towel (2.4.7),
                            # need digit \d to distinguish pillow0 and pillowcase0 (3.3.5)
                            # Need to escape character ( and ) in aloe (green bottle) (4.4.5)
                            # Either use xy or depth (z) distance to get nearest object names
                            if use_depth:
                                df = objhand_df.loc[index, :].filter(regex=f"^{re.escape(c)}\d").filter(regex='_dist_z$')
                                nearest_name = df.index[df.argmin()].replace('_dist_z',
                                                                             '')  # e.g. pillow0, pillowcase0, towel0, paper towel0
                            else:
                                df = objhand_df.loc[index, :].filter(regex=f"^{re.escape(c)}\d").filter(regex='_dist$')
                                nearest_name = df.index[df.argmin()].replace('_dist',
                                                                             '')  # e.g. pillow0, pillowcase0, towel0, paper towel0
                            # select nearest object's coordinates
                            # need anchor ^ to distinguish between towel0 and paper towel0
                            s = objhand_df.loc[index, :].filter(regex=f"^{re.escape(nearest_name)}")
                            frame_series = frame_series.append(s)
                        frame_series.name = index
                        coordinates = coordinates.append(frame_series)
                    self.data_frames.append(coordinates)

                add_category_and_coordinates(self.categories, use_depth=False)

            # logger.info(f'Features: {self.combine_df.columns}')
            # Note that without copy=True, this code will return a view and subsequent changes to x_train will change self.combine_df
            # e.g. x_train /= np.sqrt(x_train.shape[1]) or x_train[2] = ...
            x_train = self.combine_df.to_numpy(copy=True)
            # PCA transform input features. Also, get inverted vector for visualization
            if int(self.configs.pca):
                # pca = PCA(float(self.configs.pca_explained), whiten=True)
                pca = PCA(int(self.configs.pca_dim), whiten=True)
                try:
                    x_train_pca = pca.fit_transform(x_train[:, 2:])
                    x_train_inverted = pca.inverse_transform(x_train_pca)
                except Exception as e:
                    print(repr(e))
                    x_train_pca = pca.fit_transform(x_train[:, 2:])
                    x_train_inverted = pca.inverse_transform(x_train_pca)
                x_train = np.hstack([x_train[:, :2], x_train_pca])
                x_train_inverted = np.hstack([x_train[:, :2], x_train_inverted])
                df_x_train_inverted = pd.DataFrame(data=x_train_inverted, index=self.data_frames[0].index,
                                                   columns=self.combine_df.columns)
                self.data_frames.append(df_x_train_inverted)
            else:
                df_x_train = pd.DataFrame(data=x_train, index=self.data_frames[0].index, columns=self.combine_df.columns)
                self.data_frames.append(df_x_train)

            # Note that this is different from x_train = x_train / np.sqrt(x_train.shape[1]), the below code will change values of
            # the memory allocation, to which self.combine_df refer -> change to be safer
            # x_train /= np.sqrt(x_train.shape[1])
            # x_train is already has unit variance for all features (pca whitening) -> scale to have unit length.
            # I read in SEM's comment that it should be useful to have unit length stimulus.
            x_train = x_train / np.sqrt(x_train.shape[1])
            # appear, x_train = np.split(x_train, [2], axis=1)  # remove appear features
            # This function train and change sem event models
            self.run_sem_and_plot(x_train)
            if store_dataframes:
                # Transform predicted vectors to the original vector space for visualization
                if int(self.configs.pca):
                    x_inferred_inverted = self.sem_model.results.x_hat
                    # x_inferred_inverted = np.hstack([appear, x_inferred_inverted])  # concat appear feature as if it's used for consistency
                    # Scale back to PCA whitening results
                    x_inferred_inverted = x_inferred_inverted * np.sqrt(x_train.shape[1])
                    x_inferred_inverted = pca.inverse_transform(x_inferred_inverted[:, 2:])
                    x_inferred_inverted = np.hstack([x_inferred_inverted[:, :2], x_inferred_inverted])
                    df_x_inferred_inverted = pd.DataFrame(data=x_inferred_inverted, index=self.data_frames[0].index,
                                                          columns=self.combine_df.columns)
                    self.data_frames.append(df_x_inferred_inverted)
                else:
                    x_inferred_ori = self.sem_model.results.x_hat * np.sqrt(x_train.shape[1])
                    df_x_inferred_ori = pd.DataFrame(data=x_inferred_ori, index=self.data_frames[0].index,
                                                     columns=self.combine_df.columns)
                    self.data_frames.append(df_x_inferred_ori)

                # Adding categories_z and coordinates_z to data_frame
                # after training to avoid messing up with indexing in input_viz_refactored.ipynb
                # adding categories_z here instead of above to avoid messing up with indexing of inputdf
                # TODO: uncomment this line after getting objhand features with depth
                add_category_and_coordinates(self.categories_z, use_depth=True)
                with open('output/run_sem/' + self.title + f'_inputdf_{self.current_epoch}.pkl', 'wb') as f:
                    pkl.dump(self.data_frames, f)

            logger.info(f'Done SEM {self.run}')
            with open('output/run_sem/sem_complete.txt', 'a') as f:
                f.write(self.run + f'_{tag}' + '\n')
            # sem's Results() is initialized and different for each run
        except Exception as e:
            with open('output/run_sem/sem_error.txt', 'a') as f:
                f.write(self.run + f'_{tag}' + '\n')
                f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc())

    def process_features(self):
        """
        This method load pre-processed features then combine and align them temporally
        :return:
        """
        # For some reason, some optical flow videos have inf value, TODO: investigate
        pd.set_option('use_inf_as_na', True)
        logger.info(f'Config {self.configs}')

        objhand_csv = os.path.join(self.configs.objhand_csv, self.run + '_objhand.csv')
        skel_csv = os.path.join(self.configs.skel_csv, self.run + '_skel_features.csv')
        appear_csv = os.path.join(self.configs.appear_csv, self.run + '_appear.csv')
        optical_csv = os.path.join(self.configs.optical_csv, self.run + '_video_features.csv')

        # Load csv files and preprocess to get a scene vector
        appear_df = preprocess_appear(appear_csv)
        skel_df = preprocess_skel(skel_csv, use_position=int(self.configs.use_position), standardize=True)
        optical_df = preprocess_optical(optical_csv, standardize=True)
        # _, obj_handling_embs, categories = preprocess_objhand(objhand_csv, standardize=True, use_depth=False)
        _, _, categories = preprocess_objhand(objhand_csv, standardize=True, use_depth=False,
                                              num_objects=int(self.configs.num_objects))
        # TODO: uncomment to test depth
        # _, _, categories_z = preprocess_objhand(objhand_csv, standardize=True, use_depth=True)
        _, obj_handling_embs, categories_z = preprocess_objhand(objhand_csv, standardize=True, use_depth=True,
                                                                num_objects=int(self.configs.num_objects))
        # Get consistent start-end times and resampling rate for all features
        combine_df, first_frame, data_frames = combine_dataframes([appear_df, optical_df, skel_df, obj_handling_embs],
                                                                  rate=self.configs.rate, fps=self.fps)
        self.first_frame = first_frame
        self.last_frame = data_frames[0].index[-1]
        # This parameter is used to limit the time of ground truth video according to feature data
        self.end_second = math.ceil(self.last_frame / self.fps)
        self.data_frames = data_frames
        self.combine_df = combine_df
        self.categories = categories
        self.categories_z = categories_z

    def run_sem_and_plot(self, x_train):
        """
        This method run SEM and plot
        :param x_train:
        :return:
        """
        self.sem_model.run(x_train, train=self.is_train, **self.run_kwargs)
        # Process results returned by SEM
        pred_boundaries = get_binned_prediction(self.sem_model.results.post, second_interval=self.second_interval,
                                                sample_per_second=self.sample_per_second)
        # Padding prediction boundaries, could be changed to have higher resolution but not necessary
        pred_boundaries = np.hstack([[0] * round(self.first_frame / self.fps / self.second_interval), pred_boundaries]).astype(
            int)
        logger.info(f'Total # of pred_boundaries: {sum(pred_boundaries)}')
        with open('output/run_sem/' + self.title + f'_diagnostic_{self.current_epoch}.pkl', 'wb') as f:
            pkl.dump(self.sem_model.results.__dict__, f)

        # Process segmentation data (ground_truth)
        data_frame = pd.read_csv(self.configs.seg_path)
        seg_video = SegmentationVideo(data_frame=data_frame, video_path=self.movie)
        seg_video.get_segments(n_annotators=100, condition=self.grain, second_interval=self.second_interval)
        biserials = seg_video.get_biserial_subjects(second_interval=self.second_interval, end_second=self.end_second)
        # logger.info(f'Subjects mean_biserial={np.nanmean(biserials):.3f}')
        # Compare SEM boundaries versus participant boundaries
        gt_freqs = seg_video.gt_freqs
        gt_freqs = gaussian_filter1d(gt_freqs, 2)
        last = min(len(pred_boundaries), len(gt_freqs))
        # compute biserial correlation and pearson_r of model boundaries
        bicorr = get_point_biserial(pred_boundaries[:last], gt_freqs[:last])
        pred_boundaries_gaussed = gaussian_filter1d(pred_boundaries.astype(float), 1)
        pearson_r, p = stats.pearsonr(pred_boundaries_gaussed[:last], gt_freqs[:last])
        percentile = percentileofscore(biserials, bicorr)
        logger.info(f'Tag={tag}: Bicorr={bicorr:.3f} cor. Percentile={percentile:.3f},  '
                    f'Subjects_median={np.nanmedian(biserials):.3f}')
        with open('output/run_sem/' + self.title + '_gtfreqs.pkl', 'wb') as f:
            pkl.dump(gt_freqs, f)
        plot_diagnostic_readouts(gt_freqs, self.sem_model.results, frame_interval=self.second_interval * self.sample_per_second,
                                 offset=self.first_frame / self.fps / self.second_interval,
                                 title=self.title + f'_diagnostic_{self.grain}_{self.current_epoch}',
                                 bicorr=bicorr, percentile=percentile, pearson_r=pearson_r)

        plot_pe(self.sem_model.results, frame_interval=self.second_interval * self.sample_per_second,
                offset=self.first_frame / self.fps / self.second_interval,
                title=self.title + f'_PE_{self.grain}_{self.current_epoch}')
        mean_pe = self.sem_model.results.pe.mean()
        std_pe = self.sem_model.results.pe.std()
        with open('output/run_sem/results_sem_corpus.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.run, self.grain, bicorr, percentile, len(self.sem_model.event_models), self.current_epoch,
                             sum(pred_boundaries), sem_init_kwargs, tag, mean_pe, std_pe, pearson_r, self.is_train])


if __name__ == "__main__":
    args = parse_config()

    if not os.path.exists('output/run_sem/results_sem_corpus.csv'):
        csv_headers = ['run', 'grain', 'bicorr', 'percentile', 'n_event_models', 'epoch',
                       'model_boundaries', 'sem_params', 'tag', 'mean_pe', 'std_pe', 'pearson_r', 'is_train']
        with open('output/run_sem/results_sem_corpus.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # Initialize keras model and running
    # Define model architecture, hyper parameters and optimizer
    f_class = GRUEvent
    # f_class = LSTMEvent
    optimizer_kwargs = dict(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    # these are the parameters for the event model itself.
    f_opts = dict(var_df0=50., var_scale0=0.06, l2_regularization=0.0, dropout=0.5,
                  n_epochs=10, t=4, batch_update=True, n_hidden=16, variance_window=None, optimizer_kwargs=optimizer_kwargs)
    # set the hyper parameters for segmentation
    lmda = float(args.lmda)  # stickyness parameter (prior)
    alfa = float(args.alfa)  # concentration parameter (prior)
    kappa = int(args.kappa)
    sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'kappa': kappa, 'f_opts': f_opts,
                       'f_class': f_class}
    # set default hyper parameters for each run, can be overridden later
    run_kwargs = dict()
    sem_model = SEM(**sem_init_kwargs)
    tag = 'mar_21_depth_pos_3'
    context_sem = SEMContext(sem_model=sem_model, run_kwargs=run_kwargs, tag=tag, configs=args)
    try:
        context_sem.read_train_valid_list()
        context_sem.iterate(is_eval=True)
    except Exception as e:
        with open('output/run_sem/sem_error.txt', 'a') as f:
            f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc(e))
