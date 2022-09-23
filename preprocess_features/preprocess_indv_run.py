import glob
import numpy as np
import pandas as pd
import math
import os
import pickle as pkl
import re
import traceback
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from copy import deepcopy
from utils import parse_config, logger, contain_substr, DictObj
from typing import Tuple, List

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
with open('gen_sim_glove_50.pkl', 'rb') as f:
    glove_vectors = pkl.load(f)
# glove_vectors = gensim.downloader.load('word2vec-ruscorpora-300')
emb_dim = glove_vectors['apple'].size


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(float)
    return appear_df


def preprocess_optical(vid_csv, standardize=True):
    vid_df = pd.read_csv(vid_csv, index_col='frame')
    for c in vid_df.columns:
        if not standardize:
            vid_df.loc[:, c] = (vid_df[c] - min(vid_df[c].dropna())) / (
                    max(vid_df[c].dropna()) - min(vid_df[c].dropna()))
        else:
            vid_df.loc[:, c] = (vid_df[c] - vid_df[c].dropna().mean()) / vid_df[
                c].dropna().std()
    return vid_df


def preprocess_skel(skel_csv, use_position=0, standardize=True, feature_tag='',
                    ratio_features=0.8, ratio_samples=0.8) -> (pd.DataFrame, bool):
    skel_df = pd.read_csv(skel_csv, index_col='frame')
    skel_df.drop(['sync_time', 'raw_time', 'body', 'J1_dist_from_J1', 'J1_3D_rel_X', 'J1_3D_rel_Y', 'J1_3D_rel_Z'], axis=1,
                 inplace=True)
    if use_position:
        keeps = ['accel', 'speed', 'dist', 'interhand', '2D', 'rel']
    else:
        keeps = ['accel', 'speed', 'dist', 'interhand', 'rel']

    for c in skel_df.columns:
        if contain_substr(c, keeps):
            continue
        else:
            skel_df.drop([c], axis=1, inplace=True)

    # Using global statistics to filter skeleton-defective runs
    defective = 0
    # load sampled skel features, 200 samples for each video.
    combined_runs = pd.read_csv(f'output/sampled_skel_features_{feature_tag}.csv')
    # mask outliers with N/A
    select_indices = (skel_df < combined_runs.quantile(.95)) & (skel_df > combined_runs.quantile(.05))
    skel_df = skel_df[select_indices]
    qualified_columns = (select_indices.sum() > int(len(skel_df) * ratio_samples))
    if qualified_columns.sum() / len(qualified_columns) <= ratio_features:
        logger.info(f"Video {skel_csv} has {len(qualified_columns) - qualified_columns.sum()} "
                    f"un-qualified columns out of {len(qualified_columns)}!!!")
        defective = 1
        # open(filtered_txt, 'a').write(f"{skel_csv}\n")

    # fill N/A
    skel_df = skel_df.ffill()

    if standardize:
        # standardize using global statistics
        select_indices = (combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))
        combined_runs_q = combined_runs[select_indices]
        stats = combined_runs_q.describe().loc[['mean', 'std']]
        skel_df = (skel_df - stats.loc['mean', skel_df.columns]) / stats.loc['std', skel_df.columns]

    return skel_df, defective


def remove_number(string):
    for i in range(100):
        string = string.replace(str(i), '')
    return string


def get_emb(category_weights, emb_dim) -> np.ndarray:
    average = np.zeros(shape=(1, emb_dim))
    for category, prob in category_weights.iteritems():
        r = np.zeros(shape=(1, emb_dim))
        try:
            r += glove_vectors[category]
        except Exception as e:
            # if this object is not in glove, use separate words with equal contribution
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        # weighted average
        average += r * prob
    return average


def get_emb_distance(category_distances, emb_dim=100) -> np.ndarray:
    # Add 1 to avoid 3 objects with 0 distances (rare but might happen), then calculate inverted weights
    category_distances = category_distances + 1
    # Add 1 to avoid cases there is only one object
    category_weights = 1 - category_distances / (category_distances.sum() + 1)
    if category_weights.sum() == 0:
        logger.error('Sum of probabilities is zero')
    average = get_emb(category_weights, emb_dim=emb_dim)
    return average / category_weights.sum()


def get_embs_and_categories(objhand_df: pd.DataFrame, emb_dim=100, num_objects=3) -> (np.ndarray, pd.DataFrame):
    obj_handling_embs = np.zeros(shape=(0, emb_dim))
    categories = pd.DataFrame()

    for i, row in objhand_df.iterrows():
        all_categories = list(row.index[row.notna()])
        if len(all_categories):
            # pick the nearest object
            nearest = row.argmin()
            assert nearest != -1
            # obj_handling_emb = get_emb_category(row.index[nearest], emb_dim)
            new_row = pd.Series(data=row.dropna().sort_values().index[:num_objects], name=row.name)
            categories = categories.append(new_row)
            # Interestingly, some words such as towel are more semantically central than mouthwash
            # glove.most_similar(glove['towel'] + ['mouthwash']) yields towel and words close to mouthwash, but not mouthwash!
            obj_handling_emb = get_emb_distance(row.dropna().sort_values()[:num_objects], emb_dim)
        else:
            obj_handling_emb = np.full(shape=(1, emb_dim), fill_value=np.nan)
        obj_handling_embs = np.vstack([obj_handling_embs, obj_handling_emb])
    return obj_handling_embs, categories


def preprocess_objhand(objhand_csv, standardize=False, use_depth=False, num_objects=3, feature='objhand') \
        -> (pd.DataFrame, pd.DataFrame):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')

    def filter_objhand():
        if use_depth:
            filtered_df = objhand_df.filter(regex=f'_dist_z$')
        else:
            filtered_df = objhand_df.filter(regex=f'_dist$')
        # be careful that filter function return a view, thus filtered_df is a view of objhand_df.
        # deepcopy to avoid unwanted bugs
        filtered_df = deepcopy(filtered_df)
        s = [re.split('([a-zA-Z\s\(\)]+)([0-9]+)', x)[1] for x in filtered_df.columns]
        instances = set(s)
        for i in instances:
            filtered_df.loc[:, i + '_mindist'] = filtered_df[[col for col in filtered_df.columns if i in col]].min(axis=1)
        filtered_df = filtered_df.filter(regex='_mindist')
        # remove mindist
        filtered_df.rename(lambda x: x.replace('_mindist', ''), axis=1, inplace=True)

        return filtered_df

    objhand_df = filter_objhand()

    obj_handling_embs, categories = get_embs_and_categories(objhand_df, emb_dim=emb_dim, num_objects=num_objects)

    obj_handling_embs = pd.DataFrame(obj_handling_embs, index=objhand_df.index,
                                     columns=list(map(lambda x: f'{feature}_{x}', range(emb_dim))))
    # Standardizing using a single video might project embedded vectors to weird space (e.g. mouthwash -> srishti)
    # Moreover, the word2vec model already standardize for the whole corpus, thus we don't need to standardize.
    if standardize:
        obj_handling_embs = (obj_handling_embs - obj_handling_embs.mean()) / obj_handling_embs.std()

    return obj_handling_embs, categories


def interpolate_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    first_frame = dataframe.index[0]
    last_frame = dataframe.index[-1]
    dummy_frame = pd.DataFrame(np.NaN, index=range(first_frame, last_frame),
                               columns=dataframe.columns)
    dummy_frame = dummy_frame.combine_first(dataframe).interpolate(limit_area='inside')
    return dummy_frame


class FeatureProcessor:
    """
    This class load individual features and pre-process them before feeding to the network
    """

    def __init__(self, configs):
        self.feature_tag = configs.feature_tag
        self.objhand_csv_dir = configs.objhand_csv
        self.num_objects = configs.num_objects
        self.skel_csv_dir = configs.skel_csv
        self.ratio_samples = float(configs.ratio_samples)
        self.ratio_features = float(configs.ratio_features)
        self.filtered_txt = f"filtered_skel_{self.feature_tag}_{self.ratio_samples}_{self.ratio_features}.txt"
        self.use_skel_position = configs.use_skel_position
        self.optical_csv_dir = configs.optical_csv
        self.appear_csv_dir = configs.appear_csv
        self.run = configs.run
        # have a run to fps df
        self.run_specs_df = pd.read_csv('run_specs.csv')
        run = self.run + ('_kinect' if '_kinect' not in self.run else '')
        self.fps = float(self.run_specs_df.loc[self.run_specs_df.run == run, 'fps'].iloc[0])
        self.rate = configs.rate
        self.df_dict = dict()
        self.complete_txt = f"preprocessed_complete_{args.feature_tag}.txt"
        # This class process at a run level, list level processes should be in parallel_preprocess_indv_run.py
        # if os.path.exists(self.complete_txt):
        #     os.remove(self.complete_txt)
        self.error_txt = f"preprocessed_error_{args.feature_tag}.txt"
        # if os.path.exists(self.error_txt):
        #     os.remove(self.error_txt)

    def resample_df(self, df) -> pd.DataFrame:
        # fps matter hear, we need feature vector at anchor timepoints to correspond to segmentation
        out_df = df.set_index(pd.to_datetime(df.index / self.fps, unit='s'), drop=False, verify_integrity=True)
        resample_index = pd.date_range(start=out_df.index[0], end=out_df.index[-1], freq=self.rate)
        dummy_frame = pd.DataFrame(np.NaN, index=resample_index, columns=out_df.columns)
        out_df = out_df.combine_first(dummy_frame).interpolate(method='time', limit_area='inside').resample(self.rate).mean()
        return out_df

    def combine_dataframes(self, data_frames) -> pd.DataFrame:
        # Some features such as optical flow are calculated not for all frames, interpolate first
        data_frames = [interpolate_frame(df) for df in data_frames]
        combine_df = pd.concat(data_frames, axis=1)
        # After dropping null values, variances are not unit anymore, some are around 0.8.
        combine_df.dropna(axis=0, inplace=True)
        combine_df['frame'] = combine_df.index
        # After resampling, some variances drop to 0.3 or 0.4
        combine_df = self.resample_df(combine_df)
        # because resample use mean, need to adjust categorical variables
        combine_df['appear'] = combine_df['appear'].apply(math.ceil).astype(float)
        combine_df['disappear'] = combine_df['disappear'].apply(math.ceil).astype(float)

        assert combine_df.isna().sum().sum() == 0
        combine_df.set_index('frame', inplace=True)
        combine_df.drop(['sync_time', 'frame'], axis=1, inplace=True, errors='ignore')
        return combine_df

    def pre_process_all_features(self) -> None:
        """
        This method load individual features then combine and align them temporally
        :return:
        """
        skel_df = self.pre_process_skel_feature()
        appear_df = self.preprocess_appear_feature()
        optical_df = self.preprocess_optical_feature()
        obj_handling_embs = self.preprocess_objhand_feature()
        scene_embs = self.preprocess_scene_feature()
        # Get consistent start-end times and resampling rate for all features
        combined_resampled_df = self.combine_dataframes([appear_df, optical_df, skel_df, obj_handling_embs, scene_embs])
        self.df_dict['combined_resampled_df'] = combined_resampled_df

    def preprocess_scene_feature(self) -> pd.DataFrame:
        logger.info(f'Processing Scene features...')
        objhand_csv = os.path.join(self.objhand_csv_dir, f'{self.run}_{self.feature_tag}_objhand.csv')
        scene_embs, _ = preprocess_objhand(objhand_csv, standardize=False,
                                           num_objects=30, use_depth=True,
                                           feature='scene')
        self.df_dict['scene_post'] = scene_embs
        return scene_embs

    def preprocess_objhand_feature(self) -> pd.DataFrame:
        logger.info(f'Processing Objhand features...')
        objhand_csv = os.path.join(self.objhand_csv_dir, f'{self.run}_{self.feature_tag}_objhand.csv')
        obj_handling_embs, categories_z = preprocess_objhand(objhand_csv, standardize=False,
                                                             num_objects=int(self.num_objects),
                                                             use_depth=True, feature='objhand')
        self.df_dict['objhand_post'] = obj_handling_embs
        self.df_dict['categories_z'] = categories_z
        return obj_handling_embs

    def preprocess_optical_feature(self) -> pd.DataFrame:
        logger.info(f'Processing Optical features...')
        optical_csv = os.path.join(self.optical_csv_dir, f'{self.run}_{self.feature_tag}_video_features.csv')
        optical_df = preprocess_optical(optical_csv, standardize=True)
        self.df_dict['optical_post'] = optical_df
        return optical_df

    def preprocess_appear_feature(self) -> pd.DataFrame:
        logger.info(f'Processing Appear features...')
        # For some reason, some optical flow videos have inf value
        pd.set_option('use_inf_as_na', True)
        appear_csv = os.path.join(self.appear_csv_dir, f'{self.run}_{self.feature_tag}_appear.csv')
        appear_df = preprocess_appear(appear_csv)
        self.df_dict['appear_post'] = appear_df
        return appear_df

    def pre_process_skel_feature(self) -> pd.DataFrame:
        logger.info(f'Processing Skel features...')
        skel_csv = os.path.join(self.skel_csv_dir, f'{self.run}_{self.feature_tag}_skel_features.csv')
        skel_df, defective = preprocess_skel(skel_csv, use_position=int(self.use_skel_position),
                                             standardize=True, feature_tag=self.feature_tag,
                                             ratio_samples=self.ratio_samples, ratio_features=self.ratio_features)
        if defective:
            open(self.filtered_txt, 'a').write(f"{self.run}\n")

        self.df_dict['skel_post'] = skel_df
        return skel_df

    def save_df_dict(self) -> None:
        if not os.path.exists('../output/preprocessed_features'):
            os.mkdir('../output/preprocessed_features')
        logger.info(f"Saving output/preprocessed_features/{self.run}_{self.feature_tag}.pkl")
        pkl.dump(self.df_dict, open(f'output/preprocessed_features/{self.run}_{self.feature_tag}.pkl', 'wb'))
        logger.info(f"Saved output/preprocessed_features/{self.run}_{self.feature_tag}.pkl")
        open(self.complete_txt, 'a').write(f"{self.run}\n")


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config: {args}')
    assert '.txt' not in args.run, f"run argument should be a video name, e.g. 1.2.3_kinect, fed {args.run}"
    processor = FeatureProcessor(configs=args)
    try:
        processor.pre_process_all_features()
        processor.save_df_dict()
    except Exception as e:
        open(processor.error_txt, 'a').write(f"{processor.run}\n{traceback.format_exc()}\n")
