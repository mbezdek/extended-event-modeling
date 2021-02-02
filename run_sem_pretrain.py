import numpy as np
import matplotlib.pyplot as plt
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

sys.path.append('../pysot')
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore
from sem.event_models import GRUEvent, LinearEvent
from sem import SEM
from utils import SegmentationVideo, get_binned_prediction, get_point_biserial, \
    logger, parse_config, contain_substr
from joblib import Parallel, delayed
import gensim.downloader

# glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
with open('gen_sim_glove_50.pkl', 'rb') as f:
    glove_vectors = pkl.load(f)
# glove_vectors = gensim.downloader.load('word2vec-ruscorpora-300')
emb_dim = glove_vectors['apple'].size


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(bool).astype(int)
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
    # Add 1 to avoid 3 objects with 0 distances (rare but might happen), then calculate inversed weights
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


def get_embeddings(objhand_df: pd.DataFrame, emb_dim=100):
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
            new_row = pd.Series(data=row.dropna().sort_values().index[:3], name=row.name)
            categories = categories.append(new_row)
            obj_handling_emb = get_emb_category(row.dropna().sort_values()[:3], emb_dim)
        else:
            scene_embedding = np.full(shape=(1, emb_dim), fill_value=np.nan)
            obj_handling_emb = np.full(shape=(1, emb_dim), fill_value=np.nan)
        scene_embs = np.vstack([scene_embs, scene_embedding])
        obj_handling_embs = np.vstack([obj_handling_embs, obj_handling_emb])
    return scene_embs, obj_handling_embs, categories


def preprocess_objhand(objhand_csv, standardize=True):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')
    # Drop non-distance columns
    for c in objhand_df.columns:
        if 'dist' not in c:
            objhand_df.drop([c], axis=1, inplace=True)
    # remove track number
    objhand_df.rename(remove_number, axis=1, inplace=True)
    all_categories = set(objhand_df.columns.values)
    # get neareast distance for the same category at each row
    for category in all_categories:
        if isinstance(objhand_df[category], pd.Series):
            objhand_df[category.replace('_dist', '')] = objhand_df[category]
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
    scene_embs, obj_handling_embs, categories = get_embeddings(objhand_df, emb_dim=emb_dim)
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
    combine_df.dropna(axis=0, inplace=True)
    first_frame = combine_df.index[0]
    combine_df['frame'] = combine_df.index
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

    plt.text(0.1, 0.8, f'bicorr={bicorr:.3f}, perc={percentile:.3f}', fontsize=14)
    plt.legend(loc='upper left')
    plt.ylim([0, 1.0])
    sns.despine()
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()
    plt.close()


def plot_diagnostic_readouts(gt_freqs, sem_readouts, frame_interval=3.0, offset=0.0, title='', show=False, save_fig=True):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.title(title)
    colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}

    latest = 0
    current = 0
    post = sem_readouts.e_hat
    switch = []
    for i in post:
        if i != current:
            if i > latest:
                switch.append('new_post')
                latest = i
            else:
                switch.append('old_post')
            current = i
        else:
            switch.append('current_post')

    df = pd.DataFrame(switch, columns=['switch'])
    plt.vlines(df[df['switch'] == 'new_post'].index / frame_interval + offset, ymin=0, ymax=1, alpha=0.5, label='Switch to New '
                                                                                                                'Event',
               color=colors['new'], linestyles='dotted')
    plt.vlines(df[df['switch'] == 'old_post'].index / frame_interval + offset, ymin=0, ymax=1, alpha=0.5, label='Switch to Old '
                                                                                                                'Event',
               color=colors['old'], linestyles='dotted')
    plt.legend()
    plt.ylim([0, 1.0])
    if save_fig:
        plt.savefig('output/run_sem/' + title + '.png')
    if show:
        plt.show()


def infer_on_video(args, run, tag):
    try:
        # For some reason, some optical flow videos have inf value, TODO: investigate
        pd.set_option('use_inf_as_na', True)
        args.run = run
        args.tag = tag
        logger.info(f'Config {args}')
        # FPS is used to pad prediction boundaries, should be inferred from run
        if 'kinect' in run:
            fps = 25
        else:
            fps = 30
        movie = run + '_trim.mp4'
        objhand_csv = os.path.join(args.objhand_csv, run + '_objhand.csv')
        skel_csv = os.path.join(args.skel_csv, run + '_skel_features.csv')
        appear_csv = os.path.join(args.appear_csv, run + '_appear.csv')
        optical_csv = os.path.join(args.optical_csv, run + '_video_features.csv')

        # Load csv files and preprocess to get a scene vector
        appear_df = preprocess_appear(appear_csv)
        skel_df = preprocess_skel(skel_csv, use_position=True, standardize=True)
        optical_df = preprocess_optical(optical_csv, standardize=True)
        scene_embs, obj_handling_embs, categories = preprocess_objhand(objhand_csv, standardize=True)
        # Get consistent start-end times and resampling rate for all features
        combine_df, first_frame, data_frames = combine_dataframes([appear_df, optical_df, skel_df, obj_handling_embs],
                                                                  rate=args.rate, fps=fps)
        last_frame = data_frames[0].index[-1]
        # This parameter is used to limit the time of ground truth video
        end_second = math.ceil(last_frame / fps)
        # Readout to visualize object-hand features
        # Re-index: some frames there are no objects near hand (which is not possible, this bug is due to min(89, NaN)=NaN
        # categories = categories.reindex(range(categories.index[-1])).ffill()
        categories = categories.ffill()
        categories = categories.loc[data_frames[0].index, :]
        data_frames.append(categories)
        coordinates = pd.DataFrame()
        objhand_df = pd.read_csv(objhand_csv)
        objhand_df = objhand_df.loc[data_frames[0].index, :]
        for index, r in categories.iterrows():
            frame_series = pd.Series(dtype=float)
            # There could be a case where there are two spray bottles near the hand: 6.3.6
            # TODO: filter seen instances
            all_categories = set(r.values)
            for c in list(all_categories):
                # Filter by category name and select distance
                # Note: paper towel and towel causes duplicated columns in series,
                # Need anchor ^ to distinguish towel and paper towel (2.4.7),
                # need digit \d to distinguish pillow0 and pillowcase0 (3.3.5)
                # Need to escape character ( and ) in aloe (green bottle) (4.4.5)
                df = objhand_df.loc[index, :].filter(regex=f"^{re.escape(c)}\d").filter(like='_dist')
                # select nearest object's coordinates
                nearest_name = df.index[df.argmin()].replace('_dist', '')  # e.g. pillow0, pillowcase0, towel0, paper towel0
                # need anchor ^ to distinguish between towel0 and paper towel0
                s = objhand_df.loc[index, :].filter(regex=f"^{re.escape(nearest_name)}")
                frame_series = frame_series.append(s)
            frame_series.name = index
            coordinates = coordinates.append(frame_series)
        data_frames.append(coordinates)

        title = os.path.basename(movie[:-4]) + tag
        # logger.info(f'Features: {combine_df.columns}')
        # Note that without copy=True, this code will return a view and subsequent changes to x_train will change combine_df
        # e.g. x_train /= np.sqrt(x_train.shape[1]) or x_train[2] = ...
        x_train = combine_df.to_numpy(copy=True)
        if int(args.pca):
            pca = PCA(float(args.pca_explained), whiten=True)
            try:
                x_train_pca = pca.fit_transform(x_train[:, 2:])
                x_train_inversed = pca.inverse_transform(x_train_pca)
            except Exception as e:
                print(repr(e))
                x_train_pca = pca.fit_transform(x_train[:, 2:])
            x_train = np.hstack([x_train[:, :2], x_train_pca])
            x_train_inversed = np.hstack([x_train[:, :2], x_train_inversed])
            df_x_train_inversed = pd.DataFrame(data=x_train_inversed, index=data_frames[0].index, columns=combine_df.columns)
            data_frames.append(df_x_train_inversed)
        else:
            df_x_train = pd.DataFrame(data=x_train, index=data_frames[0].index, columns=combine_df.columns)
            data_frames.append(df_x_train)


        # VAE feature from washing dish video
        # x_train_vae = np.load('video_color_Z_embedded_64_5epoch.npy')
        def run_sem_and_plot(x_train, tag=''):
            # Initialize keras model and running
            # set the parameters for the segmentation
            f_class = GRUEvent
            # f_class = LinearEvent
            # these are the parameters for the event model itself.
            f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, dropout=0.5,
                          n_epochs=10, t=4, batch_update=True)
            # f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, n_epochs=10)
            # f_opts = dict(l2_regularization=0.5, n_epochs=10)
            lmda = float(args.lmda)  # stickyness parameter (prior)
            alfa = float(args.alfa)  # concentration parameter (prior)
            sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'f_opts': f_opts,
                               'f_class': f_class}
            run_kwargs = dict()
            sem_model = SEM(**sem_init_kwargs)
            # sem_model.run(x_train_vae[5537 + 10071: 5537 + 10071 + 7633, :], **run_kwargs)
            sem_model.run(x_train, **run_kwargs)
            # Process results returned by SEM
            # sample_per_second = 30  # washing dish video
            sample_per_second = 1000 / float(args.rate.replace('ms', ''))
            second_interval = 1  # interval to group boundaries
            pred_boundaries = get_binned_prediction(sem_model.results.post, second_interval=second_interval,
                                                    sample_per_second=sample_per_second)
            # Padding prediction boundaries, could be changed to have higher resolution but not necessary
            pred_boundaries = np.hstack([[0] * round(first_frame / fps / second_interval), pred_boundaries]).astype(bool)
            logger.info(f'Total # of pred_boundaries: {sum(pred_boundaries)}')
            with open('output/run_sem/' + title + '_diagnostic.pkl', 'wb') as f:
                pkl.dump(sem_model.results.__dict__, f)

            def evaluate_and_plot(grain='fine'):
                # Process segmentation data (ground_truth)
                data_frame = pd.read_csv(args.seg_path)
                seg_video = SegmentationVideo(data_frame=data_frame, video_path=movie)
                seg_video.get_segments(n_annotators=100, condition=grain, second_interval=second_interval)
                biserials = seg_video.get_biserial_subjects(second_interval=second_interval, end_second=end_second)
                # logger.info(f'Subjects mean_biserial={np.nanmean(biserials):.3f}')
                # Compare SEM boundaries versus participant boundaries
                gt_freqs = seg_video.gt_freqs
                gt_freqs = gaussian_filter1d(gt_freqs, 2)
                # data_old = pd.read_csv('./data/zachs2006_data021011.dat', delimiter='\t')
                # _, _, gt_freqs = load_comparison_data(data_old)
                last = min(len(pred_boundaries), len(gt_freqs))
                bicorr = get_point_biserial(pred_boundaries[:last], gt_freqs[:last])
                percentile = percentileofscore(biserials, bicorr)
                logger.info(f'Tag={tag}: Bicorr={bicorr:.3f} cor. Percentile={percentile:.3f},  '
                            f'Subjects_median={np.nanmedian(biserials):.3f}')
                plot_subject_model_boundaries(gt_freqs, pred_boundaries, title=title + f'_{grain}',
                                              show=False, bicorr=bicorr, percentile=percentile)
                with open('output/run_sem/' + title + '_gtfreqs.pkl', 'wb') as f:
                    pkl.dump(gt_freqs, f)
                plot_diagnostic_readouts(gt_freqs, sem_model.results, frame_interval=second_interval * sample_per_second,
                                         offset=first_frame / fps / second_interval, title=title + f'_diagnostic_{grain}')
                with open('output/run_sem/results_sem_run.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([run, grain, bicorr, percentile, np.nanmedian(biserials), pred_boundaries,
                                     sum(pred_boundaries), sem_init_kwargs, tag])
                return bicorr, percentile

            # bicorr, percentile = evaluate_and_plot('fine')
            bicorr, percentile = evaluate_and_plot('coarse')
            return sem_model, bicorr, percentile

        # Note that this is different from x_train = x_train / np.sqrt(x_train.shape[1]), the below code will change values of
        # the memory allocation, change to be safer
        # x_train /= np.sqrt(x_train.shape[1])
        x_train = x_train / np.sqrt(x_train.shape[1])
        # appear, x_train = np.split(x_train, [2], axis=1)  # remove appear features
        sem_model, bicorr, percentile = run_sem_and_plot(x_train, tag=f'{tag}')
        if int(args.pca):
            x_infered_inversed = sem_model.results.x_hat
            # x_infered_inversed = np.hstack([appear, x_infered_inversed])  # concat appear feature as if it's used for consistency
            x_infered_inversed = x_infered_inversed * np.sqrt(x_train.shape[1])
            x_infered_inversed = pca.inverse_transform(x_infered_inversed[:, 2:])
            x_infered_inversed = np.hstack([x_infered_inversed[:, :2], x_infered_inversed])
            df_x_infered_inversed = pd.DataFrame(data=x_infered_inversed, index=data_frames[0].index, columns=combine_df.columns)
            data_frames.append(df_x_infered_inversed)
        else:
            x_infered_ori = sem_model.results.x_hat * np.sqrt(x_train.shape[1])
            df_x_infered_ori = pd.DataFrame(data=x_infered_ori, index=data_frames[0].index, columns=combine_df.columns)
            data_frames.append(df_x_infered_ori)

        with open('output/run_sem/' + title + '_inputdf.pkl', 'wb') as f:
            pkl.dump(data_frames, f)

        logger.info(f'Done SEM {run}')
        with open('output/run_sem/sem_complete.txt', 'a') as f:
            f.write(run + '\n')
        return sem_model.results, bicorr, percentile
    except Exception as e:
        with open('output/run_sem/sem_error.txt', 'a') as f:
            f.write(args.run + '\n')
            # f.write(repr(e) + '\n')
            f.write(traceback.format_exc() + '\n')
        return None, None, None


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


if __name__ == "__main__":
    args = parse_config()
    if '.txt' in args.run:
        # merge_feature_lists(args.run)
        choose = ['kinect']
        # choose = ['C1']
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if contain_substr(run, choose)]
    else:
        runs = [args.run]

    tag = 'feb_02_train_multiple'
    if not os.path.exists('output/run_sem/results_sem_run.csv'):
        csv_headers = ['run', 'grain', 'bicorr', 'percentile', 'mean_subject', 'pred_boundaries',
                       'model_boundaries', 'sem_params', 'tag']
        with open('output/run_sem/results_sem_run.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    # Uncomment this for debugging
    # sem_model, bicorr, percentile = infer_on_video(args, runs[0], tag)
    res = Parallel(n_jobs=8)(delayed(infer_on_video)(args, run, tag) for run in runs)
    sem_results, bicorrs, percentiles = zip(*res)
    # Average metric for all runs
    bis = []
    pers = []
    for b, p in zip(bicorrs, percentiles):
        if b is not None:
            bis.append(b)
            pers.append(p)
    if not os.path.exists('output/run_sem/results_sem_agg.csv'):
        csv_headers = ['bicorr', 'percentile', 'alfa', 'lmda', 'tag']
        with open('output/run_sem/results_sem_agg.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    with open('output/run_sem/results_sem_agg.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([sum(bis) / len(bis), sum(pers) / len(pers), float(args.alfa), float(args.lmda), tag])
