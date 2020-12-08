import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import os
from sklearn.decomposition import PCA
from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent, RecurrentEvent, \
    GRUEvent, LSTMEvent
from sem import sem_run
from utils import resample_df, SegmentationVideo, get_binned_prediction, get_point_biserial, \
    get_frequency_ground_truth, logger, parse_config


def preprocess_appear(appear_csv):
    appear_df = pd.read_csv(appear_csv, index_col='frame')
    for c in appear_df.columns:
        appear_df.loc[:, c] = appear_df[c].astype(bool).astype(int)
    return appear_df


def preprocess_optical(vid_csv):
    vid_df = pd.read_csv(vid_csv, index_col='frame')
    vid_df.drop(['pixel_correlation'], axis=1, inplace=True)
    for c in vid_df.columns:
        vid_df.loc[:, c] = (vid_df[c] - min(vid_df[c].dropna())) / (
                max(vid_df[c].dropna()) - min(vid_df[c].dropna()))
    return vid_df


def preprocess_skel(skel_csv):
    skel_df = pd.read_csv(skel_csv, index_col='frame')
    skel_df.drop(['sync_time', 'raw_time', 'body'], axis=1, inplace=True)
    for c in skel_df.columns:
        if '_ID' in c or 'Tracked' in c:
            skel_df.drop([c], axis=1, inplace=True)
        else:
            skel_df.loc[:, c] = (skel_df[c] - min(skel_df[c].dropna())) / (
                    max(skel_df[c].dropna()) - min(skel_df[c].dropna()))
    return skel_df


def preprocess_objhand(objhand_csv):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')
    for c in objhand_df.columns:
        if '2D' in c:
            objhand_df.drop([c], axis=1, inplace=True)
        else:
            objhand_df.loc[:, c] = (objhand_df[c] - min(objhand_df[c].dropna())) / (
                    max(objhand_df[c].dropna()) - min(objhand_df[c].dropna()))
    return objhand_df


def combine_dataframes(data_frames, rate='40ms', fps=30):
    combine_df = pd.concat(data_frames, axis=1)
    combine_df.fillna(0, inplace=True)
    combine_df['sync_time'] = combine_df.index / fps
    combine_df = resample_df(combine_df, rate=rate)
    return combine_df


def plot_subject_model_boundaries(gt_freq, pred_boundaries, title=''):
    plt.figure()
    plt.plot(gt_freq, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.title(title)
    b = np.arange(len(pred_boundaries))[pred_boundaries][0]
    plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75)
    for b in np.arange(len(pred_boundaries))[pred_boundaries][1:]:
        plt.plot([b, b], [0, 1], 'k:', alpha=0.75)

    plt.legend(loc='upper left')
    plt.ylim([0, 0.6])
    sns.despine()
    plt.show()


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config {args}')
    # Load csv files and preprocess to get a scene vector
    appear_df = preprocess_appear(args.appear_csv)
    skel_df = preprocess_skel(args.skel_csv)
    optical_df = preprocess_optical(args.optical_csv)
    objhand_df = preprocess_objhand(args.objhand_csv)
    # combine_df = combine_dataframes([appear_df, optical_df], rate=args.rate, fps=int(args.fps))
    combine_df = combine_dataframes([appear_df, optical_df, skel_df], rate=args.rate)
    # combine_df = combine_dataframes([appear_df, optical_df, skel_df, objhand_df], rate=args.rate)
    x_train = combine_df.drop(['sync_time'], axis=1).to_numpy()
    end = math.ceil(1000 / int(args.rate[:-2]) * int(args.end_second))
    # end = 300
    x_train = x_train[: end]
    pca = PCA(n_components=int(args.pca_dim))
    x_train_pca = pca.fit_transform(x_train[:, 3:])
    x_train_pca = np.hstack([x_train[:, :3], x_train_pca])
    # Initialize keras model
    # set the parameters for the segmentation
    f_class = GRUEvent
    # f_class = LinearEvent
    # these are the parameters for the event model itself.
    f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, dropout=0.5,
                  n_epochs=10, t=4)
    # f_opts = dict(l2_regularization=0.5, n_epochs=10)
    lmda = float(args.lmda)  # stickyness parameter (prior)
    alfa = float(args.alfa)  # concentration parameter (prior)
    sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'f_opts': f_opts, 'f_class': f_class}
    sem_results = sem_run(x_train, sem_init_kwargs)
    # sem_results = sem_run(x_train_pca, sem_init_kwargs)
    # Process results returned by SEM
    pred_boundaries = get_binned_prediction(sem_results.post, fps=3)
    logger.info(f'Total # of boundaries: {sum(pred_boundaries)}')
    # Process segmentation data (ground_truth)
    data_frame = pd.read_csv(args.seg_path)
    seg_video = SegmentationVideo(data_frame=data_frame, video_path=args.movie)
    seg_video.get_segments(n_annotators=100, condition=args.grain)
    seg_points = np.hstack(seg_video.seg_points)
    end_second = int(args.end_second)
    gt_boundaries, _ = get_frequency_ground_truth(seg_points, end_second=end_second)
    gt_freq = gt_boundaries / seg_video.n_participants
    # Compare SEM boundaries versus participant boundaries
    bicorr = get_point_biserial(pred_boundaries[:end_second], gt_freq[:end_second])
    plot_subject_model_boundaries(gt_freq, pred_boundaries,
                                  title=os.path.basename(args.movie))
