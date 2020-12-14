import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
import os
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.stats import percentileofscore
from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent, RecurrentEvent, \
    GRUEvent, LSTMEvent
from sem import sem_run, SEM
from utils import resample_df, SegmentationVideo, get_binned_prediction, get_point_biserial, \
    get_frequency_ground_truth, logger, parse_config, load_comparison_data


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
    skel_df.drop(['sync_time', 'raw_time', 'body', 'J1_dist_from_J1'], axis=1, inplace=True)
    for c in skel_df.columns:
        if '_ID' in c or 'Tracked' in c or skel_df.loc[:, c].isnull().all():
            skel_df.drop([c], axis=1, inplace=True)
        else:
            skel_df.loc[:, c] = (skel_df[c] - min(skel_df[c].dropna())) / (
                    max(skel_df[c].dropna()) - min(skel_df[c].dropna()))
    return skel_df


def preprocess_objhand(objhand_csv):
    objhand_df = pd.read_csv(objhand_csv, index_col='frame')
    for c in objhand_df.columns:
        if '2D' in c or objhand_df.loc[:, c].isnull().all():
            objhand_df.drop([c], axis=1, inplace=True)
        else:
            objhand_df.loc[:, c] = (objhand_df[c] - min(objhand_df[c].dropna())) / (
                    max(objhand_df[c].dropna()) - min(objhand_df[c].dropna()))
    return objhand_df


def combine_dataframes(data_frames, rate='40ms', fps=30):
    combine_df = pd.concat(data_frames, axis=1)
    combine_df.dropna(axis=0, inplace=True)
    combine_df.fillna(0, inplace=True)
    combine_df['sync_time'] = combine_df.index / fps
    combine_df = resample_df(combine_df, rate=rate)
    return combine_df


def plot_subject_model_boundaries(gt_freqs, pred_boundaries, title=''):
    plt.figure()
    plt.plot(gt_freqs, label='Subject Boundaries')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Boundary Probability')
    plt.title(title)
    b = np.arange(len(pred_boundaries))[pred_boundaries][0]
    plt.plot([b, b], [0, 1], 'k:', label='Model Boundary', alpha=0.75, color='b')
    for b in np.arange(len(pred_boundaries))[pred_boundaries][1:]:
        plt.plot([b, b], [0, 1], 'k:', alpha=0.75, color='b')

    plt.legend(loc='upper left')
    plt.ylim([0, 0.8])
    sns.despine()
    plt.show()


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config {args}')
    end_second = int(args.end_second)
    fps = int(args.fps)
    # Load csv files and preprocess to get a scene vector
    appear_df = preprocess_appear(args.appear_csv)
    skel_df = preprocess_skel(args.skel_csv)
    optical_df = preprocess_optical(args.optical_csv)
    objhand_df = preprocess_objhand(args.objhand_csv)
    # combine_df = combine_dataframes([appear_df, optical_df], rate=args.rate, fps=fps)
    combine_df = combine_dataframes([appear_df, optical_df, skel_df], rate=args.rate)
    combine_df['frame'] = (combine_df['sync_time'] * fps).apply(math.ceil)
    first_frame = combine_df['frame'][0]
    # combine_df = combine_dataframes([appear_df, optical_df, skel_df, objhand_df], rate=args.rate)
    x_train = combine_df.drop(['sync_time'], axis=1).to_numpy()
    end_index = math.ceil(1000 / int(args.rate[:-2]) * end_second)
    # end = 300
    x_train = x_train[:end_index]
    standardizer = preprocessing.StandardScaler()
    x_train_standardized = standardizer.fit_transform(x_train[:, 2:])
    x_train_standardized /= np.sqrt(x_train_standardized.shape[1])
    x_train_standardized = np.hstack([x_train[:, :2], x_train_standardized])
    pca = PCA(n_components=int(args.pca_dim))
    x_train_standardized_pca = pca.fit_transform(x_train_standardized[:, 2:])
    x_train_standardized_pca = np.hstack([x_train_standardized[:, :2], x_train_standardized_pca])
    # VAE feature from washing dish video
    # x_train_vae = np.load('video_color_Z_embedded_64_5epoch.npy')
    # Initialize keras model and running
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
    run_kwargs = dict()
    sem_model = SEM(**sem_init_kwargs)
    # sem_model.run(x_train_vae[5537 + 10071: 5537 + 10071 + 7633, :], **run_kwargs)
    # sem_model.run(x_train, **run_kwargs)
    sem_model.run(x_train_standardized_pca, **run_kwargs)
    # sem_model.run(x_train_standardized_pca, **run_kwargs)
    # Process results returned by SEM
    # sample_per_second = 30
    sample_per_second = 3
    second_interval = 1
    pred_boundaries = get_binned_prediction(sem_model.results.post, second_interval=second_interval, sample_per_second=sample_per_second)
    pred_boundaries = np.hstack([[0] * round(first_frame / fps / second_interval), pred_boundaries]).astype(bool)
    logger.info(f'Total # of pred_boundaries: {sum(pred_boundaries)}')
    # Process segmentation data (ground_truth)
    data_frame = pd.read_csv(args.seg_path)
    seg_video = SegmentationVideo(data_frame=data_frame, video_path=args.movie)
    seg_video.get_segments(n_annotators=100, condition=args.grain)
    seg_video.get_biserial_subjects(second_interval=1, end_second=end_second)
    print(f'Mean subjects biserial={np.mean(seg_video.biserials):.3f}')
    # Compare SEM boundaries versus participant boundaries
    gt_freqs = seg_video.gt_freqs
    # data_old = pd.read_csv('./data/zachs2006_data021011.dat', delimiter='\t')
    # _, _, gt_freqs = load_comparison_data(data_old)
    bicorr = get_point_biserial(pred_boundaries[:end_second], gt_freqs[:end_second])
    percentile = percentileofscore(seg_video.biserials, bicorr)
    print(f'bicorr={bicorr:.3f} correspond to percentile={percentile:.3f}')
    plot_subject_model_boundaries(gt_freqs, pred_boundaries,
                                  title=os.path.basename(args.movie))
