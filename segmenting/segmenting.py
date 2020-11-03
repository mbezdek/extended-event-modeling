import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import sys
import os
sys.path.append(os.getcwd())
from scipy.stats import zscore
from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent
from sem.event_models import RecurrentEvent, GRUEvent, LSTMEvent
from sem import sem_run
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from utils import parse_config, logger


def process_features(features_dataframe: pd.DataFrame) -> np.ndarray:
    for col in features_dataframe.columns:
        if col != 'sync_time':
            features_dataframe[col] = zscore(features_dataframe[col])

    x_train = features_dataframe.drop(['sync_time'], axis=1).to_numpy()
    return x_train


def plot_features_and_posterior(features_train, post):
    cluster_id = np.argmax(post, axis=1)
    cc = sns.color_palette('Dark2', post.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw=dict(width_ratios=[1, 2]))
    for clt in cluster_id:
        idx = np.nonzero(cluster_id == clt)[0]
        axes[0].scatter(features_train[idx, 0], features_train[idx, 1], color=cc[clt], alpha=.5)
    axes[0].set_xlabel(r'$\mathbf{x}_{s,1}$')
    axes[0].set_ylabel(r'$\mathbf{x}_{s,2}$')

    sns.set_palette('Dark2')
    axes[1].plot(post)
    y_hat = np.argmax(post, axis=1)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Posterior Probability')
    # print(np.argmax(post, axis=1))


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')
    # Create SEM parameters
    sem_kwargs = dict(lmda=float(args.stickyness), alfa=float(args.concentration), f_class=LinearEvent,
                      f_opts=dict(l2_regularization=float(args.l2_regularization)))
    features_df = pd.read_csv(args.features_csv)
    features_train = process_features(features_dataframe=features_df)
    segmentation_results = sem_run(features_train, sem_kwargs)
    plot_features_and_posterior(features_train=features_train, post=segmentation_results.post)
    plt.show()
