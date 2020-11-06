import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns
import sys
import os
import pickle as pkl
# Normally, RNN runs faster on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.append(os.getcwd())
from scipy.stats import zscore
from sem.event_models import LinearEvent, NonLinearEvent, RecurrentLinearEvent
from sem.event_models import RecurrentEvent, GRUEvent, LSTMEvent
from sem import sem_run
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from utils import parse_config, logger


def process_features(features_dataframe: pd.DataFrame) -> np.ndarray:
    for col in features_dataframe.columns:
        if col != 'sync_time':
            features_dataframe[col] = zscore(features_dataframe[col])

    x_train = features_dataframe.drop(['sync_time'], axis=1).to_numpy()
    return x_train


def plot_features_and_posterior(features_train, post) -> None:
    cluster_id = np.argmax(post, axis=1)
    cc = sns.color_palette('Dark2', post.shape[1])
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_train)
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    tsne_result = tsne.fit_transform(features_train)
    fig = plt.figure(figsize=(14, 4))
    sns.set_palette('Dark2')
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    for clt in cluster_id:
        idx = np.nonzero(cluster_id == clt)[0]
        ax.scatter(pca_result[idx, 0],
                   pca_result[idx, 1],
                   pca_result[idx, 2],
                   color=tuple(list(cc[clt])+[0.5]),
                   alpha=0.5)
    ax.set_title('PCA')
    ax = fig.add_subplot(1, 3, 2)
    for clt in cluster_id:
        idx = np.nonzero(cluster_id == clt)[0]
        ax.scatter(tsne_result[idx, 0],
                   tsne_result[idx, 1],
                   color=tuple(list(cc[clt])+[0.5]),
                   alpha=0.5)
    ax.set_title('t-SNE')
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(post, alpha=0.5)
    y_hat = np.argmax(post, axis=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Posterior Probability')
    # print(np.argmax(post, axis=1))


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    logger.info(f'Config {args}')
    # Create SEM parameters
    f_opts = dict(
        l2_regularization=args.l2_regularization,
        n_epochs=args.epochs,
    )
    sem_kwargs = dict(lmda=float(args.stickyness), alfa=float(args.concentration),
                      f_class=LinearEvent,
                      f_opts=f_opts)
    features_df = pd.read_csv(args.features_csv)
    features_train = process_features(features_dataframe=features_df)
    segmentation_results = sem_run(features_train, sem_kwargs)
    with open(args.posterior_output, 'wb') as f:
        pkl.dump(segmentation_results, f, protocol=pkl.HIGHEST_PROTOCOL)
    plot_features_and_posterior(features_train=features_train, post=segmentation_results.post)
    plt.show()
