import pickle as pkl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from typing import List, Tuple, Dict
import logging
from scipy.spatial.distance import mahalanobis as mahalanobis_scipy

# reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.debug("DEBUG Mode")
logger.info("INFO Mode")


# a function to calculate mahalanobis distance for a scene vector, given the centroid and covariance matrix.
def mahalanobis(x, centroid, cov):
    """
    Calculate the mahalanobis distance for a scene vector, given the centroid and covariance matrix.
    """
    # scipy.spatial.distance.mahalanobis
    mah_d = mahalanobis_scipy(x, centroid, np.linalg.inv(cov))
    # covariance matrix might not be positive definite, which results in negative
    # value for dot products.
    # if np.isnan(mah_d):
    # logger.debug(f"Mahalanobis distance is np.nan")
    # logger.debug(f"inv_cov={np.linalg.inv(cov)}")
    return mah_d


# a class to manage variance and mahalanobis distance for a run
class VarMahalanobis:
    def __init__(self, run, epoch, dropout_rate, recurrent_dropout_rate, is_val):
        self.run = run
        self.epoch = epoch
        self.dropout_rate = dropout_rate
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.is_val = is_val

        self.is_exist = False
        self.dropout_pickle = self.get_pickle()
        self.var = None
        self.mahalanobis = None
        self.centroid = None
        self.df = None

    def get_pickle(self):
        """
        Get the pickle file for a given run.
        """
        path = os.path.join(res_dropout_dir,
                            f"res_dropout_{self.run}_{self.epoch}_{self.dropout_rate}_{self.recurrent_dropout_rate}.pkl")
        try:
            res = pkl.load(open(path, 'rb'))
            self.is_exist = True
        except FileNotFoundError:
            logger.debug(f"File not found for {path}")
            return None
        return res

    def get_scene_vectors(self):
        """
        Get the scene vectors for a given run, for all time steps within the run.
        """
        if self.dropout_pickle is not None:
            return self.dropout_pickle['diagnostic']['x']

        return None

    def get_centroid_run(self):
        """
        Get the centroid of the scene vectors for a given run, for all time steps within the run.
        """
        if self.centroid is not None:
            return self.centroid

        if self.dropout_pickle is not None:
            centroid = np.mean(self.dropout_pickle['diagnostic']['x'], axis=0)
            self.centroid = centroid
            return centroid

        return None

    def get_variance(self):
        """
        Get the variance of the results for a given run, for all time steps within the run.
        """
        if self.var is not None:
            return self.var

        if self.dropout_pickle is not None:
            # concatenate all the resamples, then groupby timestep and get the variance.
            df = pd.DataFrame(np.concatenate(self.dropout_pickle['resamples'], axis=0),
                              index=[j for i in range(len(self.dropout_pickle['resamples'])) for j in
                                     range(self.dropout_pickle['resamples'][0].shape[0])])
            variance = df.groupby(df.index).apply(np.var).mean(axis=1)
            self.var = pd.DataFrame({'variance': variance})
            return self.var

        return None

    def get_mahalanobis(self, train_centroid, train_cov):
        """
        Get the mahalanobis distance for each time step within a run.
        """
        if self.mahalanobis is not None:
            return self.mahalanobis

        if self.dropout_pickle is not None:
            # calculate mahalanobis distance for each time step
            mahalanobis_distance = [mahalanobis(x, train_centroid, train_cov) for x in self.dropout_pickle['diagnostic']['x']]
            self.mahalanobis = pd.DataFrame({'mahalanobis': mahalanobis_distance})
            return self.mahalanobis

        return None

    def get_var_mahalanobis(self, centroid, cov):
        """
        Get the variance and mahalanobis distance for a given run.
        Return: a dataframe with run, epoch, dropout_rate, recurrent_dropout_rate, variance, mahalanobis
        """
        if self.df is not None:
            return self.df

        if self.var is None:
            self.var = self.get_variance()
        if self.mahalanobis is None:
            self.mahalanobis = self.get_mahalanobis(centroid, cov)
        if self.var is not None and self.mahalanobis is not None:
            lim = min(self.var.shape[0], self.mahalanobis.shape[0])
            self.df = pd.concat([self.var.iloc[:lim], self.mahalanobis.iloc[:lim]], axis=1)

            self.df['run'] = self.run
            self.df['epoch'] = self.epoch
            self.df['dropout_rate'] = self.dropout_rate
            self.df['recurrent_dropout_rate'] = self.recurrent_dropout_rate
            self.df['is_val'] = self.is_val
            return self.df
        else:
            return None


# a factory class to create a list of VarMahalanobis objects
# given a list of tuples (run, epoch, dropout_rate, recurrent_dropout_rate)
# and calculate the centroid and covariance matrix up to an epoch and for either train or validation runs.
class VarMahalanobisFactory:
    def __init__(self, varmah_identifiers: List[Tuple]):
        self.varmahs = [VarMahalanobis(run, epoch, dropout_rate, recurrent_dropout_rate, is_val)
                        for run, epoch, dropout_rate, recurrent_dropout_rate, is_val in varmah_identifiers]
        self.varmahs = [varmah for varmah in self.varmahs if varmah.is_exist]
        self.exist_identifiers = [(varmah.run, varmah.epoch, varmah.dropout_rate, varmah.recurrent_dropout_rate, varmah.is_val)
                                  for varmah in self.varmahs]
        logger.info(f"Created {len(self.varmahs)} VarMahalanobis objects.")

    def get_covariance(self, last_epoch=101, is_val=False):
        """
        Get the covariance matrix for all runs.
        """
        # get the covariance matrix for all runs
        cov = np.cov(np.concatenate([varmah.get_scene_vectors() for varmah in self.varmahs
                                     if varmah.epoch <= last_epoch and varmah.is_val == is_val], axis=0).T)
        return cov

    def get_centroid_all(self, last_epoch=101, is_val=False):
        """
        Get the centroid for all runs.
        """
        # get the centroid for all runs
        centroid = np.mean(np.concatenate([varmah.get_scene_vectors() for varmah in self.varmahs
                                           if varmah.epoch <= last_epoch and varmah.is_val == is_val], axis=0), axis=0)
        return centroid

    def get_df_from_identifiers(self, varmah_identifiers: List[Tuple], last_epoch=None, is_val=False):
        # compute variance and mahalanobis distance for all runs
        dfs = []
        for identifier in varmah_identifiers:
            if identifier in self.exist_identifiers:
                varmah = self.varmahs[self.exist_identifiers.index(identifier)]
                if last_epoch is None:
                    epoch = varmah.epoch
                else:
                    epoch = last_epoch
                logger.info(
                    f"Getting VarMahalanobis for {identifier}, using centroid and covariance "
                    f"until last_epoch={epoch} and for is_val={is_val} runs")
                dfs.append(varmah.get_var_mahalanobis(
                    centroid=self.get_centroid_all(last_epoch=epoch, is_val=is_val),
                    cov=self.get_covariance(last_epoch=epoch, is_val=is_val)))
            else:
                logger.debug(f"VarMahalanobis object for {identifier} does not exist.")
        return pd.concat(dfs, axis=0)


res_dropout_dir = '../../output/diagnose/dropout_epoch/'
uncertainty_output_dir = '../../output/diagnose/uncertainty/'
os.makedirs(uncertainty_output_dir, exist_ok=True)
train_runs = open('../../output/train_sep_09.txt', 'rt').readlines()
train_runs = [x.strip() for x in train_runs]
val_runs = open('../../output/valid_sep_09.txt', 'rt').readlines()
val_runs = [x.strip() for x in val_runs]

# build a list of tuple of run, epoch, dropout_rate, recurrent_dropout_rate, is_val
varmah_identifiers = []
dropout_rates = [0.5]
recurrent_dropout_rates = [0.5]
epochs = np.arange(1, 61, 10)
for epoch in epochs:
    for dropout_rate in dropout_rates:
        for recurrent_dropout_rate in recurrent_dropout_rates:
            for run in train_runs:
                varmah_identifiers.append((run, epoch, dropout_rate, recurrent_dropout_rate, False))
            for run in val_runs:
                varmah_identifiers.append((run, epoch, dropout_rate, recurrent_dropout_rate, True))
# for each identifier, create a VarMahalanobis object
varmah_factory = VarMahalanobisFactory(varmah_identifiers)

# for each identifier, compute variance and mahalanobis distance, concat into a dataframe
df = varmah_factory.get_df_from_identifiers(varmah_identifiers[::-1])
# plot violin plot for Mahalabobis distance for each epoch, color by is_val, plotly
fig = px.violin(df[df.epoch >= 1], y="mahalanobis", x="epoch",
                color="is_val",
                points=None, hover_data=df.columns)
fig.write_html(f"{uncertainty_output_dir}/mahalanobis_across_training.html")
# plot violin plot for Mahalabobis distance for each epoch, color by is_val, plotly
fig = px.violin(df[df.epoch >= 1], y="variance", x="epoch",
                color="is_val",
                points=None, box=True, hover_data=df.columns)
fig.write_html(f"{uncertainty_output_dir}/variance_across_training.html")
