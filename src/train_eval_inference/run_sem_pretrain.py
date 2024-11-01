import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import math
import os
import sys
import traceback
import csv
import pickle as pkl
import scipy.stats as stats

import ray

# this n_cpus setting seems to limit # active threads for each ray actor method.
# add env_vars to the ray.init to set PYTHONPATH for the remote actors
print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
# you might encounter ModuleNotFoundError: No module named 'sem', the runtime_env is to catch this error (assuming you already exported SEM2 to PYTHONPATH)
# however, in ray 1.7.0 and ray 1.8.0, the runtime_env is not working, and you will encounter this message and then Ray will hang:
# (raylet) [2024-03-08 19:34:02,614 E 482138 482138] agent_manager.cc:134: Not all required Ray dependencies for the
# runtime_env feature were found. To install the required dependencies, please run `pip install 'ray[default]'`
if float(ray.__version__[:3]) > 1.8:
    ray.init(num_cpus=12,
         runtime_env={"env_vars": {"PYTHONPATH": os.environ.get('PYTHONPATH', '')}}
         )
else:
    # Replace the direct ray.init() call with a more robust initialization
    try:
        ray.init(num_cpus=12, ignore_reinit_error=True)
    except TypeError as e:
        print(f"Error initializing Ray: {e}")
        print("Attempting to initialize Ray without specifying CPUs...")
        ray.init(ignore_reinit_error=True)

sys.path.append('../pysot')
sys.path.append('../SEM2')
sys.path.append('.')
from scipy.stats import percentileofscore
from sem.event_models import GRUEvent
from sem.sem import SEM
from src.utils import SegmentationVideo, get_binned_prediction, get_point_biserial, DictObj, \
    logger, parse_config, get_coverage, get_purity, event_label_to_interval, PermutationBiserial, adjust_n_boundaries, \
    remove_flurries, remove_random_boundaries
from src.preprocess_features.compute_pca_all_runs import PCATransformer
from scipy.ndimage import gaussian_filter1d
from typing import List

seed = int(os.environ.get('SEED', '1111'))
logger.info(f'Setting seed in run_sem_pretrain.py, seed={seed}')
np.random.seed(seed)
tf.random.set_seed(seed)


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
    NUM_COLORS = post.max()
    # Hard-code 40 events for rainbow to be able to compare across events
    # NUM_COLORS = 30
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
                plt.axvline(second, linestyle='dotted', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Old Event')
            elif b == 2:
                plt.axvline(second, linestyle='solid', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='New Event')
            elif b == 3:
                plt.axvline(second, linestyle='dashdot', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Restart Event')
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
    plt.close()


def plot_pe(sem_readouts, frame_interval, offset, title):
    fig, ax = plt.subplots()
    df = pd.DataFrame({'prediction_error': sem_readouts.pe}, index=range(len(sem_readouts.pe)))
    df['second'] = df.index / frame_interval + offset
    df.plot(kind='line', x='second', y='prediction_error', alpha=1.00, ax=ax)
    plt.savefig('output/run_sem/' + title + '.png')
    plt.close()


class SEMContext:
    """
    This class maintain global variables for SEM training and inference
    """

    def __init__(self, sem_model=None, run_kwargs=None, configs=None):
        self.sem_model = sem_model
        self.run_kwargs = run_kwargs
        self.sem_tag = configs.sem_tag
        self.configs = configs
        path = os.path.join(self.configs.sem_results_path, self.sem_tag)
        if not os.path.exists(path):
            os.makedirs(path)
        self.epochs = int(self.configs.epochs)
        self.train_stratified = int(self.configs.train_stratified)
        self.second_interval = 1
        self.sample_per_second = 1000 / float(self.configs.rate.replace('ms', ''))
        self.perm_biserial = PermutationBiserial(n_permutations=1000)

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
        self.first_second = None
        self.end_second = None
        self.df_object = None
        self.categories = None
        self.categories_z = None

    def iterate(self, is_eval=True):
        # assuming no stratification, a list of *_kinect instead of a list of *.txt
        self.train_dataset = self.train_list
        # Randomize order of training video
        # random.shuffle(self.train_dataset)
        self.train_dataset = np.random.permutation(self.train_dataset)
        self.valid_dataset = np.random.permutation(self.valid_dataset)
        for e in range(self.epochs):
            # epoch counting from 1 for inputdf and diagnostic.
            self.current_epoch = e + 1
            logger.info('Training')
            self.training()
            if is_eval and self.current_epoch % 10 == 1:
                # if is_eval and self.current_epoch % 5 == 0:
                logger.info('Evaluating')
                self.evaluating()
            # break
            # LR annealing
            # if self.current_epoch % 10 == 0:
            #     self.sem_model.general_event_model.decrease_lr()
            #     self.sem_model.general_event_model_x2.decrease_lr()
            #     self.sem_model.general_event_model_x3.decrease_lr()
            #     self.sem_model.general_event_model_yoke.decrease_lr()
            #     for k, e in self.sem_model.event_models.items():
            #         e.decrease_lr.remote()

        logger.info(f"Training set: \n{self.train_dataset}")

    def training(self):
        run = self.train_dataset[(self.current_epoch - 1) % len(self.train_dataset)]
        self.is_train = True
        self.sem_model.kappa = float(self.configs.kappa)
        self.sem_model.alfa = float(self.configs.alfa)
        logger.info(f'Training video {run} at epoch {self.current_epoch}')
        self.set_run_variables(run)
        self.infer_on_video(store_dataframes=int(self.configs.store_frames))

    def evaluating(self):
        for index, run in enumerate(self.valid_dataset):
            self.is_train = False
            self.sem_model.kappa = 0
            self.sem_model.alfa = 1e-30
            logger.info(f'Evaluating video {run} at epoch {self.current_epoch}')
            self.set_run_variables(run)
            self.infer_on_video(store_dataframes=int(self.configs.store_frames))
            # break

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
            # To be consistent when config is stratified
            if is_stratified:
                return [[token]]
            # degenerate case, e.g. 4.4.4_kinect
            return [token]

    def read_train_valid_list(self):
        self.valid_dataset = self.parse_input(self.configs.valid)
        self.train_list = self.parse_input(self.configs.train, is_stratified=self.train_stratified)

    def set_run_variables(self, run):
        self.run = run
        self.movie = run + '_trim.mp4'
        self.title = os.path.join(self.sem_tag, os.path.basename(self.movie[:-4]) + self.sem_tag)
        # self.grain = 'coarse'
        # FPS is used to pad prediction boundaries, should be inferred from run
        if 'kinect' in run:
            self.fps = 25
        else:
            self.fps = 30

    def infer_on_video(self, store_dataframes=1):
        try:
            self.load_features()

            # Note that without copy=True, this code will return a view and subsequent changes
            # to x_train will change self.df_object.combined_resampled_df
            # e.g. x_train /= np.sqrt(x_train.shape[1]) or x_train[2] = ...
            logger.info(f"Done loading combined_resampled features "
                        f"{self.df_object.combined_resampled_df.shape}")
            x_train = self.df_object.combined_resampled_df.to_numpy(copy=True)
            resampled_indices = pd.Series(self.df_object.combined_resampled_df.index)
            # PCA transform input features. Also, get inverted vector for visualization
            if int(self.configs.pca):
                logger.info(f"Perform PCA on combined_resampled features")
                if int(self.configs.use_ind_feature_pca):
                    pca_transformer = PCATransformer(feature_tag=self.configs.feature_tag, pca_tag='all')
                    x_train_pca = pca_transformer.transform(x_train)
                    x_train_inverted = pca_transformer.invert_transform(x_train_pca)
                else:
                    # pca = pkl.load(open(f'output/{self.configs.feature_tag}_{self.configs.pca_tag}_pca.pkl', 'rb'))
                    pca = PCATransformer.load_pca_version_agnostic(f'output/pca_estimator_from_all_runs/{self.configs.feature_tag}_{self.configs.pca_tag}_pca')
                    assert x_train.shape[1] == pca.n_features_, f'MISMATCH: pca.n_features_ = {pca.n_features_} ' \
                                                                f'vs. input features={x_train.shape[1]}!!!'
                    x_train_pca = pca.transform(x_train)
                    x_train_inverted = pca.inverse_transform(x_train_pca)

                x_train = x_train_pca
                df_x_train = pd.DataFrame(data=x_train, index=resampled_indices)
                setattr(self.df_object, 'x_train_pca', df_x_train)
                df_x_train_inverted = pd.DataFrame(data=x_train_inverted, index=resampled_indices,
                                                   columns=self.df_object.combined_resampled_df.columns)
                setattr(self.df_object, 'x_train_inverted', df_x_train_inverted)
            else:
                df_x_train = pd.DataFrame(data=x_train, index=resampled_indices,
                                          columns=self.df_object.combined_resampled_df.columns)
                setattr(self.df_object, 'x_train', df_x_train)

            logger.info(f"Feeding PCA-ed features {x_train.shape} to SEM")
            # x_train is already has unit variance for all features (pca whitening) -> scale to have unit length.
            x_train = x_train / np.sqrt(x_train.shape[1])
            # x_train = np.random.permutation(x_train)
            # This function train and change sem event models
            self.run_sem_and_log(x_train, resampled_indices)
            # Transform predicted vectors to the original vector space for visualization
            if int(self.configs.pca):
                x_inferred_pca = self.sem_model.results.x_hat
                # Scale back to PCA whitening results
                x_inferred_pca = x_inferred_pca * np.sqrt(x_train.shape[1])
                df_x_inferred = pd.DataFrame(data=x_inferred_pca, index=resampled_indices)
                setattr(self.df_object, 'x_inferred_pca', df_x_inferred)
                if int(self.configs.use_ind_feature_pca):
                    pca_transformer = PCATransformer(feature_tag=self.configs.feature_tag, pca_tag=self.configs.pca_tag)
                    x_inferred_inverted = pca_transformer.invert_transform(x_inferred_pca)
                else:
                    pca = pkl.load(open(f'output/{self.configs.feature_tag}_{self.configs.pca_tag}_pca.pkl', 'rb'))
                    x_inferred_inverted = pca.inverse_transform(x_inferred_pca)
                df_x_inferred_inverted = pd.DataFrame(data=x_inferred_inverted, index=resampled_indices,
                                                      columns=self.df_object.combined_resampled_df.columns)
                setattr(self.df_object, 'x_inferred_inverted', df_x_inferred_inverted)
            else:
                x_inferred_ori = self.sem_model.results.x_hat * np.sqrt(x_train.shape[1])
                df_x_inferred_ori = pd.DataFrame(data=x_inferred_ori, index=resampled_indices,
                                                 columns=self.df_object.combined_resampled_df.columns)
                setattr(self.df_object, 'x_inferred', df_x_inferred_ori)

            if store_dataframes:
                path = os.path.join(self.configs.sem_results_path, self.title + f'_input_output_df_{self.current_epoch}.pkl')
                with open(path, 'wb') as f:
                    pkl.dump(self.df_object.__dict__, f)

            logger.info(f'Done SEM {self.run} at {self.current_epoch} epoch. is_train={self.is_train}!!!\n')
            if not os.path.exists(f'logs'):
                os.makedirs(f'logs')
            with open(f'logs/sem_complete_{self.sem_tag}.txt', 'a') as f:
                f.write(self.run + f'_{self.sem_tag}' + '\n')
        except Exception as e:
            with open(f'logs/sem_error_{self.sem_tag}.txt', 'a') as f:
                logger.error(f'{e}')
                f.write(self.run + f'_{self.sem_tag}' + '\n')
                f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc())

    def load_features(self):
        """
        :return:
        """
        df_dict = pkl.load(open(os.path.join(self.configs.preprocessed_features_path,
                                             f'{self.run}_{self.configs.feature_tag}.pkl'), 'rb'))
        df_object = DictObj(df_dict)

        self.last_frame = df_object.combined_resampled_df.index[-1]
        self.first_frame = df_object.combined_resampled_df.index[0]
        # This parameter is used to limit the time of ground truth video according to feature data
        self.end_second = math.ceil(self.last_frame / self.fps)
        self.first_second = math.ceil(self.first_frame / self.fps)
        self.df_object = df_object

    def calculate_correlation(self, pred_boundaries, grain='coarse'):
        # Process segmentation data (ground_truth)
        data_frame = pd.read_csv(self.configs.seg_path)
        seg_video = SegmentationVideo(data_frame=data_frame, video_path=self.movie)
        seg_video.get_human_segments(n_annotators=100, condition=grain, second_interval=self.second_interval)
        # Compare SEM boundaries versus participant boundaries
        last = min(len(pred_boundaries), self.end_second)
        # this function aggregate subject boundaries, apply a gaussian kernel
        seg_video.get_gt_freqs(second_interval=self.second_interval, end_second=last)
        # calculate correlations for all subjects
        biserials = seg_video.get_biserial_subjects(second_interval=self.second_interval, end_second=last)
        # compute biserial correlation and pearson_r of model boundaries
        bicorr = get_point_biserial(pred_boundaries[:last], seg_video.gt_freqs[:last])
        null_biserials = self.perm_biserial.get_null_bicorrs(
            run=self.run, nb=(pred_boundaries[:last] != 0).sum(), gt_freqs=seg_video.gt_freqs[:last])
        distance_to_null = bicorr - np.median(null_biserials)
        # get t-statistic of a value from a distribution
        t_stat_biserial = (bicorr - np.mean(null_biserials)) / np.std(null_biserials)
        pred_boundaries_gaussed = gaussian_filter1d(pred_boundaries.astype(float), 2)
        pearson_r, p = stats.pearsonr(pred_boundaries_gaussed[:last], seg_video.gt_freqs[:last])
        percentile_to_humans = percentileofscore(biserials, bicorr)
        logger.info(f'Tag={tag}: Scaled_bicorr={bicorr:.3f} cor. Distance_to_null={distance_to_null:.3f},  '
                    f'Subjects_median={np.nanmedian(biserials):.3f}')

        return bicorr, percentile_to_humans, pearson_r, seg_video, distance_to_null, t_stat_biserial

    def compute_clustering_metrics(self):
        start_second = self.first_frame / self.fps
        event_to_intervals = event_label_to_interval(self.sem_model.results.e_hat, start_second)
        df = pd.read_csv(f'{self.configs.action_path}')
        run_df = df[df['run'] == self.run.split('_')[0]]
        # calculate coverage
        coverage_df = pd.DataFrame(
            columns=['annotated_event', 'annotated_length', 'sem_max_overlap', 'max_coverage', 'epoch', 'run', 'tag', 'is_train'])
        for i, annotations in run_df.iterrows():
            ann_event, max_coverage_event, max_coverage = get_coverage(annotations, event_to_intervals)
            coverage_df.loc[len(coverage_df.index)] = [ann_event, annotations['endsec'] - annotations['startsec'],
                                                       max_coverage_event, max_coverage,
                                                       self.current_epoch, self.run, self.sem_tag, self.is_train]
        # calculate purity
        purity_df = pd.DataFrame(
            columns=['sem_event', 'sem_length', 'annotated_max_overlap', 'max_purity', 'epoch', 'run', 'tag', 'is_train'])
        for sem_event, sem_intervals in event_to_intervals.items():
            sem_event, max_purity_ann_event, max_purity = get_purity(sem_event, sem_intervals, run_df)
            sem_length = sum([interval[1] - interval[0] for interval in sem_intervals])
            purity_df.loc[len(purity_df.index)] = [sem_event, sem_length, max_purity_ann_event, max_purity,
                                                   self.current_epoch, self.run, self.sem_tag, self.is_train]
        purity_df.to_csv(path_or_buf=os.path.join(self.configs.sem_results_path, 'purity.csv'), index=False, header=False,
                         mode='a')
        coverage_df.to_csv(path_or_buf=os.path.join(self.configs.sem_results_path, 'coverage.csv'), index=False, header=False,
                           mode='a')
        average_coverage = np.average(coverage_df['max_coverage'], weights=coverage_df['annotated_length'])
        average_purity = np.average(purity_df['max_purity'], weights=purity_df['sem_length'])

        from sklearn.metrics import adjusted_mutual_info_score
        tempdf = pd.DataFrame()
        tempdf['ev'] = 'none'
        tempdf['sec'] = self.df_object.combined_resampled_df.index / self.fps
        for i in range(len(run_df)):
            ev = run_df.iloc[i]
            start = ev['startsec']
            end = ev['endsec']
            tempdf.loc[(tempdf['sec'] >= start) & (tempdf['sec'] <= end), 'ev'] = ev['evname']
        tempdf['ev_fact'] = tempdf['ev'].factorize()[0]
        mi = adjusted_mutual_info_score(self.sem_model.results.e_hat, tempdf['ev_fact'])
        return average_purity, average_coverage, mi

    def run_sem_and_log(self, x_train: np.ndarray, resampled_indices: pd.Series):
        """
        This method run SEM, log diagnostic results and plot prediction errors and boundaries
        :param x_train:
        :param resampled_indices:
        :return:
        """
        self.sem_model.run(x_train, train=self.is_train, **self.run_kwargs)
        # set k_prev to None in order to run the next video
        self.sem_model.k_prev = None
        # set x_prev to None in order to train the general event
        self.sem_model.x_prev = None

        ## Process results returned by SEM

        average_purity, average_coverage, mi = self.compute_clustering_metrics()

        timesteps = x_train.shape[0]
        logger.info(f'Total # of Steps={timesteps}')
        # Logging information about trigger timepoints
        trigger = self.sem_model.results.triggers.sum()
        logger.info(f'Total # of Triggers: {trigger}')
        # Logging some information about types of boundaries
        switch_old = (self.sem_model.results.boundaries == 1).sum()
        switch_new = (self.sem_model.results.boundaries == 2).sum()
        switch_current = (self.sem_model.results.boundaries == 3).sum()
        n_boundaries = (self.sem_model.results.boundaries != 0).sum()
        boundary_to_trigger = (switch_old + switch_new + switch_current) / trigger
        logger.info(f'Ratio of boundary to trigger: {boundary_to_trigger:.2f}')
        logger.info(f'Total # of OLD switches: {switch_old}')
        logger.info(f'Total # of NEW switches: {switch_new}')
        logger.info(f'Total # of RESTART switches: {switch_current}')
        if self.is_train:
            entropy = stats.entropy(self.sem_model.results.c) / np.log((self.sem_model.results.c > 0).sum())
        else:
            entropy = stats.entropy(self.sem_model.results.c_eval) / np.log((self.sem_model.results.c_eval > 0).sum())

        # replacing flurries of boundaries as a single boundary
        boundaries_no_flurry = remove_flurries(self.sem_model.results.boundaries.astype(bool).astype(int), k=3)
        n_boundaries_adjusted = np.sum(boundaries_no_flurry)
        pred_boundaries_adjusted = get_binned_prediction(boundaries_no_flurry, second_interval=self.second_interval,
                                                         sample_per_second=self.sample_per_second)
        removed = n_boundaries - n_boundaries_adjusted
        boundaries_random_removed = remove_random_boundaries(self.sem_model.results.boundaries.astype(bool).astype(int), removed)
        pred_boundaries_random_adjusted = get_binned_prediction(boundaries_random_removed, second_interval=self.second_interval,
                                                                sample_per_second=self.sample_per_second)

        pred_boundaries = get_binned_prediction(self.sem_model.results.boundaries, second_interval=self.second_interval,
                                                sample_per_second=self.sample_per_second)
        # Padding prediction boundaries, could be changed to have higher resolution but not necessary
        pred_boundaries = np.hstack([[0] * round(self.first_frame / self.fps / self.second_interval),
                                     pred_boundaries]).astype(int)
        pred_boundaries_adjusted = np.hstack([[0] * round(self.first_frame / self.fps / self.second_interval),
                                              pred_boundaries_adjusted]).astype(int)
        pred_boundaries_random_adjusted = np.hstack([[0] * round(self.first_frame / self.fps / self.second_interval),
                                                     pred_boundaries_random_adjusted]).astype(int)
        logger.info(f'Total # of 1s binned pred_boundaries: {sum(pred_boundaries)}')
        logger.info(f'Total # of event models: {len(self.sem_model.event_models) - 1}')
        threshold = 600
        active_event_models = np.count_nonzero(self.sem_model.c > threshold)
        logger.info(f'Total # of event models active more than {threshold // 3}s: {active_event_models}')

        mean_pe = self.sem_model.results.pe.mean()
        std_pe = self.sem_model.results.pe.std()
        # correlation of pe and uncertainty 1D arrays
        pe_unc_corr = np.corrcoef(self.sem_model.results.pe, self.sem_model.results.uncertainty)[0, 1]
        logger.info(f"Correlation of PE and Uncertainty: {pe_unc_corr:.2f}")
        # logging average scores for this run, these scores are used to plot scatter matrix across training
        path = os.path.join(self.configs.sem_results_path, 'stats_with_ratio.csv')
        with open(path, 'a') as f:
            writer = csv.writer(f)
            grain = 'coarse'
            bicorr, percentile_to_humans, pearson_r, seg_video, distance_to_null, t_stat_biserial = self.calculate_correlation(
                pred_boundaries=pred_boundaries, grain=f'{grain}')
            plot_diagnostic_readouts(seg_video.gt_freqs, self.sem_model.results,
                                     frame_interval=self.second_interval * self.sample_per_second,
                                     offset=self.first_frame / self.fps / self.second_interval,
                                     title=self.title + f'_diagnostic_{grain}_{self.current_epoch}',
                                     bicorr=bicorr, percentile=percentile_to_humans, pearson_r=pearson_r)
            bicorr_adjusted, _, _, _, _, t_stat_biserial_adjusted = self.calculate_correlation(
                pred_boundaries=pred_boundaries_adjusted, grain=f'{grain}')
            bicorr_random_adjusted, _, _, _, _, _ = self.calculate_correlation(
                pred_boundaries=pred_boundaries_random_adjusted, grain=f'{grain}')

            path = os.path.join(self.configs.sem_results_path, self.title + f'_gtfreqs_{grain}.pkl')
            with open(path, 'wb') as f:
                pkl.dump(seg_video.gt_freqs, f)
            # len adds 1, and the buffer model adds 1 => len() - 2
            writer.writerow(
                [self.run, grain, bicorr, percentile_to_humans, len(self.sem_model.event_models) - 2, active_event_models,
                 self.current_epoch, (self.sem_model.results.boundaries != 0).sum(), sem_init_kwargs, tag, mean_pe,
                 std_pe, pearson_r, self.is_train,
                 switch_old, switch_new, switch_current, entropy,
                 average_purity, average_coverage,
                 self.sem_model.results.triggers.sum(), timesteps,
                 distance_to_null, pe_unc_corr, t_stat_biserial, mi,
                 n_boundaries_adjusted, t_stat_biserial_adjusted, bicorr_adjusted, bicorr_random_adjusted])

            grain = 'fine'
            bicorr, percentile_to_humans, pearson_r, seg_video, distance_to_null, t_stat_biserial = self.calculate_correlation(
                pred_boundaries=pred_boundaries, grain=f'{grain}')
            bicorr_adjusted, _, _, _, _, t_stat_biserial_adjusted = self.calculate_correlation(
                pred_boundaries=pred_boundaries_adjusted, grain=f'{grain}')
            bicorr_random_adjusted, _, _, _, _, _ = self.calculate_correlation(
                pred_boundaries=pred_boundaries_random_adjusted, grain=f'{grain}')
            path = os.path.join(self.configs.sem_results_path, self.title + f'_gtfreqs_{grain}.pkl')
            with open(path, 'wb') as f:
                pkl.dump(seg_video.gt_freqs, f)
            writer.writerow(
                [self.run, grain, bicorr, percentile_to_humans, len(self.sem_model.event_models) - 2, active_event_models,
                 self.current_epoch, (self.sem_model.results.boundaries != 0).sum(), sem_init_kwargs, tag, mean_pe,
                 std_pe, pearson_r, self.is_train,
                 switch_old, switch_new, switch_current, entropy,
                 average_purity, average_coverage,
                 self.sem_model.results.triggers.sum(), timesteps,
                 distance_to_null, pe_unc_corr, t_stat_biserial, mi,
                 n_boundaries_adjusted, t_stat_biserial_adjusted, bicorr_adjusted, bicorr_random_adjusted])


        # logging SEM's diagnostic scores, these scores are used to inspect individual runs.
        self.sem_model.results.first_frame = self.first_frame
        self.sem_model.results.end_second = self.end_second
        self.sem_model.results.fps = self.fps
        self.sem_model.results.current_epoch = self.current_epoch
        self.sem_model.results.is_train = self.is_train
        self.sem_model.results.resampled_indices = resampled_indices
        path = os.path.join(self.configs.sem_results_path, self.title + f'_diagnostic_{self.current_epoch}.pkl')
        with open(path, 'wb') as f:
            pkl.dump(self.sem_model.results.__dict__, f)
        path = os.path.join(self.configs.sem_results_path, self.title + f'_gtfreqs.pkl')
        with open(path, 'wb') as f:
            pkl.dump(seg_video.gt_freqs, f)


if __name__ == "__main__":
    args = parse_config()
    logger.info(f'Config: {args}')

    csv_headers = ['run', 'grain', 'bicorr', 'percentile_to_human', 'n_event_models', 'active_event_models', 'epoch',
                   'n_boundaries', 'sem_params', 'tag', 'mean_pe', 'std_pe', 'pearson_r', 'is_train',
                   'switch_old', 'switch_new', 'switch_current', 'entropy',
                   'purity', 'coverage', 'n_triggers', 'n_timesteps',
                   'distance_to_null', 'pe_unc_corr', 't_stat_biserial', 'mi',
                   'n_boundaries_adjusted', 't_stat_biserial_adjusted', 'bicorr_adjusted', 'bicorr_random_adjusted']
    if not os.path.exists(f'{args.sem_results_path}'):
        os.makedirs(f'{args.sem_results_path}')
    path = os.path.join(args.sem_results_path, 'stats_with_ratio.csv')
    if not os.path.exists(path):
        path = os.path.join(args.sem_results_path, 'stats_with_ratio.csv')
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    else:
        # check if the intended header and the header in the existed file are the same
        df_metrics = pd.read_csv(path)
        diff = list(set(csv_headers).difference(set(df_metrics)))
        if len(diff) > 0:
            logger.info(f'set(csv_headers).difference(set(df_metrics))={diff},'
                        f'Adding these new columns and 0s values to the csv file.')
            for col in diff:
                df_metrics[col] = 0
            df_metrics.to_csv(path, index=False)
    if not os.path.exists(f'{args.sem_results_path}purity.csv'):
        csv_headers = ['sem_event', 'sem_length', 'annotated_max_overlap', 'max_purity', 'epoch', 'run', 'tag', 'is_train']
        path = os.path.join(args.sem_results_path, 'purity.csv')
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
    path = os.path.join(args.sem_results_path, 'coverage.csv')
    if not os.path.exists(path):
        csv_headers = ['annotated_event', 'annotated_length', 'sem_max_overlap', 'max_coverage', 'epoch', 'run', 'tag',
                       'is_train']
        path = os.path.join(args.sem_results_path, 'coverage.csv')
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    # Initialize keras model and running
    # Define model architecture, hyper parameters and optimizer
    f_class = GRUEvent
    # f_class = LSTMEvent
    optimizer_kwargs = dict(learning_rate=float(args.lr), beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=False)
    # these are the parameters for the event model itself.
    f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, dropout=0.5,
                  n_epochs=1, t=4, batch_update=True, n_hidden=int(args.n_hidden), variance_window=None,
                  optimizer_kwargs=optimizer_kwargs)
    # set the hyper-parameters for segmentation
    lmda = float(args.lmda)  # stickyness parameter (prior)
    alfa = float(args.alfa)  # concentration parameter (prior)
    kappa = float(args.kappa)
    threshold = float(args.threshold)  # threshold to trigger event selection based on PE or uncertainty
    sem_init_kwargs = {'lmda': lmda, 'alfa': alfa, 'kappa': kappa, 'f_opts': f_opts, 'f_class': f_class,
                       'threshold': threshold, 'trigger': args.trigger,
                       'equal_sigma': args.equal_sigma,
                       }
    logger.info(f'SEM parameters: {sem_init_kwargs}')
    # set default hyper-parameters for each run, can be overridden later
    run_kwargs = {'progress_bar': False}
    # run_kwargs = {'progress_bar': True}
    sem_model = SEM(**sem_init_kwargs)
    tag = args.sem_tag

    context_sem = SEMContext(sem_model=sem_model, run_kwargs=run_kwargs, configs=args)
    try:
        context_sem.read_train_valid_list()
        context_sem.iterate(is_eval=True)
        with open('output/tag_complete.txt', 'a') as f:
            f.write(f'{context_sem.sem_tag} completed after {context_sem.current_epoch} epochs' + '\n')
    except Exception as e:
        with open(f'output/tag_error_{context_sem.sem_tag}.txt', 'a') as f:
            f.write(traceback.format_exc() + '\n')
            print(traceback.format_exc())
