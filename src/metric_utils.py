"""
@author: Tan Nguyen
"""

import pickle as pkl
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.metrics import adjusted_mutual_info_score
import scipy.interpolate as interp
import re

import glob
import os
import sys
from joblib import Parallel, delayed
import random
from glob import glob
from copy import deepcopy

from copy import deepcopy
from tqdm import tqdm
from random import shuffle
import traceback

import cv2
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib import animation
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# py.offline.init_notebook_mode(connected=True)
sys.path.append(os.getcwd())
from src.utils import get_point_biserial, get_binned_prediction, SegmentationVideo, remove_flurries, logger
from typing import List, Dict

valid_runs = ['1.2.3', '2.4.1', '3.1.3', '6.3.9', '1.3.3', '2.2.10', '4.4.3', '6.1.8', '2.2.9', '1.1.6', '3.4.3', '1.3.6',
              '2.2.1', '6.3.4', '1.2.7', '4.4.2', '6.2.3', '4.3.5', '6.3.8', '2.4.9', '2.4.2', '6.1.5', '1.1.8', '3.1.7']
grain = 'fine'
annot_df = pd.read_csv('./resources/event_annotation_timing_average.csv')
run_specs = pd.read_csv('./resources/run_specs.csv')
seg_data = pd.read_csv('./resources/seg_data_analysis_clean.csv')
event_to_color = {'none': 'gray', 'perform jumping jacks': 'blue', 'do stair steps': 'blue', 'eat a granola bar': 'blue',
                  'drink water': 'blue', 'look at text message': 'black', 'bicep curls': 'blue', 'shoulder press': 'blue',
                  'take objects out of drawers': 'black', 'wash face': 'orange', 'shave face': 'orange', 'brush teeth': 'orange',
                  'apply chapstick': 'orange', 'brush hair': 'orange', 'drink sport drink': 'blue',
                  'put objects in drawers': 'black',
                  'jump rope': 'blue', 'fold socks': 'green', 'fold shirts or pants': 'green', 'put cases on pillows': 'green',
                  'put on bed sheets': 'green', 'clean a surface': 'black', 'use vacuum attachment': 'green',
                  'use hand duster': 'green',
                  'prepare a bagel': 'red', 'prepare hot oatmeal': 'red', 'prepare milk': 'red', 'prepare tea': 'red',
                  'prepare toast': 'red', 'prepare fresh fruit': 'red', 'sit ups': 'blue', 'torso rotations': 'black',
                  'vacuum floor': 'green', 'fold towels': 'green', 'fold blanket or comforter': 'green',
                  'prepare yogurt with granola': 'red',
                  'prepare orange juice': 'red', 'take a pill': 'black', 'push ups': 'blue', 'comb hair': 'orange',
                  'use mouthwash': 'orange', 'apply lotion': 'orange', 'floss teeth': 'orange',
                  'use hair gel': 'orange', 'prepare cereal': 'red',
                  'prepare instant coffee': 'red'}
# replace spaces with underscores
event_to_color = {re.sub(' ', '_', k): v for k, v in event_to_color.items()}


def movie_boundary_from_run(df_run) -> np.ndarray:
    movie_boundaries = np.hstack([[0], np.array(df_run)[1:] != np.array(df_run)[:-1]])
    return movie_boundaries


def boundary_from_ehat(df_e_hat) -> np.ndarray:
    boundaries = np.hstack([[0], np.array(df_e_hat)[1:] != np.array(df_e_hat)[:-1]])

    return boundaries


def in_intervals(boundary, intervals):
    for interval in intervals:
        if interval[0] < boundary < interval[1]:
            return True
    return False


def fair_shuffle(df_e_hat, df_run):
    # identify no boundary intervals
    concatenated_boundaries = np.hstack([[0], np.array(df_e_hat)[1:] != np.array(df_e_hat)[:-1]])
    b_indices = np.where(concatenated_boundaries == 1)[0]
    no_boundary_intervals = []
    run_switch_indices = np.hstack([[0], np.array(df_run)[1:] != np.array(df_run)[:-1]])
    run_switch_indices = np.where(run_switch_indices == 1)[0]
    for si in run_switch_indices:
        right = b_indices > si
        left = b_indices < si
        try:
            no_boundary_intervals.append([si - (np.min(si - b_indices[left])), np.min(b_indices[right] - si) + si])
        except Exception as e:
            # logger.info(e)
            pass
    # logger.info(f"intervals={no_boundary_intervals}")

    # a list of event lengths and event labels
    prev_e = np.array(df_e_hat)[0]
    length = 0
    grouped_events = []
    for i, e in enumerate(np.array(df_e_hat)):
        if e != prev_e:
            grouped_events.append([length, [prev_e] * length])
            prev_e = e
            length = 1
        else:
            length += 1
    ## for the last event
    grouped_events.append([length, [prev_e] * length])

    # permute without violating no boundary intervals
    grouped_events_cp = deepcopy(grouped_events)
    random_grouped_events = []
    last_boundary = -1
    while len(grouped_events_cp):
        random_index = np.random.randint(0, len(grouped_events_cp))
        count = 0
        while (in_intervals(last_boundary + grouped_events_cp[random_index][0], no_boundary_intervals)):
            # logger.info(f"{last_boundary} + {event_lengths[random_index]} in intervals={no_boundary_intervals}")
            count += 1
            if count > 100:
                #                 logger.info(f"Must violate at {last_boundary} + {grouped_events_cp[random_index][0]}")
                break
            random_index = np.random.randint(0, len(grouped_events_cp))
        last_boundary += grouped_events_cp[random_index][0]
        random_grouped_events.append(deepcopy(grouped_events_cp[random_index]))
        del grouped_events_cp[random_index]

    random_df_e_hat = pd.Series(sum([x[1] for x in random_grouped_events], []))
    random_df_e_hat.name = "e_hat"
    return random_df_e_hat


def shuffle_label_lengths(labels):
    x = list(labels)
    indexes = [0] + [index for index, _ in enumerate(x) if x[index] != x[index - 1]]
    indexes.append(len(x))
    groupedlist = [x[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if i != len(indexes) - 1]
    shuffle(groupedlist)
    final = pd.Series(np.concatenate(groupedlist).flat)
    final.name = "e_hat"
    return final


def shuffle_lengths(labels):
    x = list(labels)
    indexes = [0] + [index for index, _ in enumerate(x) if x[index] != x[index - 1]]
    indexes.append(len(x))
    groupedlist = [x[indexes[i]:indexes[i + 1]] for i, _ in enumerate(indexes) if i != len(indexes) - 1]
    schemas = list(set(labels))
    lengths = [len(e) for e in groupedlist]
    shuffle(lengths)
    # shuffle(groupedlist)
    final = [[np.random.choice(schemas)] * l for l in lengths]
    final = pd.Series(np.concatenate(final).flat)
    final.name = "e_hat"
    return final


def compute_schema_df(e_files):
    # build a dataframe of active SEM models and high-level event annotations
    schema_df = pd.DataFrame()
    last_sec = 0
    for dfile in tqdm(e_files):
        try:
            run = dfile.split('/')[-1].split('_')[0]
            epoch = dfile.split('/')[-1].split('_')[-1].split('.')[0]
            video_path = run + '_kinect_trim.mp4'
            # capture = cv2.VideoCapture(os.path.join('data/small_videos/', f'{video_path}'))
            # fps = capture.get(cv2.CAP_PROP_FPS)
            fps = float(run_specs[run_specs.run == run + "_kinect_trim"]['fps'])

            readout_dataframes = pkl.load(open(dfile, 'rb'))
            e_hat = readout_dataframes['e_hat']  # SEM active events
            boundaries = readout_dataframes['boundaries']
            pe = readout_dataframes['pe']
            pe_w = readout_dataframes['pe_w']
            pe_w2 = readout_dataframes['pe_w2']
            pe_w3 = readout_dataframes['pe_w3']
            # find frame index from input df:
            # input_file = dfile.replace('diagnostic', 'inputdf')   # for older tags
            input_file = dfile.replace('diagnostic', 'input_output_df')
            input_dataframes = pkl.load(open(input_file, 'rb'))
            # sec = input_dataframes.x_train.index / fps  # for older tags
            # fps can be weird if the video is corrupted, making sec infs
            sec = input_dataframes['combined_resampled_df'].index / fps

            tempdf = pd.DataFrame({'run': run, 'epoch': epoch, 'e_hat': e_hat, 'sec': sec,
                                   'pe': pe, 'pe_w': pe_w, 'pe_w2': pe_w2, 'pe_w3': pe_w3, 'tag': dfile.split('/')[-2],
                                   'gt_freqs': None,
                                   # 'boundary_prob': None,
                                   'eb_pe_rs': 0, 'eb_pe_w_rs': 0, 'eb_pe_w2_rs': 0, 'eb_pe_w3_rs': 0,
                                   'eb_pe': 0, 'eb_pe_w': 0, 'eb_pe_w2': 0, 'eb_pe_w3': 0})
            tempdf['ev'] = 'none'
            rundf = annot_df[annot_df['run'] == run]
            for i in range(len(rundf)):
                ev = rundf.iloc[i]
                start = ev['startsec']
                end = ev['endsec']
                tempdf.loc[(tempdf['sec'] >= start) & (tempdf['sec'] <= end), 'ev'] = ev['evname']

            # get human segmentation with the same interval
            seg_data = pd.read_csv('./resources/seg_data_analysis_clean.csv')
            seg_video = SegmentationVideo(data_frame=seg_data, video_path=video_path)
            seg_video.get_human_segments(n_annotators=100, condition=grain, second_interval=1)
            first_second = tempdf.sec[0]
            last_second = tempdf.sec[len(tempdf) - 1]
            # logger.info(last_second)
            # add 1 to have larger range of gt_freqs, avoid out of bound interpolation
            seg_video.get_gt_freqs(second_interval=1, end_second=np.ceil(last_second) + 1)
            arr1_interp = interp.interp1d(np.arange(len(seg_video.gt_freqs)), seg_video.gt_freqs)
            # add 0.1 to include the last time-point
            # the formula in np.arange ensure that gt_freqs_stretch and tempdf have the same length
            gt_freqs_stretch = arr1_interp(
                np.arange(first_second, last_second + 0.1, (last_second - first_second) / (len(tempdf) - 1)))
            tempdf['gt_freqs'] = gt_freqs_stretch

            # add last_sec for concatenation purpose
            tempdf['sec'] = tempdf['sec'] + last_sec
            last_sec += last_second

            # for validation purpose of sketching gt_freqs
            # plt.figure()
            # plt.plot(seg_video.gt_freqs, label='Original')
            # plt.plot(np.arange(first_second, last_second + 0.1, (last_second - first_second)/(len(tempdf) - 1)),
            #          gt_freqs_stretch, label='Interpolate')
            # plt.xlabel('Time (seconds)')
            # plt.ylabel('Boundary Probability')
            # plt.title(f"{video_path}.png")
            # plt.legend()
            # plt.savefig(f"check_interpolation_{video_path}.png")
            # plt.close()

            # boundaries from PEs
            tempdf['event_boundary'] = boundary_from_ehat(tempdf['e_hat'])
            tempdf['movie_boundary'] = movie_boundary_from_run(tempdf['run'])
            n_b = int(tempdf.event_boundary.sum())
            tempdf.loc[tempdf.nlargest(n_b, f'pe').index, f'eb_pe'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w2').index, f'eb_pe_w2'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w3').index, f'eb_pe_w3'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w').index, f'eb_pe_w'] = 1
            tempdf['event_boundary_rs'] = boundaries.astype(bool)
            n_b = int(tempdf.event_boundary_rs.sum())
            tempdf.loc[tempdf.nlargest(n_b, f'pe').index, f'eb_pe_rs'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w2').index, f'eb_pe_w2_rs'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w3').index, f'eb_pe_w3_rs'] = 1
            tempdf.loc[tempdf.nlargest(n_b, f'pe_w').index, f'eb_pe_w_rs'] = 1

            schema_df = pd.concat([schema_df, tempdf])

        except Exception as e:
            logger.info(traceback.format_exc())
            logger.info('error', dfile)
            return pd.DataFrame()
    # factorize event labels for numeric analyses:
    # logger.info(schema_df)
    schema_df['ev_fact'] = pd.factorize(schema_df['ev'])[0]
    schema_df['epoch'] = schema_df['epoch'].astype(int)
    return schema_df


def compute_schema_df_epoch(epoch, diag_files, eval_interval=10, last_epoch=108):
    # select only runs at validation point (both train and valid)
    if int(epoch) > last_epoch or int(epoch) % eval_interval != 1:
        return pd.DataFrame()
    e_files = [dfile for dfile in diag_files if dfile.split('/')[-1].split('_')[-1].split('.')[0] == epoch]
    if len(e_files) == 0:
        return pd.DataFrame()
    e_schema_df = compute_schema_df(e_files)
    return e_schema_df


def compute_schema_df_tag(tag, is_save=True):
    if os.path.exists(f"./output/dataframes/tag_schema_df_{tag}.csv"):
        logger.info(f'Load cached schema_df for tag {tag}!')
        return pd.read_csv(f"./output/dataframes/tag_schema_df_{tag}.csv")
    logger.info(f'Compute schema_df for tag {tag}!')
    diag_files = glob(f'output/run_sem/{tag}/*_diagnostic*.pkl')
    diag_files = [x.replace("\\", "/") for x in diag_files]
    if len(diag_files) == 0:
        logger.info(f'No diagnostic files for tag {tag}! return empty schema_df.')
        return pd.DataFrame()
    epochs = list(set([dfile.split('/')[-1].split('_')[-1].split('.')[0] for dfile in diag_files]))
    # the kernal keeps crashing, need to convert to serial mode
    e_schema_dfs = []
    for epoch in epochs:
        e_schema_dfs.append(compute_schema_df_epoch(epoch, diag_files))
#     e_schema_dfs = Parallel(n_jobs=11)(delayed(compute_schema_df_epoch)(epoch, diag_files)
#                                        for epoch in epochs)
    schema_df = pd.concat(e_schema_dfs)
    schema_df = schema_df.reset_index(drop=True)
    schema_df['epoch'] = schema_df['epoch'].astype(int)
    if is_save:
        if not os.path.exists("./output/dataframes/"):
            os.makedirs("./output/dataframes/")
        schema_df.to_csv(f"./output/dataframes/tag_schema_df_{tag}.csv", index=False)
    return schema_df


def average_biserial_run(sdf, run, boundary_from='event_boundary_rs', scale=True, remove_flurry=True):
    sdf = sdf[sdf.run == run]
    if len(sdf) == 0:
        return pd.DataFrame()
    # pred_boundaries = get_binned_prediction(sdf[boundary_from].to_numpy().astype(int), second_interval=1,
    #                                     sample_per_second=3).astype(bool)
    # human_segmentation = get_binned_prediction(sdf['gt_freqs'].to_numpy(), second_interval=1,
    #                                     sample_per_second=3).astype(float)
    # biserial = get_point_biserial(pred_boundaries, human_segmentation)
    if remove_flurry:
        boundaries_no_flurry = remove_flurries(sdf[boundary_from].to_numpy().astype(bool))
        biserial = get_point_biserial(np.array(boundaries_no_flurry), sdf['gt_freqs'], scale=scale)
    else:
        biserial = get_point_biserial(sdf[boundary_from], sdf['gt_freqs'], scale=scale)
    return pd.DataFrame([{'run': run, 'bicorr': biserial}])


def compute_biserial_epoch(tag, epoch, sdf, permute=10, scale=True, remove_flurry=True):
    try:
        # note: biserial excluding training videos
        edf = sdf[(sdf['epoch'] == epoch) & (sdf['tag'] == tag) & (sdf['run'].isin(valid_runs))]
        if len(edf) == 0:
            logger.info(f'epoch {epoch} is empty for {tag}! return empty biserial_df')
            return pd.DataFrame()
        # e_hat = np.array(edf['e_hat'])
        # concatenated_boundaries = np.hstack([[0], e_hat[1:] != e_hat[:-1]])
        run_bicorrs = Parallel(n_jobs=12)(delayed(average_biserial_run)(edf, run, 'event_boundary_rs', scale, remove_flurry)
                                          for run in edf.run.unique())
        biserial = pd.concat(run_bicorrs).reset_index(drop=True).mean(numeric_only=True)[0]
        # run_bicorrs = Parallel(n_jobs=12)(delayed(average_biserial_run)(edf, run, 'eb_pe_rs', scale, remove_flurry)
        #                                                                for run in sdf.run.unique())
        # bi_pe = pd.concat(run_bicorrs).reset_index(drop=True).mean(numeric_only=True)[0]
        # run_bicorrs = Parallel(n_jobs=12)(delayed(average_biserial_run)(edf, run, 'eb_pe_w_rs', scale, remove_flurry)
        #                                                                for run in sdf.run.unique())
        # bi_pe_w = pd.concat(run_bicorrs).reset_index(drop=True).mean(numeric_only=True)[0]
        # run_bicorrs = Parallel(n_jobs=12)(delayed(average_biserial_run)(edf, run, 'eb_pe_w2_rs', scale, remove_flurry)
        #                                                                for run in sdf.run.unique())
        # bi_pe_w2 = pd.concat(run_bicorrs).reset_index(drop=True).mean(numeric_only=True)[0]
        # run_bicorrs = Parallel(n_jobs=12)(delayed(average_biserial_run)(edf, run, 'eb_pe_w3_rs', scale, remove_flurry)
        #                                                                for run in sdf.run.unique())
        # bi_pe_w3 = pd.concat(run_bicorrs).reset_index(drop=True).mean(numeric_only=True)[0]

        bi_epoch_tag_df = pd.DataFrame({'epoch': epoch, 'tag': tag, 'bi': [biserial],
                                        'bi_shuffle': np.nan,
                                        # 'bi_shuffle_h_fair': np.nan,
                                        #    'bi_pe': [bi_pe], 'bi_pe_w':[bi_pe_w],
                                        #    'bi_pe_w2': [bi_pe_w2], 'bi_pe_w3': [bi_pe_w3]
                                        })
        for i in range(permute):
            # shuffled = np.array(fair_shuffle(df_e_hat=edf['e_hat'], df_run=edf['run']))
            # r_concatenated_boundaries = np.hstack([[0], shuffled[1:] != shuffled[:-1]])
            # r_biserial_fair = get_point_biserial(r_concatenated_boundaries, edf['gt_freqs'], scale=scale)
            shuffled = np.array(shuffle_label_lengths(labels=edf['e_hat']))
            r_concatenated_boundaries = np.hstack([[0], shuffled[1:] != shuffled[:-1]])
            r_biserial = get_point_biserial(r_concatenated_boundaries, edf['gt_freqs'], scale=scale)
            bi_epoch_tag_df = pd.concat([bi_epoch_tag_df, pd.DataFrame({'epoch': epoch, 'tag': tag,
                                                                        'bi': [biserial],
                                                                        'bi_shuffle': r_biserial,
                                                                        # 'bi_shuffle_h_fair': r_biserial_fair,
                                                                        #   'bi_pe': [bi_pe], 'bi_pe_w':[bi_pe_w],
                                                                        #   'bi_pe_w2': [bi_pe_w2], 'bi_pe_w3': [bi_pe_w3]
                                                                        })])
        return bi_epoch_tag_df
    except Exception as e:
        logger.info(f"Failed for {tag} at epoch={epoch} with error={e}")
        return pd.DataFrame()


def compute_biserial_tag(tag, remove_flurry=True, scale=True, is_save=True, debug=False):
    if os.path.exists(f'./output/dataframes/tag_biserial_df_{tag}_rmf{remove_flurry}.csv'):
        logger.info(f'Load cached biserial for {tag}')
        return pd.read_csv(f'./output/dataframes/tag_biserial_df_{tag}_rmf{remove_flurry}.csv')

    logger.info(f"Computing biserial_df for {tag} from its schema_df")
    tag_schema_df = compute_schema_df_tag(tag)
    if len(tag_schema_df) == 0:
        logger.info(f'schema_df is empty for {tag}! return empty biserial_df')
        return pd.DataFrame()
    if debug:
        tag_epoch_biserial_dfs = []
        for epoch in (tag_schema_df.epoch.unique()):  
            tag_epoch_biserial_dfs.append(compute_biserial_epoch(tag, epoch, tag_schema_df, scale=scale, remove_flurry=remove_flurry))
    else:
        tag_epoch_biserial_dfs = Parallel(n_jobs=11)(
            delayed(compute_biserial_epoch)(tag, epoch, tag_schema_df, scale=scale, remove_flurry=remove_flurry)
            for epoch in (tag_schema_df.epoch.unique()))
    tag_biserial_df = pd.concat(tag_epoch_biserial_dfs)
    tag_biserial_df['epoch'] = tag_biserial_df['epoch'].astype('int')
    if is_save:
        if not os.path.exists("./output/dataframes/"):
            os.makedirs("./output/dataframes/")
        tag_biserial_df.to_csv(f'./output/dataframes/tag_biserial_df_{tag}_rmf{remove_flurry}.csv', index=False)
    logger.info(f"Successfully computed biserial_df for {tag}")
    return tag_biserial_df


def compute_mi_epoch(tag, epoch, sdf, permute=10):
    # note: cached mi_df (saved before 07/29/23) includes training videos, recompute these caches
    edf = sdf[(sdf['epoch'] == epoch) & (sdf['tag'] == tag) & (sdf['run'].isin(valid_runs))]
    if len(edf) == 0:
        logger.info(f'epoch {epoch} is empty for {tag}! return empty mi_df')
        return pd.DataFrame()

    mi = adjusted_mutual_info_score(edf['e_hat'], edf['ev_fact'])
    mi_tag_epoch_df = pd.DataFrame({'epoch': epoch, 'tag': tag, 'mi': [mi],
                                    'mi_shuffle': np.nan,
                                    })
    for i in range(permute):
        shuffled = shuffle_label_lengths(edf['e_hat'])
        mip = adjusted_mutual_info_score(shuffled, edf['ev_fact'])
        temp_df = pd.DataFrame(
            {'epoch': epoch, 'tag': tag, 'mi': [mi], 'mi_shuffle': mip})
        mi_tag_epoch_df = pd.concat([mi_tag_epoch_df, temp_df])
    return mi_tag_epoch_df


def compute_mi_tag(tag, is_save=True):
    if os.path.exists(f'./output/dataframes/tag_mi_df_{tag}.csv'):
        logger.info(f'Load cached mi for {tag}')
        return pd.read_csv(f'./output/dataframes/tag_mi_df_{tag}.csv')

    logger.info(f"Computing mi_df for {tag} from its schema_df")
    tag_schema_df = compute_schema_df_tag(tag)
    if len(tag_schema_df) == 0:
        logger.info(f'schema_df is empty for {tag}! return empty mi_df')
        return pd.DataFrame()

    tag_epoch_mi_dfs = Parallel(n_jobs=11)(
        delayed(compute_mi_epoch)(tag, epoch, tag_schema_df)
        for epoch in (tag_schema_df.epoch.unique()))
    tag_mi_df = pd.concat(tag_epoch_mi_dfs)
    tag_mi_df['epoch'] = tag_mi_df['epoch'].astype('int')
    if is_save:
        if not os.path.exists("./output/dataframes/"):
            os.makedirs("./output/dataframes/")
        tag_mi_df.to_csv(f'./output/dataframes/tag_mi_df_{tag}.csv', index=False)
    logger.info(f"Successfully computed mi_df for {tag}")
    return tag_mi_df


def compute_pc(edf):
    """
    Compute purity and coverage. The two metrics could have the same or different ways of calculation,
    depending on what we're interested in. Thus, separate the two.
    """
    e_hats = edf.e_hat.unique()
    ev_facts = edf.ev_fact.unique()

    purity_df = pd.DataFrame(columns=["e_hat", "max_ev_fact", "purity", 'len_e_hat'])
    for e_hat in e_hats:
        max_ev_fact = -1
        max_purity = -1
        len_e_hat = len(edf[(edf.e_hat == e_hat)])
        for ev_fact in ev_facts:
            overlap = edf[(edf.e_hat == e_hat) & (edf.ev_fact == ev_fact)]
            purity = len(overlap) / len_e_hat
            if purity > max_purity:
                max_ev_fact = ev_fact
                max_purity = purity
        purity_df.loc[len(purity_df)] = [e_hat, max_ev_fact, max_purity, len_e_hat]

    coverage_df = pd.DataFrame(columns=["ev_fact", "max_e_hat", "coverage", 'len_ev_fact'])
    for ev_fact in ev_facts:
        max_e_hat = -1
        max_coverage = -1
        len_ev_fact = len(edf[(edf.ev_fact == ev_fact)])
        for e_hat in e_hats:
            overlap = edf[(edf.e_hat == e_hat) & (edf.ev_fact == ev_fact)]
            coverage = len(overlap) / len_ev_fact
            if coverage > max_coverage:
                max_e_hat = e_hat
                max_coverage = coverage
        coverage_df.loc[len(coverage_df)] = [ev_fact, max_e_hat, max_coverage, len_ev_fact]

    # average_purity = purity_df.purity.mean()
    # average_coverage = coverage_df.coverage.mean()

    average_purity = np.average(purity_df.purity, weights=purity_df.len_e_hat)
    average_coverage = np.average(coverage_df.coverage, weights=coverage_df.len_ev_fact)

    return average_purity, average_coverage


def compute_pc_epoch(tag, epoch, sdf, permute=10):
    # note: cached mi_df (saved before 07/29/23) includes training videos, recompute these caches
    edf = sdf[(sdf['epoch'] == epoch) & (sdf['tag'] == tag) & (sdf['run'].isin(valid_runs))]
    if len(edf) == 0:
        logger.info(f'epoch {epoch} is empty for {tag}! return empty pc_df')
        return pd.DataFrame()

    p, c = compute_pc(edf[["e_hat", "ev_fact"]])
    pc_tag_epoch_df = pd.DataFrame({'epoch': epoch, 'tag': tag, 'purity': [p], 'coverage': [c],
                                    'purity_shuffle': np.nan, 'coverage_shuffle': np.nan,
                                    })

    for i in range(permute):
        shuffled = shuffle_label_lengths(edf['e_hat'])
        p_shuffled, c_shuffled = compute_pc(
            pd.concat([shuffled.reset_index(drop=True), edf['ev_fact'].reset_index(drop=True)], axis=1))
        temp_df = pd.DataFrame({'epoch': epoch, 'tag': tag, 'purity': [p], 'coverage': [c],
                                'purity_shuffle': p_shuffled, 'coverage_shuffle': c_shuffled,
                                })
        pc_tag_epoch_df = pd.concat([pc_tag_epoch_df, temp_df])
    return pc_tag_epoch_df


def compute_pc_tag(tag, is_save=True):
    if os.path.exists(f'./output/dataframes/tag_pc_df_{tag}.csv'):
        logger.info(f'Load cached pc for {tag}')
        return pd.read_csv(f'./output/dataframes/tag_pc_df_{tag}.csv')

    logger.info(f"Computing pc_df for {tag} from its schema_df")
    tag_schema_df = compute_schema_df_tag(tag)
    if len(tag_schema_df) == 0:
        logger.info(f'schema_df is empty for {tag}! return empty pc_df')
        return pd.DataFrame()

    tag_epoch_pc_dfs = Parallel(n_jobs=11)(
        delayed(compute_pc_epoch)(tag, epoch, tag_schema_df)
        for epoch in (tag_schema_df.epoch.unique()))
    tag_pc_df = pd.concat(tag_epoch_pc_dfs)
    tag_pc_df['epoch'] = tag_pc_df['epoch'].astype('int')
    if is_save:
        if not os.path.exists("./output/dataframes/"):
            os.makedirs("./output/dataframes/")
        tag_pc_df.to_csv(f'./output/dataframes/tag_pc_df_{tag}.csv', index=False)
    logger.info(f"Successfully computed pc_df for {tag}")
    return tag_pc_df


def plot_instance_confusion_matrix(epoch, tag, schemas='active', permute=False):
    # epoch : int
    # schemas : 'all' includes blank rows for all schemas, 'active' is just schemas active in selected epoch
    schema_df = compute_schema_df_tag(tag)
    schema_df['ev_instance'] = schema_df['ev'] + ' ' + schema_df['run']
    schema_df['ev_fact_instance'] = pd.factorize(schema_df['ev_instance'])[0]
    f, ax = plt.subplots(figsize=(30, 20))
    tdf = schema_df[(schema_df['run'].isin(valid_runs)) & (schema_df['epoch'] == epoch) & (schema_df['ev_instance'] != 'none')]
    if permute:
        tdf['e_hat'] = np.random.permutation(tdf.e_hat.values)
    evdict = {}
    for evfac in tdf.ev_fact_instance.unique():
        evdict[evfac] = tdf[tdf.ev_fact_instance == evfac]['ev_instance'].iloc[0]
    # compute cross-tabs of SEM events and ground-truth annotations:
    ct = pd.crosstab(tdf.e_hat, tdf.ev_instance)

    # Omit 'none' timepoints:
    ct = ct[[x for x in ct.columns if 'none' not in x]]

    # Omit scripted actions that appear in a single instance in validation set:
    single_actions = ['prepare yogurt with granola', 'prepare orange juice',
                      'prepare milk', 'prepare fresh fruit', 'apply chapstick',
                      'use hair gel', 'sit ups', 'push ups', 'take a pill']
    # replace spaces with underscores
    single_actions = [re.sub(' ', '_', x) for x in single_actions]
    for single in single_actions:
        ct = ct[[x for x in ct.columns if single not in x]]

    # compute linear sum assignment to sort ground-truth labels maximizing match to SEM events:
    # Duplicate SEM rows to match number of ground truth labels if there are fewer SEM events than labels:
    if np.shape(ct)[1] > np.shape(ct)[0]:
        fac = int(np.ceil(np.shape(ct)[1] / np.shape(ct)[0]))
        padded = pd.concat([ct] * fac)
        padded = padded.sort_index()
        row_ind, col_ind = linear_sum_assignment(padded * -1)
    else:
        row_ind, col_ind = linear_sum_assignment(ct * -1)
    if schemas == 'all':
        for ehat in range(schema_df.e_hat.max() + 1):
            if ehat not in ct.index:
                df1 = pd.DataFrame([[np.nan] * len(ct.columns)], columns=ct.columns, index=[ehat])
                ct = ct.append(df1)
    elif schemas == 'active':
        pass
    ct = ct.sort_index()
    # Convert to proportions by column:
    ct = ct / ct.sum()
    '''
    # Add percent purity of schemas to index:
    schema_purity = zip(ct.index,round(ct.max(axis=1)/ct.sum(axis=1),2))
    ct.index=[str(x[0]) + ' - ' + str(x[1]) for x in schema_purity]
    # Add coverage:
    label_coverage = zip(ct.columns,round(ct.max(axis=0)/ct.sum(axis=0),2))
    ct.columns=[str(x[0]) + ' - ' + str(x[1]) for x in label_coverage]
    '''
    if len(ct.columns) == len(col_ind):
        plt.cla()
        # sort columns based on column index:
        # ctplot=ct.iloc[:,col_ind]
        # alphabetize columns:
        # ctplot=ct[sorted(ct.columns)]
        # sort by chapter type:
        cinst = [event_to_color[x.rstrip(' -.0123456789')] + ' ' + x for x in ct.columns]
        ctplot = ct[ct.columns[np.argsort(cinst)[::-1]]]
        # ctplot=ct[ct.columns[np.argsort(ct.columns.str.split('.').str[-3])]]
        # sns.heatmap(ctplot,annot=False,cbar=True,cmap='viridis',vmin=0.000,vmax=150,norm=LogNorm(),square=True)
        # ax = sns.heatmap(ctplot,annot=False,cbar=True,cmap='viridis',vmin=0,vmax=0.5,square=True,linewidths=0.1,linecolor='gray')
        ax = sns.heatmap(ctplot, annot=False, cbar=True, cmap='viridis', vmin=0, vmax=0.5, square=True, linewidths=0.1,
                         linecolor='gray', xticklabels=ctplot.columns)

        # sns.heatmap(ctplot, mask=ctplot != 0, cbar=False, color = "white")
        ax.invert_yaxis()
        for tick in ax.get_xticklabels():
            tick.set_color(event_to_color[tick._text.rstrip(' -.0123456789')])

        # Add activity type labels to x axis:
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', size=35))
                 for text, color in zip(['breakfast   ', 'bathroom                           ',
                                         'cleaning                                 ', 'exercise         ', 'multi-activity'],
                                        ['red', 'orange', 'green', 'blue', 'black'])]
        xbox = HPacker(children=boxes, align="center", pad=-175, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0, frameon=False, bbox_to_anchor=(0.1, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

        # Set colorbar font size:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=25)

        # Set axis labels
        plt.ylabel('SEM Schema', fontsize=35)
        plt.xlabel('')
        plt.title(f'{tag}\nProportion of scripted action label scenes assigned to each SEM schema', fontsize=35)
        # plt.title(f'Proportion of scripted action label scenes assigned to each SEM schema', fontsize=35)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_instance_epoch_{epoch}_{tag}.png', dpi=300)


if "__name__" == "__main__":
    name_to_tags = dict()
    # uncertainty_tags = [f'jan_01_uncertainty3E-03_s{x}_1E+00_1E-02' for x in range(1010, 1090, 10)]  # local
    uncertainty_tags = [f'jan_03_uncertainty3E-03_s{x}_1E+00_5E-03' for x in range(1010, 1090, 10)]  # remote
    name_to_tags['uncertainty'] = uncertainty_tags
    key = 'uncertainty'
    # schema_df = pd.DataFrame()
    # for tag in name_to_tags[key]:
    #     tag_schema_df = compute_schema_df_tag(tag)
    #     schema_df = pd.concat([schema_df, tag_schema_df])
    # paralelize the above
    tag_schema_dfs = Parallel(n_jobs=11)(delayed(compute_schema_df_tag)(tag) for tag in name_to_tags[key])
    schema_df = pd.concat(tag_schema_dfs)
    schema_df.to_csv(f'./output/dataframes/group_schema_df_{key}.csv', index=False)

    # biserial_df = pd.DataFrame()
    # for tag in name_to_tags[key]:
    #     tag_biserial_df = compute_biserial_tag(tag)
    #     biserial_df = pd.concat([biserial_df, tag_biserial_df])
    # paralelize the above
    tag_biserial_dfs = Parallel(n_jobs=11)(delayed(compute_biserial_tag)(tag) for tag in name_to_tags[key])
    biserial_df = pd.concat(tag_biserial_dfs)
    biserial_df.to_csv(f'./output/dataframes/group_biserial_df_{key}.csv', index=False)
