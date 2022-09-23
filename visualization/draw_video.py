"""
This script is used to draw and overlay predictions on video,
Run this file on the cluster instead of downloading pickle files and run locally
"""
import logging
import pickle as pkl
import pandas as pd
import numpy as np

import cv2
from scipy.ndimage import gaussian_filter1d
from glob import glob
import os
import sys
import matplotlib
import panel as pn
import sys
from typing import List

matplotlib.use('agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import joblib
from utils import get_point_biserial, get_binned_prediction
import scipy.stats as stats
import traceback
from utils import ColorBGR
import time
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap
import colorcet as cc
import matplotlib as mpl
from matplotlib.lines import Line2D
import math
from scipy.spatial.transform import Rotation as R


def remove_number(string):
    for i in range(100):
        string = string.replace(str(i), '')
    return string


def get_emb_category(category_distances, emb_dim=100):
    # Add 1 to avoid 3 objects with 0 distances (rare but might happen), then calculate inversed weights
    category_distances = category_distances + 1
    # Add 1 to avoid cases there is only one object
    category_weights = 1 - category_distances / (category_distances.sum() + 1)
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


def drawskel(frame_number, frame, skel_df, color=(255, 0, 0), thickness=2):
    # frame_number : video frame to select skeleton joints
    # frame : image frame to draw on
    # skel_df : df of skeleton joint coordinates
    # color : RGB tuple for color
    # thinkness : thickness to draw bone lines

    # Find scale factor of video frame. Original skeleton 2D dimensions are 1080 x 1920
    s = frame.shape[0] / 1080.0
    # ys = frame.shape[1]/1920.0
    # xs=1.0
    # ys=1.0
    r = skel_df[skel_df['frame'] == frame_number]

    def seg(bone, f):
        # draw segment between two joints
        ax = int(r['J' + str(bone[0]) + '_2D_X'].values[0] * s)
        ay = int(r['J' + str(bone[0]) + '_2D_Y'].values[0] * s)
        bx = int(r['J' + str(bone[1]) + '_2D_X'].values[0] * s)
        by = int(r['J' + str(bone[1]) + '_2D_Y'].values[0] * s)
        # print(ax,ay,bx,by)
        if all(x > 0 for x in (ax, ay, bx, by)):
            cv2.line(f, (ax, ay), (bx, by), color, thickness)
        return f

    for bone in [(3, 2), (2, 20), (20, 1), (1, 0),
                 (21, 7), (7, 6), (22, 6), (6, 5), (5, 4), (4, 20),
                 (20, 8), (8, 9), (9, 10), (24, 10), (10, 11), (11, 23),
                 (0, 12), (12, 13), (13, 14), (14, 15),
                 (0, 16), (16, 17), (17, 18), (18, 19)]:
        frame = seg(bone, frame)
    # Draw tracked vs inferred joints:
    for joint in range(25):
        jx = int(r['J' + str(joint) + '_2D_X'].values[0] * s)
        jy = int(r['J' + str(joint) + '_2D_Y'].values[0] * s)
        # jtrack = r['J' + str(joint) + '_Tracked'].values[0]
        if (all(x > 0 for x in [jx, jy])):
            if joint == 11:
                cv2.circle(frame, (jx, jy), 5, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (jx, jy), 3, color, -1)
    #             if jtrack=='Tracked':
    #                 #draw tracked joint
    #                 cv2.circle(frame,(jx,jy),4,(0,255,0),-1)
    #             elif jtrack=='Inferred':
    #                 #draw inferred joint
    #                 cv2.circle(frame,(jx,jy),4,(0,0,255),-1)
    #             elif jtrack=='Predicted':
    #                 cv2.circle(frame,(jx,jy),4,(255,0,0),-1)
    return frame


def get_nearest(emb_vector: List, space='glove'):
    if space == 'glove':
        # nearest_objects = glove_vectors.most_similar(emb_vector, restrict_vocab=10000)
        nearest_objects = glove_vectors.most_similar(emb_vector)
        nearest_objects = [(nr[0], round(nr[1], 2)) for nr in nearest_objects]
        return nearest_objects
    elif space == 'scene':
        # res = {kv[0]: np.linalg.norm(kv[1] - emb_vector) for kv in word2vec.items()}
        res = {kv[0]: cosine_similarity(kv[1], emb_vector)[0][0] for kv in scene_word2vec.items()}
        res = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
        res = [(nr[0], round(nr[1], 2)) for nr in res]
        return res
    elif space == 'corpus':
        res = {kv[0]: cosine_similarity(kv[1], emb_vector)[0][0] for kv in corpus_word2vec.items()}
        res = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
        res = [(nr[0], round(nr[1], 2)) for nr in res]
        return res


def drawobj(instances, frame, odf, color=(255, 0, 0), thickness=1, draw_name=False, tint=False, draw_rect=True):
    s = frame.shape[0] / 1080.0
    def contain_number(string: str):
        for i in range(100):
            if str(i) in string:
                return 1
        return 0
    tmp = []
    for i, ins in enumerate(instances):
        if contain_number(ins):
            tmp.append(ins)
            continue
        else:
            # choose the nearest instance of the category
            all_instances = odf.filter(like=ins).dropna().filter(like='dist_z')
            if len(all_instances.columns):
                tmp.append(all_instances.idxmax(axis='columns').values[0].replace('_dist_z', ''))
    instances = tmp
    for i in instances:
        xmin = odf[i + '_x'] * s
        ymin = odf[i + '_y'] * s
        xmax = xmin + (odf[i + '_w'] * s)
        ymax = ymin + (odf[i + '_h'] * s)
        try:
            conf_score = float(odf[i + '_confidence'])
            color_scaled = tuple(map(int, np.array(color) * conf_score))
        except:
            color_scaled = (0, 0, 255)
        if draw_rect:
            cv2.rectangle(frame, pt1=(int(xmin), int(ymin)),
                          pt2=(int(xmax), int(ymax)),
                          color=color_scaled, thickness=thickness)
        if tint:
            frame = frame.astype(int)

            def colorize(image, hue, saturation=1):
                """ Add color of the given hue to an RGB image.

                By default, set the saturation to 1 so that the colors pop!
                """
                hsv = color.rgb2hsv(image)
                hsv[:, :, 1] = saturation
                hsv[:, :, 0] = hue
                return color.hsv2rgb(hsv)
            frame[int(ymin): int(ymax), int(xmin): int(xmax), 1:] += int(50 * conf_score)
            frame = np.clip(frame, 0, 255)
            frame = frame.astype(np.uint8)
        if draw_name:
            cv2.putText(frame, text=i,
                        org=(int(xmin), int(ymax - 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color_scaled)
    # Code to also get nearest objects in Glove for input categories
    # try:
    #     sr = objdf.loc[frame_number].dropna().filter(regex='_dist$').rename(lambda x: remove_number(x).replace('_dist', ''))
    #     arr = get_emb_category(sr, emb_dim=50)
    #     nearest_objects = get_nearest(arr, glove=True)
    #     for index, instance in enumerate(nearest_objects[:3]):
    #         cv2.putText(frame, text=str(instance), org=(frame.shape[1] - 420, 20 + 20 * index),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
    #                     color=(255, 0, 0))
    # except Exception as e:
    #     print(e)
    return frame


def draw_frame_resampled(frame_slider, skel_checkbox, obj_checkbox, run_select, get_img=False, black=False):
    outframe = deepcopy(anchored_frames[frame_slider])
    if black:
        outframe[:, :, :] = 0
    # draw skeleton here
    if skel_checkbox:
        try:
            outframe = drawskel(frame_slider, outframe, skel_df, color=(255, 0, 0))
            # TODO: comment these lines if not using position in training SEM.
            # outframe = drawskel(frame_slider, outframe, pca_input_df, color=(255, 0, 0))
            # outframe = drawskel(frame_slider, outframe, pred_skel_df, color=(0, 255, 0))
        except Exception as e:
            cv2.putText(outframe, 'No skeleton data', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # print(traceback.format_exc())
    else:
        outframe = anchored_frames[frame_slider]
    if obj_checkbox:
        try:
            # Draw boxes for nearest objects and background objects
            odf_z = objhand_df[objhand_df.index == frame_slider]
            odf_z = odf_z[odf_z.columns[~odf_z.isna().any()].tolist()]
            sorted_objects = list(pd.Series(odf_z.filter(regex='dist_z$').iloc[0, :]).sort_values().index)
            sorted_objects = [object.replace('_dist_z', '') for object in sorted_objects]
            outframe = drawobj(sorted_objects[:3], outframe, odf_z, color=ColorBGR.red, draw_name=False)
            outframe = drawobj(sorted_objects[3:], outframe, odf_z, color=ColorBGR.cyan, draw_name=False)

            # Draw nearest words (in the video)
            # nearest_objects = get_nearest(pred_objhand.loc[frame_slider, :].values)
            # for index, instance in enumerate(nearest_objects[:3]):
            #     cv2.putText(outframe, text=str(instance), org=(outframe.shape[1] - 140, 20 + 20 * index),
            #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            #                 color=ColorBGR.green)
            # Draw nearest objects (Glove corpus)
            # Put three nearest words from glove to pre-sampling input vector
            # input_nearest_objects = get_nearest(
            #     [np.array(inputdf.objhand_pre.loc[frame_slider].values, dtype=np.float32)],
            #     glove=True)
            # for index, instance in enumerate(input_nearest_objects[:3]):
            #     cv2.putText(outframe, text=str(instance), org=(outframe.shape[1] - 450, 20 + 20 * index),
            #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            #                 color=ColorBGR.red)

            # Put three nearest words from glove to post-sampling, pre-PCA input vector
            # input_nearest_objects = get_nearest(
            #     [np.array(inputdf.objhand_post.loc[frame_slider].values, dtype=np.float32)],
            #     glove=True)
            # for index, instance in enumerate(input_nearest_objects[:3]):
            #     cv2.putText(outframe, text=str(instance), org=(outframe.shape[1] - 450, 20 + 20 * index),
            #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            #                 color=ColorBGR.blue)
            out_obj_frame = np.zeros(shape=(outframe.shape[0] // 2, outframe.shape[1], outframe.shape[2]), dtype=np.uint8)
            # Put three nearest words from scene to input vector
            input_nearest_objects = get_nearest(
                [np.array(inputdf.x_train_inverted.loc[frame_slider, inputdf.objhand_post.columns].values, dtype=np.float32)],
                space='scene')
            for index, instance in enumerate(input_nearest_objects[:3]):
                cv2.putText(out_obj_frame, text=str(instance), org=(out_obj_frame.shape[1] - 450, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.red)
            cv2.putText(out_obj_frame, text=f'Input (Scene)', org=(out_obj_frame.shape[1] - 450, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.red)
            # Put three nearest words from scene to prediction vector
            pred_nearest_objects = get_nearest([np.array(pred_objhand.loc[frame_slider, :].values, dtype=np.float32)],
                                               space='scene')
            for index, instance in enumerate(pred_nearest_objects[:3]):
                cv2.putText(out_obj_frame, text=str(instance), org=(out_obj_frame.shape[1] - 300, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.green)
            # draw boxes for objects predicted to be near the hand
            instances = [pr[0] for pr in pred_nearest_objects]
            outframe = drawobj(instances[:3], outframe, odf_z, color=ColorBGR.green, draw_name=False, tint=True, draw_rect=False)
            cv2.putText(out_obj_frame, text=f'Predicted (Scene)', org=(out_obj_frame.shape[1] - 300, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.green)
            # Put three nearest words from corpus to prediction vector
            pred_nearest_objects = get_nearest([np.array(pred_objhand.loc[frame_slider, :].values, dtype=np.float32)],
                                               space='corpus')
            for index, instance in enumerate(pred_nearest_objects[:3]):
                cv2.putText(out_obj_frame, text=str(instance), org=(out_obj_frame.shape[1] - 150, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.magenta)
            cv2.putText(out_obj_frame, text=f'Predicted (Corpus)', org=(out_obj_frame.shape[1] - 150, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.magenta)

        except Exception as e:
            e
            # print(traceback.format_exc())

    cv2.putText(outframe, text=f'RED: 3 Nearest Objects (GT)', org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.red)

    cv2.putText(outframe, text=f'CYAN: Background Objects (GT)', org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.cyan)
    cv2.putText(outframe, text=f'TINTED: Predicted Nearest Objects', org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.yellow)

    # Testing the effect of each stage of processing objhand feature
    # cv2.putText(outframe, text=f'RED: Pre-sampling', org=(10, 120),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
    #             color=ColorBGR.red)
    #
    # cv2.putText(outframe, text=f'Blue: Post-sampling, Pre-PCA', org=(10, 140),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
    #             color=ColorBGR.blue)
    # cv2.putText(outframe, text=f'Magenta: Post-PCA', org=(10, 160),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
    #             color=ColorBGR.magenta)

    # add frameID
    cv2.putText(outframe, text=f'FrameID: {frame_slider}', org=(10, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0))
    # add Segmentation flag
    index = pred_objhand.index.get_indexer([frame_slider])[0]
    # if sem_readouts['e_hat'][index] != sem_readouts['e_hat'][index - 1]:
    if sem_readouts['boundaries'][index] == 1:
            cv2.putText(outframe, text='SWITCH TO OLD', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))
    if sem_readouts['boundaries'][index] == 2:
        cv2.putText(outframe, text='CREATE NEW', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))
    if sem_readouts['boundaries'][index] == 3:
        cv2.putText(outframe, text='RESTART CURRENT', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))

    if get_img:
        return outframe, out_obj_frame

    # embedding image on axis to align
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(cv2.cvtColor(outframe, cv2.COLOR_BGR2RGB))
    plt.close(fig)
    return fig, out_obj_frame


def impose_rainbow_events(ax, fig):
    cm = plt.get_cmap('gist_rainbow')
    post = sem_readouts['e_hat']
    boundaries = sem_readouts['boundaries']
    NUM_COLORS = post.max()
    # Hard-code 40 events for rainbow to be able to compare across epochs
    # NUM_COLORS = 30
    for i, (b, e) in enumerate(zip(boundaries, post)):
        if b != 0:
            second = i / frame_interval + offset
            if b == 1:
                ax.axvline(second, linestyle=(0, (5, 10)), alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Old Event')
            elif b == 2:
                ax.axvline(second, linestyle='solid', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='New Event')
            elif b == 3:
                ax.axvline(second, linestyle='dotted', alpha=0.3, color=cm(1. * e / NUM_COLORS), label='Restart Event')
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(vmin=0, vmax=NUM_COLORS, clip=False)),
                 orientation='horizontal')


def impose_metrics(ax, fig):
    pred_boundaries = get_binned_prediction(sem_readouts['post'], second_interval=second_interval,
                                            sample_per_second=3)
    # Padding prediction boundaries, could be changed to have higher resolution but not necessary
    pred_boundaries = np.hstack([[0] * round(first_frame / fps / second_interval), pred_boundaries]).astype(bool)
    #     gt_freqs_local = gaussian_filter1d(gt_freqs, 2)
    last = min(len(pred_boundaries), len(gt_freqs))
    bicorr = get_point_biserial(pred_boundaries[:last], gt_freqs[:last])
    pred_boundaries_gaussed = gaussian_filter1d(pred_boundaries.astype(float), 1)
    pearson_r, p = stats.pearsonr(pred_boundaries_gaussed[:last], gt_freqs[:last])
    ax.text(0.1, 0.3, f'bicorr={bicorr:.3f}, pearson={pearson_r:.3f}', fontsize=14)
    ax.set_ylim([0, 0.4])


def impose_line_boundaries(ax, fig):
    from matplotlib.lines import Line2D
    linestyles = ['dashed', 'solid', 'dotted']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=ls) for ls in linestyles]
    labels = ['Old Event', 'New Event', 'Restart Event']
    ax.legend(lines, labels, loc='upper right')


def plot_boundaries(ax, fig):
    num_colors = 0
    num_colors = max(num_colors, sem_readouts['e_hat'].max())
    cmap = ListedColormap(cc.glasbey_dark[:num_colors + 1])
    semmin = 0
    semmax = num_colors
    cmap1 = ListedColormap(cc.glasbey_dark[semmin:semmax + 1])
    norm = mpl.colors.Normalize(vmin=semmin, vmax=semmax + 1, clip=False)
    cbar1_ax = fig.add_axes([.91, .1, .02, .8])
    cbar1 = mpl.colorbar.ColorbarBase(ax=cbar1_ax, cmap=cmap1, norm=norm)
    r1 = cbar1.vmax - cbar1.vmin
    cbar1.set_ticks([((cbar1.vmin + r1) / (semmax + 1)) * (0.5 + i) for i in range(semmin, semmax + 1)])
    cbar1.set_ticklabels(range(semmin, semmax + 1))
    cbar1.set_label('SEM Events')

    boundaries = sem_readouts['boundaries']
    e_hat = sem_readouts['e_hat']
    for i, (b, e) in enumerate(zip(boundaries, e_hat)):
        if b != 0:
            second = i / frame_interval + offset
            if b == 1:  # Switch to an old event
                ax.axvline(second, linestyle=(0, (5, 10)), linewidth=1, alpha=1, color=cmap(1. * e / num_colors), label='Old Event')
            elif b == 2:  # Create a new event
                ax.axvline(second, linestyle='solid', linewidth=1, alpha=1, color=cmap(1. * e / num_colors), label='New Event')
            elif b == 3:  # Restart the current event
                ax.axvline(second, linestyle='dotted', linewidth=1, alpha=1, color=cmap(1. * e / num_colors), label='Restart Event')
    linestyles = ['dashed', 'solid', 'dotted']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=ls) for ls in linestyles]
    labels = ['Switch to an Old Event', 'Switch to a New Event', 'Restart the Current Event']
    ax.legend(lines, labels, loc='upper right')


def plot_diagnostic_readouts(frame_slider, run_select, get_img=False, ax=None, fig=None):
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gaussian_filter1d(gt_freqs, 1), label='Subject Boundaries')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Boundary Probability')
    ax.set_ylim([0, max(gt_freqs) + max(gt_freqs) * 0.2])
    ax.axvline(frame_slider / fps, linewidth=3, alpha=0.5, color='r')
    ax.set_title(f"SEM's and humans' boundaries for activity {run_select}")
    plot_boundaries(ax=ax, fig=fig)
    if get_img:
        fig.canvas.draw()
        image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return image_from_plot
    # plt.close()
    return fig


def plot_pe(run_select, tag_select, epoch_select, frame_slider, get_img=True, ax=None, fig=None):
    # print('Plot Prediction Error')
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    fig.suptitle(f'{tag_select} - {run_select}')
    df = pd.DataFrame(
        columns=['pe', 'pe_w',
                 # 'pe_w2', 'pe_w3', 'pe_w3_yoke',
                 'feature_change', 'prediction_change', 'second', 'epoch'])

    prediction_change = np.linalg.norm(np.vstack([np.zeros(shape=(1, sem_readouts['x_hat'].shape[1])),
                                                  (sem_readouts['x_hat'][1:] - sem_readouts['x_hat'][:-1])]), axis=1)

    input_feature = inputdf.x_train_pca.to_numpy() / np.sqrt(inputdf.x_train_pca.shape[1])
    feature_change = np.linalg.norm(np.vstack([np.zeros(shape=(1, input_feature.shape[1])),
                                               (input_feature[1:] - input_feature[:-1])]), axis=1)
    df1 = pd.DataFrame({'pe': sem_readouts['pe'], 'pe_w': sem_readouts['pe_w'] if 'oct' in tag_select else sem_readouts['pe'],
                        # 'pe_w2': v['pe_w2'] if 'oct' in tag_select else v['pe'],
                        # 'pe_w3': v['pe_w3'] if 'oct' in tag_select else v['pe'],
                        # 'pe_w3_yoke': v['pe_yoke'] if 'oct' in tag_select else v['pe'],
                        'feature_change': feature_change,
                        'prediction_change': prediction_change,
                        'second': inputdf.appear_post.index / 25,
                        'epoch': epoch_select}, index=inputdf.appear_post.index)
    df = df.append(df1)
    df.epoch = pd.to_numeric(df.epoch, errors='coerce')

    ax.plot(df['second'], df['pe'])
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Prediction Error')
    ax.axvline(frame_slider / fps, linewidth=2, alpha=0.5, color='r')
    ax.set_title('Prediction Error for ' + run_select)
    plot_boundaries(ax=ax, fig=fig)
    if get_img:
        fig.canvas.draw()
        image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return image_from_plot
    # plt.close()
    return fig


def plot_pe_and_diag(run_select, tag_select, epoch_select, frame_slider, get_img=True):
    fig, axes = plt.subplots(nrows=1, figsize=(8, 4.5), sharex='all', squeeze=False)
    plot_diagnostic_readouts(frame_slider, run_select, ax=axes[0][0], fig=fig, get_img=False)
    # plot_pe(run_select, tag_select, epoch_select, frame_slider, ax=axes[0], fig=fig, get_img=False)
    # plt.savefig('test.png')
    if get_img:
        fig.canvas.draw()
        image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return image_from_plot


def draw_static_body(frame_number, skel_df, prop='speed'):
    # frame_number : video frame to select skeleton joints
    # skel_df : df of skeleton joint coordinates
    # prop : string, property of joint to draw. One of: ['speed', 'acceleration', 'dist_from_J1']

    # Find scale factor of video frame. Original skeleton 2D dimensions are 1080 x 1920
    # s = frame.shape[0] / 1080.0
    # ys = frame.shape[1]/1920.0
    # xs=1.0
    # ys=1.0
    bodybase = './outline.png'
    frame = cv2.imread(bodybase, 1)
    r = skel_df[skel_df['frame'] == frame_number]

    j_coords = [(150, 275),  # 0 SpineBase
                (150, 175),  # 1 SpineMid
                (150, 90),  # 2 Neck
                (150, 50),  # 3 Head
                (200, 130),  # 4 ShoulderLeft
                (220, 210),  # 5 ElbowLeft
                (245, 265),  # 6 WristLeft
                (260, 300),  # 7 HandLeft
                (100, 130),  # 8 ShoulderRight
                (80, 210),  # 9 ElbowRight
                (55, 265),  # 10 WristRight
                (40, 300),  # 11 HandRight
                (190, 275),  # 12 HipLeft
                (185, 420),  # 13 KneeLeft
                (185, 500),  # 14 AnkleLeft
                (185, 550),  # 15 FootLeft
                (110, 275),  # 16 HipRight
                (115, 420),  # 17 KneeRight
                (105, 500),  # 18 AnkleRight
                (105, 550),  # 19 FootRight
                (150, 130),  # 20 SpineShoulder
                (270, 330),  # 21 HandTipLeft
                (290, 300),  # 22 ThumbLeft
                (30, 330),  # 23 HandTipRight
                (10, 300),  # 24 ThumbRight
                ]
    for j in range(25):
        jmax = combined_runs['J' + str(j) + '_' + prop].quantile(.95)
        jmin = combined_runs['J' + str(j) + '_' + prop].quantile(.05)
        jval = r['J' + str(j) + '_' + prop].values[0]
        if jval < jmin:
            jval = jmin
        if jval > jmax:
            jval = jmax
        if jmax == jmin:
            # J1 to J1
            p = 0
        else:
            p = (jval - jmin) / (jmax - jmin)
        cv2.circle(frame, j_coords[j], 15, (0, 255 * p, 255 - 255 * p), -1)
    cv2.putText(frame, text=f'{prop}', org=(10, 45),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55,
                color=(255, 255, 255))
    return frame


def draw_interhand_features(frame_number, skel_df, feature, location, frame=None):
    # frame_number : video frame to select skeleton joints
    # skel_df : df of skeleton joint coordinates
    # feature : string, property of joint to draw. One of: ['interhand_dist', 'interhand_speed', 'interhand_acceleration']
    # location : coordinates of where to draw feature
    # Find scale factor of video frame. Original skeleton 2D dimensions are 1080 x 1920
    # s = frame.shape[0] / 1080.0
    # ys = frame.shape[1]/1920.0
    # xs=1.0
    # ys=1.0
    # bodybase='/Users/bezdek/Desktop/outline.jpeg'
    r = skel_df[skel_df['frame'] == frame_number]
    jmax = combined_runs[feature].quantile(.95)
    jmin = combined_runs[feature].quantile(.05)
    jval = r[feature].values[0]
    if jval < jmin:
        jval = jmin
    if jval > jmax:
        jval = jmax
    p = (jval - jmin) / (jmax - jmin)
    cv2.circle(frame, (location[0] + 40, location[1] + 40), 15, (0, 255 * p, 255 - 255 * p), -1)
    cv2.putText(frame, text=feature, org=(location),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55,
                color=(255, 255, 255))

    return frame


def draw_skeleton_ball(frame_slider):
    try:
        frame = np.zeros((600, 200, 3))
        for f, l in [('interhand_dist', (10, 20)),
                     ('interhand_speed', (10, 100)),
                     ('interhand_acceleration', (10, 200))]:
            frame = draw_interhand_features(frame_slider, skel_df, f, l, frame)
        frame = frame.astype(np.uint8)
        outframe = cv2.hconcat([draw_static_body(frame_slider, skel_df, prop='speed'),
                                draw_static_body(frame_slider, skel_df, prop='acceleration'),
                                draw_static_body(frame_slider, skel_df, prop='dist_from_J1'),
                                frame])
    except Exception as e:
        outframe = np.zeros((600, 1100, 3))
        cv2.putText(outframe, 'No skeleton data', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # logging.error(f'{traceback.format_exc()}')

    # add frameID
    cv2.putText(outframe, text=f'Run: {run_select} FrameID: {frame_slider}', org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55,
                color=(255, 255, 255))

    return outframe


def calc_joint_rel_position(df):
    # left shoulder : J4
    # right shoulder : J8
    # df : skeleton tracking dataframe with 3D joint coordinates
    # returns df with columns for joints translated to SpineMid as origin and shoulders rotated to same Z value
    # Translate all joints to J1 as origin:
    for j in range(25):
        for dim in ['X', 'Y', 'Z']:
            df[f'J{j}_3D_rel_{dim}'] = df[f'J{j}_3D_{dim}'] - df[f'J1_3D_{dim}']
    # calculate angle for rotation and rotate around the Y-axis:
    rotation_radians = math.atan2(df['J4_3D_rel_Z'] - df['J8_3D_rel_Z'], df['J4_3D_rel_X'] - df['J8_3D_rel_X'])
    rotation_vector = rotation_radians * np.array([0, 1, 0])  # Y-axis
    rotation = R.from_rotvec(rotation_vector)
    for j in range(25):
        jxout = 'J' + str(j) + '_3D_rel_X'
        jyout = 'J' + str(j) + '_3D_rel_Y'
        jzout = 'J' + str(j) + '_3D_rel_Z'
        df[jxout], df[jyout], df[jzout] = rotation.apply((df[jxout], df[jyout], df[jzout]))
    return df


def anim_event_series(e_array, view, title=''):
    # PARAMETERS:
    # e_series: array of 25 joints in X,Y,Z depth space
    #    i.e., [X1,X2,X3,...,X25,Y1,Y2,Y3,...,Y25,Z1,Z2,Z3,...,Z25],
    #    with a row for each timepoint of joint positions recorded.
    #
    # out_file: output name for the animation. HTML format works.
    # view: front or side
    # t: frame id

    # plt.rcParams['animation.ffmpeg_path'] = '/home/n.tan/.conda/envs/tf-37-new/bin/ffmpeg'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if view == 'front':
        ax.view_init(elev=5., azim=270.)  # frontward
    elif view == 'side':
        ax.view_init(elev=5., azim=180.)  # side
    # ax = Axes3D(fig)
    # Setting the axes properties
    ax.set_xlim3d([-1.5, 1.0])
    ax.set_xlabel('X')
    ax.set_zlim3d([-1.5, 1.0])
    ax.set_zlabel('Y')
    ax.set_ylim3d([-1.5, 1.0])
    ax.set_ylabel('Z')
    ax.set_title(title)

    # def update_graph(t):
    #     plt.cla()
    #     print(t)
    if view == 'front':
        ax.view_init(elev=5., azim=270.)  # frontward
    elif view == 'side':
        ax.view_init(elev=5., azim=180.)  # side
        # ax.set_xlim3d([-1.5, 1.0])
        # ax.set_xlabel('X')
        # ax.set_zlim3d([-1.5, 1.0])
        # ax.set_zlabel('Y')
        # ax.set_ylim3d([-1.5, 1.0])
        # ax.set_ylabel('Z')

    x = [(-1) * i for i in e_array[0:25]]
    # x = e_series[t, 0:25]
    y = e_array[25:50]
    z = e_array[50:75]

    # Plotting bones by connecting the joints
    spine_x = [x[i] for i in [3, 2, 20, 1, 0]]
    spine_y = [y[i] for i in [3, 2, 20, 1, 0]]
    spine_z = [z[i] for i in [3, 2, 20, 1, 0]]
    ax.plot(spine_x, spine_z, spine_y, color='b')

    l_arm_x = [x[i] for i in [20, 4, 5, 6, 7, 21]]
    l_arm_y = [y[i] for i in [20, 4, 5, 6, 7, 21]]
    l_arm_z = [z[i] for i in [20, 4, 5, 6, 7, 21]]
    ax.plot(l_arm_x, l_arm_z, l_arm_y, color='b')

    r_arm_x = [x[i] for i in [20, 8, 9, 10, 11, 23]]
    r_arm_y = [y[i] for i in [20, 8, 9, 10, 11, 23]]
    r_arm_z = [z[i] for i in [20, 8, 9, 10, 11, 23]]
    ax.plot(r_arm_x, r_arm_z, r_arm_y, color='b')

    l_leg_x = [x[i] for i in [0, 12, 13, 14, 15]]
    l_leg_y = [y[i] for i in [0, 12, 13, 14, 15]]
    l_leg_z = [z[i] for i in [0, 12, 13, 14, 15]]
    ax.plot(l_leg_x, l_leg_z, l_leg_y, color='b')

    r_leg_x = [x[i] for i in [0, 16, 17, 18, 19]]
    r_leg_y = [y[i] for i in [0, 16, 17, 18, 19]]
    r_leg_z = [z[i] for i in [0, 16, 17, 18, 19]]
    ax.plot(r_leg_x, r_leg_z, r_leg_y, color='b')

    fig.canvas.draw()
    image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    return image_from_plot

    # Creating the Animation object
    # line_ani = animation.FuncAnimation(fig, update_graph, len(e_series),
    #                                    interval=1, blit=False)
    # writervideo = animation.FFMpegWriter(fps=3, codec='xvid')
    # writervideo = animation.MovieWriter(fps=3)
    # line_ani.save('output/videos/test.png', writer='imagemagick')
    # line_ani.save(out_file, writer=writervideo)


def draw_video():
    output_video_path = f'output/videos/{run_select}_{tag}_{epoch}.avi'
    if not os.path.exists(f'../output/videos'):
        os.makedirs('../output/videos')
    if os.path.exists(output_video_path):
        print('Video already drawn!!! Deleting...')
        # return
        os.remove(output_video_path)
    print(f'Drawing {output_video_path}')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # cv2_writer = cv2.VideoWriter(output_video_path, fourcc=fourcc, fps=15,
    #                              frameSize=(640, 480), isColor=True)
    # cv2_writer_long = cv2.VideoWriter(output_video_path[:-4] + '_input.avi', fourcc=fourcc, fps=15,
    #                                   # frameSize=(640, 720),
    #                                   frameSize=(640, 480),
    #                                   isColor=True)
    # cv2_writer_obj = cv2.VideoWriter(output_video_path[:-4] + '_obj.avi', fourcc=fourcc, fps=15,
    #                                  frameSize=(640, 480),
    #                                  isColor=True)
    # cv2_writer_diag = cv2.VideoWriter(output_video_path[:-4] + '_diagnostic.avi', fourcc=fourcc, fps=15,
    #                                   frameSize=(640, 240),
    #                                   isColor=True)
    # cv2_writer_pe = cv2.VideoWriter(output_video_path[:-4] + '_pe.avi', fourcc=fourcc, fps=15,
    #                                 frameSize=(640, 240),
    #                                 isColor=True)
    # cv2_writer_combined = cv2.VideoWriter(output_video_path[:-4] + '_combined.avi', fourcc=fourcc, fps=15,
    #                                       frameSize=(640, 480),
    #                                       isColor=True)
    # cv2_writer_ball = cv2.VideoWriter(output_video_path[:-4] + '_ball.avi', fourcc=fourcc, fps=15,
    #                                       frameSize=(640, 480),
    #                                       isColor=True)
    cv2_writer_comp = cv2.VideoWriter(output_video_path, fourcc=fourcc, fps=15,
                                          frameSize=(1600, 900),
                                          isColor=True)
    count = 0

    for frame_id, frame in anchored_frames.items():
        # old_id = frame_id
        # indices = list(set(pca_input_df.index).intersection(set(skel_df.index)))
        # frame_id = align(frame_id, indices)
        # if frame_id == -1:
        #     continue
        # del anchored_frames[old_id]
        # anchored_frames[frame_id] = frame
        count += 1
        # if frame_id > 400:
        #     break

        def get_side_and_front(df, frame_id, title=''):
            x_array = df.filter(like='rel_X').loc[frame_id].to_numpy()
            y_array = df.filter(like='rel_Y').loc[frame_id].to_numpy()
            z_array = df.filter(like='rel_Z').loc[frame_id].to_numpy()
            e_array = np.hstack([x_array, y_array, z_array])
            if len(e_array) == 72:
                e_array = np.insert(e_array, 1, 0)
                e_array = np.insert(e_array, 26, 0)
                e_array = np.insert(e_array, 51, 0)
            else:
                assert len(e_array) == 75, f'len(e_array)={len(e_array)} is not 72 or 75'
            side_3d = anim_event_series(e_array=e_array, view='side', title=title)
            front_3d = anim_event_series(e_array=e_array, view='front', title=title)
            return side_3d, front_3d

        # side_before_preprocess, front_before_preprocess = get_side_and_front(skel_df, frame_id,
        #                                                                      title='Before Preprocessing')
        # before_preprocess = cv2.hconcat([side_before_preprocess, front_before_preprocess])
        # before_preprocess = cv2.resize(before_preprocess, dsize=(800, 450))
        # side_before_pca, front_before_pca = get_side_and_front(skel_df_unscaled, frame_id, title='Before Pca')
        # before_pca = cv2.hconcat([side_before_pca, front_before_pca])
        # before_pca = cv2.resize(before_pca, dsize=(800, 450))
        side_after_pca, front_after_pca = get_side_and_front(pca_input_df, frame_id, title='Ground Truth (GT)')
        after_pca = cv2.hconcat([side_after_pca, front_after_pca])
        after_pca = cv2.resize(after_pca, dsize=(800, 450))
        side_pred, front_pred = get_side_and_front(pred_skel_df, frame_id, title="SEM's Prediction")
        prediction = cv2.hconcat([side_pred, front_pred])
        prediction = cv2.resize(prediction, dsize=(800, 450))

        img, out_obj_frame = draw_frame_resampled(frame_id, skel_checkbox=True, obj_checkbox=True, run_select=run_select,
                                                  get_img=True,
                                                  black=False)
        # out_obj_frame = cv2.resize(out_obj_frame, dsize=(800, 450))
        img = cv2.resize(img, dsize=(800, 450))
        # diagnostic = plot_diagnostic_readouts(frame_id, run_select, get_img=True)
        # diagnostic = cv2.resize(diagnostic, dsize=(800, 450))
        # pe = plot_pe(run_select, tag, epoch, frame_slider=frame_id)
        # pe = cv2.resize(pe, dsize=(640, 240))
        pe_and_diag = plot_pe_and_diag(run_select, tag, epoch, frame_id, get_img=True)
        pe_and_diag = cv2.resize(pe_and_diag, dsize=(800, 450))
        # skeleton_ball = draw_skeleton_ball(frame_slider=frame_id)
        # skeleton_ball = cv2.resize(skeleton_ball, dsize=(640, 480))
        # combined = cv2.hconcat([cv2.vconcat([img, skeleton_ball]), cv2.vconcat([out_obj_frame, pe_and_diag])])
        combined = cv2.hconcat([cv2.vconcat([img, after_pca]),
                                cv2.vconcat([pe_and_diag, prediction])])
        # cv2.imwrite('test.png', skeleton_ball)
        # diagnostic = np.concatenate([pe, diagnostic], axis=0)
        # cv2_writer_long.write(img)
        # cv2_writer_obj.write(out_obj_frame)
        # cv2_writer_diag.write(diagnostic)
        # cv2_writer_pe.write(pe)
        # cv2_writer_combined.write(pe_and_diag)
        cv2_writer_comp.write(combined)
    cv2_writer_comp.release()
    print(f'Done {output_video_path}')


if __name__ == "__main__":

    t1 = time.perf_counter()
    if len(sys.argv) == 2:
        with open('runs_to_draw.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            run_select, tag, epoch = [x.strip() for x in line.split(' ')]
    else:
        run_select = sys.argv[1]
        tag = sys.argv[2]
        epoch = sys.argv[3]
    run_select = run_select.replace('_kinect', '')
    second_interval = 1  # interval to group boundaries
    frame_per_second = 3  # sampling rate to input to SEM
    cv2_reader = cv2.VideoCapture(f'data/small_videos/{run_select}_kinect_trim.mp4')
    fps = cv2_reader.get(cv2.CAP_PROP_FPS)
    # fps = 25.0  # kinect videos
    frame_interval = frame_per_second * second_interval
    skel_df = pd.read_csv(f'output/skel/{run_select}_kinect_skel_features.csv')
    skel_df.set_index('frame', drop=False, inplace=True)
    objhand_df = pd.read_csv(os.path.join(f'output/objhand/{run_select}_kinect_objhand.csv'))
    anchored_frames = joblib.load(f'output/run_sem/frames/{run_select}_kinect_trimmar_20_individual_depth_scene_frames.joblib')
    inputdf = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_inputdf_{epoch}.pkl', 'rb'))
    # glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    glove_vectors = pkl.load(open('gen_sim_glove_50.pkl', 'rb'))
    gt_freqs = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_gtfreqs.pkl', 'rb'))
    sem_readouts = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_diagnostic_{epoch}.pkl', 'rb'))

    first_frame = inputdf.appear_post.index[0]
    offset = first_frame / fps / second_interval

    categories_z = inputdf.categories_z
    skel_df_post = inputdf.skel_post
    objhand_post = inputdf.objhand_post

    # Prepare dataframes to plot input skeleton and predicted skeleton
    pca_input_df = inputdf.x_train_inverted
    pred_skel_df = inputdf.x_inferred_inverted
    skel_df_unscaled = skel_df_post.copy().loc[:, skel_df_post.columns]
    pred_skel_df = pred_skel_df.loc[:, skel_df_post.columns]
    pca_input_df = pca_input_df.loc[:, skel_df_post.columns]

    # TODO: cross-check with run_sem_pretrain for consistency
    # load sampled skel features, 200 samples for each video.
    combined_runs = pd.read_csv('sampled_skel_features_dec_6.csv')
    # standardize using global statistics
    select_indices = (combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))
    combined_runs_q = combined_runs[select_indices]
    stats = combined_runs_q.describe().loc[['mean', 'std']]
    # skel_df = (skel_df - stats.loc['mean', skel_df.columns]) / stats.loc['std', skel_df.columns]

    skel_df_unscaled = skel_df_unscaled * stats.loc['std', skel_df_post.columns] + stats.loc['mean', skel_df_post.columns]
    pred_skel_df = pred_skel_df * stats.loc['std', skel_df_post.columns] + stats.loc['mean', skel_df_post.columns]
    pca_input_df = pca_input_df * stats.loc['std', skel_df_post.columns] + stats.loc['mean', skel_df_post.columns]

    skel_df_unscaled['frame'] = skel_df_unscaled.index
    pred_skel_df['frame'] = pred_skel_df.index
    pca_input_df['frame'] = pca_input_df.index
    # combined sampled runs to extract global, drawing ball plots
    combined_runs = pd.read_csv('sampled_skel_features.csv')
    combined_runs['J1_dist_from_J1'] = np.zeros(shape=(len(combined_runs), 1))

    for i in range(25):
        new_column = f'J{i}_Tracked'
        skel_df_unscaled[new_column] = 'Inferred'
        pred_skel_df[new_column] = 'Predicted'
        pca_input_df[new_column] = 'Inferred'

        pred_objhand = inputdf.x_inferred_inverted
        pred_objhand = pred_objhand.loc[:, objhand_post.drop(['euclid', 'cosine'], axis=1, errors='ignore').columns]
        # Prepare a dictionary of word2vec for this particular run
        # categories = set()
        # for c in inputdf.categories_z.columns:
        #     categories.update(inputdf.categories_z.loc[:, c].dropna())
        # if None in categories:
        #     categories.remove(None)
        #
        # corpus categories
        corpus_categories = pkl.load(open('corpus_categories.pkl', 'rb'))
        corpus_word2vec = dict()
        for category in corpus_categories:
            r = np.zeros(shape=(1, pred_objhand.shape[1]))
            try:
                r += glove_vectors[category]
            except Exception as e:
                words = category.split(' ')
                for w in words:
                    w = w.replace('(', '').replace(')', '')
                    r += glove_vectors[w]
                r /= len(words)
            corpus_word2vec[category] = r

        # scene categories
        scene_categories = pkl.load(open('scene_categories.pkl', 'rb'))
        scene_word2vec = dict()
        for category in scene_categories[run_select + '_kinect']:
            r = np.zeros(shape=(1, pred_objhand.shape[1]))
            try:
                r += glove_vectors[category]
            except Exception as e:
                words = category.split(' ')
                for w in words:
                    w = w.replace('(', '').replace(')', '')
                    r += glove_vectors[w]
                r /= len(words)
            scene_word2vec[category] = r

    # align frame id from anchored frames to frame id from output
    def align(index, indices, window=3):
        for i in range(index, index + window):
            if i in indices:
                return i
        for i in range(index, index - window, -1):
            if i in indices:
                return i
        return -1
    new_anchored_frames = dict()
    indices = list(set(pca_input_df.index).intersection(set(skel_df.index)))
    for frame_id, frame in anchored_frames.items():
        frame_id = align(frame_id, indices)
        if frame_id == -1:
            continue
        # del anchored_frames[old_id]
        new_anchored_frames[frame_id] = frame
    anchored_frames = new_anchored_frames

    draw_video()
    t2 = time.perf_counter()
    print(f'Time elapsed for {run_select}: {t2 - t1}')
