"""
This script is used to draw and overlay predictions on video,
Run this file on the cluster instead of downloading pickle files and run locally
"""
import pickle as pkl
import pandas as pd
import numpy as np

import cv2
from scipy.ndimage import gaussian_filter1d
import os
import matplotlib
import sys

sys.path.append('')
# get the directory this file is in
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from typing import List

matplotlib.use('agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import joblib
from src.utils import get_point_biserial, get_binned_prediction
import scipy.stats as stats
from src.utils import ColorBGR
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


def draw_skel_and_obj_box(frame_slider, skel_checkbox, obj_checkbox, run_select, get_img=False, black=False):
    skel_obj_imposed_frame = deepcopy(anchored_frames[frame_slider])
    if black:
        skel_obj_imposed_frame[:, :, :] = 0
    # draw skeleton here
    if skel_checkbox:
        try:
            skel_obj_imposed_frame = drawskel(frame_slider, skel_obj_imposed_frame, skel_df_from_csv, color=(255, 0, 0))
            # comment these lines if not using position in training SEM, in that case,
            # back-projected skeleton and predicted should be in relative pose, side and front view visualization
            # outframe = drawskel(frame_slider, outframe, pca_input_df, color=(255, 0, 0))
            # outframe = drawskel(frame_slider, outframe, pred_skel_df, color=(0, 255, 0))
        except Exception as e:
            cv2.putText(skel_obj_imposed_frame, 'No skeleton data', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # print(traceback.format_exc())
    else:
        skel_obj_imposed_frame = anchored_frames[frame_slider]
    if obj_checkbox:
        try:
            # Draw boxes for nearest objects and background objects
            odf_z = objhand_df_from_csv[objhand_df_from_csv.frame == frame_slider]
            odf_z = odf_z[odf_z.columns[~odf_z.isna().any()].tolist()]
            sorted_objects = list(pd.Series(odf_z.filter(regex='dist_z$').iloc[0, :]).sort_values().index)
            sorted_objects = [object.replace('_dist_z', '') for object in sorted_objects]
            skel_obj_imposed_frame = drawobj(sorted_objects[:3], skel_obj_imposed_frame, odf_z, color=ColorBGR.red, draw_name=False)
            skel_obj_imposed_frame = drawobj(sorted_objects[3:], skel_obj_imposed_frame, odf_z, color=ColorBGR.cyan, draw_name=False)

            # this frame has names of nearest objects imposed on it, this
            # visualization is harder for humans to track than the yellow-tinted boxes visualization.
            obj_name_imposed_frame = np.zeros(shape=(skel_obj_imposed_frame.shape[0] // 2, skel_obj_imposed_frame.shape[1],
                                            skel_obj_imposed_frame.shape[2]), dtype=np.uint8)
            # Put three nearest words from scene to input vector
            input_nearest_objects = get_nearest(
                [np.array(input_objhand_inverted.loc[input_objhand_inverted.frame == frame_slider,
                                                     objhand_columns].values.squeeze(),
                          dtype=np.float32)],
                space='scene')
            for index, instance in enumerate(input_nearest_objects[:3]):
                cv2.putText(obj_name_imposed_frame, text=str(instance), org=(obj_name_imposed_frame.shape[1] - 450, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.red)
            cv2.putText(obj_name_imposed_frame, text=f'Input (Scene)', org=(obj_name_imposed_frame.shape[1] - 450, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.red)
            # Put three nearest words from scene to prediction vector
            pred_nearest_objects = get_nearest([np.array(pred_objhand_inverted.loc[pred_objhand_inverted.frame == frame_slider, objhand_columns].values.squeeze(), dtype=np.float32)],
                                               space='scene')
            for index, instance in enumerate(pred_nearest_objects[:3]):
                cv2.putText(obj_name_imposed_frame, text=str(instance), org=(obj_name_imposed_frame.shape[1] - 300, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.green)
            # This is the yellow-tinted boxes visualization (along with red and cyan boxes),
            # easier for humans to track
            pred_nearest_objects = get_nearest([np.array(pred_objhand_inverted.loc[pred_objhand_inverted.frame == frame_slider, objhand_columns].values.squeeze(), dtype=np.float32)],
                                               space='scene')
            instances = [pr[0] for pr in pred_nearest_objects]
            skel_obj_imposed_frame = drawobj(instances[:3], skel_obj_imposed_frame, odf_z, color=ColorBGR.green, draw_name=False, tint=True, draw_rect=False)
            cv2.putText(obj_name_imposed_frame, text=f'Predicted (Scene)', org=(obj_name_imposed_frame.shape[1] - 300, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.green)
            # Put three nearest words from corpus to prediction vector
            pred_nearest_objects = get_nearest([np.array(pred_objhand_inverted.loc[pred_objhand_inverted.frame == frame_slider, objhand_columns].values.squeeze(), dtype=np.float32)],
                                               space='corpus')
            for index, instance in enumerate(pred_nearest_objects[:3]):
                cv2.putText(obj_name_imposed_frame, text=str(instance), org=(obj_name_imposed_frame.shape[1] - 150, 50 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.45,
                            color=ColorBGR.magenta)
            cv2.putText(obj_name_imposed_frame, text=f'Predicted (Corpus)', org=(obj_name_imposed_frame.shape[1] - 150, 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=ColorBGR.magenta)

        except Exception as e:
            print("Exception", e)
            obj_name_imposed_frame = None
            # print(traceback.format_exc())

    cv2.putText(skel_obj_imposed_frame, text=f'RED: 3 Nearest Objects (GT)', org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.red)

    cv2.putText(skel_obj_imposed_frame, text=f'CYAN: Background Objects (GT)', org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.cyan)
    cv2.putText(skel_obj_imposed_frame, text=f'TINTED: Predicted Nearest Objects', org=(10, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=ColorBGR.yellow)

    # add frameID
    cv2.putText(skel_obj_imposed_frame, text=f'FrameID: {frame_slider}', org=(10, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 255, 0))
    # add Segmentation flag
    index = pd.Index(pred_objhand_inverted.frame).get_indexer([frame_slider])[0]
    # if sem_readouts['e_hat'][index] != sem_readouts['e_hat'][index - 1]:
    if sem_readouts['boundaries'][index] == 1:
        cv2.putText(skel_obj_imposed_frame, text='SWITCH TO OLD', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))
    if sem_readouts['boundaries'][index] == 2:
        cv2.putText(skel_obj_imposed_frame, text='CREATE NEW', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))
    if sem_readouts['boundaries'][index] == 3:
        cv2.putText(skel_obj_imposed_frame, text='RESTART CURRENT', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 255, 0))

    if get_img:
        return skel_obj_imposed_frame, obj_name_imposed_frame

    # embedding image on axis to align
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(cv2.cvtColor(skel_obj_imposed_frame, cv2.COLOR_BGR2RGB))
    plt.close(fig)
    return fig, obj_name_imposed_frame


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
                ax.axvline(second, linestyle=(0, (5, 10)), linewidth=1, alpha=1, color=cmap(1. * e / num_colors),
                           label='Old Event')
            elif b == 2:  # Create a new event
                ax.axvline(second, linestyle='solid', linewidth=1, alpha=1, color=cmap(1. * e / num_colors), label='New Event')
            elif b == 3:  # Restart the current event
                ax.axvline(second, linestyle='dotted', linewidth=1, alpha=1, color=cmap(1. * e / num_colors),
                           label='Restart Event')
    linestyles = ['dashed', 'solid', 'dotted']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle=ls) for ls in linestyles]
    labels = ['Switch to an Old Event', 'Switch to a New Event', 'Restart the Current Event']
    ax.legend(lines, labels, loc='upper right')


def plot_diagnostic_readouts(frame_slider, run_select, get_img=False, ax=None, fig=None):
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gaussian_filter1d(gt_freqs, 1), label='Subject Boundaries')
    ax.set_ylim([0, max(gt_freqs) + max(gt_freqs) * 0.2])
    ax.axvline(frame_slider / fps, linewidth=3, alpha=0.5, color='r')
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

    input_feature = input_output_df['x_train_pca'].to_numpy() / np.sqrt(input_output_df['x_train_pca'].shape[1])
    feature_change = np.linalg.norm(np.vstack([np.zeros(shape=(1, input_feature.shape[1])),
                                               (input_feature[1:] - input_feature[:-1])]), axis=1)
    df1 = pd.DataFrame({'pe': sem_readouts['pe'], 'pe_w': sem_readouts['pe_w'] if 'oct' in tag_select else sem_readouts['pe'],
                        # 'pe_w2': v['pe_w2'] if 'oct' in tag_select else v['pe'],
                        # 'pe_w3': v['pe_w3'] if 'oct' in tag_select else v['pe'],
                        # 'pe_w3_yoke': v['pe_yoke'] if 'oct' in tag_select else v['pe'],
                        'feature_change': feature_change,
                        'prediction_change': prediction_change,
                        'second': input_output_df['combined_resampled_df'].index / 25,
                        'epoch': epoch_select}, index=input_output_df['combined_resampled_df'].index)
    df = df.append(df1)
    # df.epoch = pd.to_numeric(df.epoch, errors='coerce')

    ax.plot(df['second'], df['pe'])
    ax.axvline(frame_slider / fps, linewidth=2, alpha=0.5, color='r')
    plot_boundaries(ax=ax, fig=fig)
    if get_img:
        fig.canvas.draw()
        image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return image_from_plot
    # plt.close()
    return fig


def plot_pe_and_diag(run_select, tag_select, epoch_select, frame_slider, get_img=True):
    # fig, axes = plt.subplots(nrows=1, figsize=(8, 4.5), sharex='all', squeeze=False)
    fig, axes = plt.subplots(nrows=2, figsize=(8, 4.5), sharex='all', squeeze=False)
    plot_diagnostic_readouts(frame_slider, run_select, ax=axes[0][0], fig=fig, get_img=False)
    # ax.set_xlabel('Time (second)')
    axes[0][0].set_ylabel('Boundary Probability')
    axes[0][0].set_title(f"SEM's and humans' boundaries for activity {run_select}")
    plot_pe(run_select, tag_select, epoch_select, frame_slider, ax=axes[1][0], fig=fig, get_img=False)
    axes[1][0].set_xlabel('Time (second)')
    axes[1][0].set_ylabel('Prediction Error')
    axes[1][0].set_title('Prediction Error for ' + run_select)
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
            frame = draw_interhand_features(frame_slider, skel_df_from_csv, f, l, frame)
        frame = frame.astype(np.uint8)
        outframe = cv2.hconcat([draw_static_body(frame_slider, skel_df_from_csv, prop='speed'),
                                draw_static_body(frame_slider, skel_df_from_csv, prop='acceleration'),
                                draw_static_body(frame_slider, skel_df_from_csv, prop='dist_from_J1'),
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
    """
    This function draws skeleton from Kinect,
    near-hand objects from input and output of SEM,
    relative coordinates of joints (input and output of SEM), side view and front view,
    human boundaries and SEM's boundaries.
    :return:
    """
    output_video_path = f'output/videos/{run_select}_{tag}_{epoch}.avi'
    if not os.path.exists(f'output/videos'):
        os.makedirs('output/videos')
    if os.path.exists(output_video_path):
        print('Video already drawn!!! Deleting...')
        # return
        os.remove(output_video_path)
    print(f'Drawing {output_video_path}')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cv2_writer_comp = cv2.VideoWriter(output_video_path, fourcc=fourcc, fps=15,
                                      frameSize=(1600, 900),
                                      isColor=True)
    count = 0
    for frame_id, frame in anchored_frames.items():
        count += 1

        # if frame_id > 400:
        #     break

        def get_side_and_front(df, frame_id, title=''):
            """
            This function draws relative 3D pose, side and front view
            :param df:
            :param frame_id:
            :param title:
            :return:
            """
            x_array = df[df.frame == frame_id].filter(like='rel_X').to_numpy().squeeze()
            y_array = df[df.frame == frame_id].filter(like='rel_Y').to_numpy().squeeze()
            z_array = df[df.frame == frame_id].filter(like='rel_Z').to_numpy().squeeze()
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

        side_after_pca, front_after_pca = get_side_and_front(input_skel_inverted, frame_id, title='Ground Truth (GT)')
        after_pca = cv2.hconcat([side_after_pca, front_after_pca])
        after_pca = cv2.resize(after_pca, dsize=(800, 450))
        side_pred, front_pred = get_side_and_front(pred_skel_inverted, frame_id, title="SEM's Prediction")
        prediction = cv2.hconcat([side_pred, front_pred])
        prediction = cv2.resize(prediction, dsize=(800, 450))

        # draw skeleton and near-hand objects
        img, out_obj_frame = draw_skel_and_obj_box(frame_id, skel_checkbox=True, obj_checkbox=True, run_select=run_select,
                                                   get_img=True,
                                                   black=False)
        img = cv2.resize(img, dsize=(800, 450))
        # diagnostic = plot_diagnostic_readouts(frame_id, run_select, get_img=True)
        # diagnostic = cv2.resize(diagnostic, dsize=(800, 450))
        # pe = plot_pe(run_select, tag, epoch, frame_slider=frame_id)
        # pe = cv2.resize(pe, dsize=(640, 240))
        # draw human and SEM boundaries
        pe_and_diag = plot_pe_and_diag(run_select, tag, epoch, frame_id, get_img=True)
        pe_and_diag = cv2.resize(pe_and_diag, dsize=(800, 450))
        # skeleton_ball = draw_skeleton_ball(frame_slider=frame_id)
        # skeleton_ball = cv2.resize(skeleton_ball, dsize=(640, 480))
        # combined = cv2.hconcat([cv2.vconcat([img, skeleton_ball]), cv2.vconcat([out_obj_frame, pe_and_diag])])
        combined = cv2.hconcat([cv2.vconcat([img, after_pca]),
                                cv2.vconcat([pe_and_diag, prediction])])
        # cv2.imwrite('test.png', skeleton_ball)
        # diagnostic = np.concatenate([pe, diagnostic], axis=0)
        cv2_writer_comp.write(combined)
    cv2_writer_comp.release()
    print(f'Done {output_video_path}')


if __name__ == "__main__":

    t1 = time.perf_counter()
    # python draw_video.py 1.2.7 sep_09_n15_1030_1E-03_1E-01_1E+07 101
    if len(sys.argv) == 2:
        with open('runs_to_draw.txt', 'r') as f:
            lines = f.readlines()
        for line in lines:
            run_select, tag, epoch = [x.strip() for x in line.split(' ')]
    else:
        assert len(sys.argv) == 4, f'len(sys.argv)={len(sys.argv)} is not 2 or 4'
        run_select = sys.argv[1]
        tag = sys.argv[2]
        epoch = sys.argv[3]
    run_select = run_select.replace('_kinect', '')
    # load true skeleton data to impose it on the frame, because SEM doesn't use absolute coordinates,
    # SEM's in/output will be visualized in relative coordinates, as side and front views
    skel_df_from_csv = pd.read_csv(f'output/skel/{run_select}_kinect_sep_09_skel_features.csv')
    skel_df_from_csv.set_index('frame', drop=False, inplace=True)
    skel_df_from_csv['frame'] = skel_df_from_csv.index.to_numpy().astype(int)
    # load object_hand data to retrieve coordinates of objects nearest to the hand (input and SEM's prediction)
    objhand_df_from_csv = pd.read_csv(os.path.join(f'output/objhand/{run_select}_kinect_sep_09_objhand.csv'))
    # need to round frame so that we can locate using integer frame_id for
    # skel, anchored, and objhand embeddings, and objects names/locations.
    objhand_df_from_csv['frame'] = objhand_df_from_csv.index.to_numpy().astype(int)
    anchored_frames = joblib.load(f'output/frames/{run_select}_kinect_trimsep_09_n15_1030_1E-03_1E-01_1E+07_frames.joblib')
    # this dataframe contains a lot of dataframes, for both input and output of SEM
    input_output_df = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_input_output_df_{epoch}.pkl', 'rb'))
    # parameters to plot boundary lines
    second_interval = 1  # interval to group boundaries
    frame_per_second = 3  # sampling rate to input to SEM
    cv2_reader = cv2.VideoCapture(f'data/small_videos/{run_select}_kinect_trim.mp4')
    fps = cv2_reader.get(cv2.CAP_PROP_FPS)
    # fps = 25.0  # kinect videos
    frame_interval = frame_per_second * second_interval
    gt_freqs = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_gtfreqs_fine.pkl', 'rb'))
    sem_readouts = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_diagnostic_{epoch}.pkl', 'rb'))
    first_frame = input_output_df['appear_post'].index[0]
    offset = first_frame / fps / second_interval

    ## SKELETON FEATURES
    # this dataframe are used to identify columns of interest in the combined dataframe
    skel_df_post = input_output_df['skel_post']
    skel_columns = skel_df_post.columns
    # Prepare dataframes to plot input skeleton and predicted skeleton, in relative coordinates,
    # as side and front views
    input_skel_inverted = input_output_df['x_train_inverted'].loc[:, skel_columns]
    pred_skel_inverted = input_output_df['x_inferred_inverted'].loc[:, skel_columns]

    # cross-check with run_sem_pretrain to make sure the same global statistics are used to normalize
    # input skeleton data.
    # load sampled skel features, 200 samples for each video.
    combined_runs = pd.read_csv('output/sampled_skel_features_sep_09.csv')
    # standardize using global statistics
    select_indices = (combined_runs < combined_runs.quantile(.95)) & (combined_runs > combined_runs.quantile(.05))
    combined_runs_q = combined_runs[select_indices]
    stats = combined_runs_q.describe().loc[['mean', 'std']]

    pred_skel_inverted = pred_skel_inverted * stats.loc['std', skel_columns] + stats.loc['mean', skel_columns]
    input_skel_inverted = input_skel_inverted * stats.loc['std', skel_columns] + stats.loc['mean', skel_columns]

    # need to round frame so that we can locate using integer frame_id for
    # skel, anchored, and objhand embeddings, and objects names/locations.
    pred_skel_inverted['frame'] = pred_skel_inverted.index.to_numpy().astype(int)
    input_skel_inverted['frame'] = input_skel_inverted.index.to_numpy().astype(int)
    # in case we want different colors for different joints, because Kinect infer some joints.
    for i in range(25):
        new_column = f'J{i}_Tracked'
        pred_skel_inverted[new_column] = 'Predicted'
        input_skel_inverted[new_column] = 'Inferred'
        skel_df_from_csv[new_column] = 'Tracked'
    # this combined_runs dataframe is used in drawing ball plots and
    # inter-hand features (not used in this current script)
    # combined_runs['J1_dist_from_J1'] = np.zeros(shape=(len(combined_runs), 1))

    ## OBJECT HAND FEATURES
    # this dataframe are used to identify columns of interest in the combined dataframe
    objhand_post = input_output_df['objhand_post']
    objhand_columns = objhand_post.drop(['euclid', 'cosine'], axis=1, errors='ignore').columns
    pred_objhand_inverted = input_output_df['x_inferred_inverted']
    pred_objhand_inverted = pred_objhand_inverted.loc[:, objhand_columns]
    input_objhand_inverted = input_output_df['x_train_inverted']
    input_objhand_inverted = input_objhand_inverted.loc[:, objhand_columns]
    # need to round frame so that we can locate using integer frame_id for
    # skel, anchored, and objhand embeddings, and objects names/locations.
    pred_objhand_inverted['frame'] = pred_objhand_inverted.index.to_numpy().astype(int)
    input_objhand_inverted['frame'] = input_objhand_inverted.index.to_numpy().astype(int)
    # corpus categories, used to determine the closest word to SEM's predictions
    # glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    glove_vectors = pkl.load(open('./resources/gen_sim_glove_50.pkl', 'rb'))
    corpus_categories = pkl.load(open('src/visualization/corpus_categories.pkl', 'rb'))
    corpus_word2vec = dict()
    for category in corpus_categories:
        r = np.zeros(shape=(1, pred_objhand_inverted.loc[:, objhand_columns].shape[1]))
        try:
            r += glove_vectors[category]
        except Exception as e:
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        corpus_word2vec[category] = r
    # scene categories, used to determine and plot the closest box to SEM's predictions
    scene_categories = pkl.load(open('src/visualization/scene_categories.pkl', 'rb'))
    scene_word2vec = dict()
    for category in scene_categories[run_select + '_kinect']:
        r = np.zeros(shape=(1, pred_objhand_inverted.loc[:, objhand_columns].shape[1]))
        try:
            r += glove_vectors[category]
        except Exception as e:
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        scene_word2vec[category] = r

    ## anchored frames are saved frames to facilitate the speed of video generation
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
    # make sure to plot just frames that have pca input/output, skel_df (depend on Kinect), and object hand (depends on tracking)
    indices = list(set(input_skel_inverted.frame).intersection(
        set(skel_df_from_csv.frame)).intersection(
        set(objhand_df_from_csv.frame)).intersection(
        set(input_objhand_inverted.frame)))
    if len(indices) > 300:
        indices = indices[::len(indices) // 300]
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
