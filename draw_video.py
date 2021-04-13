"""
This script is used to draw and overlay predictions on video,
Run this file on the cluster instead of downloading pickle files and run locally
"""

import pickle as pkl
import pandas as pd
import numpy as np

import cv2
from scipy.ndimage import gaussian_filter1d
from glob import glob
import os
import sys
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import joblib
from utils import get_point_biserial, get_binned_prediction
import scipy.stats as stats
import traceback
from utils import ColorBGR


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
        jtrack = r['J' + str(joint) + '_Tracked'].values[0]
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


def get_nearest(emb_vector: np.ndarray, glove=False):
    if glove:
        nearest_objects = glove_vectors.most_similar(emb_vector)
        nearest_objects = [(nr[0], round(nr[1], 2)) for nr in nearest_objects]
        return nearest_objects

    res = {kv[0]: np.linalg.norm(kv[1] - emb_vector) for kv in word2vec.items()}
    res = sorted(res.items(), key=lambda kv: kv[1])
    res = [(nr[0], round(nr[1], 2)) for nr in res]
    return res


def drawobj(instances, frame, odf, color=(255, 0, 0), thickness=1, draw_name=False):
    s = frame.shape[0] / 1080.0
    for i in instances:
        xmin = odf[i + '_x'] * s
        ymin = odf[i + '_y'] * s
        xmax = xmin + (odf[i + '_w'] * s)
        ymax = ymin + (odf[i + '_h'] * s)
        try:
            conf_score = float(odf[i + '_confidence'])
            color = tuple(map(int, np.array(color) * conf_score))
        except:
            color = (0, 0, 255)
        cv2.rectangle(frame, pt1=(int(xmin), int(ymin)),
                      pt2=(int(xmax), int(ymax)),
                      color=color, thickness=thickness)
        if draw_name:
            cv2.putText(frame, text=i,
                        org=(int(xmin), int(ymax - 5)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=color)
    # Code to also get nearest objects in Glove for input categories
    # try:
    #     sr = objdf.loc[frame_number].dropna().filter(regex='_dist$').rename(lambda x: remove_number(x).replace('_dist', ''))
    #     arr = get_emb_category(sr, emb_dim=50)
    #     nearest_objects = get_nearest(arr, glove=True)
    #     for index, instance in enumerate(nearest_objects[:3]):
    #         cv2.putText(frame, text=str(instance), org=(frame.shape[1] - 420, 20 + 20 * index),
    #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
    #                     color=(255, 0, 0))
    # except Exception as e:
    #     print(e)
    return frame


def draw_frame_resampled(frame_slider, skel_checkbox, obj_checkbox, run_select, get_img=False, black=True):
    outframe = deepcopy(anchored_frames[frame_slider])
    if black:
        outframe[:, :, :] = 0
    # draw skeleton here
    if skel_checkbox:
        try:
            outframe = drawskel(frame_slider, outframe, skel_df)
            # TODO: comment these lines if not using position in training SEM.
            # outframe = drawskel(frame_slider, outframe, pca_input_df, color=(255, 0, 0))
            # outframe = drawskel(frame_slider, outframe, pred_skel_df, color=(0, 255, 0))
        except Exception as e:
            cv2.putText(outframe, 'No skeleton data', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print(traceback.format_exc())
    else:
        outframe = anchored_frames[frame_slider]
    if obj_checkbox:
        try:
            odf_z = objhand_df[objhand_df.index == frame_slider]
            odf_z = odf_z[odf_z.columns[~odf_z.isna().any()].tolist()]
            sorted_objects = list(pd.Series(odf_z.filter(regex='dist_z$').iloc[0, :]).sort_values().index)
            sorted_objects = [object.replace('_dist_z', '') for object in sorted_objects]
            outframe = drawobj(sorted_objects[:3], outframe, odf_z, color=ColorBGR.red, draw_name=True)
            outframe = drawobj(sorted_objects[3:], outframe, odf_z, color=ColorBGR.cyan, draw_name=True)
            # Draw nearest words (in the video)
            nearest_objects = get_nearest(pred_objhand.loc[frame_slider, :].values)
            for index, instance in enumerate(nearest_objects[:3]):
                cv2.putText(outframe, text=str(instance), org=(outframe.shape[1] - 140, 20 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 255, 0))
            # Draw nearest objects (Glove corpus)
            glove_nearest_objects = get_nearest([np.array(pred_objhand.loc[frame_slider, :].values, dtype=np.float32)],
                                                glove=True)
            for index, instance in enumerate(glove_nearest_objects[:3]):
                cv2.putText(outframe, text=str(instance), org=(outframe.shape[1] - 280, 20 + 20 * index),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 255, 255))
        except Exception as e:
            print(traceback.format_exc())

    cv2.putText(outframe, text=f'RED: 3 Nearest Objects', org=(10, 120),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                color=ColorBGR.red)

    cv2.putText(outframe, text=f'CYAN: Background Objects', org=(10, 140),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                color=ColorBGR.cyan)

    # add frameID
    cv2.putText(outframe, text=f'FrameID: {frame_slider}', org=(10, 200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                color=(0, 255, 0))
    # add Segmentation flag
    index = pred_objhand.index.get_indexer([frame_slider])[0]
    if sem_readouts['e_hat'][index] != sem_readouts['e_hat'][index - 1]:
        cv2.putText(outframe, text='SEGMENT', org=(10, 220),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(0, 255, 0))

    if get_img:
        return outframe

    # embedding image on axis to align
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(cv2.cvtColor(outframe, cv2.COLOR_BGR2RGB))
    plt.close(fig)
    return fig


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


def plot_diagnostic_readouts(frame_slider, run_select, title='', get_img=False):
    fig, ax = plt.subplots()
    ax.plot(gaussian_filter1d(gt_freqs, 1), label='Subject Boundaries')
    ax.set_xlabel('Time (second)')
    ax.set_ylabel('Boundary Probability')
    ax.set_title('Diagnostic Readouts ' + run_select)
    colors = {'new': 'red', 'old': 'green', 'restart': 'blue', 'repeat': 'purple'}

    impose_rainbow_events(ax, fig)
    impose_line_boundaries(ax, fig)
    impose_metrics(ax, fig)

    ax.axvline(frame_slider / fps, linewidth=2, alpha=0.5, color='r')
    if get_img:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        #         canvas = FigureCanvas(fig)
        #         canvas.draw()
        fig.canvas.draw()
        image_from_plot = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return image_from_plot
    plt.close(fig)
    return fig


def draw_video():
    output_video_path = f'output/videos/{run_select}_{tag}_{epoch}.avi'
    if not os.path.exists(f'output/videos'):
        os.makedirs('output/videos')
    if os.path.exists(output_video_path):
        print('Video already drawn!!!')
        return
    else:
        print(f'Drawing {output_video_path}')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    cv2_writer = cv2.VideoWriter(output_video_path, fourcc=fourcc, fps=15,
                                 frameSize=(640, 480), isColor=True)
    for frame_id, frame in anchored_frames.items():
        img = draw_frame_resampled(frame_id, skel_checkbox=True, obj_checkbox=True, run_select=run_select, get_img=True)
        img = cv2.resize(img, dsize=(640, 480))
        cv2_writer.write(img)
        # diagnostic = plot_diagnostic_readouts(frame_id, run_select, title='', get_img=True)
        # diagnostic = cv2.resize(diagnostic, dsize=(640, 480))
        # concat = np.concatenate([img, diagnostic], axis=0)
        # cv2_writer.write(concat)
    cv2_writer.release()
    print(f'Done {output_video_path}')


import sys

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
fps = 25.0  # kinect videos
frame_interval = frame_per_second * second_interval
# def listen_to_run(run_select):
skel_df = pd.read_csv(f'output/skel/{run_select}_kinect_skel_features.csv')
objhand_df = pd.read_csv(os.path.join(f'output/objhand/{run_select}_kinect_objhand.csv'))
anchored_frames = joblib.load(f'output/run_sem/frames/{run_select}_kinect_trimmar_20_individual_depth_scene_frames.joblib')
inputdf = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_inputdf{epoch}.pkl', 'rb'))
glove_vectors = pkl.load(open('gen_sim_glove_50.pkl', 'rb'))
gt_freqs = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_gtfreqs.pkl', 'rb'))
sem_readouts = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_diagnostic{epoch}.pkl', 'rb'))

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

skel_df_unscaled = skel_df_unscaled * skel_df.std() + skel_df.mean()
pred_skel_df = pred_skel_df * skel_df.std() + skel_df.mean()
pca_input_df = pca_input_df * skel_df.std() + skel_df.mean()

skel_df_unscaled['frame'] = skel_df_unscaled.index
pred_skel_df['frame'] = pred_skel_df.index
pca_input_df['frame'] = pca_input_df.index
for i in range(25):
    new_column = f'J{i}_Tracked'
    skel_df_unscaled[new_column] = 'Inferred'
    pred_skel_df[new_column] = 'Predicted'
    pca_input_df[new_column] = 'Inferred'

    # Prepare a dictionary of word2vec for this particular run
    categories = set()
    for c in inputdf.categories.columns:
        categories.update(inputdf.categories.loc[:, c].dropna())
    if None in categories:
        categories.remove(None)

    pred_objhand = inputdf.x_inferred_inverted
    pred_objhand = pred_objhand.loc[:, objhand_post.drop(['euclid', 'cosine'], axis=1, errors='ignore').columns]
    word2vec = dict()
    for category in categories:
        r = np.zeros(shape=(1, pred_objhand.shape[1]))
        try:
            r += glove_vectors[category]
        except Exception as e:
            words = category.split(' ')
            for w in words:
                w = w.replace('(', '').replace(')', '')
                r += glove_vectors[w]
            r /= len(words)
        word2vec[category] = r

draw_video()
