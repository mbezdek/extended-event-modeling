import sys

sys.path.append('.')
sys.path.append('../pysot')
import json
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from utils import logger, parse_config, SegmentationVideo, Canvas


def plot_appear_features(args, run, tag):
    # FPS is used to visualize, should be inferred from run
    if 'kinect' in run:
        fps = 25
    else:
        fps = 30
    # Load saved csv file
    output_csv_appear = os.path.join(args.output_csv_appear, run + '_appear.csv')
    input_csv_tracking = os.path.join(args.input_csv_tracking, run + '_r50.csv')
    changing_points = pd.read_csv(output_csv_appear)
    appear_points = changing_points[changing_points['appear'] != 0]['frame'] / fps
    appear_points = appear_points.to_numpy()
    disappear_points = changing_points[changing_points['disappear'] != 0]['frame'] / fps
    disappear_points = disappear_points.to_numpy()
    # Process segmentation data (ground_truth)
    data_frame = pd.read_csv('database.200731.1.csv')
    input_video_path = ('_').join(
        os.path.basename(input_csv_tracking).split('_')[:-1]) + '_trim.mp4'
    seg_video = SegmentationVideo(data_frame=data_frame, video_path=input_video_path)
    # calculate segmentation points according to fps
    # consider only one annotation
    condition = 'coarse'
    seg_points = seg_video.get_segments(n_annotators=100, condition=condition)
    seg_points = np.hstack(seg_points)
    # Plot participant boundaries and appear/disappear timepoints
    canvas_agg = Canvas(rows=1, columns=1)
    h, b, _ = canvas_agg.axes[0].hist(seg_points, bins=99, alpha=0.5)[:]
    # sns.histplot(seg_points, bins=200, kde=False)
    canvas_agg.axes[0].vlines(appear_points, ymin=max(h // 2), ymax=max(h), alpha=0.2,
                              colors='green', label='appear')
    canvas_agg.axes[0].vlines(disappear_points, ymin=0, ymax=max(h // 2), alpha=0.2,
                              colors='red', label='disappear')
    canvas_agg.axes[0].set_title(f'{os.path.splitext(output_csv_appear)[0]}')
    canvas_agg.figure.savefig(
        f'{os.path.splitext(output_csv_appear)[0]}' + '.jpg')
    # plt.show()


def gen_appear_features(args, run, tag):
    try:
        args.run = run
        args.tag = tag
        logger.info(f'Config {args}')
        output_csv_appear = os.path.join(args.output_csv_appear, run + '_appear.csv')
        input_csv_tracking = os.path.join(args.input_csv_tracking, run + '_r50.csv')
        csv_headers = ['frame', 'appear', 'disappear']
        with open(output_csv_appear, 'w') as g:
            writer = csv.writer(g)
            writer.writerow(csv_headers)
        # read tracking csv
        label_df = pd.read_csv(input_csv_tracking)
        frames = sorted(label_df['frame'].unique())
        # loop all frame
        previous_types = set(label_df['name'].unique())
        previous_types = {object_type: 0 for object_type in previous_types}
        # all categories in the video
        all_types = previous_types
        for frame in frames:
            appear = 0
            disappear = 0
            result_csv_row = [frame]
            current_df = label_df[label_df['frame'] == frame]
            current_types = set(current_df['name'].unique())
            current_types = {object_type: len(current_df[current_df['name'] == object_type])
                             for object_type in current_types}
            # number of objects for all categories at current frame
            current_types = {**all_types, **current_types}
            logger.debug(f'all {all_types}')
            logger.debug(f'prev {previous_types}')
            logger.debug(f'cur {current_types}')
            for object_type in all_types.keys():
                if current_types[object_type] < previous_types[object_type]:
                    disappear += 1
                elif current_types[object_type] > previous_types[object_type]:
                    appear += 1
            # assign current to previous
            previous_types = current_types
            logger.debug(f'appear {appear}')
            logger.debug(f'disappear {disappear}')
            # write csv file
            result_csv_row += [appear, disappear]
            with open(output_csv_appear, 'a') as g:
                writer = csv.writer(g)
                writer.writerow(result_csv_row)

        plot_appear_features(args, run, tag)
        logger.info(f'Done Appear {run}')
        with open('appear_complete.txt', 'a') as f:
            f.write(run + '\n')
        return input_csv_tracking, output_csv_appear
    except Exception as e:
        with open('appear_error.txt', 'a') as f:
            f.write(run + '\n')
            f.write(repr(e) + '\n')
        return None, None


if __name__ == '__main__':
    args = parse_config()
    if '.txt' in args.run:
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if 'Stats' not in run]
    else:
        runs = [args.run]

    # runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    tag = '_dec_28'
    res = Parallel(n_jobs=8)(delayed(
        gen_appear_features)(args, run, tag) for run in runs)
    input_tracking_csvs, output_appear_csvs = zip(*res)
    results = dict()
    for i, run in enumerate(runs):
        results[run] = dict(input_tracking_csv=input_tracking_csvs[i],
                            output_appear_csv=output_appear_csvs[i])
    with open('results_appear_features.json', 'w') as f:
        json.dump(results, f)
