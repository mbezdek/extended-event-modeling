import sys
sys.path.append('.')
sys.path.append('../pysot')
from utils import logger, parse_config
from utils import calc_center, boxDistance
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import json
from joblib import Parallel, delayed


def gen_feature_video(args, run, tag):
    # FPS is used to concat track_df and skel_df, should be inferred from run
    if 'kinect' in run:
        fps = 25
    else:
        fps = 30
    args.run = run
    args.tag = tag
    logger.info(f'Config {args}')
    track_csv = os.path.join(args.input_track_csv, run + '_r50.csv')
    skel_csv = os.path.join(args.input_skel_csv, run.replace('_C1', '') + '_skel_clean.csv')
    output_csv = os.path.join(args.output_objhand_csv, run + '_objhand.csv')
    # Read tracking result
    track_df = pd.read_csv(track_csv)
    track_df = calc_center(track_df)
    # Read skeleton result and set index by frame to merge tracking and skeleton
    skel_df = pd.read_csv(skel_csv)
    # sync_time, Right Hand: J11_2D_X, J11_2D_Y
    hand_df = skel_df.loc[:, ['sync_time', 'J11_2D_X', 'J11_2D_Y']]
    hand_df['frame'] = (hand_df.loc[:, 'sync_time'] * fps).apply(round).astype(np.int)
    hand_df.set_index('frame', drop=False, verify_integrity=True, inplace=True)
    # Process tracking result to create tracking dataframe
    final_frameid = max(max(hand_df['frame']), max(track_df['frame']))
    objs = track_df['name'].unique()
    objs_df = pd.DataFrame(index=range(final_frameid))
    objs_df.index.name = 'frame'
    for obj in objs:
        obj_df = track_df[track_df['name'] == obj].set_index('frame', drop=False,
                                                             verify_integrity=True)
        objs_df[obj + '_x_cent'] = obj_df['x_cent']
        objs_df[obj + '_y_cent'] = obj_df['y_cent']
        objs_df[obj + '_x'] = obj_df['x']
        objs_df[obj + '_y'] = obj_df['y']
        objs_df[obj + '_w'] = obj_df['w']
        objs_df[obj + '_h'] = obj_df['h']
        objs_df[obj + '_confidence'] = obj_df['confidence']
    logger.info('Combine hand dataframe and objects dataframe')
    # objhand_df = pd.concat([hand_df, objs_df], axis=1)
    objhand_df = hand_df.combine_first(objs_df)
    objhand_df = objhand_df.sort_index()
    # Process null entry by interpolation
    logger.info('Interpolate')
    for obj in objs:
        objhand_df[obj + '_x_cent'] = objhand_df[obj + '_x_cent'].interpolate(method='linear')
        objhand_df[obj + '_y_cent'] = objhand_df[obj + '_y_cent'].interpolate(method='linear')
        objhand_df[obj + '_x'] = objhand_df[obj + '_x'].interpolate(method='linear')
        objhand_df[obj + '_y'] = objhand_df[obj + '_y'].interpolate(method='linear')
        objhand_df[obj + '_w'] = objhand_df[obj + '_w'].interpolate(method='linear')
        objhand_df[obj + '_h'] = objhand_df[obj + '_h'].interpolate(method='linear')
    objhand_df['J11_2D_X'] = objhand_df['J11_2D_X'].interpolate(method='linear')
    objhand_df['J11_2D_Y'] = objhand_df['J11_2D_Y'].interpolate(method='linear')
    objhand_df['J11_2D_X'] = objhand_df['J11_2D_X'] / 2
    objhand_df['J11_2D_Y'] = objhand_df['J11_2D_Y'] / 2
    # Smooth movements
    logger.info('Gaussian filtering')
    for obj in objs:
        objhand_df[obj + '_x_cent'] = gaussian_filter1d(objhand_df[obj + '_x_cent'], 3)
        objhand_df[obj + '_y_cent'] = gaussian_filter1d(objhand_df[obj + '_y_cent'], 3)
        objhand_df[obj + '_x'] = gaussian_filter1d(objhand_df[obj + '_x'], 3)
        objhand_df[obj + '_y'] = gaussian_filter1d(objhand_df[obj + '_y'], 3)
        objhand_df[obj + '_w'] = gaussian_filter1d(objhand_df[obj + '_w'], 3)
        objhand_df[obj + '_h'] = gaussian_filter1d(objhand_df[obj + '_h'], 3)
    objhand_df['J11_2D_X'] = gaussian_filter1d(objhand_df['J11_2D_X'], 3)
    objhand_df['J11_2D_Y'] = gaussian_filter1d(objhand_df['J11_2D_Y'], 3)
    # Let do resampling when combining with other features while running SEM, not here
    # objhand_df.loc[:, 'sync_time'] = objhand_df.index / fps
    # objhand_df.loc[:, 'frame'] = objhand_df.index
    # resampledf = resample_df(objhand_df, rate='333ms')
    # resampledf['frame'] = resampledf['frame'].apply(round)
    resampledf = objhand_df
    resampledf['frame'] = resampledf.index  # To concatenate with other features
    # Calculate distances between all objects and hand
    logger.info('Calculate object-hand distances')
    for obj in objs:
        resampledf[obj + '_dist'] = resampledf[
            [obj + '_x', obj + '_y', obj + '_w', obj + '_h', 'J11_2D_X',
             'J11_2D_Y']].apply(
            lambda x: boxDistance(x[0], x[1], x[2], x[3], x[4], x[5]) if (
                np.all(pd.notnull(x))) else np.nan, axis=1)
    resampledf.to_csv(output_csv, index=False)
    return track_csv, skel_csv, output_csv


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    if '.txt' in args.run:
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if 'Stats' not in run]
    else:
        runs = [args.run]

    # gen_feature_video(track_csv=args.input_track_csv, skel_csv=args.input_skel_csv,
    #                   output_csv=args.output_objhand_csv)

    runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    tag = '_dec_26'
    res = Parallel(n_jobs=8)(delayed(
        gen_feature_video)(args, run, tag) for run in runs)
    track_csvs, skel_csvs, output_csvs = zip(*res)
    results = dict()
    for i, run in enumerate(runs):
        results[run] = dict(track_csv=track_csvs[i], skel_csv=skel_csvs[i], output_csv=output_csvs[i])
    with open('results_objhand.json', 'w') as f:
        json.dump(results, f)

