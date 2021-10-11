import sys

import numpy as np
import traceback

sys.path.append('.')
sys.path.append('../pysot')
import pandas as pd
import os
import json
from joblib import Parallel, delayed
from utils import logger, parse_config, contain_substr


def calc_joint_dist(df, joint):
    # df : skeleton tracking dataframe with 3D joint coordinates
    # joint : integer 0 to 24 corresponding to a Kinect skeleton joint
    # returns df with column of distance between the joint and spine mid:
    # some key joints are spine_mid:J1,left hand:J7,right hand: J11, foot left J15, foot right: J19
    j = str(joint)
    jx = 'J' + j + '_3D_X'
    jy = 'J' + j + '_3D_Y'
    jz = 'J' + j + '_3D_Z'
    jout = 'J' + j + '_dist_from_J1'
    df[jout] = np.sqrt(
        (df[jx] - df.J1_3D_X) ** 2 + (df[jy] - df.J1_3D_Y) ** 2 + (df[jz] - df.J1_3D_Z) ** 2)
    return df


def calc_joint_speed(df, joint):
    j = str(joint)
    jx = 'J' + j + '_3D_X'
    jy = 'J' + j + '_3D_Y'
    jz = 'J' + j + '_3D_Z'
    jout = 'J' + j + '_speed'
    df[jout] = np.sqrt((df[jx] - df[jx].shift(1)) ** 2 + (df[jy] - df[jy].shift(1)) ** 2 + (
            df[jz] - df[jz].shift(1)) ** 2) / (df.sync_time - df.sync_time.shift(1))
    return df


# speed doesn't have direction -> opposite direction is an issue, but rare.
def calc_joint_acceleration(df, joint):
    j = str(joint)
    js = 'J' + j + '_speed'
    if js not in df.columns:
        df = calc_joint_speed(df, joint)
    jout = 'J' + j + '_acceleration'
    df[jout] = (df[js] - df[js].shift(1)) / (df.sync_time - df.sync_time.shift(1))
    return df


def calc_interhand_dist(df):
    df['interhand_dist'] = np.sqrt(
        (df.J11_3D_X - df.J7_3D_X) ** 2 + (df.J11_3D_Y - df.J7_3D_Y) ** 2 + (
                df.J11_3D_Z - df.J7_3D_Z) ** 2)
    return df


# speed doesn't have direction -> interhand speed and acceleration?
# actually, intuitively correct calculation is not necessarily useful. More, consider with other features.
def calc_interhand_speed(df):
    # Values are positive when right hand (J11) is faster than left hand (J7)
    if 'J7_speed' not in df.columns:
        df = calc_joint_speed(df, 7)
    if 'J11_speed' not in df.columns:
        df = calc_joint_speed(df, 11)
    df['interhand_speed'] = df.J11_speed - df.J7_speed
    return df


def calc_interhand_acceleration(df):
    # Values are positive when right hand (J11) is faster than left hand (J7)
    if 'J7_acceleration' not in df.columns:
        df = calc_joint_acceleration(df, 7)
    if 'J11_acceleration' not in df.columns:
        df = calc_joint_acceleration(df, 11)
    df['interhand_acceleration'] = df.J11_acceleration - df.J7_acceleration
    return df


def gen_skel_feature(args, run, tag):
    try:
        # FPS is used to index output and later used to concat, should be inferred from run
        if 'kinect' in run:
            fps = 25
            skel_csv_in = os.path.join(args.skel_csv_in,
                                       run.replace('_C1', '').replace('_C2', '').replace(
                                           '_kinect', '') + '_skel_clean.csv')
        else:
            fps = 30
            skel_csv_in = os.path.join(args.skel_csv_in, run + '_skel.csv')
        args.run = run
        args.tag = tag
        logger.info(f'Config {args}')
        # Load skeleton dataframe
        skel_csv_out = os.path.join(args.skel_csv_out,
                                    run + '_skel_features.csv')
        skeldf = pd.read_csv(skel_csv_in)
        if args.joints == "all":
            joints = list(range(25))
        for j in joints:
            skeldf = calc_joint_dist(skeldf, j)
            skeldf = calc_joint_speed(skeldf, j)
            skeldf = calc_joint_acceleration(skeldf, j)
        skeldf = calc_interhand_dist(skeldf)
        skeldf = calc_interhand_speed(skeldf)
        skeldf = calc_interhand_acceleration(skeldf)
        skeldf['frame'] = (skeldf['sync_time'] * fps).apply(round).astype(int)
        skeldf = skeldf[~skeldf['frame'].duplicated(keep='first')]
        skeldf.to_csv(skel_csv_out, index=False)

        logger.info(f'Done Skel {run}')
        with open('skel_complete.txt', 'a') as f:
            f.write(run + '\n')
        return skel_csv_in, skel_csv_out
    except Exception as e:
        with open('skel_error.txt', 'a') as f:
            f.write(run + '\n')
            f.write(repr(e) + '\n')
            f.write(traceback.format_exc() + '\n')
        return None, None


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    if '.txt' in args.run:
        choose = ['kinect']
        # choose = ['C1']
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if contain_substr(run, choose)]
    else:
        runs = [args.run]

    # runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    # runs = ['1.1.5_C1', '4.4.5_C1']
    tag = '_dec_28'
    # skel_in_csv, skel_out_csv = gen_skel_feature(args, runs[0], tag)
    res = Parallel(n_jobs=16)(delayed(
        gen_skel_feature)(args, run, tag) for run in runs)
    skel_in_csvs, skel_out_csvs = zip(*res)
    results = dict()
    for i, run in enumerate(runs):
        results[run] = dict(skel_in_csv=skel_in_csvs[i], skel_out_csv=skel_out_csvs[i])
    with open('results_skel_features.json', 'w') as f:
        json.dump(results, f)
