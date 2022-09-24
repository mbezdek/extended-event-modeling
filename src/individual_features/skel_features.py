import sys
import os
import numpy as np
import traceback

# dir_name, filename = os.path.split(os.path.abspath(__file__))
# sys.path.append(dir_name)
# sys.path.append(os.getcwd())
import pandas as pd
from joblib import Parallel, delayed
from src.utils import logger, parse_config, contain_substr
from scipy.spatial.transform import Rotation as R
import math


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


def calc_interhand_speed(df):
    # Values are positive when hands moving away negative when hands get closer
    if 'interhand_dist' not in df.columns:
        df = calc_interhand_dist(df)
    df['interhand_speed'] = (df['interhand_dist'] - df['interhand_dist'].shift(1)) / (
            df.sync_time - df.sync_time.shift(1))
    return df


def calc_interhand_acceleration(df):
    # Values are positive when interhand speed is increasing and negative when decreasing
    if 'interhand_speed' not in df.columns:
        df = calc_interhand_speed(df)
    df['interhand_acceleration'] = (df['interhand_speed'] - df['interhand_speed'].shift(1)) / (
            df.sync_time - df.sync_time.shift(1))
    return df


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
        logger.info(f'Config {args}')
        # Load skeleton dataframe
        skel_csv_out = os.path.join(args.skel_csv_out,
                                    f'{run}_{tag}_skel_features.csv')
        skeldf = pd.read_csv(skel_csv_in)
        sync_time = skeldf['sync_time'].copy()
        skeldf = skeldf.rolling(7).mean()
        skeldf['sync_time'] = sync_time
        if args.joints == "all":
            joints = list(range(25))
        for j in joints:
            skeldf = calc_joint_dist(skeldf, j)
            skeldf = calc_joint_speed(skeldf, j)
            skeldf = calc_joint_acceleration(skeldf, j)
        skeldf = calc_interhand_dist(skeldf)
        skeldf = calc_interhand_speed(skeldf)
        skeldf = calc_interhand_acceleration(skeldf)
        skeldf = skeldf.apply(calc_joint_rel_position, axis=1)
        skeldf['frame'] = (skeldf['sync_time'] * fps).apply(round).astype(int)
        skeldf = skeldf[~skeldf['frame'].duplicated(keep='first')]
        skeldf.to_csv(skel_csv_out, index=False)

        logger.info(f'Done Skel {run}')
        print(f'Done Skel {run}')
        with open(f'output/skel_complete_{tag}.txt', 'a') as f:
            f.write(run + '\n')
        return skel_csv_in, skel_csv_out
    except Exception as e:
        with open(f'output/skel_error_{tag}.txt', 'a') as f:
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
    # skel_in_csv, skel_out_csv = gen_skel_feature(args, runs[0], tag)
    if os.path.exists(f'output/skel_complete_{args.feature_tag}.txt'):
        os.remove(f'output/skel_complete_{args.feature_tag}.txt')
    if os.path.exists(f'output/skel_error_{args.feature_tag}.txt'):
        os.remove(f'output/skel_error_{args.feature_tag}.txt')
    if not os.path.exists(args.skel_csv_out):
        os.makedirs(args.skel_csv_out)
    res = Parallel(n_jobs=8, prefer="threads")(delayed(
        gen_skel_feature)(args, run, args.feature_tag) for run in runs)

    from src.preprocess_features.pool_skel_features_all_run import pool_features
    pool_features(complete_skel_path=f'output/skel_complete_{args.feature_tag}.txt',
                  output_stats_path=f'{args.skel_stats_out}sampled_skel_features_{args.feature_tag}.csv',
                  tag=args.feature_tag)
