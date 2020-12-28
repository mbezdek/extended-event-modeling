import sys

sys.path.append('.')
sys.path.append('../pysot')
import pandas as pd
import os
import json
from joblib import Parallel, delayed
from utils import logger, parse_config, \
    calc_joint_dist, calc_joint_speed, calc_joint_acceleration, \
    calc_interhand_dist, calc_interhand_speed, calc_interhand_acceleration


def gen_skel_feature(args, run, tag):
    # FPS is used to index output and later used to concat, should be inferred from run
    if 'kinect' in run:
        fps = 25
    else:
        fps = 30
    args.run = run
    args.tag = tag
    logger.info(f'Config {args}')
    # Load skeleton dataframe
    skel_csv_in = os.path.join(args.skel_csv_in, run.replace('_C1', '') + '_skel_clean.csv')
    skel_csv_out = os.path.join(args.skel_csv_out, run.replace('_C1', '') + '_skel_features.csv')
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
    skeldf.to_csv(skel_csv_out, index=False)

    return skel_csv_in, skel_csv_out


if __name__ == '__main__':
    # Parse config file
    args = parse_config()
    if '.txt' in args.run:
        with open(args.run, 'r') as f:
            runs = f.readlines()
            runs = [run.strip() for run in runs if 'Stats' not in run]
    else:
        runs = [args.run]

    runs = ['1.1.5_C1', '6.3.3_C1', '4.4.5_C1', '6.2.4_C1', '2.2.5_C1']
    tag = '_dec_26'
    res = Parallel(n_jobs=8)(delayed(
        gen_skel_feature)(args, run, tag) for run in runs)
    skel_in_csvs, skel_out_csvs = zip(*res)
    results = dict()
    for i, run in enumerate(runs):
        results[run] = dict(skel_in_csv=skel_in_csvs[i], skel_out_csv=skel_out_csvs[i])
    with open('results_skel_features.json', 'w') as f:
        json.dump(results, f)
