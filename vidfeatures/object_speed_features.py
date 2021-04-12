import pickle as pkl
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd
from utils import parse_config, contain_substr, logger
from joblib import Parallel, delayed
import traceback


def calculate_object_speed(args, run, save=False):
    try:
        # inputdf = pkl.load(open(f'output/run_sem/{tag}/{run_select}_kinect_trim{tag}_inputdf_{epoch}.pkl', 'rb'))
        # object_df = inputdf[5]
        # object_df = deepcopy(object_df.filter(regex='_cent'))
        objhand_df = pd.read_csv(os.path.join(args.input_objhand_csv, f'{run}_objhand.csv'))
        # Filter to distance columns:
        object_df = deepcopy(objhand_df.filter(regex='_cent'))
        instances = set([x.split('_')[0] for x in object_df.columns])

        # Calculate max speeds across all instances of an object class, (e.g. glass0,glass1,...):
        for i in instances:
            object_df[i + '_speed'] = np.sqrt(
                (object_df[i + '_x_cent'] - object_df[i + '_x_cent'].shift(1)) ** 2 + (object_df[i + '_y_cent'] - object_df[i + '_y_cent'].shift(1)) ** 2)
        object_df = deepcopy(object_df.filter(regex='_speed'))
        s = [re.split('([a-zA-Z\s\(\)]+)([0-9]+)', x)[1] for x in object_df.columns]
        objects = list(set(s))
        for o in objects:
            # Note: paper towel and towel causes duplicated columns in series,
            # Need anchor ^ to distinguish towel and paper towel (2.4.7),
            # need digit \d to distinguish pillow0 and pillowcase0 (3.3.5)
            # Need to escape character ( and ) in aloe (green bottle) (4.4.5)
            object_df.loc[:, o + '_maxspeed'] = object_df.filter(regex=f"^{re.escape(o)}\d").max(axis=1)
        speed_df = deepcopy(object_df.filter(regex='maxspeed'))

        # Frames to seconds
        # speed_df.index = speed_df.index / (25.0)
        speed_df['frame'] = speed_df.index
        speed_df.to_csv(f'{args.output_objspeed_csv}/{run}_objspeed.csv', index=False)

        fig = plt.figure(figsize=(20, 10))
        ax = sns.heatmap(speed_df.T, cmap="Spectral_r", xticklabels=300, robust=True)
        ax.set(title=f'{run} Object Distances', xlabel='Time')
        if save:
            fig.savefig(f'{args.output_objspeed_csv}/{run}_objspeed.png')
        plt.close(fig)
        logger.info(f'Done Object Speed {run}')
        with open('objspeed_complete.txt', 'a') as f:
            f.write(run + '\n')
    except Exception as e:
        with open('objspeed_error.txt', 'a') as f:
            f.write(run + '\n')
            f.write(repr(e) + '\n')
            f.write(traceback.format_exc() + '\n')


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

    import os

    if not os.path.exists(args.output_objspeed_csv):
        os.makedirs(args.output_objspeed_csv)
    if '.txt' not in args.run:
        calculate_object_speed(args, runs[0], save=True)
    else:
        Parallel(n_jobs=8)(delayed(calculate_object_speed)(args, run) for run in runs)
