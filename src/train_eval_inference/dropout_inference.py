# example usage: python dropout_inference.py 0.5 0.5
import os
import sys

# this doesn't solve the no module named 'sem' error when running the script on terminal
# in pycharm, adding content root to PYTHONPATH solve that problem
sys.path.extend([os.path.join(os.path.dirname(os.getcwd()), "SEM2")])

import pandas as pd

## Running this script on the cluster: weirdly, if --mem-per-cpu is specified, the script seems to halt
# actually, some nodes can run the script, but some nodes can't. --mem-per-cpu is not the problem.
# (no error message, no output)
import pickle as pkl
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import ray

# When Ray starts on a machine,
# a number of Ray workers will be started automatically (1 per CPU by default).
# A Ray Actor is also a “Ray worker” but is instantiated at runtime (upon actor_cls.remote()).
# All of its methods will run on the same process, using the same resources
# (designated when defining the Actor). Note that unlike tasks, the python processes that runs
# Ray Actors are not reused and will be terminated when the Actor is deleted.
# https://docs.ray.io/en/master/actors.html
# ray.init(num_cpus=16)
from sem.event_models import GRUEvent
# from importlib import reload  # Not needed in Python 2
import logging

# reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.debug("DEBUG Mode")
logger.info("INFO Mode")

# read command line arguments from sys.argv
# sys.argv[0] is the name of the script
# sys.argv[1] is the first argument
# sys.argv[2] is the second argument
if len(sys.argv) > 2:
    dropout_rate = float(sys.argv[1])
else:
    dropout_rate = 0.5
if len(sys.argv) > 3:
    recurrent_dropout_rate = float(sys.argv[2])
else:
    recurrent_dropout_rate = 0.5

output_dir = "../../output/diagnose/dropout_epoch/"

# This also implies that tasks are scheduled more flexibly, and that if you don’t need
# the stateful part of an actor, you’re mostly better off using tasks.
# https://docs.ray.io/en/master/actors.html
# num_cpus=1 is the default value, take ~13s to run, num_cpus=4 takes ~24s to run
# @ray.remote(num_cpus=1)
def generate_predictions(scene_vectors, e_hats, weights, n_predictions=1000):
    """
    Generate dropout inferences for a given scene vector and e_hat, for all timesteps.
    """
    optimizer_kwargs = dict(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
    f_opts = dict(var_df0=10., var_scale0=0.06, l2_regularization=0.0, 
                  n_epochs=1, t=4, batch_update=True, n_hidden=15, variance_window=None, optimizer_kwargs=optimizer_kwargs,
                  recurrent_dropout=recurrent_dropout_rate, dropout=dropout_rate)
    new_model = GRUEvent(scene_vectors.shape[1], **f_opts)
    new_model.init_model()
    new_model.do_reset_weights()
    # set True so that the model doesn't return the input
    new_model.f_is_trained = True
    # predict 1733 steps for 1734 time-steps because we don't predict for the first step.
    steps = min(scene_vectors.shape[0] - 1, n_predictions)
    x_hat_both_dropout = np.zeros(shape=(steps + 1, 30))
    e_hat_prev = None
    for i in range(steps):
        e_hat = e_hats[i]
        if e_hat != e_hat_prev:
            new_model.x_history = [np.zeros(shape=(0, 30), dtype=np.float64)]
        new_model.set_model_weights(weights[e_hat])
        # predict_next is a wrapper for _predict_next to return the same vector if the model is untrained.
        # setting trained before so it's the same to use either.
        _, both_dropout = new_model.predict_next(scene_vectors[i, :], dropout=True, recurrent_dropout=True)
        # update_estimate=False so that weights and noise_variance is not updated and
        # only concatenate the current vector to history.
        new_model.update(scene_vectors[i, :], scene_vectors[i + 1, :], update_estimate=False)
        x_hat_both_dropout[i + 1, :] = both_dropout
        e_hat_prev = e_hat
    x_hat_both_dropout[0] = scene_vectors[0]
    return x_hat_both_dropout


valid_runs = open('../../output/valid_sep_09.txt', 'rt').readlines()
valid_runs = [x.strip() for x in valid_runs]
train_runs = open('../../output/train_sep_09.txt', 'rt').readlines()
train_runs = [x.strip() for x in train_runs]
runs = train_runs + valid_runs

# for each epoch, iterate over all runs, perform dropout inference and save the results for each run.
# runs = ['1.2.3_kinect']
res = {}
for epoch in range(1, 108, 10):
    for run in runs:
        import glob

        files = glob.glob(f'output/run_sem/oct_13_refactor_seed1080_1E-03_1E-01_1E+07/{run}*_diagnostic_{epoch}.pkl')
        if len(files) == 0:
            logger.info(f"No files for {run} at epoch {epoch}")
            continue
        assert len(files) == 1, f"More than one file found for run {run} at epoch {epoch}: {files}"
        # last_e_file = sorted(files)[-1]
        e_file = files[0]
        # logger.info(f"Latest epoch for {run} is :{os.path.basename(e_file)}")
        diagnostic = pkl.load(open(f'{e_file}', 'rb'))
        # diagnostic = pkl.load(open(f'output/run_sem/oct_13_refactor_seed1080_1E-03_1E-01_1E+07/{run}_trimoct_13_refactor_seed1080_1E-03_1E-01_1E+07_diagnostic_101.pkl', 'rb'))
        n_resample = 8
        res[run] = {}
        res[run]['diagnostic'] = diagnostic
        # futures = []
        res[run]['resamples'] = []
        for i in range(n_resample):
            x_hat_both_dropout = generate_predictions(scene_vectors=diagnostic['x'],
                                                  e_hats=diagnostic['e_hat'],
                                                  weights=diagnostic['weights'])
            res[run]['resamples'].append(x_hat_both_dropout)
            # logger.info(f'Done {i+1} resample')
            # futures.append(generate_predictions.remote(scene_vectors=diagnostic['x'],
            #                                         e_hats=diagnostic['e_hat'],
            #                                         weights=diagnostic['weights']))
            # logger.info(f'Queued {i + 1} resample for {run}')
        # res[run]['resamples'] = ray.get(futures)
        pkl.dump(res[run], open(os.path.join(output_dir,
                                            f"res_dropout_{run}_{epoch}_{dropout_rate}_{recurrent_dropout_rate}.pkl"),
                                'wb'))
        logger.info(f"Done generating predictions for {run} at epoch {epoch}")

        # for each run, plot predictions, prediction error, and variance timecourse
        is_val = run in valid_runs
        # plot dropout SEM predictions, full SEM's predictions, and ground truth
        features = list(range(30))
        fig = make_subplots(rows=len(features), cols=1,
                            subplot_titles=list(f'Feature {feature}' for feature in features))
        for r, feature in enumerate(features):
            for i in range(len(res[run]['resamples'])):
                fig.add_trace(go.Line(x=np.arange(1000), y=res[run]['resamples'][i][:, feature], name=f'x_hat_dropout_r{i}',
                                    showlegend=(feature == 25)), row=r + 1, col=1)
            fig.add_trace(go.Line(x=np.arange(1000), y=res[run]['diagnostic']['x_hat'][:1000, feature], name='x_hat',
                                showlegend=(feature == 25)), row=r + 1, col=1)
            fig.add_trace(go.Line(x=np.arange(1000), y=res[run]['diagnostic']['x'][:1000, feature], name='x_input',
                                showlegend=(feature == 25)), row=r + 1, col=1)

        fig.update_layout(title_text="Prediction of full SEM and dropout SEM",
                        height=len(features) * 500)
        fig.write_html(os.path.join(output_dir, f"{run}_{epoch}_{'val' if is_val else 'train'}_{dropout_rate}_{recurrent_dropout_rate}_predictions.html"))

        # plotting prediction error for dropout SEM resamples and full SEM
        fig = go.Figure()

        for i in range(len(res[run]['resamples'])):
            pe_both_dropout = np.linalg.norm(res[run]['diagnostic']['x'][:1000, :] - res[run]['resamples'][i][:1000, :],
                                            axis=1)
            fig.add_trace(go.Line(x=np.arange(1000), y=pe_both_dropout, name=f'pe_dropout_r{i}'))
        fig.add_trace(go.Line(x=np.arange(1000), y=res[run]['diagnostic']['pe'][:1000], name='pe'))

        fig.update_layout(title_text="Prediction Error of full SEM and dropout SEM")
        fig.write_html(os.path.join(output_dir, f"{run}_{epoch}_{'val' if is_val else 'train'}_{dropout_rate}_{recurrent_dropout_rate}_pe.html"))

        # plot variance of dropout SEM resamples and full SEM's prediction error
        df = pd.DataFrame(np.concatenate(res[run]['resamples'], axis=0),
                        index=[i for j in range(len(res[run]['resamples'])) for i in
                                range(res[run]['resamples'][0].shape[0])])
        variance = df.groupby(df.index).apply(np.var).mean(axis=1)
        fig = go.Figure()

        fig.add_trace(go.Line(x=np.arange(1000), y=res[run]['diagnostic']['pe'][:1000] / 100, name='pe/100'))
        fig.add_trace(go.Line(x=np.arange(1000), y=variance[:1000], name='variance'))

        corr = stats.pearsonr(res[run]['diagnostic']['pe'][:1000], variance[:1000])[0]
        fig.update_layout(title_text=f"Prediction Errors of full SEM (/100) and Variance of dropout SEM, Pearson={corr:.02f}")
        fig.write_html(os.path.join(output_dir, f"{run}_{epoch}_{'val' if is_val else 'train'}_{dropout_rate}_{recurrent_dropout_rate}_variance_and_pe.html"))
        logger.info(f"Done plotting for {run} at epoch {epoch}")
