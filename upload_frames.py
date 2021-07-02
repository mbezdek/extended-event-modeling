from joblib import delayed, Parallel
import glob
import os
import subprocess

file_names = glob.glob('output/run_sem/frames/*frames.joblib')
# file_names = file_names[:4]


def upload(name):
    print(f'Uploading {name}...')
    subprocess.run(['scp', f'{name}', 'n.tan@login3-02.chpc.wustl.edu:/scratch/n.tan/extended-event-modeling/output/run_sem/frames/'])

    print(f'Done {name}!')


Parallel(n_jobs=8)(delayed(upload)(file_name) for file_name in file_names)
