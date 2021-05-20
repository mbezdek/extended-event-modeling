import subprocess

alfa = [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8]
# alfa = [1e-8]
lmda = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
# lmda = [1e4]

for a in alfa:
    for l in lmda:
        proc = subprocess.Popen(
            ['sbatch', '--job-name', 'partition', 'model_corpus.sh', 'train.txt', 'valid.txt',
             f'{a:.0E}', f'{l:.0E}', f'may_12_grid_alfa{a:.0E}_lmda{l:.0E}'])
