import subprocess

alfa = [1e6, 1e4, 1e2, 1e0, 1e-2]
# alfa = [1e-8]
lmda = [1e0, 1e2, 1e4, 1e6, 1e8]
# lmda = [1e4]

for a in alfa:
    for l in lmda:
        if a > l:
            print(f'Alfa={a} > Lmda={l}, skip this run')
            continue
        tag = f'june_17_pca30_lr1e-3_grid_alfa{a:.0E}_lmda{l:.0E}'
        proc = subprocess.Popen(
            ['sbatch', '--job-name', tag, 'model_corpus.sh',
             'train.txt', 'valid.txt', f'{a:.0E}', f'{l:.0E}', tag])
