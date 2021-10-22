import subprocess

# alfa = [1e2, 1e0, 1e-2, 1e-4]
# alfa = [1e-2, 1e-4]
alfa = [1e-2]
# lmda = [1e2, 1e4, 1e6, 1e8]
lmda = [1e4, 1e6]
# lmda = [1e4]
# lrs = [1e-1, 1e-2, 1e-4, 1e-5]
# lrs = [1e-4]
lrs = [1e-3]

for lr in lrs:
    for a in alfa:
        for l in lmda:
            if a >= l:
                print(f'Alfa={a} > Lmda={l}, skip this run')
                continue
            tag = f'oct_22_h30_global_std_skel_grid_lr{lr:.0E}_alfa{a:.0E}_lmda{l:.0E}'
            proc = subprocess.Popen(
                ['sbatch', '--job-name', tag, 'model_corpus.sh',
                 'train.txt', 'valid.txt', f'{a:.0E}', f'{l:.0E}', tag, f'{lr:.0E}'])
