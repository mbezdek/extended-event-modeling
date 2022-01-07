import subprocess

# alfa = [1e2, 1e0, 1e-2, 1e-4]
# alfa = [1e-2, 1e-4]
alfa = [1e-1, 1e-2]
# lmda = [1e2, 1e4, 1e6, 1e8]
lmda = [1e4, 1e5]
# lrs = [1e-1, 1e-2, 1e-4, 1e-5]
# lrs = [1e-4]
lrs = [1e-3]

node = 3
for lr in lrs:
    for a in alfa:
        for l in lmda:
            if a >= l:
                print(f'Alfa={a} > Lmda={l}, skip this run')
                continue
            tag = f'jan_06_scale_bicorr_grid_lr{lr:.0E}_alfa{a:.0E}_lmda{l:.0E}'
            proc = subprocess.Popen(
                ['sbatch', '--job-name', tag,
                 '--nodelist', f'node1{node}',
                 'model_corpus.sh',
                 'inter_qualified_train.txt', 'inter_qualified_valid.txt', f'{a:.0E}', f'{l:.0E}', tag, f'{lr:.0E}'])
            node += 1
