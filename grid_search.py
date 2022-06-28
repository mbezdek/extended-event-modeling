import subprocess

alfa = [1e-1]
lmda = [1e5]
lrs = [1e-3]
seeds = [x for x in [1010, 1050, 1070, 1140, 1150]]
node = 14
for lr in lrs:
    for a in alfa:
        for l in lmda:
            for s in seeds:
                if a >= l:
                    print(f'Alfa={a} > Lmda={l}, skip this run')
                    continue
                tag = f'june_25_{s}_deprived_emb'
                proc = subprocess.Popen(
                    ['sbatch', '--job-name', tag,
                     '--nodelist', f'node{node:02d}',
                     'model_corpus.sh',
                     'inter_qualified_train.txt', 'inter_qualified_valid.txt', f'{a:.0E}', f'{l:.0E}', tag, f'{lr:.0E}',
                     f'{s}'])
                node += 1
