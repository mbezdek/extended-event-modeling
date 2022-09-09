import subprocess

alfa = [1e-1]
lmda = [1e7]
lrs = [1e-3]
seeds = [x for x in range(1090, 1170, 10)]
node_id = 0
nodes = list(range(14, 14+8))
for lr in lrs:
    for a in alfa:
        for l in lmda:
            for s in seeds:
                if a >= l:
                    print(f'Alfa={a} > Lmda={l}, skip this run')
                    continue
                tag = f'aug_18_no_semantics_off_update_{s}_{lr:.0E}_{a:.0E}_{l:.0E}'
                proc = subprocess.Popen(
                    ['sbatch', '--job-name', tag,
                     '--nodelist', f'node{nodes[node_id]:02d}',
                     'model_corpus.sh',
                     'qualified_train_.8_.8.txt', 'qualified_valid_.8_.8.txt', f'{a:.0E}', f'{l:.0E}', tag, f'{lr:.0E}',
                     f'{s}'])
                node_id += 1
