import subprocess

trigger = 'pe'
thresholds = [5e-1]
# trigger = 'uncertainty'
# thresholds = [3e-3]

alfa = [1e-2]  # for PE
lmda = [1e-1]  # for PE
# alfa = [1e0]  # for uncertainty
# lmda = [1e-2]  # for uncertainty
seeds = [x for x in range(1010, 1090, 10)]
node_id = 0
nodes = list(range(13, 13+12))
# for pe_threshold in pe_thresholds:
for threshold in thresholds:
    for a in alfa:
        for l in lmda:
            for seed in seeds:
                # if a >= l:
                #     print(f'Alfa={a} > Lmda={l}, skip this run')
                #     continue
                tag = f'dec_24_{trigger}{threshold:.0E}_s{seed}_{a:.0E}_{l:.0E}'
                # make sure to double-check the argument order in model_corpus.sh
                proc = subprocess.Popen(
                    ['sbatch', '--job-name', tag,
                     '--nodelist', f'node{nodes[node_id]:02d}',
                     # hard-code here is dangerous, changing name of directory in local wouldn't refactor this, combined
                     # with not deleting in remote can result in running old code
                     'src/train_eval_inference/model_corpus.sh',
                     'output/train_sep_09.txt', #$1
                     'output/valid_sep_09.txt', #$2
                     f'{a:.0E}', #$3
                     f'{l:.0E}', #$4
                     tag, #$5
                     trigger, #$6
                     f'{threshold:.0E}', #$7
                     f'{seed}']) #$8
                node_id += 1
