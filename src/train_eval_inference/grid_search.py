import subprocess

trigger = 'pe'
thresholds = [4.3e-1]
alfa = [2e-2]  # for PE
lmda = [5e0]  # for PE

# trigger = 'always'
# thresholds = [1e-1]  # just as a placeholder
# alfa = [1e-1]  # for always
# lmda = [5e5]  # for always

# trigger = 'uncertainty'
# thresholds = [2.5e-3]
# alfa = [1e0]  # for uncertainty
# lmda = [5e-3]  # for uncertainty
seeds = [x for x in range(1010, 1110, 10)]
# seeds = [x for x in [1050, 1080] + list(range(1090, 1170, 10))]
node_id = 0
nodes = list(range(15, 33))
# for pe_threshold in pe_thresholds:
for lr in [1e-3]:
    for threshold in thresholds:
        for a in alfa:
            for l in lmda:
                for seed in seeds:
                    # if a >= l:
                    #     print(f'Alfa={a} > Lmda={l}, skip this run')
                    #     continue
                    tag = f'aug_04_{trigger}_nosema_{threshold:.1E}_s{seed}_{a:.0E}_{l:.0E}'
                    # tag = f'april_24_nosema_s{seed}_{a:.0E}_{l:.0E}'
                    # make sure to double-check the argument order in model_corpus.sh
                    proc = subprocess.Popen(
                        ['sbatch', '--job-name', tag,
                         '--nodelist', f'node{nodes[node_id]:02d}',
                         # NOTE: hard-code here is dangerous, changing name of directory in local wouldn't refactor this, combined
                         # with not deleting in remote can result in running old code
                         'src/train_eval_inference/model_corpus.sh',
                         'output/train_sep_09.txt', #$1
                         # 'output/train_sep_09_for_fmri.txt', #$1
                         'output/valid_sep_09.txt', #$2
                         # 'output/valid_sep_09_for_fmri.txt', #$2
                         f'{a:.0E}', #$3
                         f'{l:.0E}', #$4
                         tag, #$5
                         trigger, #$6
                         f'{threshold:.1E}', #$7
                         f'{seed}',
                         f'{lr:.0E}']) #$9
                    node_id += 1
