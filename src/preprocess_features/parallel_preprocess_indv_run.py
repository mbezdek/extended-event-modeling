import subprocess
import os
from src.utils import parse_config


args = parse_config()

path = args.run
assert '.txt' in path, f"run argument should be a *.txt, fed {path}"
with open(path, 'r') as f:
    runs = f.readlines()
    runs = [run.strip() for run in runs]
print(f"Total runs={len(runs)}")

complete_txt = f"output/preprocessed_complete_{args.feature_tag}.txt"
if os.path.exists(complete_txt):
    print(f"{complete_txt} existed, deleting..")
    os.remove(complete_txt)
error_txt = f"output/preprocessed_error_{args.feature_tag}.txt"
if os.path.exists(error_txt):
    print(f"{error_txt} existed, deleting..")
    os.remove(error_txt)
filtered_txt = f"output/filtered_skel_{args.feature_tag}_{args.ratio_samples}_{args.ratio_features}.txt"
if os.path.exists(filtered_txt):
    print(f"{filtered_txt} existed, deleting..")
    os.remove(filtered_txt)

for run in runs:
    proc = subprocess.Popen(
        ['srun', '--time=4:00:00', '--cpus-per-task', '1',
         'python', 'src/preprocess_features/preprocess_indv_run.py', '-c', 'configs/config_preprocess.ini',
         '--run', run])
