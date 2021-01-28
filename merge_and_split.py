import sys

with open('appear_complete.txt', 'r') as f:
    appears = f.readlines()

with open('vid_complete.txt', 'r') as f:
    vids = f.readlines()

with open('skel_complete.txt', 'r') as f:
    skels = f.readlines()

with open('objhand_complete.txt', 'r') as f:
    objhands = f.readlines()

sem_runs = set(appears).intersection(set(skels)).intersection(set(vids)).intersection(set(objhands))
sem_runs = list(sem_runs)
if len(sys.argv) > 1:
    number_of_files = int(sys.argv[1])
else:
    number_of_files = 10
number_of_runs = len(sem_runs)
files = []
for i in range(number_of_files):
    files.append(f'intersect_features_{i+1}.txt')
    with open(f'intersect_features_{i+1}.txt', 'w') as f:
        f.writelines(sem_runs[int(number_of_runs * i/number_of_files):int(number_of_runs * (i+1)/number_of_files)])
with open('job_input.txt', 'w') as f:
    f.writelines('\n'.join(files))
