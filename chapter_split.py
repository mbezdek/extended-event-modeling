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
files = []
chapters = ['1', '2', '3', '4']
for c in chapters:
    files.append(f'chapter_{c}.txt')
    chapter_runs = [r for r in sem_runs if f'.{c}.' in r]
    with open(f'chapter_{c}.txt', 'w') as f:
        f.writelines(chapter_runs)
with open('chapter_input.txt', 'w') as f:
    f.writelines('\n'.join(files))
