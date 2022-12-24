import subprocess

with open('src/visualization/runs_to_draw.txt', 'r') as f:
    lines = f.readlines()
for line in lines:
    run_select, tag, epoch = [x.strip() for x in line.split(' ')]

    proc = subprocess.Popen(
          [f'python', './src/visualization/draw_video.py',  f'{run_select}', f'{tag}',  f'{epoch}'])
