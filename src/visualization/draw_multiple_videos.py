# import subprocess
#
# with open('src/visualization/runs_to_draw.txt', 'r') as f:
#     lines = f.readlines()
# for line in lines:
#     run_select, tag, epoch = [x.strip() for x in line.split(' ')]
#
#     proc = subprocess.Popen(
#           [f'python', './src/visualization/draw_video.py',  f'{run_select}', f'{tag}',  f'{epoch}'])

import subprocess
from concurrent.futures import ProcessPoolExecutor

# Function to read the commands from the file and return them as a list
def generate_commands_from_file(filename):
    commands = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        run_select, tag, epoch = [x.strip() for x in line.split(' ')]
        command = ['python', './src/visualization/draw_video.py', run_select, tag, epoch]
        commands.append(command)
    return commands

def run_command(command):
    """Run a command (list of strings) using subprocess."""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Done: {' '.join(command)}. Output: {result.stdout.decode('utf-8')}")

# Generate the list of commands
commands = generate_commands_from_file('src/visualization/runs_to_draw.txt')

# Limiting to 2 processes at a time (you can adjust this number as needed)
with ProcessPoolExecutor(max_workers=6) as executor:
    executor.map(run_command, commands)
