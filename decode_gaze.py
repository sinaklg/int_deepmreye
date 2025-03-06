"""
-----------------------------------------------------------------------------------------
decode_gaze.py
-----------------------------------------------------------------------------------------
Goal of the script:
Decode gaze using DeepMReye network (https://github.com/DeepMReye/DeepMReye) 
fine tuned for experiments at MRI INT.
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1]: data output main folder ('/data/yourfolder'')
sys.argv[2]: project name ('your_project')
-----------------------------------------------------------------------------------------
Output(s):

-----------------------------------------------------------------------------------------
To run:
1. cd to function
2. python decode_gaze.py [data main folder] [project name]
-----------------------------------------------------------------------------------------
"""

import json
import os 
import sys

settings_file = "settings.json"
main_dir = sys.argv[1]
project_name = sys.argv[2]

output_folder = f"{main_dir}/{project_name}/derivatives/int_deepmreye"

# Define settings
if not os.path.exists(settings_file):
    raise FileNotFoundError(f"Settings file {settings_file} not found.")
with open(settings_file) as f:
    settings = json.load(f)

# Ensure log directory exists
log_dir = f"{output_folder}/log_outputs"
os.makedirs(log_dir, exist_ok=True)
job_dir = f"{output_folder}/jobs"
os.makedirs(job_dir, exist_ok=True)

# Define the SLURM script template
slurm_gpu_script = f"""#!/bin/bash
#SBATCH -p {settings['partition']}
#SBATCH -A {settings['account']}
#SBATCH --nodes={settings['nodes']}
#SBATCH --gres=gpu:{settings['gpus']}
#SBATCH --time={settings['time']}
#SBATCH -e {log_dir}/deepmreye_%N_%j_%a.err
#SBATCH -o {log_dir}/deepmreye_%N_%j_%a.out

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
"""

# Define python script
python_script = f"python int_deepmreye.py {main_dir} {project_name} {settings['task']}"


# Generate and run shell
sh_file = f"{job_dir}/gaze_decode_run.sh"
print(f"Run script written to {sh_file}")
if settings['gpu_use']:
    with open(sh_file, "w") as f:
        f.write(slurm_gpu_script)
        f.write(python_script)
    os.system(f"sbatch {sh_file}")
else:
    # Write the script to file
    with open(sh_file, "w") as f:
        f.write(python_script)
    os.system(f"sh {sh_file}")
