import json
import os 

def generate_slurm_script(settings_file, output_file):
    # Define settings
    settings_file = "settings.json"
    if not os.path.exists(settings_file):
        raise FileNotFoundError(f"Settings file {settings_file} not found.")
    with open(settings_file) as f:
        settings = json.load(f)

    # Ensure log directory exists
    log_dir = os.path.join(settings['log_dir'], "log_outputs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Define the SLURM script template
    slurm_script = f"""#!/bin/bash
#SBATCH -p {settings['partition']}
#SBATCH -A {settings['account']}
#SBATCH --nodes={settings['nodes']}
#SBATCH --gres=gpu:{settings['gpus']}
#SBATCH --time={settings['time']}
#SBATCH -e {log_dir}/deepmreye_%N_%j_%a.err
#SBATCH -o {log_dir}/deepmreye_%N_%j_%a.out
#SBATCH -J {settings['job_name']}

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python {settings['script']} {settings['data_path']} {settings['project_name']} {settings['project_id']}"""
    
    # Write the script to file
    with open(output_file, "w") as f:
        f.write(slurm_script)
    
    print(f"SLURM script written to {output_file}")

# Example usage
generate_slurm_script("settings.json", "run_pretraining.sh")
