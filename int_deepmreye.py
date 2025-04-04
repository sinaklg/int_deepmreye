"""
-----------------------------------------------------------------------------------------
int_deepmreye.py
-----------------------------------------------------------------------------------------
Goal of the script:
Run deepmreye using INT fine tuned weights on fmriprep output 
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1] -> main directory (str)
sys.argv[2] -> project name (str)
sys.argv[3] -> task name (str)
-----------------------------------------------------------------------------------------
Output(s):
BIDS extended folder structure with deepmreye output: 
- figures: prediction_visualizer.html (Overview of prediction for all subjects)
- masks: MNI space registered eyevoxel masks 
- pp_data: labels used by model 
- pred: tsv.gz (timestamp, x, y) of prediction in median (1/TR) and subTR (10/TR) resolution 
for each subject as well as evaluations dictionary including predictions and scores with 
evaluation of prediction to labels
- report: eye voxel mask extraction report per subject 
-----------------------------------------------------------------------------------------
"""

# Import modules and add library to path
import sys
import json
import os
import pickle
import glob
import warnings
import numpy as np
import pandas as pd
import shutil
import re

# DeepMReye imports
from deepmreye import analyse, preprocess, train
from deepmreye.util import data_generator, model_opts 

sys.path.append("{}/utils".format(os.getcwd()))
from training_utils import adapt_evaluation

# Define paths 
main_dir = os.path.join(sys.argv[1], sys.argv[2], "derivatives", "int_deepmreye") 
project_name = sys.argv[2]
task = sys.argv[3]
fig_dir = f"{main_dir}/figures"
func_dir = f"{main_dir}/func"  
model_dir = f"{main_dir}/model"
model_file = f"{model_dir}/int_deepmreye_weights.h5" 
pp_dir = f"{main_dir}/pp_data/"
mask_dir = f"{main_dir}/mask"
report_dir = f"{main_dir}/report"
pred_dir = f"{main_dir}/pred"

# Make directories
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(pp_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

# Define settings
settings_file = "settings.json"
if not os.path.exists(settings_file):
    raise FileNotFoundError(f"Settings file {settings_file} not found.")
with open(settings_file) as f:
    settings = json.load(f)

subjects = settings['subjects']
ses = settings["session"]
num_run = settings["num_run"]
subTRs = settings['subTRs']
TR = settings['TR']

opts = model_opts.get_opts()
opts["train_test_split"] = settings["train_test_split"]

# Define environment cuda
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # stop warning
if settings["partition"] == "volta" or settings["partition"] == "kepler":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # use 3 gpu cards 


# -------------- Preload masks to save time within subject loop---------------------------------
# Load masks/templates once
(eyemask_small, eyemask_big, dme_template, mask, x_edges, y_edges, z_edges) = preprocess.get_masks()

for subject in subjects:
    print(f"Running {subject}")
    func_sub_dir = f"{func_dir}/{subject}"
    mask_sub_dir = f"{mask_dir}/{subject}"
    os.makedirs(mask_sub_dir, exist_ok=True)

    # Get all run files
    func_files = sorted(glob.glob(f"{func_sub_dir}/*.nii.gz"))  # one .nii.gz per run
    print(func_files)

    for func_file in func_files:
        # Extract run number from filename
        run_match = re.search(r'run-(\d+)', func_file)
        run_number = run_match.group(1) if run_match else "01"

        # Desired mask filename for this run
        mask_filename = f"mask_{subject}_ses-02_task-{task}_run-{run_number}_space-T1w_desc-preproc_bold.p"
        mask_path = os.path.join(mask_sub_dir, mask_filename)

        # Check if mask already exists for this specific run
        if os.path.exists(mask_path):
            print(f"Mask for {subject} run-{run_number} already exists. Skipping.")
            continue

        print(f"Mask for {subject} run-{run_number} missing. Running mask generation...")

        # Generate the mask
        preprocess.run_participant(
            fp_func=func_file, 
            dme_template=dme_template, 
            eyemask_big=eyemask_big, 
            eyemask_small=eyemask_small,
            x_edges=x_edges, y_edges=y_edges, z_edges=z_edges,
            transforms=['Affine', 'Affine', 'SyNAggro']
        )

        # Move new mask to mask directory
        new_mask_files = glob.glob(f"{func_sub_dir}/*.p")
        if new_mask_files:
            for mask_file in new_mask_files:
                shutil.move(mask_file, mask_sub_dir)
            print(f"Mask files for run-{run_number} moved to {mask_sub_dir}.")
        else:
            print(f"WARNING: No mask files found after processing run-{run_number}.")



# -------------------- Pre-process data ---------------------------------------------
# Pre-process data
for subject in subjects:    
    print(f"Preprocessing {subject}")
    subject_data = []
    subject_labels = [] 
    subject_ids = []

    for run in range(num_run): 
         # Identify mask and label files
            mask_filename = f"mask_{subject}_{ses}_task-{task}_run-0{run + 1}_space-T1w_desc-preproc_bold.p"
            mask_path = os.path.join(mask_dir, subject, mask_filename)

            if not os.path.exists(mask_path):
                print(f"WARNING --- Mask file {mask_filename} not found for Subject {subject} Run {run + 1}.")
                continue

            # Load mask and normalize it
            this_mask = pickle.load(open(mask_path, "rb"))
            this_mask = preprocess.normalize_img(this_mask)

            # No labels bc pretrained
            this_label = this_label = np.zeros(
                (this_mask.shape[3], 10, 2)
            )

            # Check if each functional image has a corresponding label
            if this_mask.shape[3] != this_label.shape[0]:
                print(
                    f"WARNING --- Skipping Subject {subject} Run {run + 1} "
                    f"--- Wrong alignment (Mask {this_mask.shape} - Label {this_label.shape})."
                )
                continue

            # Store across runs
            subject_data.append(this_mask)  # adds data per run to list
            subject_labels.append(this_label)
            subject_ids.append(([subject] * this_label.shape[0],
                                    [run + 1] * this_label.shape[0]))
            
    # Save participant file
    preprocess.save_data(participant=f"{subject}_{task}_no_label",
                            participant_data=subject_data,
                            participant_labels=subject_labels,
                            participant_ids=subject_ids,
                            processed_data=pp_dir,
                            center_labels=False)
try:
    os.system(f'rm {pp_dir}.DS_Store')
    print('.DS_Store file deleted successfully.')
except Exception as e:
    print(f'An error occurred: {e}')

# Define paths to dataset
datasets = [pp_dir + p for p in os.listdir(pp_dir) if "no_label" in p]

# Load data from one participant to showcase input/output
X, y = data_generator.get_all_subject_data(datasets[0])
print(f"Input: {X.shape}, Output: {y.shape}")

test_participants = [pp_dir + p for p in os.listdir(pp_dir) if "no_label" in p]
generators = data_generator.create_generators(test_participants,
                                              test_participants)
generators = (*generators, test_participants, test_participants)  # Add participant list

# -------------------- Train and evaluate model -----------------------------------------
# Get untrained model and load with trained weights
(model, model_inference) = train.train_model(dataset=f"{task}_PT",
                                             generators=generators,
                                             opts=opts,
                                             return_untrained=True)
model_inference.load_weights(model_file)

(evaluation, scores) = train.evaluate_model(dataset=f"{task}_PT",
    model=model_inference,
    generators=generators,
    save=False,
    model_path=model_dir,
    model_description="",
    verbose=2,
    percentile_cut=80,
)    

fig = analyse.visualise_predictions_slider(evaluation, scores, ylim=settings["ylim"])
fig.write_html(f"{fig_dir}/prediction_visualizer.html")
   
# Sava data      
np.save(f"{pred_dir}/evaluation_dict_{task}.npy",evaluation)
np.save(f"{pred_dir}/scores_dict_{task}.npy",scores)


# Save predictions as tsv
labels_list = os.listdir(pp_dir)

for label in labels_list:
    subject = label.split("_")[0]  
    print(f"saving subject: {subject}")
    
    
    # Get predictions
    df_pred_median, df_pred_subtr = adapt_evaluation(evaluation[f'{main_dir}/pp_data/{label}'])
    print(len(df_pred_median))

    # Split BEFORE adding timestamps
    df_pred_median_parts = np.array_split(df_pred_median, num_run)
    df_pred_subtr_parts = np.array_split(df_pred_subtr, num_run * 10) 


    for i in range(num_run):
        subtr_start = i * 10
        subtr_end = (i + 1) * 10
        df_subtr_part = pd.concat(df_pred_subtr_parts[subtr_start:subtr_end], ignore_index=True)
        df_median_part = df_pred_median_parts[i].reset_index(drop=True)

        # Add timestamps
        df_median_part.insert(0, 'timestamp', df_median_part.index.astype(int) * TR)
        df_subtr_part.insert(0, 'timestamp', df_subtr_part.index.astype(int) * (TR / 10))  # Assuming 10 Hz

        # Keep only relevant columns
        df_median_part = df_median_part[['timestamp', 'X', 'Y']]
        df_subtr_part = df_subtr_part[['timestamp', 'X', 'Y']]

        # Save
        df_median_part.to_csv(f'{pred_dir}/{subject}_ses-{ses}_task-{task}_run-0{i+1}_pred_median.tsv.gz', sep='\t', index=False, compression='gzip')
        df_subtr_part.to_csv(f'{pred_dir}/{subject}_ses-{ses}_task-{task}_run-0{i+1}_pred_subtr.tsv.gz', sep='\t', index=False, compression='gzip')



# Move .p and .html to destination folders
for subject in subjects:
	func_sub_dir = f"{func_dir}/{subject}"
	
	# move .html files
	rsync_report_cmd = f"rsync -avuz --remove-source-files {func_sub_dir}/*.html {report_dir}/{subject}/"
	rm_report_cmd = f"rm -Rf {func_sub_dir}/*.html"
	os.system(rsync_report_cmd)
	os.system(rm_report_cmd)

