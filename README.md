# int_deepmreye

By :       Sina KLING<br/>
Project :  INT_DeepMReye <br/>
With :     Matthias NAU, Martin SZINTE<br/>
Version:   1.0<br/>

## Version description
Gaze decoder using DeepMReye network (https://github.com/DeepMReye/DeepMReye) fine tuned for experiments at 
INT/CERIMED. 


## Installation of deepmreye 

It is recommmended to work using virtual environments. 
Install DeepMReye with a CPU/GPU version of TensorFlow using the following command.

```
conda create --name deepmreye python=3.9
conda activate deepmreye
pip install deepmreye
pip install -r requirements.txt

```

For GPU support, install correct tensorflow version and GPU toolkits:

```
pip install tensorflow==2.11.0
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

```

## Model Weights and Example Data 
Model weights and example data are available on figshare. Download everything automatically by running: 

```
python example_usage.py [extraction directory]

```

## Pipeline 

Folder structure extents extisting BIDS structure. This is how the data repository will be structured. Download the weights from (). 

```bash
main_dir/project_name/derivatives
├── figures
├── func
│   ├── sub-0X
│   │   ├── sub-0X_ses-00_task-project_name_run-01_space-T1w_desc-preproc_bold.nii.gz
│   │   ├── sub-0X_ses-00_task-project_name_run-02_space-T1w_desc-preproc_bold.nii.gz
├── log_outputs
│   ├── deepmreye_gpu014_8969452_4294967294.err
│   ├── deepmreye_gpu014_8969452_4294967294.out
├── mask
│   ├── sub-0X
│   │   ├── mask_sub-0X_ses-00_task-project_name_run-01_space-T1w_desc-preproc_bold.p
│   │   ├── mask_sub-0X_ses-00_task-project_name_run-02_space-T1w_desc-preproc_bold.p
│   └── sub-0Y
│       ├── mask_sub-0Y_ses-00_task-project_name_run-01_space-T1w_desc-preproc_bold.p
│       ├── mask_sub-0Y_ses-00_task-project_name_run-02_space-T1w_desc-preproc_bold.p
├── model
│    └── modelinference_DeepMReyeCalib.h5
├── pp_data
│   ├── sub-0X_project_name_no_label.npz
│   ├── sub-0Y_project_name_no_label.npz
├── pred
│   ├── evaluation_project_name.npy
│   ├── scores_project_name.npy
│   ├── sub-0X
│   │   ├── sub-0X_pred_median.tsv.gz
│   │   ├── sub-0Y_pred_subTR.tsv.gz
├── report
│   ├── sub-0X
│   │   ├── report_sub-0X_ses-00_task-project_name_run-01_space-T1w_desc-preproc_bold.html

```

All parameters are set in settings. Specify your experiment details beforehand. For example:

```
"subjects": ["sub-0X","sub-0Y"], 
"session": "ses-00", 
"eye": "eye1", 
"num_run": 3,
"num_TR": 154, 

```

# Steps to follow 

1. Generate SLURM script by running 

``` 
python generate_sh.py 

```

2. Run SLURM script to run the main deepmreye pipeline on mesocentre

```
sbatch run_pretraining.sh 

```