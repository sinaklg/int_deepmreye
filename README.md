# int_deepmreye

By :       Sina KLING<br/>
Project :  int_deepmreye <br/>
With :     Matthias NAU, Martin SZINTE<br/>
Version:   0.1<br/>

## Version description
Gaze decoder using DeepMReye network (https://github.com/DeepMReye/DeepMReye) fine tuned for experiments at 
INT/CERIMED. 


## Installation of deepmreye 

It is recommmended to work using conda virtual environments. 
Install DeepMReye with a CPU/GPU version of TensorFlow using the following command.

```
conda create --name deepmreye python=3.9
conda activate deepmreye
pip install deepmreye
git clone git@github.com:sinaklg/int_deepmreye.git
cd int_deepmreye
pip install -r requirements.txt
```

For GPU support (if available and nvidia), install GPU toolkits:

```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## Model Weights and Example Data 
Model weights and example data are available [here](https://figshare.com/s/a381947ea8564235cdd0). 
Download the example dataset and model weigths by running: 

```
python example_usage.py int_dataset_example
```

The example contains 3 participants BOLD timeseries preprocessed and registered to individual T1w structural scans using fmriprep 23.1.4.

Data repository will be structured as such:

```
int_dataset_example/DeepMReyeCalib/derivatives/int_deepmreye
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

All analaysis parameters are set in [settings.json](settings.json).
Make sure to specify this file to your experiment settings beforehand. 

# Steps to follow for GPU usage AMU Mesocentre

Make sure the partition in settings is set to "volta" or "kepler". 

1. Generate SLURM script by running 

``` 
python generate_sh.py 

```

2. Run SLURM script to run the main deepmreye pipeline on mesocentre

```
sbatch run_pretraining.sh 

```

# Steps to follow for local running 

Just run the main script with

```
python cerimed_deepmreye.py [main_directory] [project_name] [task] [group]

```