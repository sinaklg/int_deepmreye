"""
-----------------------------------------------------------------------------------------
download_dataset_weights.py
-----------------------------------------------------------------------------------------
Goal of the script:
Download example data from figshare 
-----------------------------------------------------------------------------------------
Input(s):
sys.argv[1] -> extraction directory (str)
-----------------------------------------------------------------------------------------
Output(s):
Downloaded int_deepmreye.zip
Extracted DeepMreyeCalib folder structure of example data 
-----------------------------------------------------------------------------------------
To run:
1. cd to function
2. python example_usage.py [extraction directory] 
-----------------------------------------------------------------------------------------
"""
import requests
import zipfile
import os
import sys
from tqdm import tqdm


URL = "https://figshare.com/ndownloader/articles/28513283?private_link=a381947ea8564235cdd0"
extract_dir = sys.argv[1]
zip_path = f"{extract_dir}/int_deepmreye_example_data.zip"

# Ensure the directory exists
os.makedirs(os.path.dirname(zip_path), exist_ok=True)

# Start the file download with progress tracking
response = requests.get(URL, stream=True)
total_size = int(response.headers.get('content-length', 0))  # Get file size

if response.status_code == 200:
    with open(zip_path, "wb") as file, tqdm(
        desc="Downloading", total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as progress:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            progress.update(len(chunk))  # Update progress bar
    print(f"\nDownload complete: {zip_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Extract ZIP file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)
    os.system(f'rm {zip_path}')

print(f"Extraction complete to: {extract_dir}")