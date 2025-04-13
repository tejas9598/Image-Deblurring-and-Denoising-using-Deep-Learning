import os
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path
import shutil

def download_file(url, filename):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def prepare_data():
    """Download and prepare DIV2K dataset."""
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)
    
    # Download DIV2K validation set (smaller and sufficient for testing)
    print("Downloading DIV2K validation set...")
    url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    zip_path = "data/DIV2K_valid_HR.zip"
    
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    
    # Extract images
    print("Extracting images...")
    extract_path = "data/temp_extract"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    # Move images to raw directory
    print("Organizing images...")
    raw_dir = Path('data/raw')
    temp_dir = Path(extract_path)
    
    # Find all PNG files in the extracted directory
    for img in temp_dir.rglob('*.png'):
        # Move the image to the raw directory
        shutil.move(str(img), str(raw_dir / img.name))
    
    # Clean up
    print("Cleaning up...")
    os.remove(zip_path)
    shutil.rmtree(extract_path)
    
    print(f"Dataset preparation complete! Images are in {raw_dir}")
    print(f"Number of images downloaded: {len(list(raw_dir.glob('*.png')))}")

if __name__ == '__main__':
    prepare_data() 