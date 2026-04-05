import os
import requests
import zipfile
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
# The official MS-COCO 2017 Unlabeled Dataset URL (~19 GB)
COCO_URL = "http://images.cocodataset.org/zips/unlabeled2017.zip"
ZIP_FILE_PATH = "./data/unlabeled2017.zip"
EXTRACT_DIR = "./data/unlabeled2017"
FINAL_DIR = "./data/fast_patches"

def download_file(url, dest_path):
    """Downloads a file with a streaming progress bar."""
    print(f"[*] Starting download from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the total file size from headers
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)
            
    print("[*] Download complete!")

def extract_and_organize(zip_path, extract_to, final_dest):
    """Extracts the massive zip file and moves images to your training folder."""
    print(f"[*] Extracting {zip_path}... (This will take a few minutes)")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("./data/")
        
    print(f"[*] Extraction complete. Moving files to {final_dest}...")
    
    # MS-COCO extracts to a folder called 'unlabeled2017' by default.
    # We move the contents to your 'fast_patches' folder to match Student C's script.
    os.makedirs(final_dest, exist_ok=True)
    
    extracted_folder = "./data/unlabeled2017"
    if os.path.exists(extracted_folder):
        files = os.listdir(extracted_folder)
        for file_name in tqdm(files, desc="Moving Images"):
            shutil.move(os.path.join(extracted_folder, file_name), os.path.join(final_dest, file_name))
            
        # Clean up the empty directory and the massive zip file to save disk space
        os.rmdir(extracted_folder)
        os.remove(zip_path)
        print(f"[*] Cleanup finished. All {len(files)} images are ready in {final_dest}!")
    else:
        print("[!] Error: Could not find the extracted folder.")

if __name__ == "__main__":
    print("--- INITIATING HIGH-RES DATASET ACQUISITION ---")
    os.makedirs("./data", exist_ok=True)
    
    # Step 1: Download
    if not os.path.exists(FINAL_DIR) or len(os.listdir(FINAL_DIR)) < 1000:
        if not os.path.exists(ZIP_FILE_PATH):
            download_file(COCO_URL, ZIP_FILE_PATH)
        else:
            print("[*] Zip file already exists, skipping download.")
            
        # Step 2: Extract & Clean up
        extract_and_organize(ZIP_FILE_PATH, EXTRACT_DIR, FINAL_DIR)
    else:
        print(f"[*] Dataset already populated! Found {len(os.listdir(FINAL_DIR))} images in {FINAL_DIR}.")
        
    print("\n--- ACQUISITION COMPLETE. YOU MAY NOW RUN train_highres.py ---")