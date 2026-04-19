import os
import urllib.request
import zipfile

def download_div2k():
    print("Downloading DIV2K High-Res Dataset (approx 3.5 GB)...")
    print("This might take a few minutes depending on your internet speed.")
    
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = "DIV2K_train_HR.zip"
    
    # Download the massive zip file
    urllib.request.urlretrieve(url, zip_path)
    
    print("\nDownload complete! Extracting images...")
    extract_dir = "./data/high_res_images"
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print(f"\nSuccess! Extracted 800 massive 2K images into {extract_dir}/DIV2K_train_HR")
    
    # Clean up the heavy zip file
    os.remove(zip_path)

if __name__ == "__main__":
    download_div2k()