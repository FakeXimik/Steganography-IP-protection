import os
from PIL import Image

def chunk_high_res_images():
    # Pointing exactly to the new DIV2K folder you just downloaded
    source_dir = "./data/high_res_images/DIV2K_train_HR"
    target_dir = "./data/fast_patches"
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    patch_size = 512
    
    print(f"Found {len(image_files)} massive images. Generating 512x512 patches...")
    
    patch_count = 0
    for img_name in image_files:
        img_path = os.path.join(source_dir, img_name)
        try:
            # Load the massive image once
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            
            # Slice it into a grid of 512x512 patches
            for x in range(0, width - patch_size, patch_size):
                for y in range(0, height - patch_size, patch_size):
                    box = (x, y, x + patch_size, y + patch_size)
                    patch = img.crop(box)
                    
                    # Save the small, fast patch
                    patch_filename = f"patch_{patch_count}_{img_name}"
                    patch.save(os.path.join(target_dir, patch_filename))
                    patch_count += 1
                    
            print(f"Processed {img_name}...")
        except Exception as e:
            print(f"Skipped {img_name}: {e}")
            
    print(f"\nSuccess! Created {patch_count} fast-loading patches in {target_dir}")

if __name__ == "__main__":
    chunk_high_res_images()