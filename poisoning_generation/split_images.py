"""
Script to split 2x2 grid images in subfolders into four separate images.
"""

import os
from PIL import Image

def split_gridded_image(folder_path):
    """
    Splits all gridded images found in the specified folder. 
    Each image is assumed to be a 2x2 grid and will be split into 4 separate images.
    
    :param folder_path: Path to the folder containing subfolders with images.
    """
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(('.png', '.jpg')):
                    image_path = os.path.join(subfolder_path, filename)
                    try:
                        with Image.open(image_path) as img:
                            w, h = img.size
                            coords = [
                                (0, 0, w//2, h//2),
                                (w//2, 0, w, h//2),
                                (0, h//2, w//2, h),
                                (w//2, h//2, w, h)
                            ]
                            for i, coord in enumerate(coords):
                                part_path = os.path.join(subfolder_path, f"{os.path.splitext(filename)[0]}_part{i+1}.png")
                                img.crop(coord).save(part_path)
                    except Exception as e:
                        print(f"Failed to split image {image_path}: {e}")

# Example usage:
if __name__ == "__main__":
    base_folder_path = "einstein_hat_writing"  # Set your folder here
    split_gridded_image(base_folder_path)
