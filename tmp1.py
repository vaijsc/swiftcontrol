

import os
from PIL import Image

# Define the folder containing the images
folder_path = "../generate/controlnet_sd15"
  # Replace with your folder path

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # Try to open the image using PIL
        with Image.open(file_path) as img:
            img.verify()  # Verify that this is a valid image
    except Exception as e:
        # If there's an error, print the filename and the error
        print(f"Cannot open {file_name}: {e}")
