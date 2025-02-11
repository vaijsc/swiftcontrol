import os
from datasets import load_dataset
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def save_image(image, file_path):
    """
    Saves a PIL Image to the specified file path.
    """
    image.save(file_path)

def save_images_from_hf_parallel(dataset_name, output_image_folder, output_conditioning_folder, image_column='image', conditioning_column='conditioning_image', image_format='png', conditioning_format='png', max_workers=8):
    """
    Saves images and conditioning images from a Hugging Face dataset to separate folders using parallel processing.

    Args:
        dataset_name (str): Name or path of the dataset on Hugging Face.
        output_image_folder (str): Path to the folder where images will be saved.
        output_conditioning_folder (str): Path to the folder where conditioning images will be saved.
        image_column (str): The column name in the Hugging Face dataset that contains the image data.
        conditioning_column (str): The column name that contains conditioning images.
        image_format (str): Format to save the images ('jpg', 'png', etc.).
        conditioning_format (str): Format to save the conditioning images ('png', etc.).
        max_workers (int): Number of threads to use for parallel processing.

    Returns:
        None
    """
    # Load the Hugging Face dataset
    ds = load_dataset(dataset_name, split='train')

    # Ensure output folders exist
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_conditioning_folder):
        os.makedirs(output_conditioning_folder)

    # Create a ThreadPoolExecutor to handle parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Iterate over each item in the dataset
        for idx, row in tqdm(enumerate(ds)):
            # Prepare file paths
            image_file_name = f'image_{idx}.{image_format}'
            image_file_path = os.path.join(output_image_folder, image_file_name)
            
            conditioning_file_name = f'conditioning_image_{idx}.{conditioning_format}'
            conditioning_file_path = os.path.join(output_conditioning_folder, conditioning_file_name)

            # Submit tasks to the thread pool for both image and conditioning_image
            futures.append(executor.submit(save_image, row[image_column], image_file_path))
            futures.append(executor.submit(save_image, row[conditioning_column], conditioning_file_path))

        # Ensure all futures are completed
        for future in as_completed(futures):
            future.result()  # Will raise exceptions if any occurred

    print(f"Images have been successfully saved to {output_image_folder} and conditioning images to {output_conditioning_folder}")

# Example usage
# dataset_name = '~/.cache/huggingface/datasets/AmritaBha___mscoco-controlnet-canny/snapshots/bb1a6942883c1d2afc9d8987c1ee06928bfcd569/data'
dataset_name = 'AmritaBha/mscoco-controlnet-canny'
output_image_folder = 'data/images'
output_conditioning_folder = 'data/conditioning_images'

save_images_from_hf_parallel(dataset_name, output_image_folder, output_conditioning_folder, image_column='image', conditioning_column='conditioning_image', image_format='png', conditioning_format='png', max_workers=16)
