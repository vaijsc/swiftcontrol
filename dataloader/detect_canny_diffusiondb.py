import cv2 as cv
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

start_idx = 0
end_idx = -1
image_dir = "../data/diffusiondb/train"
canny_conditioning_save_path = "../data/canny_laion/conditioning"
os.makedirs(canny_conditioning_save_path, exist_ok=True)
image_names = [f for f in tqdm(os.listdir(image_dir)) if f.endswith(".jpg")]

def process_image(i):
    image_name = image_names[i]
    image_path = osp.join(image_dir, image_name)
    if osp.exists(image_path):
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            edges = cv.Canny(img, 100, 200)
            cv.imwrite(osp.join(canny_conditioning_save_path, image_name), edges)
    return None

with ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_image, range(len(image_names))), total=len(image_names), desc="Processing images"))
