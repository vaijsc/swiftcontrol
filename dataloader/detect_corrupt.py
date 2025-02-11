import cv2
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

image_dir = "/lustre/scratch/client/vinai/users/ngannh9/face/data/LAION/train"
image_names = [f for f in tqdm(os.listdir(image_dir)) if f.endswith(".jpg")]
for file in tqdm(image_names):
    with open(os.path.join(image_dir, file), 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        print('Not complete image', file)
    else:
        imrgb = cv2.imread(os.path.join(image_dir, file), 1)