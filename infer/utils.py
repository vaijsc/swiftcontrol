from dataloader.utils import DATASET_DIR
import json
import torch
import os
import os.path as osp
from PIL import Image
from dataloader.utils import conditioning_transforms

def get_test_data(dataname=['canny118k', 'cannylaion_val'], resolution=512):
    cond_images = []
    prompts = []
    for name in dataname: #=['canny118k', 'cannylaion_val']
        if name == 'canny118k': 
            text_path = DATASET_DIR['canny118k']['embed_txt_path']
            cond_img_dir = DATASET_DIR['canny118k']['cond_img_dir']

            info = open(text_path, 'r')
            info = json.load(info)[100004:100008]
            # load testing data
            for sample in info:
                img_name = sample["image_path"]
                prompts.append(sample["text"])
                cond_images.append(conditioning_transforms(resolution)(Image.open(osp.join(cond_img_dir, "conditioning_" + img_name)).convert("RGB")))
        if name == 'cannylaion_val':
            text_path = DATASET_DIR['cannylaion_val']['text_path']
            cond_img_dir = DATASET_DIR['cannylaion_val']['cond_img_dir']
            f = open(text_path, 'r')
            prompt_data = f.read().splitlines()
            prompts.extend(prompt_data)
            cond_files = sorted(os.listdir(cond_img_dir))
            print(prompt_data)
            print(cond_files)
            for file in cond_files:
                cond_images.append(conditioning_transforms(resolution)(Image.open(osp.join(cond_img_dir, file)).convert("RGB")))
    return torch.stack(cond_images).to(device="cuda"), prompts