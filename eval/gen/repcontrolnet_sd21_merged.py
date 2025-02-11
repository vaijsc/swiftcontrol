from torchvision import transforms
import torch
import os
import os.path as osp
from typing import List
from diffusers import UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from torchvision.utils import save_image
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler
from repcontrolnet.model.unet_2d_condition import RepUNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from utils import set_seed
import numpy as np
from tqdm import tqdm
import gc
import json
from utils import process_cond_image
from repcontrolnet.infer import RepControlNetInfer
from utils import process_cond_image, set_seed
cond_path = "../data/eval/MSCOCO/conditioning"
caption_path = "../data/eval/MSCOCO/processed_coco2.json"
save_path = "../data/eval/generate/repcontrolnet_sd21_merged"
os.makedirs(save_path, exist_ok=True)
set_seed(0)
if __name__ == "__main__":
    bs = 16
    f = open(caption_path, 'r')
    metadata = json.load(f)
    cond_images = [osp.join(cond_path, image_prompt["image_path"]) for image_prompt in metadata][:1000]
    prompts = [image_prompt['text'] for image_prompt in metadata][:1000]
    step = 140000
    repcontrolnet_model_path = f"./output/rep_canny_3m/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"
    alpha = 0.9
    beta = 0.27
    with torch.no_grad():
        infer = RepControlNetInfer(repcontrolnet_model_path, alpha=round(alpha), beta=round(beta))
        for i in tqdm(range(len(prompts) // bs +1)):
            prompt = prompts[i*bs:i*bs+bs]
            if len(prompt) == 0:
                continue
            images = infer(prompt=prompt, cond_image=cond_images[i*bs:i*bs+bs])
            for image_idx, image in enumerate(images):
                save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
            del images
            torch.cuda.empty_cache()
            gc.collect()