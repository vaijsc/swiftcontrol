import torch
import os
import os.path as osp
from typing import List
from torchvision.utils import save_image
from utils import set_seed
from tqdm import tqdm
import gc
import json
from distill_controlnet.infer import DistillControlTester
from utils import set_seed
cond_path = "../data/eval/MSCOCO/conditioning"
caption_path = "../data/eval/MSCOCO/processed_coco2.json"
save_path = "../data/eval/generate/distillcontrolnet_sd21"
os.makedirs(save_path, exist_ok=True)
set_seed(0)
if __name__ == "__main__":
    bs = 64
    f = open(caption_path, 'r')
    metadata = json.load(f)
    cond_images = [osp.join(cond_path, image_prompt["image_name"]) for image_prompt in metadata]
    cond_images = cond_images[:100]
    prompts = [image_prompt['text'] for image_prompt in metadata][:100]
    controlnet_path = "output/distill_controlnet/checkpoint-40000/controlnet_ema"
    with torch.no_grad():
        infer = DistillControlTester(controlnet_path)
        for i in tqdm(range(len(prompts) // bs +1)):
            prompt = prompts[i*bs:i*bs+bs]
            if len(prompt) == 0:
                continue
            cond_image = cond_images[i*bs:i*bs+bs]
            images = infer(prompt=prompt, cond_image=cond_image, guidance_scale=1.0)
            for image_idx, image in enumerate(images):
                save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_image[image_idx])))
            del images
            torch.cuda.empty_cache()
            gc.collect()