from dataloader.utils import DATASET_DIR
from torchvision import transforms
import json
import torch
import gc
import os
import os.path as osp
from typing import List
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from transformers import CLIPTextModel, AutoTokenizer
import contextlib
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from dataloader.utils import conditioning_transforms, encode_prompt
from diffusers.utils import make_image_grid
class Validator():
    def __init__(self, tokenizer, text_encoder, save_path, is_final_validation=False, seed=0):
        # if not is_final_validation:
        #     controlnet = accelerator.unwrap_model(controlnet)
        # else:
        #     controlnet = ControlNetModel.from_pretrained(output_dir, torch_dtype=weight_dtype)
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.seed = seed
        self.inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
        self.cond_images, self.prompt_embeds = self.get_validation_data()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
    def get_validation_data(self, dataname=['canny118k', 'cannylaion_val'], resolution=512):
        cond_images = []
        promtps = []
        for name in dataname:
            if name == 'canny118k':
                text_path = DATASET_DIR['canny118k']['embed_txt_path']
                cond_img_dir = DATASET_DIR['canny118k']['cond_img_dir']

                info = open(text_path, 'r')
                info = json.load(info)[100004:100008]
                # load testing data
                for sample in info:
                    img_name = sample["image_path"]
                    promtps.append(sample["text"])
                    cond_images.append(conditioning_transforms(resolution=resolution)(Image.open(osp.join(cond_img_dir, "conditioning_" + img_name)).convert("RGB")))
            if name == 'cannylaion_val':
                text_path = DATASET_DIR['cannylaion_val']['text_path']
                cond_img_dir = DATASET_DIR['cannylaion_val']['cond_img_dir']
                f = open(text_path, 'r')
                prompt_data = f.read().splitlines()
                promtps.extend(prompt_data)
                cond_files = sorted(os.listdir(cond_img_dir))
                for file in cond_files:
                    cond_images.append(conditioning_transforms(resolution=resolution)(Image.open(osp.join(cond_img_dir, file)).convert("RGB")))
        return torch.stack(cond_images), promtps
    
    def __call__(self, controlnet, unet, pretrained_model_name_or_path, step, vae, text_encoder, tokenizer, accelerator, weight_dtype=None, guidance_scale=7.5, num_inference_steps=4):
        device = "cuda" if accelerator is None else accelerator.device
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        if not weight_dtype:
            weight_dtype = controlnet.dtype
        print("start infer")
        with torch.no_grad():
            if accelerator is not None:
                controlnet = accelerator.unwrap_model(controlnet)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                pretrained_model_name_or_path,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=controlnet, 
                safety_checker=None,
                torch_dtype=weight_dtype,
            ).to(device)
            pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
            pipeline.set_progress_bar_config(disable=True)

            with self.inference_ctx:
                image = pipeline(
                    prompt=self.prompt_embeds, image=self.cond_images.to(device=device), guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=self.generator
                ).images
            save_img = make_image_grid(image, rows=1, cols=len(image))
            save_img.save(osp.join(self.save_path, f'step_{str(step)}_cfg_{guidance_scale}.jpg'))
        print("end infer")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

# pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"

# text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").to("cuda")
# vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to("cuda")
# unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to("cuda")
# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", use_safetensors=True).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(
#         pretrained_model_name_or_path,
#         subfolder="tokenizer",
#         use_fast=False)
# validator = Validator("cuda", tokenizer, text_encoder)
# validator(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, controlnet=controlnet, pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", step=1)