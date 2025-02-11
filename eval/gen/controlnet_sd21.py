import torch
import json
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
import os
import os.path as osp
from tqdm import tqdm

from utils import process_cond_image, set_seed
cond_path = "../data/eval/MSCOCO/conditioning"
caption_path = "../data/eval/MSCOCO/processed_coco2.json"
save_path = "../data/eval/generate/controlnet_sd21"
os.makedirs(save_path, exist_ok=True)
set_seed(0)
if __name__ == "__main__":
    bs = 32
    f = open(caption_path, 'r')
    metadata = json.load(f)
    start = 0
    end = -1
    cond_images = [osp.join(cond_path, image_prompt["image_name"]) for image_prompt in metadata][start:end]    
    prompts = [image_prompt['text'] for image_prompt in metadata][start:end]
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers").to(device="cuda")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=True).to("cuda")
    pipe.enable_attention_slicing()

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda").manual_seed(0)
    print("start generating ...")
    for i in tqdm(range(len(prompts) // bs +1)):
        prompt = prompts[i*bs:i*bs+bs]
        if len(prompt) == 0 or i < 315:
            continue
        output = pipe(prompt=prompt, image=process_cond_image(cond_images[i*bs:i*bs+bs]), num_inference_steps=50).images
        for image_idx, image in enumerate(output):
            image.save(osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
        del output
        torch.cuda.empty_cache()
