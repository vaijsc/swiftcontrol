import torch
import json
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from utils import process_cond_image, set_seed
import os
import os.path as osp
from tqdm import tqdm
from infer.utils import get_test_data
import argparse
from torchvision.utils import save_image
from distill_controlnet.infer import DistillControlTester
import gc
from repcontrolnet.infer import RepControlNetInfer

cond_path = "../data/eval/MSCOCO/conditioning"
caption_path = "../data/eval/MSCOCO/processed_coco2.json"
# caption_path = "../data/eval/MSCOCO/coco.json"
set_seed(0)

if __name__ == "__main__":
    with torch.no_grad():
        # get model name to test
        parser = argparse.ArgumentParser()
        parser.add_argument("model_name", type=str, help="")
        parser.add_argument("--start", type=int, default=0, help="")
        parser.add_argument("--resume_from", type=int, default=0, help="")
        parser.add_argument("--end", type=int, default=-1, help="")
        args = parser.parse_args()
        model_name = args.model_name
        save_path = f"../generate/{model_name}"

        os.makedirs(save_path, exist_ok=True)
        # prepare prompt
        bs = 32
        f = open(caption_path, 'r')
        metadata = json.load(f)
        start = args.start
        end = args.end
        resume_from = args.resume_from
        cond_images = [osp.join(cond_path, image_prompt["image_name"]) for image_prompt in metadata][start:end]
        prompts = [image_prompt['text'] for image_prompt in metadata][start:end]
    
        if 'controlnet_sd' in model_name and all(x not in model_name for x in ['rep', 'distill']):
            num_inference_steps=50
            guidance_scale = 7.5
            if model_name == 'controlnet_sd15':
                controlnet_path = "lllyasviel/sd-controlnet-canny"
                sd_path = "runwayml/stable-diffusion-v1-5"
                controlnet = ControlNetModel.from_pretrained(controlnet_path).to(device="cuda")
            elif model_name == 'controlnet_sd21':
                controlnet_path = "thibaud/controlnet-sd21-canny-diffusers"
                sd_path = "stabilityai/stable-diffusion-2-1-base"
                controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False).to(device="cuda")
            elif model_name == 'controlnet_sdturbo':
                ckpt = 130000
                controlnet_path = f"../output/controlnet_sdturbo/checkpoint-{ckpt}/controlnet"
                # controlnet_path = "thibaud/controlnet-sd21-canny-diffusers"

                sd_path = "stabilityai/sd-turbo"
                controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True, subfolder="controlnet").to(device="cuda")
                # controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False).to(device="cuda")

                num_inference_steps=4
                guidance_scale = 8.0
                save_path = save_path+f"_{num_inference_steps}"
                os.makedirs(save_path, exist_ok=True)
            # elif model_name == 'controlnet_sdturbo_from_sd21':
            #     controlnet_path = "thibaud/controlnet-sd21-canny-diffusers"

            #     sd_path = "stabilityai/sd-turbo"
            #     controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False).to(device="cuda")

            #     num_inference_steps=20
            #     guidance_scale = 0.0
            #     save_path = save_path+f"_fromsd21_{num_inference_steps}"
            #     os.makedirs(save_path, exist_ok=True)
            else:
                raise ValueError(f'not known {model_name}')
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                sd_path, controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=True).to("cuda")
            pipe.enable_attention_slicing()
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.safety_checker = None
            pipe.set_progress_bar_config(disable=True)

            generator = torch.Generator(device="cuda").manual_seed(0)
            print("start generating ...")
            for i in tqdm(range(len(prompts) // bs +1)):
                prompt = prompts[i*bs:i*bs+bs]
                if len(prompt) == 0 or i < resume_from:
                    continue
                output = pipe(prompt=prompt, image=process_cond_image(cond_images[i*bs:i*bs+bs]), 
                              num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
                for image_idx, image in enumerate(output):
                    image.save(osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
                del output
                torch.cuda.empty_cache()
        elif model_name == "distill_controlnet_sd21":
            step = 25000
            controlnet_path = f"../output/distill_controlnet/checkpoint-{step}/controlnet_ema"
            infer = DistillControlTester(controlnet_path)
            for i in tqdm(range(len(prompts) // bs +1)):
                if i < resume_from or len(prompt) == 0:
                    continue
                prompt = prompts[i*bs:i*bs+bs]
                cond_image = cond_images[i*bs:i*bs+bs]
                images = infer(prompt=prompt, cond_image=cond_image, guidance_scale=1.0)
                # save_image(images, osp.join(save_path, f"test_{i}.jpg"), nrow=1)
                
                for image_idx, image in enumerate(images):
                    save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_image[image_idx])))
                del images
                torch.cuda.empty_cache()
                gc.collect()
        elif "repcontrolnet_sd21" in model_name:
            step = 180000
            repcontrolnet_model_path = f"../output/rep_canny_3m/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"
    
            if "branch" in model_name:
                infer = RepControlNetInfer(repcontrolnet_model_path, stage="branch")
            elif "merge" in model_name:
                alpha = 0.0
                beta = 0.0
                save_path = osp.join(save_path, f'alpha_{round(alpha,2)}_beta_{round(beta,2)}')
                os.makedirs(save_path, exist_ok=True)
                infer = RepControlNetInfer(repcontrolnet_model_path, stage="merge", alpha=alpha, beta=beta)
                breakpoint()
            else:
                raise ValueError(f'specify branch or merge for {model_name}')
                
            for i in tqdm(range(len(prompts) // bs +1)):
                prompt = prompts[i*bs:i*bs+bs]
                if len(prompt) == 0:
                    continue
                images = infer(prompt=prompt, cond_image=cond_images[i*bs:i*bs+bs])
                # save_image(images, osp.join(save_path, f"test_{i}.jpg"), nrow=1)
                for image_idx, image in enumerate(images):
                    save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
                del images
                torch.cuda.empty_cache()
                gc.collect()
        elif "repcontrolnet_sdturbo" in model_name:
            def gen_image(infer: RepControlNetInfer, save_path):
                for i in tqdm(range(len(prompts) // bs +1)):
                    prompt = prompts[i*bs:i*bs+bs]
                    if len(prompt) == 0:
                        continue
                    images = infer(prompt=prompt, cond_image=cond_images[i*bs:i*bs+bs], guidance_scale=7.5, num_inference_steps=20)
                    # save_image(images, osp.join(save_path, f"test_{i}.jpg"), nrow=1)
                    for image_idx, image in enumerate(images):
                        save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
                    del images
                    torch.cuda.empty_cache()
                    gc.collect()
            base_model = "stabilityai/sd-turbo"

            if "lite" in model_name:
                lite = True
                step = 120000
                repcontrolnet_model_path = f"../output/rep_canny_3m_sdturbo_lite/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"
            else:
                lite = False
                step = 120000
                repcontrolnet_model_path = f"../output/rep_canny_3m_sdturbo/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"

            if "branch" in model_name:
                infer = RepControlNetInfer(repcontrolnet_model_path, stage="branch", base_model=base_model, lite=lite)
                gen_image(infer=infer, save_path=save_path)
            elif "merge" in model_name:
                alpha = 1.0
                beta = 1.0
                save_path = osp.join(save_path, f'alpha_{round(alpha,2)}_beta_{round(beta,2)}')
                os.makedirs(save_path, exist_ok=True)
                infer = RepControlNetInfer(repcontrolnet_model_path, stage="merge", base_model=base_model, alpha=alpha, beta=beta, lite=lite)
                gen_image(infer=infer, save_path=save_path)
                
            else:
                raise ValueError(f'specify branch or merge for {model_name}')

        else:
            raise ValueError(f'not known model {model_name}')