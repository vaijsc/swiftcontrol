from controlnet.validator import Validator
import os
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

validation_savepath = "output/test/control_sdturbo_vary_cfg_ema"
os.makedirs(validation_savepath, exist_ok=True)
pretrained_model_name_or_path = "stabilityai/sd-turbo"
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path, subfolder="vae"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path, subfolder="tokenizer",
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet"
)
ckpt = 130000
cfgs = [7.5]
controlnet_path = f"../output/controlnet_sdturbo/checkpoint-{ckpt}/controlnet_ema"
                
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True, subfolder="controlnet").to(device="cuda")
validator = Validator(tokenizer=tokenizer, text_encoder=text_encoder, save_path=validation_savepath)
for cfg in cfgs:
    validator(
        controlnet=controlnet, 
        unet=unet, 
        pretrained_model_name_or_path=pretrained_model_name_or_path, 
        step=0,
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        accelerator=None,
        weight_dtype=controlnet.dtype,
        guidance_scale=cfg,
        num_inference_steps=4
        )