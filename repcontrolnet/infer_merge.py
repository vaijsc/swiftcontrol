import torch
import os
import os.path as osp
from typing import List
from diffusers.utils.torch_utils import randn_tensor
from torchvision.utils import save_image
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from utils import set_seed
import numpy as np
from tqdm import tqdm
import gc
from utils import process_cond_image
"""Preparing infer data: 
    - prompt.txt : img_name + prompt
    - folder: containing img_name

"""

#multistep RepControlNet inference
step = 180000
repcontrolnet_model_path = f"../output/rep_canny_3m/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"

@torch.no_grad
def reparam_unet(unet, repcontrolnet_model_path, alpha, beta):
    repcontrolnet_named_params = torch.load(repcontrolnet_model_path)
    state_dict = unet.state_dict()
    for name, param in repcontrolnet_named_params.items():
        if "rep" in name:
            base_name = name.replace('_rep', '')
            state_dict[base_name] = alpha * unet[base_name] + beta * param
        elif "controlnet_cond_embedding" in name:
            state_dict[base_name] = param
        else:
            raise ValueError(f"invalid param: {param}")
    state_dict.update(state_dict)
    unet.load_state_dict(state_dict)
    breakpoint()
        
def add_conditioning_adapter(unet):
    def custom_forward(self):
        def forward(latent_model_input, t, encoder_hidden_states, controlnet_cond, return_dict=False):
            pass
        return forward

class RepControlNetInfer():
    def __init__(self, repcontrolnet_model_path, stage, base_model, 
                alpha=None, beta=None, seed=0, lite=True):
        #stage: train: branch weight, test: merge weight
        if lite:
            from repcontrolnet.model_lite.unet_2d_condition import RepUNet2DConditionModel
        else:
            from repcontrolnet.model.unet_2d_condition import RepUNet2DConditionModel
            
        set_seed(0)
        self.device = "cuda"
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
    
        self.scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(self.device)
        self.repcontrolnet = RepUNet2DConditionModel.from_pretrained_unet(base_model, 
                    reparams_path=repcontrolnet_model_path, subfolder="unet", alpha=alpha, beta=beta, stage=stage).to(self.device)
        self.repcontrolnet.eval()
        self.weight_dtype = self.repcontrolnet.dtype
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder"
        ).to(self.device, dtype=torch.float32)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def prepare_data(self, prompt, cond_image, num_images_per_prompt=1, resolution=512):
        cond_images = process_cond_image(cond_image, resolution=resolution).to(dtype=self.weight_dtype, device=self.device)
        encoder_hidden_state = self.encode_prompt(prompt=prompt, num_images_per_prompt=num_images_per_prompt)
        return cond_images, encoder_hidden_state
    def _maybe_convert_prompt(self, prompt):  # noqa: F821
        tokens = self.tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in self.tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt
    def maybe_convert_prompt(self, prompt):  # noqa: F821
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts
    def encode_prompt(self, prompt, num_images_per_prompt=1):
        input_id = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        encoder_hidden_state = self.text_encoder(input_id)[0].repeat_interleave(num_images_per_prompt, dim=0)
        return encoder_hidden_state
    def prepare_prompt(self, prompt_embeds, do_classifier_free_guidance, num_images_per_prompt=1):
        prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype, device=self.device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        uncond_tokens = [""] * bs_embed
        uncond_tokens = self.maybe_convert_prompt(uncond_tokens)
        max_length = prompt_embeds.shape[1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(self.device)
        else:
            attention_mask = None
        negative_prompt_embeds = self.text_encoder(
            uncond_input.input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.weight_dtype, device=self.device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds
    def inference(self, prompt, cond_image, guidance_scale=7.5, num_images_per_prompt=1, resolution=512, num_inference_steps=50):
        do_classifier_free_guidance = guidance_scale > 1.0
        cond_embeds, prompt_embeds = self.prepare_data(prompt, cond_image, num_images_per_prompt, resolution=resolution)
        prompt_embeds, negative_prompt_embeds = self.prepare_prompt(prompt_embeds, do_classifier_free_guidance, num_images_per_prompt)
        
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latents = randn_tensor((prompt_embeds.shape[0], 4, 64, 64), generator=self.generator, device=self.repcontrolnet.device, dtype=self.weight_dtype)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred_uncond = self.repcontrolnet(latent_model_input,t,encoder_hidden_states=negative_prompt_embeds,controlnet_cond = cond_embeds,return_dict=False)[0]
            noise_pred_text = self.repcontrolnet(latent_model_input,t,encoder_hidden_states=prompt_embeds,controlnet_cond = cond_embeds,return_dict=False)[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=self.generator, return_dict=False)[0]
        latents = (latents / self.vae.config.scaling_factor).to(dtype=self.weight_dtype)
        images = (self.vae.decode(latents, generator=self.generator).sample + 1) / 2
        return images

    def __call__(self, prompt, cond_image, num_images_per_prompt=1, resolution=512, guidance_scale=7.5, num_inference_steps=50):
            images = self.inference(prompt, cond_image, num_images_per_prompt=num_images_per_prompt, 
                                    resolution=resolution, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
            return images
if __name__ == "__main__":
    
    #prepare
    test_cond_folder="data/val/cannylaion/cond_images"
    test_prompt_file = "data/val/cannylaion/imgname_prompt.txt"
    f = open(test_prompt_file)
    prompts = []
    for line in f:
        prompt = line.strip()  
        prompts.append(prompt)
    image_paths = sorted(os.listdir(test_cond_folder))
    cond_images = [osp.join(test_cond_folder, image_path) for image_path in image_paths]
    output_folder = "output/repcontrolnet/scaling_factor_exp/"
    
    # infer
    betas = np.arange(0.0, 2.0, 0.4)
    alphas = np.arange(0.0, 2.0, 0.4)
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for alpha in (alphas):
            for beta in tqdm(betas):
                infer = RepControlNetInfer(repcontrolnet_model_path, stage="merge", alpha=(alpha), beta=(beta))
                images = infer(prompts, cond_images)
                save_image(images, osp.join(output_folder, f"test_{(alpha)}_{(beta)}.jpg"))
                del infer
                torch.cuda.empty_cache()
                gc.collect()