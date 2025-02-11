import torch
import os
import os.path as osp
from typing import List

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image
from tqdm import tqdm
from utils import process_cond_image
from torchvision.utils import save_image
class DistillControlTester():
    def __init__(self, controlnet_path, path_ckpt_sbv2="/lustre/scratch/client/vinai/users/ngannh9/enhance/ckpt/sb_v2_ckpt/unet", base_model = "stabilityai/stable-diffusion-2-1-base", device="cuda", generator=None):
        self.device = device
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to(self.device)
        self.unet.eval()
        self.weight_dtype = self.unet.dtype
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        self.controlnet = self.controlnet.to(device=self.device, dtype=self.weight_dtype)
        self.last_timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

        self.tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            base_model, subfolder="text_encoder"
        ).to(self.device, dtype=torch.float32)
        # prepare stuff
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.alpha_t = (alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)
        if generator is None:
            self.generator = torch.Generator(device=self.device).manual_seed(0)
        else:
            self.generator = generator
        del alphas_cumprod
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
    def prepare_data(self, prompt, cond_image, num_images_per_prompt=1, resolution=512):
        cond_images = process_cond_image(cond_image, resolution=resolution).to(dtype=self.weight_dtype, device=self.device)
        encoder_hidden_state = self.encode_prompt(prompt=prompt, num_images_per_prompt=num_images_per_prompt)
        return cond_images, encoder_hidden_state
    @torch.no_grad
    def encode_prompt(self, prompt, num_images_per_prompt=1):
        input_id = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")
        encoder_hidden_state = self.text_encoder(input_id)[0].repeat_interleave(num_images_per_prompt, dim=0)
        return encoder_hidden_state
    @torch.no_grad
    def generate_latent(self, prompt, controlnet_image, guidance_scale=1.0, noise=None):
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            len_prompt = len(prompt)
        elif isinstance(prompt, torch.Tensor):
            len_prompt = prompt.shape[0]
        else:
            raise ValueError(f'not support prompt type {type(prompt)}')
        
        # prepare noise
        if noise is None:
            bs=1
            noise = torch.randn(len_prompt, 4, 64, 64, device="cuda", generator=self.generator)
        else:
            bs = noise.shape[0] // len_prompt
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            encoder_hidden_state = self.encode_prompt(prompt, bs)
        elif isinstance(prompt, torch.Tensor):
            encoder_hidden_state = prompt
        else:
            raise ValueError(f'not support prompt type {type(prompt)}')
        do_classifier_free_guidance = guidance_scale > 1.0
        encoder_hidden_state, negative_prompt_embeds = self.prepare_prompt(encoder_hidden_state, do_classifier_free_guidance)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noise,
            self.last_timestep,
            encoder_hidden_states=encoder_hidden_state,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        model_pred = self.unet(noise, self.last_timestep, encoder_hidden_state,
                    down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
                    ).sample
        
        #classifier free
        
        if do_classifier_free_guidance:
            noise_pred_uncond = self.unet(noise, self.last_timestep, negative_prompt_embeds,
                    down_block_additional_residuals=[torch.zeros_like(sample).to(dtype=self.weight_dtype) for sample in down_block_res_samples],
                    mid_block_additional_residual=torch.zeros_like(mid_block_res_sample).to(dtype=self.weight_dtype),
                    ).sample
            model_pred = noise_pred_uncond + guidance_scale * (model_pred - noise_pred_uncond)
        pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
        if self.noise_scheduler.config.thresholding:
                pred_original_sample = self.noise_scheduler._threshold_sample(
                pred_original_sample
            )
        elif self.noise_scheduler.config.clip_sample:
            clip_sample_range = self.noise_scheduler.config.clip_sample_range
            pred_original_sample = pred_original_sample.clamp(
                -clip_sample_range, clip_sample_range
            )
        return pred_original_sample
    @torch.no_grad
    def decode_image(self, latent):
        latent = latent / self.vae.config.scaling_factor
        image = (self.vae.decode(latent).sample + 1) / 2
        return image
    @torch.no_grad
    def generate_image(self, prompt, cond_images, guidance_scale=1.0, noise=None, return_latent=False):
        latent = self.generate_latent(prompt, cond_images, noise=noise, guidance_scale=guidance_scale)
        image = self.decode_image(latent)
        if return_latent:
            return latent, image
        return image
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
    @torch.no_grad
    def __call__(self, prompt, cond_image, guidance_scale=4.5, num_images_per_prompt=1, resolution=512):
        cond_images, prompt_embeds = self.prepare_data(prompt, cond_image, num_images_per_prompt, resolution)
        gen_image = self.generate_image(prompt=prompt_embeds, cond_images=cond_images, guidance_scale=guidance_scale)
        return gen_image

if __name__ == "__main__":
    test_cond_folder="data/val/cannylaion/cond_images"
    test_prompt_file = "data/val/cannylaion/prompt.txt"
    f = open(test_prompt_file)
    prompts = []
    for line in f:
        prompt = line.strip()  
        prompts.append(prompt)
    image_paths = sorted(os.listdir(test_cond_folder))
    cond_images = [osp.join(test_cond_folder, image_path) for image_path in image_paths]
    output_folder = "output/distillcontrol/guidance_exps"
    os.makedirs(output_folder)
    controlnet_path = "output/distill_controlnet/checkpoint-40000/controlnet_ema"
    tester = DistillControlTester(controlnet_path)
    guidance_scale = 0.0
    while guidance_scale < 7.0:
        gen_image = tester(prompts, cond_images, guidance_scale=guidance_scale)
        full_path_save = osp.join(output_folder, f"test_distillcontrol_{guidance_scale}.jpg")
        save_image(gen_image, full_path_save)
        guidance_scale = guidance_scale + 0.5
    print(f"saved image to {output_folder}")