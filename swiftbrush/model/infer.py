import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, CLIPTextModel
from tqdm import tqdm
import os.path as osp

class SBV2Gen():
    def __init__(self, path_ckpt_sbv2="sb_v2_ckpt/0.5", model_name = "stabilityai/stable-diffusion-2-1-base"):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to("cuda")
        self.unet.eval()

        self.last_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        
        self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder"
        ).to("cuda", dtype=torch.float32)
        
        # prepare stuff
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
    @torch.no_grad
    def encode_prompt(self, prompt, batch_size=1):
        input_id = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")
        encoder_hidden_state = self.text_encoder(input_id)[0].repeat_interleave(batch_size, dim=0)
        return encoder_hidden_state
    @torch.no_grad
    def generate_latent(self, prompt, noise=None):
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            len_prompt = len(prompt)
        elif isinstance(prompt, torch.Tensor):
            len_prompt = prompt.shape[0]
        else:
            raise ValueError(f'not support prompt type {type(prompt)}')
        
        if noise is None:
            bs=1
            noise = torch.randn(len_prompt, 4, 64, 64, device="cuda")
        else:
            bs = noise.shape[0] // len_prompt
        if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
            encode_hidden_state = self.encode_prompt(prompt, bs)
        elif isinstance(prompt, torch.Tensor):
            encode_hidden_state = prompt
        else:
            raise ValueError(f'not support prompt type {type(prompt)}')
        model_pred = self.unet(noise, self.last_timestep, encode_hidden_state).sample
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
    def generate_image(self, prompt, noise=None, return_latent=False):
        latent = self.generate_latent(prompt, noise)
        image = self.decode_image(latent)
        if return_latent:
            return latent, image
        return image
    