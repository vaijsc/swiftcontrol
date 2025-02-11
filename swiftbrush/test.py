# import torch
# import os
# import os.path as osp
# import json
# import numpy as np
# from torchvision.transforms.functional import pil_to_tensor
# from PIL import Image
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionControlNetPipeline, ControlNetModel
# from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer
# from torchvision.utils import save_image
# from tqdm import tqdm
# from dataloader.utils import DATASET_DIR
# from torchvision import transforms
# import gc
# from dataloader.utils import conditioning_transforms, encode_prompt
# from torchvision.utils import save_image
# path_save = "./test_sb"
# os.makedirs(path_save, exist_ok=True)

# #bug
# class SBTester():
#     def __init__(self, path_ckpt_sbv2="/lustre/scratch/client/vinai/users/ngannh9/enhance/sb_v2_ckpt/0.5", model_name = "stabilityai/stable-diffusion-2-1-base", generator=None):
#         self.device = "cuda"
#         self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
#         self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
#         self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to(self.device)
#         self.unet.eval()
#         self.last_timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
#         self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

#         self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
#         self.text_encoder = CLIPTextModel.from_pretrained(
#             model_name, subfolder="text_encoder"
#         ).to(self.device, dtype=torch.float32)
#         # prepare stuff
#         alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
#         self.alpha_t = (alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
#         self.sigma_t = ((1 - alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)
#         self.weight_dtype = self.unet.dtype
#         if generator is None:
#             self.generator = torch.Generator(device=self.device).manual_seed(0)
#         else:
#             self.generator = generator
        
#         del alphas_cumprod
#         # self.cond_images, self.prompt_embeds, self.prompts = self.get_validation_data()
#         # del tokenizer, text_encoder
#         # gc.collect()
#         # torch.cuda.empty_cache()
        
#     def get_validation_data(self, data_name="canny", resolution=512):
#         cond_transforms = transforms.Compose(
#             [
#                 transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
#                 transforms.CenterCrop(resolution),
#                 transforms.ToTensor(),
#             ])
#         if data_name == "canny":
#             text_path = DATASET_DIR['canny']['embed_txt_path']
#             cond_img_dir = DATASET_DIR['canny']['cond_img_dir']
#             info = open(text_path, 'r')
#             info = json.load(info)[100000:100003]
#             cond_images = []
#             prompt_embeds = []
#             prompts = []
#             img_names = []
#             for sample in info:
#                 img_name = sample["image_path"]
#                 img_names.append(img_name)
#                 prompts.append(sample["text"])
#                 prompt_embeds.append(torch.from_numpy(np.load(sample["text_embedding"], allow_pickle=True)))
#                 cond_images.append(cond_transforms(Image.open(osp.join(cond_img_dir, "conditioning_" + img_name)).convert("RGB")))
#             print(prompts)
#             print(img_names)
#             cond_images = torch.stack(cond_images).to(dtype=self.weight_dtype, device=self.device)
#             # cond_images = torch.cat([cond_images] * 2) if self.do_classifier_free_guidance else cond_images
#             return cond_images, torch.stack(prompt_embeds).to(dtype=self.weight_dtype, device=self.device), prompts
#         elif data_name == "fill50k":
#             cond_image_dir = "./val50kfill"
#             cond_image_files = sorted(os.listdir(cond_image_dir))
#             cond_images = []
            
#             for image in cond_image_files:
#                 cond_images.append(cond_transforms(Image.open(osp.join(cond_image_dir, image)).convert("RGB")))
#             cond_images = torch.stack(cond_images).to(dtype=self.weight_dtype, device=self.device)
            
#             prompts = ["red circle with blue background","cyan circle with brown floral background","light coral circle with white background","cornflower blue circle with light golden rod yellow background"]
#             input_id = self.tokenizer(prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self.device)
#             encoder_hidden_state = self.text_encoder(input_id)[0].to(dtype=self.weight_dtype, device=self.device)
#             return cond_images, encoder_hidden_state, prompts
            
#         else:
#             raise ValueError("not known val dataname")
#     @torch.no_grad
#     def encode_prompt(self, prompt, batch_size=1):
#         input_id = self.tokenizer(
#             prompt,
#             max_length=self.tokenizer.model_max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         ).input_ids.to("cuda")
#         encoder_hidden_state = self.text_encoder(input_id)[0].repeat_interleave(batch_size, dim=0)
#         return encoder_hidden_state
#     @torch.no_grad
#     def decode_image(self, latent):
#         latent = latent / self.vae.config.scaling_factor
#         image = (self.vae.decode(latent).sample + 1) / 2
#         return image
#     def generate_latent(self, noise):
#         model_pred = self.unet(noise, self.last_timestep, self.prompt_embeds).sample
#         pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
#         if self.noise_scheduler.config.thresholding:
#                 pred_original_sample = self.noise_scheduler._threshold_sample(
#                 pred_original_sample
#             )
#         elif self.noise_scheduler.config.clip_sample:
#             clip_sample_range = self.noise_scheduler.config.clip_sample_range
#             pred_original_sample = pred_original_sample.clamp(
#                 -clip_sample_range, clip_sample_range
#             )
#         return pred_original_sample
#     @torch.no_grad
#     def generate_image(self, prompt, noise=None, return_latent=False):
#         if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
#             len_prompt = len(prompt)
#         elif isinstance(prompt, torch.Tensor):
#             len_prompt = prompt.shape[0]
#         else:
#             raise ValueError(f'not support prompt type {type(prompt)}')
        
#         if noise is None:
#             bs=1
#             noise = torch.randn(len_prompt, 4, 64, 64, device="cuda")
#         else:
#             bs = noise.shape[0] // len_prompt
#         if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
#             encode_hidden_state = self.encode_prompt(prompt, bs)
#         elif isinstance(prompt, torch.Tensor):
#             encode_hidden_state = prompt
#         else:
#             raise ValueError(f'not support prompt type {type(prompt)}')
#         latent = self.generate_latent(noise, encode_hidden_state)
#         image = self.decode_image(latent)
#         if return_latent:
#             return latent, image
#         return image
    
#     @torch.no_grad
#     def __call__(self):
#         noise = torch.randn(self.cond_images.shape[0], 4, 64, 64, device=self.device)
#         gen_image = self.generate_image(noise)
#         return gen_image

# class SBControlValidator():
#     def __init__(self, path_ckpt_sbv2="/lustre/scratch/client/vinai/users/ngannh9/enhance/sb_v2_ckpt/0.5", model_name = "stabilityai/stable-diffusion-2-1-base", generator=None):
#         self.cond_images, self.prompts = self.get_validation_data()
#         self.device = "cuda"
#         self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
#         self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device)
#         self.unet = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to(self.device)
#         self.unet.eval()
#         self.last_timestep = torch.ones((1,), dtype=torch.int64, device=self.device)
#         self.last_timestep = self.last_timestep * (self.noise_scheduler.config.num_train_timesteps - 1)

#         self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
#         self.text_encoder = CLIPTextModel.from_pretrained(
#             model_name, subfolder="text_encoder"
#         ).to(self.device, dtype=torch.float32)
#         # prepare stuff
#         alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
#         self.alpha_t = (alphas_cumprod[self.last_timestep] ** 0.5).view(-1, 1, 1, 1)
#         self.sigma_t = ((1 - alphas_cumprod[self.last_timestep]) ** 0.5).view(-1, 1, 1, 1)
#         self.weight_dtype = self.unet.dtype
#         if generator is None:
#             self.generator = torch.Generator(device=self.device).manual_seed(0)
#         else:
#             self.generator = generator
        
#         del alphas_cumprod
#     def get_validation_data(self, dataname=['canny118k', 'cannylaion_val'], resolution=512):
#         cond_images = []
#         prompts = []
#         for name in dataname:
#             if name == 'canny118k':
#                 text_path = DATASET_DIR['canny118k']['embed_txt_path']
#                 cond_img_dir = DATASET_DIR['canny118k']['cond_img_dir']

#                 info = open(text_path, 'r')
#                 info = json.load(info)[100000:100004]
#                 # load testing data
#                 for sample in info:
#                     img_name = sample["image_path"]
#                     prompts.append(sample["text"])
#                     cond_images.append(conditioning_transforms()(Image.open(osp.join(cond_img_dir, "conditioning_" + img_name)).convert("RGB")))
#             if name == 'cannylaion_val':
#                 text_path = DATASET_DIR['cannylaion_val']['text_path']
#                 cond_img_dir = DATASET_DIR['cannylaion_val']['cond_img_dir']
#                 f = open(text_path, 'r')
#                 prompt_data = f.read().splitlines()
#                 prompts.extend(prompt_data)
#                 cond_files = os.listdir(cond_img_dir)
#                 for file in cond_files:
#                     cond_images.append(conditioning_transforms()(Image.open(osp.join(cond_img_dir, file)).convert("RGB")))
#         return torch.stack(cond_images).to(device=self.device), prompts
#     @torch.no_grad
#     def encode_prompt(self, prompt, batch_size=1):
#         input_id = self.tokenizer(
#             prompt,
#             max_length=self.tokenizer.model_max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         ).input_ids.to("cuda")
#         encoder_hidden_state = self.text_encoder(input_id)[0].repeat_interleave(batch_size, dim=0)
#         return encoder_hidden_state
#     @torch.no_grad
#     def generate_latent(self, noise, encoder_hidden_state, controlnet, controlnet_image, sb_unet=None):
#         down_block_res_samples, mid_block_res_sample = controlnet(
#             noise,
#             self.last_timestep,
#             encoder_hidden_states=encoder_hidden_state,
#             controlnet_cond=controlnet_image,
#             return_dict=False,
#         )
#         model_pred = self.unet(noise, self.last_timestep, encoder_hidden_state,
#                     down_block_additional_residuals=[sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples],
#                     mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
#                     ).sample
#         pred_original_sample = (noise - self.sigma_t * model_pred) / self.alpha_t
#         if self.noise_scheduler.config.thresholding:
#                 pred_original_sample = self.noise_scheduler._threshold_sample(
#                 pred_original_sample
#             )
#         elif self.noise_scheduler.config.clip_sample:
#             clip_sample_range = self.noise_scheduler.config.clip_sample_range
#             pred_original_sample = pred_original_sample.clamp(
#                 -clip_sample_range, clip_sample_range
#             )
#         return pred_original_sample
#     @torch.no_grad
#     def decode_image(self, latent):
#         latent = latent / self.vae.config.scaling_factor
#         image = (self.vae.decode(latent).sample + 1) / 2
#         return image
#     @torch.no_grad
#     def generate_image(self, controlnet, sb_unet, prompt=None, noise=None, return_latent=False):
        
#         #preparing prompt
#         if prompt is None:
#             prompt = self.prompts
#         if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
#             len_prompt = len(prompt)
#         elif isinstance(prompt, torch.Tensor):
#             len_prompt = prompt.shape[0]
#         else:
#             raise ValueError(f'not support prompt type {type(prompt)}')
        
#         # prepare noise
#         if noise is None:
#             bs=1
#             noise = torch.randn(len_prompt, 4, 64, 64, device="cuda", generator=self.generator)
#         else:
#             bs = noise.shape[0] // len_prompt
#         if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
#             encode_hidden_state = self.encode_prompt(prompt, bs)
#         elif isinstance(prompt, torch.Tensor):
#             encode_hidden_state = prompt
#         else:
#             raise ValueError(f'not support prompt type {type(prompt)}')
        
#         latent = self.generate_latent(noise, encode_hidden_state, controlnet, self.cond_images, sb_unet)
#         image = self.decode_image(latent)
#         if return_latent:
#             return latent, image
#         return image
    
#     @torch.no_grad
#     def __call__(self, controlnet, sb_unet):
#         gen_image = self.generate_image(controlnet.to(device=self.device, dtype=self.weight_dtype), sb_unet)
#         return gen_image

# if __name__ == "__main__":
#     tester = SBControlValidator()
#     controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers")
#     gen_image = tester(controlnet)
#     save_image(gen_image, osp.join(path_save, "test_sbcontrol_validator.jpg"))
#     print(f"saved image to {osp.join(path_save, "test_sbcontrol_validator.jpg")}")