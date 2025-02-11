# from dataloader.utils import DATASET_DIR
# from torchvision import transforms
# import json
# import torch
# import os
# import os.path as osp
# from typing import List
# from diffusers.utils.torch_utils import randn_tensor
# from torchvision.utils import save_image
# import numpy as np
# from PIL import Image
# import gc
# from transformers import CLIPTextModel, CLIPTokenizer
# from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

# class RepControlNetValidator():
#     def __init__(self, data_name, scheduler, tokenizer, text_encoder, vae, device, save_path, guidance_scale=7.5, num_images_per_prompt=1):
#         self.save_path = save_path
#         os.makedirs(self.save_path, exist_ok=True)
#         self.guidance_scale = guidance_scale
#         self.vae = vae
#         self.num_images_per_prompt = num_images_per_prompt
#         self.weight_dtype = vae.dtype
#         self.device = device
#         self.do_classifier_free_guidance = guidance_scale > 1.0
#         self.cond_embeds, prompt_embeds = self.get_validation_data(data_name, text_encoder, tokenizer)
#         self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(prompt_embeds, tokenizer, text_encoder)
#         self.batch_size = self.prompt_embeds.shape[0]
#         self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
#         self.scheduler = scheduler
        
#     def get_validation_data(self, data_name, text_encoder, tokenizer, resolution=512):
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
#             info = json.load(info)[100000:100008]
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
#             return cond_images, torch.stack(prompt_embeds).to(dtype=self.weight_dtype, device=self.device)
#         elif data_name == "fill50k":
#             cond_image_dir = "./val50kfill"
#             cond_image_files = sorted(os.listdir(cond_image_dir))
#             cond_images = []
            
#             for image in cond_image_files:
#                 cond_images.append(cond_transforms(Image.open(osp.join(cond_image_dir, image)).convert("RGB")))
#             cond_images = torch.stack(cond_images).to(dtype=self.weight_dtype, device=self.device)
            
#             prompts = ["red circle with blue background","cyan circle with brown floral background","light coral circle with white background","cornflower blue circle with light golden rod yellow background"]
#             input_id = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to("cuda")
#             encoder_hidden_state = text_encoder(input_id)[0].to(dtype=self.weight_dtype, device=self.device)
#             return cond_images, encoder_hidden_state
            
#         else:
#             raise ValueError("not known val dataname")
        
#     def _maybe_convert_prompt(self, prompt, tokenizer):  # noqa: F821
#         tokens = tokenizer.tokenize(prompt)
#         unique_tokens = set(tokens)
#         for token in unique_tokens:
#             if token in tokenizer.added_tokens_encoder:
#                 replacement = token
#                 i = 1
#                 while f"{token}_{i}" in tokenizer.added_tokens_encoder:
#                     replacement += f" {token}_{i}"
#                     i += 1

#                 prompt = prompt.replace(token, replacement)

#         return prompt
    
#     def maybe_convert_prompt(self, prompt, tokenizer):  # noqa: F821
#         if not isinstance(prompt, List):
#             prompts = [prompt]
#         else:
#             prompts = prompt

#         prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

#         if not isinstance(prompt, List):
#             return prompts[0]

#         return prompts
    
#     def encode_prompt(self, prompt_embeds, tokenizer, text_encoder):
#         prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype, device=self.device)
#         bs_embed, seq_len, _ = prompt_embeds.shape
#         prompt_embeds = prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
#         prompt_embeds = prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)
#         uncond_tokens = [""] * bs_embed
#         uncond_tokens = self.maybe_convert_prompt(uncond_tokens, tokenizer)
#         max_length = prompt_embeds.shape[1]
#         uncond_input = tokenizer(
#             uncond_tokens,
#             padding="max_length",
#             max_length=max_length,
#             truncation=True,
#             return_tensors="pt",
#         )
#         if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
#             attention_mask = uncond_input.attention_mask.to(self.device)
#         else:
#             attention_mask = None
#         negative_prompt_embeds = text_encoder(
#             uncond_input.input_ids.to(self.device),
#             attention_mask=attention_mask,
#         )
#         negative_prompt_embeds = negative_prompt_embeds[0]
#         if self.do_classifier_free_guidance:
#             # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
#             seq_len = negative_prompt_embeds.shape[1]
#             negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.weight_dtype, device=self.device)
#             negative_prompt_embeds = negative_prompt_embeds.repeat(1, self.num_images_per_prompt, 1)
#             negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * self.num_images_per_prompt, seq_len, -1)
#         return prompt_embeds, negative_prompt_embeds
    
#     def inference(self, repcontrolnet, seed, num_inference_steps=20):
#         self.scheduler.set_timesteps(num_inference_steps, device=self.device)
#         timesteps = self.scheduler.timesteps
#         if seed is None:
#             generator = None
#         else:
#             generator = torch.Generator(device=self.device).manual_seed(seed)
#         latents = randn_tensor((self.batch_size, 4, 64, 64), generator=generator, device=self.device, dtype=self.weight_dtype)
#         for i, t in enumerate(timesteps):
#             # expand the latents if we are doing classifier free guidance
#             latent_model_input = self.scheduler.scale_model_input(latents, t)
#             noise_pred_uncond = repcontrolnet(latent_model_input,t,encoder_hidden_states=self.negative_prompt_embeds,controlnet_cond = self.cond_embeds,return_dict=False)[0]
#             noise_pred_text = repcontrolnet(latent_model_input,t,encoder_hidden_states=self.prompt_embeds,controlnet_cond = self.cond_embeds,return_dict=False)[0]

#             # perform guidance
#             if self.do_classifier_free_guidance:
#                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
#             # compute the previous noisy sample x_t -> x_t-1
#             latents = self.scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
#         latents = (latents / self.vae.config.scaling_factor).to(dtype=self.weight_dtype)
#         images = (self.vae.decode(latents, generator=generator).sample + 1) / 2
#         return images

#     def prepare_repcontrolnet(repcontrolnet, unet_model_name, alpha, beta):
#         unet = UNet2DConditionModel.from_pretrained(unet_model_name, subfolder="unet")
#         rep_keys = set(repcontrolnet.state_dict().keys()) - set(unet.state_dict.keys())
#         for rep_key in rep_keys:
#             unet_key = rep_key.replace('_rep', '')
#             if unet_key in unet.state_dict.keys():
#                 unet.state_dict[unet_key] = alpha * repcontrolnet.state_dict[unet_key] + beta * repcontrolnet.state_dict[rep_key]
#             else:
#                 raise ValueError(f"model keys not match {rep_key}")
#         del repcontrolnet
#         gc.collect()
#         torch.cuda.empty_cache()
#         return unet
        
#     def __call__(self, repcontrolnet, seed):
#         with torch.no_grad():
#             images = self.inference(repcontrolnet, seed)
#         save_image(images, osp.join(self.save_path, nrow=1))
#         torch.cuda.empty_cache()

# if __name__ == "__main__":
#     save_path = "infer/repcontrolnet"
#     os.makedirs(save_path, exist_ok=True)
#     unet_model_path = "stabilityai/stable-diffusion-2-1-base"
#     repcontrolnet_model_path = "sd-model-finetuned/checkpoint-30000"
#     scheduler = DDPMScheduler.from_pretrained(unet_model_path, subfolder="scheduler")
#     tokenizer = CLIPTokenizer.from_pretrained(unet_model_path, subfolder="tokenizer")
#     text_encoder = CLIPTextModel.from_pretrained(unet_model_path, subfolder="text_encoder")
#     vae = AutoencoderKL.from_pretrained(unet_model_path, subfolder="vae")
#     repcontrolnet = RepUNet2DConditionModel.from_pretrained(repcontrolnet_model_path, cond_model_path=None, subfolder="repcontrolnet")
#     validator = Validator("canny", scheduler, tokenizer, text_encoder, vae, save_path)
#     validator(repcontrolnet, seed=0)
