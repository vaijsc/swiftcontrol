import torch
import os.path as osp
import argparse
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionControlNetPipeline, ControlNetModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import cv2
controlnet_path = "thibaud/controlnet-sd21-canny-diffusers"
sd_model = "stabilityai/stable-diffusion-2-1-base"
path_sb_v2_gen_model = "/lustre/scratch/client/vinai/users/ngannh9/enhance/ckpt/sb_v2_ckpt/unet"
path_sb_v2_inverse_model = "/lustre/scratch/client/vinai/users/ngannh9/enhance/inverse2enhance_ckpt/checkpoint-10000"
path_save = "./output_sb_control_test"
prompts = ["A woman with black hair and red lipstick holding a flower", 
            "Bright and cheerful clown in vibrant, multicolored costume with oversized shoes, a red nose, and a big, friendly smile.",
            "Portrait of a young woman with long, flowing chestnut hair and a warm, sun-kissed glow",
            "Anime-style illustration of a young, heroic girl with vibrant blue hair styled in twin tails, adorned with glowing, silver star-shaped clips"]
prompts = ["Designs Similar to Lake Verde In The Alps",
"Pinup: Classic Cars, Pinupgirls, Pinups, Car Pinup, Pinup Girls, Pin Ups, Pinup Vintage, Pin Up Girls",
"pumpkin time with little girl  Baby photography located in Brockville Ontario, Kingston Ontario and Ottawa Ontario",
"Plus one: The gorgeous supermodel was accompanied by her handsome husband, football star Tom Brady",
"Memories by Lord Frederic Leighton",
"Missed opportunity concept and too late symbol as slow  and delayed businesspeople stuck on a bridge because an eraser erased the path with other quick employees continuing the rac>"]
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(0)
class SBV2Gen():
    def __init__(self, path_ckpt_sbv2, model_name = sd_model):
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to("cuda")
        self.unet_gen = UNet2DConditionModel.from_pretrained(path_ckpt_sbv2, subfolder="unet_ema").to("cuda")
        self.unet_gen.eval()

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
        print(self.alpha_t)
        print(self.sigma_t)
        del alphas_cumprod
        
class SBV2Inverse():
    def __init__(self, path_ckpt, dtype="fp32"):
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        self.model_name = "stabilityai/sd-turbo"
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.model_name, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae").to(
            "cuda", dtype=torch.float32
        )

        self.unet_inverse = UNet2DConditionModel.from_pretrained(f"{path_ckpt}", subfolder="unet_ema").to(
            "cuda", dtype=self.weight_dtype
        )

        self.unet_inverse.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder"
        ).to("cuda", dtype=self.weight_dtype)
        
        self.mid_timestep = torch.ones((1,), dtype=torch.int64, device="cuda")
        self.mid_timestep = self.mid_timestep * 500
        
        # prepare stuff
        T = torch.ones((1,), dtype=torch.int64, device="cuda")
        T = T * (self.noise_scheduler.config.num_train_timesteps - 1)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to("cuda")
        self.alpha_t = (alphas_cumprod[T] ** 0.5).view(-1, 1, 1, 1)
        self.sigma_t = ((1 - alphas_cumprod[T]) ** 0.5).view(-1, 1, 1, 1)
        del alphas_cumprod
        
def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

def generate_latent(sbv2_gen, noise, encoder_hidden_state):
    model_pred = sbv2_gen.unet_gen(noise, sbv2_gen.last_timestep, encoder_hidden_state).sample
    pred_original_sample = (noise - sbv2_gen.sigma_t * model_pred) / sbv2_gen.alpha_t
    if sbv2_gen.noise_scheduler.config.thresholding:
            pred_original_sample = sbv2_gen.noise_scheduler._threshold_sample(
            pred_original_sample
        )
    elif sbv2_gen.noise_scheduler.config.clip_sample:
        clip_sample_range = sbv2_gen.noise_scheduler.config.clip_sample_range
        pred_original_sample = pred_original_sample.clamp(
            -clip_sample_range, clip_sample_range
        )
    return pred_original_sample

def decode_image(vae, pred_original_sample):
    pred_original_sample = pred_original_sample / vae.config.scaling_factor
    image = (vae.decode(pred_original_sample).sample + 1) / 2
    return image

def generate_image(sbv2_gen, noise, encoder_hidden_state):
    pred_original_sample = generate_latent(sbv2_gen, noise, encoder_hidden_state)
    image = decode_image(sbv2_gen.vae, pred_original_sample)
    return image

def compute_inverted_code(gen_image, sbv2_inverse, encoder_hidden_state):
    input_image = gen_image * 2 - 1
    latents = sbv2_inverse.vae.encode(input_image.to(torch.float32)).latent_dist.sample()
    latents = latents * sbv2_inverse.vae.config.scaling_factor
    predict_inverted_code = sbv2_inverse.unet_inverse(latents, sbv2_inverse.mid_timestep, encoder_hidden_state).sample.to(dtype=torch.float32)
    return predict_inverted_code
def tensor2cv(image):
    np_image = image.cpu().permute(1, 2, 0).numpy() * 255  # Shape (H, W, C)
    np_image = np_image.astype(np.uint8)
    cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv2tensor(images):
    tensor_images = []
    for image in images:
        tensor_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
        tensor_image = torch.from_numpy(tensor_image).permute(2, 0, 1).float()
        tensor_image = tensor_image / 255.0
        tensor_images.append(tensor_image)
    return torch.stack(tensor_images, dim=0)

class SBV2Enhance():
    def __init__(self, path_sb_v2_gen_model, path_sb_v2_inverse_model):
        # define sb gen and inverse model
        self.sbv2_gen = SBV2Gen(path_sb_v2_gen_model)
        self.sbv2_inverse = SBV2Inverse(path_sb_v2_inverse_model)
        
    @torch.no_grad()
    def __call__(self, prompts, batch_size):
        total_image = len(prompts)*batch_size
        noise = torch.randn(total_image, 4, 64, 64, device="cuda")
        input_id = tokenize_captions(prompts, self.sbv2_gen.tokenizer).to("cuda")
        encoder_hidden_state = self.sbv2_gen.text_encoder(input_id)[0].repeat_interleave(batch_size, dim=0)
        gen_image = generate_image(self.sbv2_gen, noise, encoder_hidden_state)
        gen_latents = generate_latent(self.sbv2_gen, noise, encoder_hidden_state)
        ## Stage 2: Inverse image to latent code and gen again with SBV2
        # Stage 2.1: Find inverted code
        predict_inverted_code = compute_inverted_code(gen_image, self.sbv2_inverse, encoder_hidden_state)
        # Stage 2.2: Gen img with inverted code
        refine_images = generate_image(self.sbv2_gen, predict_inverted_code, encoder_hidden_state)
        # new_refine_image = generate_image(self.sbv2_gen, new_noise, encoder_hidden_state)
        
        # refine_latents = generate_latent(self.sbv2_gen, predict_inverted_code, encoder_hidden_state)
        # controlnet 
        edges = [cv2.Canny(tensor2cv(refine_image), 100, 200) for refine_image in refine_images]
        edges = cv2tensor(edges)
        save_image(edges, "edges.jpg", nrow=edges.shape[0])
        # switch controlnet cond
        tmp = edges.chunk(2)
        edges = torch.cat((tmp[1], tmp[0]), dim=0)
        cond_path = "./test_data/images"
        image_files = os.listdir(cond_path)
        from torchvision import transforms
        
        transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        cond_images = []
        for file in image_files:
            cond_image = transforms(Image.open(osp.join(cond_path, file)).convert("RGB"))
            cond_images.append(cond_image)
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16, use_safetensors=False)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            # vae=self.vae.to(dtype=self.weight_dtype),
            # text_encoder=self.text_encoder.to(dtype=self.weight_dtype),
            # tokenizer=self.tokenizer,
            unet=self.sbv2_gen.unet_gen,
            controlnet=controlnet.to(dtype=self.sbv2_gen.unet_gen.dtype), 
            # scheduler=self.noise_scheduler,
            safety_checker=None,
            )
        
        pipeline = pipeline.to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.Generator(device="cuda").manual_seed(0)
        controlnet_noise = pipeline(
            prompt = prompts, 
            image=cond_images, 
            num_inference_steps=2, 
            generator=generator,
            output_type="latent"
        ).images
        del pipeline
        alphas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        control_refine_images = []
        for alpha in alphas:
            controlnet_noise = controlnet_noise*alpha + noise*(1-alpha)
            control_refine_images.append(generate_image(self.sbv2_gen, controlnet_noise, encoder_hidden_state))
        
        predict_inverted_code = decode_image(self.sbv2_gen.vae, predict_inverted_code)
        controlnet_noise = decode_image(self.sbv2_gen.vae, controlnet_noise)
        
        return gen_image, predict_inverted_code, refine_images, noise, gen_latents, controlnet_noise, control_refine_images
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a BrushNet training script.")
    parser.add_argument(
        "--sd21_gen",
        action="store_true",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pca_viz",
        action="store_true",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pca_pnp",
        action="store_true",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--degrade",
        type=str,
        # default="cs2,cs4,deno,deblur_uni,deblur_gauss,deblur_aniso,sr2,sr4,sr8,sr16,sr_bicubic4,sr_bicubic8,sr_bicubic16,color,inp",
        default=None,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--enhance_time",
        type=int,
        # default="cs2,cs4,deno,deblur_uni,deblur_gauss,deblur_aniso,sr2,sr4,sr8,sr16,sr_bicubic4,sr_bicubic8,sr_bicubic16,color,inp",
        default=1,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    args = parser.parse_args()
    return args
def image_grid(imgs, cols):
    rows = int(len(imgs)/cols)
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(rows*w, cols*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%rows*w, i//rows*h))
    return grid
if __name__ == "__main__":
    # for testing
    args = parse_args()
    sd21_gen = args.sd21_gen
    degrade = None
    if args.degrade:
        degrade = args.degrade.split(",")
    import os
    os.makedirs(path_save, exist_ok=True)
    sbv2_enhance = SBV2Enhance(path_sb_v2_gen_model, path_sb_v2_inverse_model)
    
    # test
    # prompt = "Portrait of a young woman with long, flowing chestnut hair and a warm, sun-kissed glow"
    # prompt = "Anime-style illustration of a young, heroic girl with vibrant blue hair styled in twin tails, adorned with glowing, silver star-shaped clips"

    bs = 1
    enhance_time = args.enhance_time
    gen_image, inverted_code, refine_image, noise, gen_latents, controlnet_noise, control_refine_image = sbv2_enhance(prompts, bs)
    path_save_enhance = osp.join(path_save, f"enhance.jpg")
    path_save_gen = osp.join(path_save, f"gen.jpg")
    path_save_controlnet_noise = osp.join(path_save, f"controlnet_noise.jpg")
    path_save_inverted_code = osp.join(path_save, f"inverted_code.jpg")
    
    save_image(gen_image, path_save_gen, nrow=gen_image.shape[0])
    save_image(refine_image, path_save_enhance, nrow=gen_image.shape[0])
    save_image(inverted_code, path_save_inverted_code, nrow=gen_image.shape[0])
    save_image(controlnet_noise, path_save_controlnet_noise, nrow=gen_image.shape[0])
    for i, img in enumerate(control_refine_image):
        path_save_control_enhance = osp.join(path_save, f"control_enhance_{i}.jpg")
        save_image(img, path_save_control_enhance, nrow=gen_image.shape[0])