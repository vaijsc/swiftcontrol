from PIL import Image
import random
import numpy as np
import torch
from torchvision import transforms
import os.path as osp
def tensor2pil(tensor_image):
    if tensor_image.dim() == 4:
        pil_imgs = []
        for image in tensor_image:
            pil_img = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            pil_imgs.append(Image.fromarray(pil_img))
        return pil_imgs
    if tensor_image.dim() == 3:
        pil_img = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return pil_img
    raise ValueError(f"tensor_image dim should be 3 or 4, got {tensor_image.shape}")

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_cond_image(image, resolution=512):
    # input is a list of path
    # output is torch tensor image [0, 1] 
    cond_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    if isinstance(image, list):
        if isinstance(image[0], str):
            cond_images = []
            for image_path in image:
                cond_images.append(cond_transforms(Image.open(image_path).convert("RGB")))
            cond_images = torch.stack(cond_images).to(dtype=torch.float32, device="cuda")
            return cond_images
    elif isinstance(image, torch.Tensor):
        return image
    else:
        raise("not support single image")