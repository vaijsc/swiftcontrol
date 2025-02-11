from torch.utils.data import ConcatDataset
import os
import os.path as osp
import random
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
from torchvision import transforms
from datasets import load_dataset
from dataloader.my_dataset import Canny118kDataset

DATASET_DIR = {
    'COCO': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/face/data/COCO/train2017",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/face/data/COCO/coco_wholebody_train_v1.0.json"
    },
    'deepfashion_mm': {

        'image_folder': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/images",
        'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/segm",
        'caption_file': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/captions.json"
    },
    'laion2m': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/LAION/preprocessed_2256k/train",
        'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/code/Self-Correction-Human-Parsing/outputs/",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/hand/LAVIS/output"
        # 'caption_file': "cap_test"
    },
    'prompt': {
        'prompt_list_path': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/prompt/prompt_list.txt",
        'embed_txt_path': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/prompt/embeds"
    },
    'canny118k': {
        # 'img_dir': "/lustre/scratch/client/vinai/users/ngannh9/hand/gen_text/llava/output/canny/captions_0_118287.jsonl",
        'img_dir': "../data/canny118k/images",
        'cond_img_dir': "../data/canny118k/conditioning_images",
        'embed_txt_path': "../data/canny118k/text_embedding.json"
    },
    'cannylaion_val': {
        'cond_img_dir': "./data/val/cannylaion/cond_images",
        'text_path': "./data/val/cannylaion/prompt.txt"
    },
    'canny_laion': {
        'folder': "../data/canny_laion"
    },
    'diffusiondb': {
        'folder': "../data/diffusiondb"
    },
}

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    captions = []
    for caption in prompts:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {"prompt_embeds": prompt_embeds.cpu()}
def center_crop(image, crop_size):
    new_width, new_height = crop_size
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))

def image_transforms(resolution=512):
    return transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def conditioning_transforms(resolution=512):
    return transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    

class CannyDataset(data.Dataset):
    def __init__(self, folder, image_size, transform, max_train_samples=None):
        self.folder = folder
        self.cond_img_dir = osp.join(folder, "conditioning")
        self.img_dir = osp.join(folder, "train")
        text_path = osp.join(folder, "text_embedding.json")
        f = open(text_path, "r") # left 10 for validation
        self.info = json.load(f)[:-10]
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self):
        return len(self.info)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        # text = self.text_embedding[index]["text"]
        text_embedding = torch.from_numpy(np.load(osp.join(self.folder, self.info[index]["text_embedding"]), allow_pickle=True))
        img_name = self.info[index]["image_path"]
        image = Image.open(osp.join(self.img_dir, img_name)).convert("RGB")
        cond_image = Image.open(osp.join(self.cond_img_dir, img_name)).convert("RGB")
        return_dict = {
                'image': [image],
                'input_ids': text_embedding,
                'img_name': img_name,
                'cond_image': [cond_image]
            }
        
        return self.transform(return_dict)
    

class PromptCannyDataset(data.Dataset):
    def __init__(self, folder, image_size, transform, max_train_samples=None):
        self.folder = folder
        self.cond_img_dir = osp.join(folder, "conditioning")
        text_path = osp.join(folder, "text_embedding.json")
        f = open(text_path, "r") # left 10 for validation
        self.info = json.load(f)[:-10]
        self.transform = transform
    
    def __len__(self):
        return len(self.info)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        # text = self.text_embedding[index]["text"]
        text_embedding = torch.from_numpy(np.load(osp.join(self.folder, self.info[index]["text_embedding"]), allow_pickle=True))
        img_name = self.info[index]["image_path"]
        cond_image = Image.open(osp.join(self.cond_img_dir, img_name)).convert("RGB")
        return_dict = {
                'input_ids': text_embedding,
                'img_name': img_name,
                'cond_image': [cond_image]
            }
        
        return self.transform(return_dict)
    
def make_fill50k_dataset(args, tokenizer, accelerator):
    dataset = load_dataset(
        "fusing/fill50k",
        cache_dir=args.cache_dir,
    )

    column_names = dataset["train"].column_names
    image_column = column_names[0]
    conditioning_image_column = column_names[1]
    caption_column = column_names[2]

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset

def get_dataset(dataset_names, split="train", transform=None, resolution=512, max_train_samples=None):
    dataset_names = dataset_names.split(',')
    dataset_list = []
    for dataset_name in dataset_names:
        if dataset_name == 'canny':
            _data =  Canny118kDataset(
                img_dir=DATASET_DIR['canny']['img_dir'],
                cond_img_dir=DATASET_DIR['canny']['cond_img_dir'],
                text_path=DATASET_DIR['canny']['embed_txt_path'],
                split=split,
                transform=transform,
                image_size=resolution,
                max_train_samples=max_train_samples
            )
        # elif dataset_name == 'fill50k':
        #     _data = make_fill50k_dataset(args, tokenizer, accelerator)
        elif dataset_name == 'canny_laion':
            _data = CannyDataset(
                folder=DATASET_DIR['canny_laion']['folder'],
                transform=transform,
                image_size=resolution
            )
        elif dataset_name == 'diffusiondb':
            _data = CannyDataset(
                folder=DATASET_DIR['diffusiondb']['folder'],
                transform=transform,
                image_size=resolution
            )
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')

        dataset_list.append(_data)
    if len(dataset_list) > 1: 
        print("using ", dataset_names)
        return ConcatDataset(dataset_list)
    else:
        return dataset_list[0]
