import torch
import time
import pandas as pd
from torch.utils.data import Dataset
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
from torchvision import transforms
class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, journeydb_path, laion_path):
        print("*** load prompt dataset: start")
        t0 = time.time()
        with open(journeydb_path) as f:
            self.prompts = f.readlines()
            self.prompts = [x.strip() for x in self.prompts]
        laion = pd.read_csv(laion_path)
        laion_prompts = list(laion.text)
        self.prompts.extend(laion_prompts)
        print(f"*** load prompt dataset: end --- {time.time()-t0} sec")
        print(f"*** Dataset length: {len(self.prompts)} ***")

    def _load(self, idx):
        return {
            "prompts": self.prompts[idx],
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        data = self._load(index)

        return data
class CannyLaionDataset(Dataset):
    def __init__(self, 
                 data_folder="/lustre/scratch/client/vinai/users/ngannh9/RepControlNet/data/canny_laion",
                 resolution=512,
                 caption_path="text_embedding.json",
                 conditioning_folder="conditioning",
                 ):
        self.data_folder = data_folder
        if osp.exists(conditioning_folder):
            self.conditioning_folder = conditioning_folder
        else:
            self.conditioning_folder = osp.join(data_folder, conditioning_folder)
        if osp.exists(caption_path):
            caption_path = caption_path
        else:
            caption_path = osp.join(data_folder, caption_path)
        print("*** load control prompt dataset: start")
        t0 = time.time()
        f = open(caption_path, 'r')
        self.metadata = json.load(f)
        print(f"*** load prompt dataset: end --- {time.time()-t0} sec")
        print(f"*** Dataset length: {len(self.metadata)} ***")
        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        prompt_data = self.metadata[index]
        image_name = prompt_data["image_path"]
        prompts = prompt_data["text"]
        text_embedding_path = prompt_data["text_embedding"]
        text_embed = torch.from_numpy(np.load(osp.join(self.data_folder ,text_embedding_path), allow_pickle=True))
        conditioning_image = Image.open(osp.join(self.conditioning_folder, image_name)).convert('RGB')
        return {
            "text_embed": text_embed,
            "conditioning_image": self.conditioning_image_transforms(conditioning_image),
            "prompts": prompts
        }
class PromptDataset(Dataset):
    def __init__(self, prompt_list_path, embed_txt_path):
        with open(embed_txt_path) as f:
            self.embed_paths = f.readlines()
            self.embed_paths = [x.strip() for x in self.embed_paths]
        with open(prompt_list_path) as f:
            self.prompts = f.readlines()
            self.prompts = [x.strip() for x in self.prompts]

        assert len(self.prompts) == len(
            self.embed_paths
        ), f"Prompt {len(self.prompts)} and embeds {len(self.embed_paths)} length mismatch"

    def _load(self, idx):
        return {
            "prompt_embeds": torch.from_numpy(np.load(self.embed_paths[idx])),
            "prompt": self.prompts[idx],
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        data = self._load(index)

        return data

    def shuffle(self, *args, **kwargs):
        ids = np.arange(len(self.prompts))
        shuffled_ids = np.random.permutation(ids)
        self.prompts = [self.prompts[idx] for idx in shuffled_ids]
        self.embed_paths = [self.embed_paths[idx] for idx in shuffled_ids]
        return self

    def select(self, selected_range):
        self.prompts = [self.prompts[idx] for idx in selected_range]
        self.embed_paths = [self.embed_paths[idx] for idx in selected_range]
        return self

class DeepFashionTextSegmDataset(Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 caption_path,
                 transform, image_size):
        self._img_path = img_dir
        self._segm_path = segm_dir
        self._caption_path = caption_path
        self._captions = json.load(open(self._caption_path, 'r'))
        self._image_fnames = self.filter_segm_images(list(self._captions.keys()))
        self._caption_path = caption_path
        self.transform = transform
        self.image_size = image_size
        self.resize_mask = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

    def filter_segm_images(self, image_fnames):
        image_fnames = list(filter(lambda fname: os.path.exists(os.path.join(self._segm_path, f'{fname[:-4]}_segm.png')), image_fnames))
        return image_fnames

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = Image.open(os.path.join(self._img_path, fname))
        return image

    def _load_segm(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_segm.png'
        segm = Image.open(os.path.join(self._segm_path, fname))
        return segm

    def _load_captions(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        return self._captions[fname]

    def __getitem__(self, index):
        image = self._load_raw_image(index)
        segm = self._load_segm(index)
        text_desc = self._load_captions(index)
        segm = self.to_tensor(segm) * 255
        segm = torch.logical_or(segm == 15, segm == 14).int() # skin & face

        segm = self.resize(segm).squeeze()
        return_dict = {
            'image': [image],
            'segm': [segm],
            'text': [text_desc],
            'img_name': self._image_fnames[index]
        }

        return self.transform(return_dict)
    

    def __len__(self):
        return len(self._image_fnames)
    
class CocoHumanDataset(Dataset):
    def __init__(self, img_dir, label_dir, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.infor = self.preprocess()
        self.imgid = list(self.infor.keys())
        self.mask = {}
        self.image_size = image_size
        self.transform = transform
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
     
    def preprocess(self):
        f = json.load(open(self.label_dir, 'rb'))
        return f
    
    def __len__(self):
        return len(self.infor)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        imgid = self.imgid[index]
        infor = self.infor[imgid]
        img_path = os.path.join(self.img_dir, infor['file_name'])
        caption = infor['caption']
        person_infor = infor['person']
      
        image = Image.open(img_path)
        bbox = [person['lefthand_box'] for person in person_infor] + [person['righthand_box'] for person in person_infor]
        
        height, width = infor['height'], infor['width']
        
        
        mask = np.zeros((height, width))
        for box in bbox:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x==y and y==w and w==h and h==0: continue
            mask[y:y+h, x:x+w] = 1
  
        mask = self.resize_mask(mask).squeeze()
        
        return_dict = {
                'image': [image],
                'text': [caption],
                'img_name': img_path,
                'segm': [mask]
            }
        
        return self.transform(return_dict)
    
class LaionDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, text_file_path, mask_path, transform, tokenizer):
        self.folder_path = folder_path
        self.transform = transform
        self.mask_path = mask_path
        self.tokenizer = tokenizer
        self.text_file_path = text_file_path
        self.captions = []
        self.images = []
        self.masks = []
        self.pixel_values = []
        with open(self.text_file_path, 'r') as file:
            for line in file:
                line = line.strip().split(' ')
                image_filename = line[0]
                caption = ' '.join(line[1:])
            
                image_path = os.path.join(self.folder_path, image_filename)
                mask_path = os.path.join(self.mask_path, image_filename)
                if os.path.isfile(mask_path):
                    mask = Image.open(mask_path)
                else:
                    mask = None
                image = Image.open(image_path).convert('RGB')  # Load image as PIL
                pixel_value = self.transform(image)
                self.captions.append(caption)
                self.images.append(image)
                self.masks.append(mask)
                self.pixel_values.append(pixel_value)
        self.input_ids = self.tokenizer(self.captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        self.input_ids = torch.stack([example for example in self.input_ids])
        self.pixel_values = torch.stack([example for example in self.pixel_values])
        self.pixel_values = self.pixel_values.to(memory_format=torch.contiguous_format).float()
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.pixel_values[idx], self.input_ids[idx]   
class LaiOn2M(Dataset):
    def __init__(self, img_dir, label_dir,segmentation_dir, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask = {}
        self.image_size = image_size
        self.transform = transform
        self.seg_dir = segmentation_dir
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
        # self.infor = self.preprocess()
        # self.imgid = list(self.infor.keys())
        self.filenames, self.captions = self.read_prompt(label_dir)
        # self.infor = self.read_segment(segmentation_dir)
        
        # self.sanity_check()

    def read_prompt(self, label_dir):
        txt_files = os.listdir(label_dir)
        filenames = []
        captions = []
        for txt_file in txt_files:
            with open(os.path.join(label_dir, txt_file), 'r') as file:
                for line in file:
                    line = line.strip().split(' ')
                    filenames.append(line[0])
                    captions.append(' '.join(line[1:]))
        return filenames, captions
    # def read_segment(self, segment_dir):
    #     filenames = []
    #     for file in os.listdir(segment_dir):
    #         if file.endswith('.npy'):
    #             if file.replace('npy', 'jpg') not in self.prompt: continue
    #             filenames.append(file.replace('.npy', '.jpg'))
    #     return filenames

    def sanity_check(self):
        # for file in self.infor:
        #     if file not in self.prompt:
        #         breakpoint()
        pass

        return
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):

        img_name = self.filenames[index]
       
        img_path = os.path.join(self.img_dir, img_name)
        caption = self.captions[index]
      
      
        image = Image.open(img_path)
        seg_path = os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy'))
        if os.path.exists(seg_path):
            mask = np.load(os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy')))
            mask = np.logical_or(mask == 13, mask == 14, mask==15).astype(np.uint8)
        else:
            width, height = image.size
            mask = np.zeros((height, width))
        mask = self.resize_mask(mask).squeeze()
        return_dict = {
                'image': [image],
                'text': [caption],
                'img_name': img_path,
                'segm': [mask]
            }
        
        return self.transform(return_dict)
class LaiOn2M(Dataset):
    def __init__(self, img_dir, label_dir,segmentation_dir, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask = {}
        self.image_size = image_size
        self.transform = transform
        self.seg_dir = segmentation_dir
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
        # self.infor = self.preprocess()
        # self.imgid = list(self.infor.keys())
        self.filenames, self.captions = self.read_prompt(label_dir)
        # self.infor = self.read_segment(segmentation_dir)
        
        # self.sanity_check()

    def read_prompt(self, label_dir):
        txt_files = os.listdir(label_dir)
        filenames = []
        captions = []
        for txt_file in txt_files:
            with open(os.path.join(label_dir, txt_file), 'r') as file:
                for line in file:
                    line = line.strip().split(' ')
                    filenames.append(line[0])
                    captions.append(' '.join(line[1:]))
        return filenames, captions
    # def read_segment(self, segment_dir):
    #     filenames = []
    #     for file in os.listdir(segment_dir):
    #         if file.endswith('.npy'):
    #             if file.replace('npy', 'jpg') not in self.prompt: continue
    #             filenames.append(file.replace('.npy', '.jpg'))
    #     return filenames

    def sanity_check(self):
        # for file in self.infor:
        #     if file not in self.prompt:
        #         breakpoint()
        pass

        return
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):

        img_name = self.filenames[index]
       
        img_path = os.path.join(self.img_dir, img_name)
        caption = self.captions[index]
      
      
        image = Image.open(img_path)
        seg_path = os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy'))
        if os.path.exists(seg_path):
            mask = np.load(os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy')))
            mask = np.logical_or(mask == 13, mask == 14, mask==15).astype(np.uint8)
        else:
            width, height = image.size
            mask = np.zeros((height, width))
        mask = self.resize_mask(mask).squeeze()
        return_dict = {
                'image': [image],
                'text': [caption],
                'img_name': img_path,
                'segm': [mask]
            }
        
        return self.transform(return_dict)
    
class Canny118kDataset(Dataset):
    def __init__(self, img_dir, cond_img_dir, text_path, split, image_size, transform, max_train_samples=None):
        self.img_dir = img_dir
        self.cond_img_dir = cond_img_dir
        f = open(text_path, "r")
        if split == "train":
            if not max_train_samples:
                self.info = json.load(f)[:50000]
            else:
                self.info = json.load(f)[:max_train_samples]
        else:
            self.info = json.load(f)[100000:]
        self.image_size = image_size
        self.transform = transform
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
    
    def __len__(self):
        return len(self.info)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        # text = self.text_embedding[index]["text"]
        text_embedding = torch.from_numpy(np.load(self.info[index]["text_embedding"], allow_pickle=True))
        img_name = self.info[index]["image_path"]
        image = Image.open(osp.join(self.img_dir, img_name)).convert("RGB")
        cond_image = Image.open(osp.join(self.cond_img_dir, "conditioning_" + img_name)).convert("RGB")
        return_dict = {
                'image': [image],
                'input_ids': text_embedding,
                'img_name': img_name,
                'cond_image': [cond_image]
            }
        
        return self.transform(return_dict)
    