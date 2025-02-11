# precomputing 
BS=128*2
import torch
import os
import os.path as osp
from transformers import AutoTokenizer, CLIPTextModel
import json
from tqdm import tqdm
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

        
def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

class SDModel():
    def __init__(self, dtype="fp32"):
        # define sb gen and inverse model
        if dtype == "fp16":
            self.weight_dtype = torch.float16
        elif dtype == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32
        
        self.model_name = "stabilityai/stable-diffusion-2-1-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder"
        ).to("cuda", dtype=self.weight_dtype)
        
    @torch.no_grad()
    def __call__(self, prompts):
        input_id = tokenize_captions(prompts, self.tokenizer).to("cuda")
        encoder_hidden_state = self.text_encoder(input_id)[0]
        return encoder_hidden_state
    
if __name__ == "__main__":
    # for testing
    path_save = "../data/diffusiondb"
    caption_folder = "../data/diffusiondb/caption"
    image_folder = "../data/diffusiondb/train"
    sd_model = SDModel()
    
    bs = BS
    json_info = []
    #output
    text_embedding_path = os.path.join(path_save, "text_embedding_path")
    path_json_info = osp.join(path_save, f"text_embedding.json")
    os.makedirs(text_embedding_path, exist_ok=True)

    # read image name and prompt
    caption_files = os.listdir(caption_folder)
    def process_file(caption_file):
        single_imgs_name = []
        total_prompts = []
        with open(osp.join(caption_folder, caption_file), 'r') as f:
            metadata = json.load(f)
            for img_name in metadata:
                single_imgs_name.append(img_name)
                total_prompts.append(metadata[img_name]["p"])
        return single_imgs_name, total_prompts

    imgs_name = []
    total_prompts = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, caption_files)

    for single_imgs_name, total_prompts in results:
        imgs_name.extend(single_imgs_name)
        total_prompts.extend(total_prompts)
    # def read_file(file_path):
    #     with open(osp.join(image_folder, file_path[:-4]+".txt"), 'r') as file:
    #         return file.read()
    # def read_files_in_parallel(file_paths):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #     # Map each file path to the read_file function and execute in parallel
    #         file_contents = list(tqdm(executor.map(read_file, file_paths)))
    #     return file_contents
    # total_prompts = read_files_in_parallel(imgs_name)

    def process_batch(i, bs, total_prompts, imgs_name, sd_model, text_embedding_path):
        prompt = total_prompts[bs * i : bs * i + bs]
        img_name = imgs_name[bs * i : bs * i + bs]

        bs_len = len(prompt)
        text_embedding = sd_model(prompt)
        json_info = []

        for j in range(bs_len):
            text_embedding_save = osp.join(text_embedding_path, f"{bs*i+j}.npy")
            np.save(text_embedding_save, text_embedding[j].cpu())

            json_info.append({
                "image_path": img_name[j],
                "text": prompt[j],
                "text_embedding": f"./text_embedding_path/{bs*i+j}.npy"
            })
        
        return json_info

    def parallel_processing(total_prompts, imgs_name, sd_model, text_embedding_path, bs):
        json_info = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_batch, i, bs, total_prompts, imgs_name, sd_model, text_embedding_path)
                for i in range(len(total_prompts) // bs)
            ]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                json_info.extend(future.result())

        return json_info

    # Gọi hàm parallel_processing
    json_info = parallel_processing(total_prompts, imgs_name, sd_model, text_embedding_path, bs)
    with open(path_json_info, "w") as fp:
        json.dump(json_info, fp, indent=4)
    print(f'saved results to {path_json_info}')
    