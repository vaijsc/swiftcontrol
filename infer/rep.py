from repcontrolnet.infer import RepControlNetInfer
import os.path as osp
import os
from tqdm import tqdm
from torchvision.utils import save_image
from infer.utils import get_test_data
import torch

model_names = ["rep_canny_3m_sdturbo_lite_merge"]
# model_names = ["rep_canny_3m_sdturbo_lite_branch", "rep_canny_3m_sdturbo_branch"]
# prepare prompt
bs = 32
cond_images, prompts = get_test_data()
def gen_image(infer: RepControlNetInfer, save_path):
    for i in tqdm(range(len(prompts) // bs +1)):
        prompt = prompts[i*bs:i*bs+bs]
        if len(prompt) == 0:
            continue
        images = infer(prompt=prompt, cond_image=cond_images[i*bs:i*bs+bs], guidance_scale=7.5, num_inference_steps=4)
        save_image(images, save_path, nrow=1)
        print("saved to ", save_path)
        # for image_idx, image in enumerate(images):
        #     save_image(image.unsqueeze(dim=0), osp.join(save_path, osp.basename(cond_images[i*bs+image_idx])))
        del images
with torch.no_grad():   
    for model_name in model_names:
        base_model = "stabilityai/sd-turbo"
        OUTPUT_FOLDER = f"output/debug/{model_name}"
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        save_image(cond_images, osp.join(OUTPUT_FOLDER, "cond.jpg"), nrow=1)
        if "lite" in model_name:
            lite = True
            step = 120000
            repcontrolnet_model_path = f"../output/rep_canny_3m_sdturbo_lite/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"
        else:
            lite = False
            step = 120000
            repcontrolnet_model_path = f"../output/rep_canny_3m_sdturbo/checkpoint-{step}/repcontrolnet_ema/repcontrolnet.pth"

        if "branch" in model_name:
            infer = RepControlNetInfer(repcontrolnet_model_path, stage="branch", base_model=base_model, lite=lite)
            gen_image(infer=infer, save_path=OUTPUT_FOLDER)
        elif "merge" in model_name:
            alphas = [0.8, 0.9, 1.0]
            # betas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
            betas = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            # for alpha in alphas:
            #     for beta in betas:
            alpha = 1.0
            beta = 1.0
            save_path = osp.join(OUTPUT_FOLDER, f'alpha_{round(alpha,2)}_beta_{round(beta,2)}.jpg')
            infer = RepControlNetInfer(repcontrolnet_model_path, stage="merge", base_model=base_model, alpha=alpha, beta=beta, lite=lite)
            gen_image(infer=infer, save_path=save_path)
                
        else:
            raise ValueError(f'specify branch or merge for {model_name}')
            
