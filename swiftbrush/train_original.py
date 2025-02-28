#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Base: few-step without improved VSD
    Update: Randomly sample #timesteps for student during training => Improve capability for 1 or 2 steps sampling
    Status: Done, Work 
"""

import argparse
import gc
import logging
import math
import os
import random
import time
import pandas as pd
import shutil
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
# import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (AutoencoderKL, AutoencoderTiny, DDPMScheduler,
                       UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from peft import LoraConfig, PeftModel
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig
from peft.utils import get_peft_model_state_dict
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.loaders import LoraLoaderMixin

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__)


def calculate_weight(
    current_step, max_train_steps, start_weight=1.0, decay_rate=0.0014
):
    return start_weight * decay_rate ** (current_step / max_train_steps)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompts, text_encoder, tokenizer, is_train=True):
    prompt_embeds_list = []

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

@torch.no_grad()
def inference(
    vae, unet, noise_scheduler, encoded_embeds, generator, device, weight_dtype
):
    input_shape = (1, 4, args.resolution // 8, args.resolution // 8)
    input_noise = torch.randn(
        input_shape, generator=generator, device=device, dtype=weight_dtype
    )

    prompt_embed = encoded_embeds["prompt_embeds"]
    prompt_embed = prompt_embed.to(device, weight_dtype)

    pred_original_sample = predict_onestep(
        unet, noise_scheduler, input_noise, prompt_embed,
    )
    pred_original_sample = pred_original_sample / vae.config.scaling_factor

    image = vae.decode(pred_original_sample.to(dtype=vae.dtype)).sample.float()
    image = (image[0].detach().cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return image

def predict_onestep(unet, noise_scheduler, input_noise, prompt_embeds):
    max_timesteps = torch.ones(
        (input_noise.shape[0],), dtype=torch.int64, device=input_noise.device
    )
    max_timesteps = max_timesteps * (noise_scheduler.config.num_train_timesteps - 1)

    alpha_T, sigma_T = 0.0047**0.5, (1 - 0.0047) ** 0.5
    model_pred = unet(input_noise, max_timesteps, prompt_embeds).sample

    noise_pred = (input_noise - sigma_T * model_pred) / alpha_T

    return noise_pred

# Define a simple prompt-only dataloader
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


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # mixed_precision=args.mixed_precision,
        log_with=args.tracker,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    teacher_lora = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    teacher_lora.b_is_student = False

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_v"],
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    teacher_lora.add_adapter(config)
    b_same_arch_as_teacher = False
    if args.student_model_name_or_path is None:
        args.student_model_name_or_path = args.pretrained_model_name_or_path
        b_same_arch_as_teacher = True

    if accelerator.is_main_process:
        print(f"Teacher arch: {args.pretrained_model_name_or_path}")
        print(f"Student arch: {args.student_model_name_or_path}")
        print(f"b_same_arch_as_teacher: {b_same_arch_as_teacher}")

    # if not b_same_arch_as_teacher:
        # Load the tokenizer
        # student_tokenizer = AutoTokenizer.from_pretrained(
        #     args.student_model_name_or_path,
        #     subfolder="tokenizer",
        #     revision=args.revision,
        #     use_fast=False,
        # )

        # import correct text encoder classes
        # student_text_encoder_cls = import_model_class_from_model_name_or_path(
        #     args.student_model_name_or_path, args.revision
        # )

        # student_text_encoder = student_text_encoder_cls.from_pretrained(
        #     args.student_model_name_or_path,
        #     subfolder="text_encoder",
        #     revision=args.revision,
        # )
    # else:
    student_text_encoder = None
    student_tokenizer = None

    student_path = args.student_model_name_or_path
    student_subfolder = "unet"
    if args.student_pretrained_path is not None:
        student_path = args.student_pretrained_path
        student_subfolder = args.student_subfolder

    if args.random_init:
        unet = UNet2DConditionModel.from_config(
            student_path, subfolder=student_subfolder
        )
    else:
        unet = UNet2DConditionModel.from_pretrained(
            student_path, subfolder=student_subfolder, revision=args.revision
        )

    # good_unet = UNet2DConditionModel.from_pretrained(
    #         args.good_unet, subfolder="unet", revision=args.revision
    #     )
    
    unet.b_is_student = True

    if not args.use_tinyvae:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    else:
        logger.info("Using tiny vae")
        vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=weight_dtype
        )

    if args.clip_weight != 0:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        if args.use_tinyvae:
            tinyvae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd", torch_dtype=weight_dtype
            )

    # Freeze vae, text encoders and teacher.
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # if not b_same_arch_as_teacher:
        # student_text_encoder.requires_grad_(False)
    # good_unet.requires_grad_(False)


    # Set unet and lora teacher as trainable.
    unet.train()
    teacher_lora.train()

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # good_unet.to(accelerator.device, dtype=weight_dtype)

    # if not b_same_arch_as_teacher:
        # student_text_encoder.to(accelerator.device, dtype=weight_dtype)
    if args.clip_weight != 0:
        clip_model, clip_tokenizer = clip_model.to(
            accelerator.device, dtype=weight_dtype
        ), open_clip.get_tokenizer("ViT-B-32")
        clip_model.eval()

        clip_mean = (
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=weight_dtype)
            .view(1, -1, 1, 1)
            .to(accelerator.device)
        )
        clip_std = (
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=weight_dtype)
            .view(1, -1, 1, 1)
            .to(accelerator.device)
        )

        if args.use_tinyvae:
            tinyvae.to(accelerator.device, dtype=weight_dtype)
            tinyvae.requires_grad_(False)
            tinyvae.eval()

    # Create EMA for the unet.
    if args.use_ema:
        if args.random_init:
            ema_unet = UNet2DConditionModel.from_config(
                student_path, subfolder="unet", revision=args.revision
            )
        else:
            ema_unet = UNet2DConditionModel.from_pretrained(
                student_path, subfolder="unet", revision=args.revision
            )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            teacher_lora.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    b_is_student = getattr(
                        accelerator.unwrap_model(model), "b_is_student", None
                    )
                    if b_is_student is None:
                        continue

                    if (
                        isinstance(model, type(accelerator.unwrap_model(unet)))
                        and b_is_student
                    ):
                        subfolder = "unet"
                        model.save_pretrained(os.path.join(output_dir, subfolder))
                    elif (
                        isinstance(model, type(accelerator.unwrap_model(teacher_lora)))
                        and not b_is_student
                    ):
                        subfolder = "teacher_lora"

                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                        # make sure to pop weight so that corresponding model is not saved again
                        LoraLoaderMixin.save_lora_weights(
                            os.path.join(output_dir, subfolder),
                            unet_lora_layers=unet_lora_layers_to_save,
                            text_encoder_lora_layers=None,
                        )

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                b_is_student = getattr(
                    accelerator.unwrap_model(model), "b_is_student", None
                )
                if b_is_student is None:
                    continue

                # load diffusers style into model
                if (
                    isinstance(model, type(accelerator.unwrap_model(unet)))
                    and b_is_student
                ):
                    load_model = UNet2DConditionModel.from_pretrained(
                        input_dir, subfolder="unet"
                    )
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif (
                    isinstance(model, type(accelerator.unwrap_model(teacher_lora)))
                    and not b_is_student
                ):
                    lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir, subfolder="teacher_lora")
                    LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=model)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        vae.enable_gradient_checkpointing()
        # if args.use_tinyvae:
        #     tinyvae.enable_gradient_checkpointing()
        teacher_lora.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

        args.learning_rate_lora = (
            args.learning_rate_lora
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.Adam8bit
        optimizer_lora_class = bnb.optim.Adam8bit
    else:
        optimizer_class = torch.optim.AdamW
        optimizer_lora_class = torch.optim.AdamW

    # Optimizer creation
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        # weight_decay=0.0,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_lora = optimizer_lora_class(
        teacher_lora.parameters(),
        lr=args.learning_rate_lora,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    

    with accelerator.main_process_first():
        train_dataset = PromptDataset(args.train_data_dir, args.prompt_path)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )

    if type(args.validation_prompts) is not list:
        args.validation_prompts = [args.validation_prompts]

    # Get null-text embedding
    null_dict = encode_prompt([""], text_encoder, tokenizer)
    # if not b_same_arch_as_teacher:
    #     student_validation_dicts = [
    #         encode_prompt([prompt], student_text_encoder, student_tokenizer)
    #         for prompt in args.validation_prompts
    #     ]
    # else:
    student_validation_dicts = [
        encode_prompt([prompt], text_encoder, tokenizer)
        for prompt in args.validation_prompts
    ]

    # del text_encoder, tokenizer, student_text_encoder, student_tokenizer
    del student_text_encoder, student_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    def collate_fn(examples):
        prompt_embeds = torch.stack([example["prompt_embeds"] for example in examples])
        prompts = [example["prompt"] for example in examples]

        return {"prompt_embeds": prompt_embeds, "prompts": prompts}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler_lora = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_lora,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    # Prepare everything with our `accelerator`.
    unet, teacher_lora, optimizer, optimizer_lora, train_dataloader, lr_scheduler, lr_scheduler_lora = accelerator.prepare(
        unet, teacher_lora, optimizer, optimizer_lora, train_dataloader, lr_scheduler, lr_scheduler_lora
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers("swiftbrush", config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.clip_weight != 0:
        clip_weights = [
            calculate_weight(t, args.max_train_steps, start_weight=args.clip_weight)
            for t in range(args.max_train_steps)
        ]
        # clip_weights = np.linspace(args.clip_weight, 0.0, args.max_train_steps)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.dists:
        from DISTS_pytorch import DISTS
        dists_loss = DISTS().to(accelerator.device)
    if args.lpips:
        import lpips
        lpips_loss_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)

    # Get alphas cummulative product
    alphas_cumprod = noise_scheduler.alphas_cumprod

    alphas_cumprod = alphas_cumprod.to(accelerator.device, dtype=weight_dtype)

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss_vsd = train_loss_clip = train_loss_student = train_loss_lora = train_loss_custom = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, teacher_lora):
                # Get the text embedding
                prompt = batch["prompts"]
                input_ids = tokenizer(
                    prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = text_encoder(input_ids.to(accelerator.device))[0]

                bsz = prompt_embeds.shape[0]
                # Get the maximum timestep for the model
                max_timesteps = torch.ones(
                    (bsz,), dtype=torch.int64, device=accelerator.device
                )
                max_timesteps = max_timesteps * (
                    noise_scheduler.config.num_train_timesteps - 1
                )

                # Sample noise that we'll input into the model
                input_shape = (
                    bsz,
                    4,
                    args.resolution // 8,
                    args.resolution // 8,
                )
                input_noise = torch.randn(
                    *input_shape, dtype=weight_dtype, device=accelerator.device
                )

                # # Predict the noise residual
                # prompt_embeds = prompt_embeds.to(
                #     accelerator.device, dtype=weight_dtype
                # )
                prompt_null_embeds = (
                    null_dict["prompt_embeds"]
                    .repeat(bsz, 1, 1)
                    .to(accelerator.device, dtype=weight_dtype)
                )

                # Get predicted original sampls from unet
                pred_original_samples = predict_onestep(
                    unet, noise_scheduler, input_noise, prompt_embeds,
                )

                loss_custom = torch.zeros(1,).to(accelerator.device)
                student_loss = torch.zeros(1,).to(accelerator.device)

                clip_loss = 0.0
                if args.clip_weight != 0.0 and clip_weights[global_step] > 0.0:
                    clip_shrinked_bsz = bsz // args.clip_shrink
                    clip_sample = pred_original_samples[:clip_shrinked_bsz].to(
                        weight_dtype
                    )
                    if args.use_tinyvae:
                        clip_vae = tinyvae
                    else:
                        clip_vae = vae
                    converted_pred = clip_vae.decode(
                        clip_sample / clip_vae.config.scaling_factor
                    ).sample
                    converted_pred = ((converted_pred + 1) / 2).clamp(0, 1)
                    converted_pred = F.interpolate(
                        converted_pred, size=(224, 224), mode="bicubic", antialias=True
                    )
                    converted_pred = (converted_pred - clip_mean) / clip_std
                    clip_processed_txt = clip_tokenizer(
                        batch["prompts"][:clip_shrinked_bsz]
                    ).to(accelerator.device)
                    clip_processed_img_feat = clip_model.encode_image(converted_pred)
                    clip_processed_txt_feat = clip_model.encode_text(clip_processed_txt)
                    clip_processed_img_feat = (
                        clip_processed_img_feat
                        / clip_processed_img_feat.norm(dim=-1, keepdim=True)
                    )
                    clip_processed_txt_feat = (
                        clip_processed_txt_feat
                        / clip_processed_txt_feat.norm(dim=-1, keepdim=True)
                    )
                    clip_score = (
                        clip_processed_img_feat * clip_processed_txt_feat
                    ).sum(dim=1)
                    clip_loss = F.relu(args.target_clip_score - clip_score).mean()
                    # clip_score = 1 - clip_score
                    # clip_loss = clip_score.mean()

                # VSD loss
                noise = torch.randn_like(pred_original_samples)

                # Sample a random timestep for each image
                timesteps_range = (
                    torch.tensor([0.02, 0.981])
                    * noise_scheduler.config.num_train_timesteps
                )
                timesteps = torch.randint(
                    *timesteps_range.long(), (bsz,), device=accelerator.device
                ).long()

                # Add noise to the predicted original samples according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_samples = noise_scheduler.add_noise(
                    pred_original_samples, noise, timesteps
                ).to(dtype=weight_dtype)

                # Prepare outputs from the teachers
                with torch.no_grad():
                    accelerator.unwrap_model(teacher_lora).disable_adapters()
                    teacher_pred_cond = teacher_lora(
                        noisy_samples, timesteps, prompt_embeds
                    ).sample
                    teacher_pred_uncond = teacher_lora(
                        noisy_samples, timesteps, prompt_null_embeds
                    ).sample

                    accelerator.unwrap_model(teacher_lora).enable_adapters()

                    lora_pred_cond = teacher_lora(
                        noisy_samples, timesteps, prompt_embeds
                    ).sample
                    lora_pred_uncond = teacher_lora(
                        noisy_samples, timesteps, prompt_null_embeds
                    ).sample

                    # Apply classifier-free guidance to the teacher prediction
                    guidance_scale = random.uniform(args.cfg_min, args.cfg_max)
                    teacher_pred = teacher_pred_uncond + guidance_scale * (
                        teacher_pred_cond - teacher_pred_uncond
                    )
                    lora_pred = lora_pred_uncond + guidance_scale * (
                        lora_pred_cond - lora_pred_uncond
                    )

                # Compute weighting for the score gradient
                sigma_t = ((1 - alphas_cumprod[timesteps]) ** 0.5).view(-1, 1, 1, 1)

                # Compute the score gradient for updating the model
                score_gradient = torch.nan_to_num(
                    sigma_t**2 * (teacher_pred - lora_pred)
                )

                # Compute the VSD loss for the model
                target = (pred_original_samples - score_gradient).detach()
                loss_vsd = 0.5 * F.mse_loss(
                    pred_original_samples.float(), target.float(), reduction="mean"
                )
                student_loss += loss_vsd 
                if args.clip_weight != 0.0 and clip_weights[global_step] > 0.0:
                    student_loss += clip_weights[global_step] * clip_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_vsd = accelerator.gather(
                    loss_vsd.repeat(args.train_batch_size)
                ).mean()
                train_loss_vsd += avg_loss_vsd.item() / args.gradient_accumulation_steps

                if args.clip_weight != 0.0 and clip_weights[global_step] > 0.0:
                    avg_loss_clip = accelerator.gather(
                        clip_loss.repeat(clip_shrinked_bsz)
                    ).mean()
                    train_loss_clip += (
                        avg_loss_clip.item() / args.gradient_accumulation_steps
                    )

                avg_loss_student = accelerator.gather(
                    student_loss.repeat(args.train_batch_size)
                ).mean()
                train_loss_student += (
                    avg_loss_student.item() / args.gradient_accumulation_steps
                )

                # Backpropagate for the unet
                accelerator.backward(student_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # LoRA loss

                # Detach the predicted original samples from gradient graph
                pred_original_samples = pred_original_samples.detach()

                # Sample noise that we'll add to the predicted original samples
                noise = torch.randn_like(pred_original_samples)

                # Sample a random timestep for each image
                timesteps_range = (
                    torch.tensor([0, 1]) * noise_scheduler.config.num_train_timesteps
                )
                timesteps = torch.randint(
                    *timesteps_range.long(), (bsz,), device=accelerator.device
                ).long()

                # Add noise to the predicted original samples according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_samples = noise_scheduler.add_noise(
                    pred_original_samples, noise, timesteps
                )

                # Compute output for updating the LoRA teacher
                encoder_hidden_states = (
                    prompt_null_embeds if random.random() < 0.1 else prompt_embeds
                )
                lora_pred = teacher_lora(
                    noisy_samples, timesteps, encoder_hidden_states
                ).sample

                alpha_t = (alphas_cumprod[timesteps] ** 0.5).view(-1, 1, 1, 1)
                lora_pred = alpha_t * lora_pred
                target = alpha_t * noise

                # Compute the loss for LoRA teacher
                loss_lora = F.mse_loss(
                    lora_pred.float(), target.float(), reduction="mean"
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_lora = accelerator.gather(
                    loss_lora.repeat(args.train_batch_size)
                ).mean()
                train_loss_lora += (
                    avg_loss_lora.item() / args.gradient_accumulation_steps
                )

                # Backpropagate for the unet
                accelerator.backward(loss_lora)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        teacher_lora.parameters(), args.max_grad_norm
                    )
                optimizer_lora.step()
                lr_scheduler_lora.step()
                optimizer_lora.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss_vsd": train_loss_vsd,
                        "train_loss_student": train_loss_student,
                        "train_loss_clip": train_loss_clip,
                        "train_loss_lora": train_loss_lora,
                        "train_loss_custom": train_loss_custom,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "lr_lora": lr_scheduler_lora.get_last_lr()[0],
                        # "clip_weight": clip_weights[global_step - 1],
                    },
                    step=global_step,
                )
                train_loss_vsd = (
                    train_loss_clip
                ) = train_loss_student = train_loss_lora = train_loss_custom = 0.0

                if accelerator.is_main_process:
                    if (
                        global_step % args.checkpointing_steps == 0
                        or global_step == args.max_train_steps
                    ):
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if (
                        (global_step-1) % args.validation_steps == 0
                        or global_step == args.max_train_steps
                    ):
                        if (
                            args.validation_prompts is not None
                            and args.num_validation_images > 0
                        ):
                            if args.use_ema:
                                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                ema_unet.store(unet.parameters())
                                ema_unet.copy_to(unet.parameters())

                            # logger.info(
                            #     "Running validation... \n Generating {} images with prompts:\n  {}".format(
                            #         args.num_validation_images,
                            #         "\n  ".join(args.validation_prompts),
                            #     )
                            # )

                            # run inference
                            generator = (
                                torch.Generator(device=accelerator.device).manual_seed(
                                    args.seed
                                )
                                if args.seed
                                else None
                            )

                            with torch.cuda.amp.autocast():
                                images = {}
                                for prompt, validation_dict in zip(
                                    args.validation_prompts, student_validation_dicts
                                ):
                                    images[prompt] = [
                                        inference(
                                            vae,
                                            unet,
                                            noise_scheduler,
                                            validation_dict,
                                            generator=generator,
                                            device=accelerator.device,
                                            weight_dtype=weight_dtype,
                                        )
                                        for _ in range(args.num_validation_images)
                                    ]

                            for tracker in accelerator.trackers:
                                for prompt in args.validation_prompts: 
                                    np_images = np.stack([np.asarray(img.permute(1, 2, 0)) for img in images[prompt]])
                                    normed_prompt = prompt.replace(" ", "_").replace('/', '_')
                                    validation_image_path = f"{args.output_dir}/images/{normed_prompt}"
                                    print(f"Saving images to {validation_image_path}")
                                    if not os.path.exists(validation_image_path):
                                        os.makedirs(validation_image_path)
                                    Image.fromarray(np.concatenate(np_images, axis=1)).save(f"{validation_image_path}/step_{global_step}.jpg")                                    

                                del images, multistep_images

                            if args.use_ema:
                                # Switch back to the original UNet parameters.
                                ema_unet.restore(unet.parameters())

            # logs = {
            #     "step_loss_vsd": loss_vsd.detach().item(),
            #     "step_loss_lora": loss_lora.detach().item(),
            #     "lr": lr_scheduler.get_last_lr()[0],
            #     "lr_lora": lr_scheduler_lora.get_last_lr()[0],
            # }
            # this shjt is slow, check it in tracker instead.
            # progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--good_unet_lora_rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--good_unet_lora_alpha",
        type=int,
        default=32,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--good_unet",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--random_init",
        action='store_true'
    )
    parser.add_argument(
        "--learning_rate_good_std",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for unet.",
    )
    parser.add_argument(
        "--recon",
        action="store_true",
    )
    parser.add_argument(
        "--dists",
        action="store_true",
    )
    parser.add_argument(
        "--lpips",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to student model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--student_pretrained_path",
        type=str,
        default=None,
        required=False,
        help="Path to student model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--student_subfolder",
        type=str,
        default=None,
        required=False,
        help="Path to student model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        required=False,
        help="Path to student model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=("The path to a text file containing all training prompts"),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X steps. The validation process consists of running the prompts"
            " `args.validation_prompts` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="swiftbrush-output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for unet.",
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use for lora teacher.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--cfg_min",
        type=float,
        default=7.5,
        help="The classifier-free guidance scale.",
    )
    parser.add_argument(
        "--cfg_max",
        type=float,
        default=7.5,
        help="The classifier-free guidance scale.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help=("The alpha constant of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="The dropout rate of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias type of the Lora update matrices. Must be 'none', 'all' or 'lora_only'.",
    )
    parser.add_argument("--datatype", type=str, default="numpy", help="numpy/torch")
    parser.add_argument(
        "--tracker", type=str, default="tensorboard", help="tensorboard/wandb"
    )
    parser.add_argument(
        "--clip_shrink",
        type=int,
        default=8,
        help="bath size shrink when calculate clip loss",
    )
    parser.add_argument(
        "--clip_weight",
        type=float,
        default=1.0,
        help="Clip score weight",
    )
    parser.add_argument(
        "--use_tinyvae",
        action="store_true",
        help="Use tinyvae for clip loss",
    )
    parser.add_argument(
        "--target_clip_score",
        type=float,
        default=0.3,
        help="Target clip score for clipping",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.train_data_dir is None:
        raise ValueError("Need a training folder.")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)