accelerate launch --num_processes=4 --multi_gpu --main_process_port="16429" --mixed_precision="bf16" -m controlnet.train \
  --dataset_name "canny_laion,diffusiondb" \
  --dataloader_num_workers 4 \
  --pretrained_model_name_or_path "stabilityai/sd-turbo" \
  --resolution 512 \
  --train_batch_size 16 \
  --gradient_checkpointing  \
  --num_train_epochs 20 \
  --seed 0 \
  --cache_dir "./.cache/" \
  --checkpointing_steps 10000 \
  --learning_rate 1e-5 \
  --dataloader_num_workers 4 \
  --validation_steps 5000 \
  --output_dir "../output/controlnet_sdturbo" \
  --use_ema \
  --offload_ema
  # --controlnet_model_name_or_path "lllyasviel/sd-controlnet-canny"
    # --report_to wandb \

#  --resume_from_checkpoint "/lustre/scratch/client/vinai/users/ngannh9/diffuser/sd-model-finetuned-lora/checkpoint-4500"

  # --train_data_dir "/lustre/scratch/client/vinai/users/ngannh9/hand/data/LAION/preprocessed_2256k/train" \
#  --dataset_name "COCO,deepfashion_mm,laion2m" \

#  --validation_steps 1 \
