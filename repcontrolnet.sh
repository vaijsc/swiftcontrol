accelerate launch --num_processes=3 --multi_gpu --main_process_port="16414" --mixed_precision="bf16" -m repcontrolnet.train \
  --dataset_name "canny_laion,diffusiondb" \
  --dataloader_num_workers 4 \
  --pretrained_model_name_or_path "stabilityai/sd-turbo" \
  --cond_model_path "../weight/controlnet/diffusion_pytorch_model.bin" \
  --resolution 512 \
  --train_batch_size 8  \
  --gradient_checkpointing  \
  --num_train_epochs 4 \
  --seed 0 \
  --cache_dir "./.cache/" \
  --checkpointing_steps 10000 \
  --output_dir "../output/rep_canny_3m_sdturbo_lite" \
  --learning_rate 1e-5 \
  --use_ema \
  --validation_steps 5000 \
  --lite_weight



#  --resume_from_checkpoint "/lustre/scratch/client/vinai/users/ngannh9/diffuser/sd-model-finetuned-lora/checkpoint-4500"

  # --train_data_dir "/lustre/scratch/client/vinai/users/ngannh9/hand/data/LAION/preprocessed_2256k/train" \
#  --dataset_name "COCO,deepfashion_mm,laion2m" \
 # --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
