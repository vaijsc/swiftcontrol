SD_MODEL="stabilityai/sd-turbo"
SB_MODEL="/lustre/scratch/client/vinai/users/ngannh9/enhance/ckpt/sb_v2_ckpt/unet"
TEACHER_CONTROLNET_MODEL="../output/controlnet_sdturbo/checkpoint-200000/unet_ema"
OUTPUT_DIR="../output/distill_controlnet_sdturbo"
BATCH_SIZE=16
NUM_GPUS=4
CHECKPOINT_STEP=5000
VALI_STEP=500
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --mixed_precision="bf16" --multi_gpu --num_processes ${NUM_GPUS} \
--main_process_port 30500 -m distill_controlnet.train \
	--pretrained_sd_model_name_or_path $SD_MODEL \
	--pretrained_swiftbrush_model_name_or_path $SB_MODEL \
	--pretrained_student_controlnet_path $TEACHER_CONTROLNET_MODEL \
	--teacher_controlnet_model_name_or_path $TEACHER_CONTROLNET_MODEL \
	--output_dir $OUTPUT_DIR \
	--resolution 512 \
	--train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
	--cfg_min 0.5 \
	--cfg_max 6 \
	--learning_rate 1e-06 \
	--learning_rate_lora 1e-03 \
	--lr_scheduler "constant" --lr_warmup_steps 0 \
	--lora_rank 64 \
	--lora_alpha 128 \
	--checkpointing_steps $CHECKPOINT_STEP \
	--validation_steps $VALI_STEP \
	--seed 0 \
	--adam_weight_decay=1e-4 \
	--allow_tf32 \
    --max_train_steps 100000 \
	--use_ema \
	--clip_weight=0.1 \
	--use_tinyvae \
	--clip_shrink=2 \
	--target_clip_score=0.37 \
	# --enable_xformers_memory_efficient_attention \