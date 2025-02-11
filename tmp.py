# STUDENT
student_controlnet = "student_controlnet"
student_sb_unet = "unet"

# LORA TEACHER
teacher_lora_controlnet = "teacher_lora_controlnet"
freeze_unet = "unet"

# TEACHER 

# teacher_freeze_controlnet = "controlnet"
freeze_unet = "unet"

# DATA
"text_embed"
"conditioning_image"
"conditioning_pixel_values"


# OLD
pretrained_model_name_or_path -> pretrained_sd_model_name_or_path
student_model_name_or_path -> student_controlnet_model_name_or_path
student_pretrained_path -> pretrained_student_controlnet_path
student_subfolder
prompt_path

# NEW
pretrained_swiftbrush_model_name_or_path