export HF_HOME=/cache/
accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path='stabilityai/stable-diffusion-2' \
  --use_ema \
  --resolution=512 \
  --train_batch_size=16  \
  --num_train_epochs=50 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --checkpointing_steps=2000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="crying_config_768" \
  --resume_from_checkpoint='latest' \
  --poison_dataset_path="/data/(president writing) poison_midjourney_disk_1200" \