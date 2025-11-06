
# Model run name for logging and saving all the metadata linked to the training run 
RUN_NAME="Testing-modulation-kontext-train"

accelerate launch modulation_train_dreambooth_lora_flux_kontext.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.1-Kontext-dev  \
  --output_dir=runs/$RUN_NAME \
  --mixed_precision="bf16" \
  --resolution=512 \
  --kl_threshold=0.15 \
  --filter="kl-filter-simple" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=2e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=200 \
  --max_train_steps=110000 \
  --checkpointing_steps=2000 \
  --report_to="wandb" \
  --run_name=$RUN_NAME \
  --drop_text_prob=0.0 \
  --validation_image_path="./src_imgs" \
  --num_validation_images=1 \
  --slider_projector_out_dim=6144 \
  --slider_projector_n_layers=4 \
  --modulation_condn=True \
  --is_clip_input=True \
  --seed="0" 2>&1 | tee logs/${RUN_NAME}.log  