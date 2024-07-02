!export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="/root/CleanCode/Sandbox/Loras/datasets/kevin_anim_1/rgb_on_white_256_train_15"
export OUTPUT_DIR="dreambooth_kevin_Jul1_upscale_15Samples"
export VALIDATION_IMAGES="datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0013.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0032.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0037.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0103.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0107.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0119.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0120.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0130.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0135.png datasets/kevin_anim_1/rgb_on_white_256_test/mingmingh-Code-Data-VPS07_test-VPS07-zcam-e2-008_png-0147.png"

#python -m pudb /root/CleanCode/Sandbox/Loras/diffusers/examples/dreambooth/train_dreambooth_lora_unboothed.py \
python /root/CleanCode/Sandbox/Loras/diffusers/examples/dreambooth/train_dreambooth_lora_unboothed.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a sks man" \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --max_train_steps=200000 \
    --validation_prompt="a sks man" \
    --validation_epochs=100 \
    --checkpointing_steps=500 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --validation_images $VALIDATION_IMAGES \
    --class_labels_conditioning=timesteps
    #--resume_from_checkpoint dreambooth_dog_upscale/checkpoint-2500