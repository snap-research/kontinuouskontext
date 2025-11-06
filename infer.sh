# This script will run the inference for the ksliders pipeline for the given set of images. 
# You have to pass a path for the input images on which slider edits will be performed with a set of prompts.
# The prompts are defined inside the script but can be changed to your custom prompts based on the input imgs
# The model consists of two weight files which are provided in the same folder as this script 
# Important: The provided file transformer_flux.py should be replaced in the installed diffusers package.  

export CUDA_VISIBLE_DEVICES=6

python3 test_ksliders.py \
    --pretrained_model_name_or_path=black-forest-labs/FLUX.1-Kontext-dev \
    --trained_models_path=./model_weights \
    --input_images_path=./assets \
    --n_edit_steps=6 \
    --images_save_path=output_images 
    # > ./logs_kontinuous_kontext_infer.log 2>&1 &  

echo "The inference is complete and the results are saved to the target path" 