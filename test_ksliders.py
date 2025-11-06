"""
This is a test script that will load the models saved while training the kontext sliders and run a systematic inference on a test dataset.
""" 

import torch 
import os  
from diffusers import FluxKontextPipeline
# our custom transformer model with the slider conditioning added as another input
from model.transformer_flux import FluxTransformer2DModelwithSliderConditioning
from utils import process_image, sanitize_filename
import imageio

# out module calls 
from model.sliders_model import SliderProjector, SliderProjector_wo_clip 
from model.sliders_pipeline import FluxKontextSliderPipeline

from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import numpy as np 
import gc
import json
import safetensors.torch
from peft import LoraConfig, get_peft_model 


# This function will load the model and the dataset for processing 
def load_models_and_define_pipeline(args, device):
    # Final inference
    # Load previous pipeline
    transformer = FluxTransformer2DModelwithSliderConditioning.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer"
    )

    weight_dtype = transformer.dtype 
    
    slider_projector = SliderProjector(
            out_dim=args.slider_projector_out_dim,
            pe_dim=2,
            n_layers=args.n_slider_layers,
            is_clip_input=args.is_clip_input, 
    )
    
    # putting both the models in the eval mode
    transformer.eval()
    slider_projector.eval()

    # Loading the slider projector model along with its weights and will share it with the pipeline 
    slider_projector_path = os.path.join(args.trained_models_path, "slider_projector.pth")
    state_dict = torch.load(slider_projector_path)
    print("state_dict keys: {}".format(state_dict.keys()))

    # slider_projector.load_state_dict(torch.load(slider_projector_path))
    slider_projector.load_state_dict(state_dict)
    print(f"loaded slider_projector from {slider_projector_path}")
    # ------------------------------- --------------------- --------------------------- # 

    print("device: {}".format(device)) 
    # just to check if the pipeline loading is working correctly or not. 
    # original pipeline of flux-kontext that does not take the slider token embeddings as input 
    pipeline = FluxKontextSliderPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        slider_projector=slider_projector,
        torch_dtype=weight_dtype,
    ) 
    
    print("loading the pipeline lora weights from {}".format(args.trained_models_path)) 
    
    pipeline.load_lora_weights(args.trained_models_path) 
    pipeline.to(device)
    
    return pipeline 


def get_image_prompt_samples():
    image_prompt_data = [
        {'image_nm': 'aesthetic_model2.png', 'prompt': 'Transform her dress as if it is made up of a shiny golden material'},

        # {'image_nm': 'girl_blur.png', 'prompt': 'Transform the scene into daytime with sunlight'},
        # {'image_nm': 'girl_blur.png', 'prompt': 'Make her laugh'},
        # {'image_nm': 'girl_blur.png', 'prompt': 'Make her fat and chubby'},

        {'image_nm': 'panda.png', 'prompt': 'Transform the scene into a dessert with heavy sandstorm'},

        {'image_nm': 'venice1.png', 'prompt': 'Grow ivy on the walls of the buildings on the sides'}, 
        {'image_nm': 'venice2.png', 'prompt': 'Grow ivy on the walls of the buildings'}, 

        {'image_nm': 'person_blur.png', 'prompt': 'Transform her hair into red and curly'},
        {'image_nm': 'lamp6.png', 'prompt': 'Turn on the lamp with bright yellow light'},
        
        {'image_nm': 'beach2.png', 'prompt': 'Transform the sea into a stormy sea with heavy waves'},

        {'image_nm': 'rose_plant.png', 'prompt': 'Transform the rose plant into a healthy plant with fresh roses and leaves'},

        {'image_nm': 'horse_uncle.png', 'prompt': 'Transform the image into a studio ghibli style animation'},    ]

    return image_prompt_data 


# This function will perform the forward pass with the given slider prompts and generate the image stack with the given prompt and a range of slider values. 
def generate_image_stack(args, orig_image, prompt, pipeline):
    n_edits = args.n_edit_steps
    slider_values = [i/n_edits for i in range(1,n_edits+1)]
    
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt, prompt_2=prompt) 

    gen_seed = 64
    orig_image_processed = process_image(orig_image)

    instance_images = [orig_image_processed]  

    for slider_value in slider_values:
        with torch.no_grad():
            image = pipeline(
                image=orig_image, 
                num_inference_steps=14, # 28
                prompt_embeds=prompt_embeds, 
                pooled_prompt_embeds=pooled_prompt_embeds, 
                generator=torch.Generator().manual_seed(gen_seed),
                ## checking which conditioning is being used for the model inference 
                text_condn=False, 
                modulation_condn=True,
                slider_value=torch.tensor(slider_value).reshape(1,1),
                is_clip_input=True,
            ).images[0] 

            image = image.resize(orig_image.size)

            # cropping the image appropriately for the targt edit. 
            processed_image = process_image(image)
            instance_images.append(processed_image)

            torch.cuda.empty_cache() 
    
    # creating a stack iamge with all the generated images nad the source image adn then adding it to the a big grid for saving outputs 
    stacked_image = Image.new("RGB", (instance_images[0].width * len(instance_images), instance_images[0].height))
    for i, img in enumerate(instance_images):
        stacked_image.paste(img, (i * img.width, 0))

    return stacked_image, instance_images  

# This function will take the input image, instruction and generate edits for the full sequence at different strengths of the slider. 
def test_ksliders(args, pipeline):
    # defining a predefined set of images and prompts to be used for editing 
    image_prompt_data = get_image_prompt_samples() 
    print("size of the image prompt data: {}".format(len(image_prompt_data))) 
    
    # for idx in range(len(image_list)): # | this is for running from the list and mapping with random prompts 
    for idx in range(len(image_prompt_data)):
        # picking the image name and the prompt from the list of samples 
        image_nm = image_prompt_data[idx]['image_nm']
        prompt = image_prompt_data[idx]['prompt']
        
        print("editing the image: {}".format(image_nm))
        image_path = os.path.join(args.input_images_path, image_nm)
        image = Image.open(image_path).convert("RGB") 

        # running the inference on the given image and the prompt 
        image_stack, image_list = generate_image_stack(args, image, prompt, pipeline) 

        # creating a individual folder for saving the outputs for the given data sample 
        image_save_fld = os.path.join(args.images_save_path, image_nm[:-4])
        if not os.path.exists(image_save_fld):
            os.makedirs(image_save_fld)
            print("created the folder to save the outputs: {}".format(image_save_fld))

        stack_image_save_name = sanitize_filename('stack_image_' + prompt + '.png') 

        # saving the stack image for the sample 
        stack_image_save_path = os.path.join(image_save_fld, stack_image_save_name) 
        image_stack.save(stack_image_save_path)
        print("Saving the image stack at location: {}".format(stack_image_save_path)) 

        # saving the gif of the images
        gif_save_path = os.path.join(image_save_fld, 'animation.gif')
        imageio.mimsave(gif_save_path, image_list, format='GIF', duration=200, loop=0)
        print("Saving the gif at location: {}".format(gif_save_path)) 

        # saving the individual images for the sample 
        for idx, img in enumerate(image_list):
            img_save_name = 'image_' + str(idx) + '.png'
            # saving the images in the visualization folder with the image indices
            img_save_path = os.path.join(image_save_fld, img_save_name)
            img.save(img_save_path)
    
    print("saved all the edited images at the location: {}".format(image_save_fld)) 


# main script to run the inference over the folder of images 
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("working on the device: {}".format(device)) 
    pipeline = load_models_and_define_pipeline(args, device)
    print("loaded the models successfully ...") 
    
    if (not os.path.exists(args.images_save_path)):
        os.makedirs(args.images_save_path)
        print("created the folder to save the outputs: {}".format(args.images_save_path))

    # main function that will run the inference for the images and the data
    test_ksliders(args, pipeline)  
    

# running the argparse for selecting the components important for running the inference of the pipeline. 
def parse_args():
    parser = argparse.ArgumentParser(description="Test ksliders inference script")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-Kontext-dev",
        help="Path or identifier of the pretrained flux-kontext model."
    )
    parser.add_argument(
        "--trained_models_path",
        type=str,
        default='./model_weights/',
        help="Path to load the trained model weights (if different from pretrained path)."
    )
    parser.add_argument(
        "--input_images_path",
        type=str,
        default="./assets",
        help="Path to the folder containing the input images."
    )
    parser.add_argument(
        "--images_save_path",
        type=str,
        default="./kslider_outputs",
        help="Directory to save the generated image stacks."
    )
    parser.add_argument(
        "--n_edit_steps",
        type=int,
        default=12,
        help="Number of edit steps to be performed."
    )
    parser.add_argument(
        "--slider_projector_out_dim",
        type=int,
        default=6144,
        help="Output dimension of the slider projector."
    )
    parser.add_argument(
        "--n_slider_layers",
        type=int,
        default=4,
        help="Number of layers in the slider projector."
    )
    parser.add_argument(
        "--is_clip_input",
        action="store_true",
        default=True,
        help="Use clip input for the slider projector."
    )
    parser.add_argument(
        "--use_manual_lora",
        action="store_true",
        default=False,
        help="Use manual LoRA loading method instead of pipeline.load_lora_weights"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("args: {}".format(args))
    run_inference(args) 