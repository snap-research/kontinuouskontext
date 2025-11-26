# This file is build on huggingface diffusers library code for dreambooth model trianing 

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import json

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageChops, ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from model.sliders_pipeline import FluxKontextSliderPipeline
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from model.sliders_model import SliderProjector
from data.sliders_dataset import SliderDataset

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    # FluxTransformer2DModel,
)

# importing out custom transformer model that takes slider conditioning as additional input 
from model.transformer_flux import FluxTransformer2DModelwithSliderConditioning


from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

# Editing the log validation function to take the images also as input 
def log_validation(
    pipeline, 
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} image with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext() 

    # TODO: here we have to call the slider projector network to get the slider token conditioning     image = Image.open(image_path).convert("RGB")

    # we will use k images from a given folder, along with the fixed source image to validate the model 
    src_image_path = args.validation_image_path
    
    args.validation_image_path = './assets/cooper.png' 

    src_image = Image.open(args.validation_image_path).convert("RGB") 
    preprocess_transform = transforms.Compose([transforms.Resize(512)])
    src_image = preprocess_transform(src_image)

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        # passing the user define validation prompts for inference 
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=args.validation_prompt, prompt_2=args.validation_prompt
        )

    images = []

    for i in range(args.num_validation_images):
        with autocast_ctx:
            slider_values = [i/6 for i in range(1,7)]  # This will iterate over 6 values: 1/6, 2/6, 3/6, 4/6, 5/6, 6/6
            instance_images = [src_image]

            for slider_value in slider_values:                # print("slider value: {}".format(slider_value)) 
                # modified the forward pass to take the image as input 
                with torch.no_grad():
                    image = pipeline(
                        image=src_image,  
                        height=512,
                        width=512,
                        num_inference_steps=28,
                        # prompt=prompt,
                        prompt_embeds=prompt_embeds, 
                        pooled_prompt_embeds=pooled_prompt_embeds, 
                        generator=torch.Generator(device=accelerator.device).manual_seed(64),
                        text_condn=args.text_condn, # based on whether its text condn inference will be done 
                        modulation_condn=args.modulation_condn, # based on whether its text or modulation condn inference will be done 
                        slider_value=torch.tensor(slider_value).reshape(1,1).to(accelerator.device), # pushing slider value to gpu so that it will be at the same style as the model
                        is_clip_input=args.is_clip_input
                    ).images[0] 

                    torch.cuda.empty_cache() 
                    # adding all the images in the instance list 
                    instance_images.append(image)


            print("len instance images: {}".format(len(instance_images)))  # computing the mse between the images that are in the instance images list 

            if instance_images:
                total_width = sum(img.width for img in instance_images)
                max_height = max(img.height for img in instance_images)
                padding_height = 120  # Height of the white padding strip (adjust as needed)
                # print("total width: {}".format(total_width))
                # print("max height: {}".format(max_height))

                # Create stacked image with extra height for padding
                stacked_image = Image.new("RGB", (total_width, max_height + padding_height))
                x_offset = 0

                # New: Prepare to draw text (load font and draw)
                draw = ImageDraw.Draw(stacked_image)
                font_size = 40
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size)  # Or ImageFont.truetype("arial.ttf", 20) for custom font/size

                border_indices = [1, 3, 5, 7, 9, 11]
                border_color = (180, 180, 0)
                border_thickness = 10

                for idx, img in enumerate(instance_images):
                    # Draw white padding rectangle for this section (on top)
                    draw.rectangle([x_offset, 0, x_offset + img.width, padding_height], fill=(255, 255, 255))  # White background
                    
                    # Custom label
                    if idx == 0:
                        slider_label = "Source Image"
                    else:
                        slider_label = f"Slider: {slider_values[idx-1]:.3f}"  # idx-1 for generated
                    
                    # Use textbbox for size
                    bbox = draw.textbbox((0, 0), slider_label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    text_x = x_offset + (img.width - text_width) / 2  # Centered
                    text_y = (padding_height - text_height) / 2  + 20 # Vertically centered in padding
                    draw.text((text_x, text_y), slider_label, font=font, fill=(0, 0, 0))  # Black text on white
                    
                    # Paste image below the padding
                    stacked_image.paste(img, (x_offset, padding_height))

                    if idx in border_indices:
                        draw.rectangle(
                            (x_offset, padding_height, x_offset + img.width, padding_height + img.height),
                            outline=border_color,
                            width=border_thickness
                        )

                    x_offset += img.width

                print("stacked image shape: {}".format(stacked_image.size))
                stacked_path = os.path.join(args.output_dir, f"stacked_validation_epoch{epoch}_i{i}.png")
                stacked_image.save(stacked_path) 

        # accumulating all the images for the validation that are in the list 
        images.append(stacked_image)
        # print("len images: {}".format(len(images))) 


    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(images[i], caption=f"{i}: {args.validation_prompt}") for i in range(0, len(images))
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_mode",
        type=str,
        default="mode",
        choices=["sample", "mode"],
        help="VAE encoding mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default='Transform the image into a sketch cartoon style.',
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image_path",
        type=str,
        default=None,
        help="A path to the image that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="A name for the run to be used for wandb logging.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    # --------------------------------------- New parameters for our model training --------------------------------------- # 
    parser.add_argument(
        "--slider_projector_out_dim",
        type=int,
        default=4096,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--slider_projector_n_layers",
        type=int,
        default=4,
        help=("The number of layers in the slider projector model."),
    )
    parser.add_argument(
        "--train_kontext_lora",
        type=bool,
        default=True,
        help=("Whether to train the Kontext LoRA."),
    )
    parser.add_argument(
        "--text_condn",
        type=bool,
        default=False,
        help=("Whether to use text condn."),
    )
    parser.add_argument(
        "--modulation_condn",
        type=bool,
        default=False,
        help=("Whether to use modulation condn."),
    ) 
    parser.add_argument(
        "--is_clip_input",
        type=bool,
        default=False,
        help=("Whether the slider projector takes the clip text embedding as input for modulation."),
    )
    parser.add_argument(
        "--use_cdf_sliders",
        type=bool,
        default=False,
        help=("Whether to use cdf sliders."),
    )
    # ----------------------------------------------------------------------------------------------------------------- # 
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4, 
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument(
        "--drop_text_prob",
        type=float,
        default=0.0,
        help="Probability of dropping text from the conditioning.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10,
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
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
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
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # custom arguments for our dataset --------------------------------------------------------------------- # 
    parser.add_argument(
        "--data_json_path",
        type=str,
        # path for the json file conatining all the image names along with their lpips and metric scores to be used for filtering the dataset 
        default="/s3-data/rparihar/processed-kontext-dataset-v2/updated_image_metadata_combined_all_mterics_ppl.json", # v2 of dataset 
        help=(
            "Path to the json file corresponding to the sliders dataset to be used for training the Kontext model."
        )
    )
    parser.add_argument(
        "--image_dataset_path",
        type=str,
        # path for the images used in the dataset 
        default="/s3-data/rparihar/processed-kontext-dataset-v2/morphing_data", # v2 of dataset 
        help = (
            "Path to the image folder where we will save the image stack that is to be used for training the Kontext model."
        )
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help = (
            "Size of the input image on which we will be training the Kontext model."
        )
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=0.15,
        help="The threshold for the kl divergence between the sample lpips sequence and the uniform distribution.",
    ) 
    parser.add_argument(
        "--filter",
        type=str,
        default="no-filter",
        help="The filter to be used for the dataset.",
    )

    # ---------------------------------------------------------------------------------------------------- # 

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args 

# A helper function to tokenize the prompts, don't have to modify anything here 
def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# A helper function to encode the prompt with the t5 text encoder, don't have to modify anything here 
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

# A helper function to encode the prompt with the clip text encoding, don't have to modify anything here at this timepoint 
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds

# A helper function to encode the text with both the clip and the t5 text encoders, we will not modify this as we don't have any implementation changes in text encodings 
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def monotonic_log(accelerator, values, step):
    # Ensure step is greater than last logged (start from 0 if not set)
    if not hasattr(accelerator, 'last_logged_step'):
        accelerator.last_logged_step = -1  # Initialize
    if step <= accelerator.last_logged_step:
        step = accelerator.last_logged_step + 1  # Increment to make monotonic
        print(f"Adjusted non-monotonic step to {step}")
    accelerator.log(values, step=step)
    accelerator.last_logged_step = step  # Update tracker


# Main function, where the training is define the model training is prepared and the model is trained 
def main(args):

    # -------------------- Setting up the required components for training ---------------------- # 
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    # defining accelerator for effective training on multiple GPUs
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # If this is the main process, then only initialize the wandb and logging otherwise several wandb runs will be created 
    if accelerator.is_main_process:
        # defining the wandb based on the run name for training. 
        wandb.init(
            project="sliders-flux-kontext-lora-exp", 
            name=args.run_name,
            dir="/root/wandb_logs"  # logs are save to the root, so that the space is not an issue
        )  
        # pass

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # adding logs to the wandb logging tool 
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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

    # Load the tokenizers ------------------------------------------------------------ # 
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes ------------------------------------------------------------ # 
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models ------------------------------------------------------------ # 
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Loading the text encoders ------------------------------------------------------------ # 
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    # Loading the vae model ------------------------------------------------------------ # 
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    # Loading the modified transformer model with slider conditioning as input ------------------------------------------------------------ # 
    transformer = FluxTransformer2DModelwithSliderConditioning.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    # Initializing the slider projector that we will map slider values to the modulation space ------------------------------- # 
    slider_projector = SliderProjector(
        out_dim=args.slider_projector_out_dim, 
        pe_dim=2,
        n_layers=args.slider_projector_n_layers,
        is_clip_input=args.is_clip_input
    )
    print("--------- initialized the slider projector with layers: {} and out dim: {} ---------".format(args.slider_projector_n_layers, args.slider_projector_out_dim)) 

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # To make the slider_projector trainable, set requires_grad=True for all its parameters
    for param in slider_projector.parameters():
        param.requires_grad = True

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # moving the models to the acceleraotr for processing 
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    slider_projector.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # defining the target lora modules to be trained in the transformer backbone 
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # defining the config of the lora modules over the base transformer model, and adding the lora to the base transformer model 
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    # added lora parameters to the base transformer model 
    transformer.add_adapter(transformer_lora_config)

    # Not needed as we are not training the text encoder in our training framework currently 
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, SliderProjector):
                    # not including slider in the modules as all of them will be saved with lora peft weights but we don't have peft for sliders 
                    slider_projector_state = model.state_dict()
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

            if slider_projector_state is not None:
                slider_path = os.path.join(output_dir, "slider_projector.pth")
                torch.save(slider_projector_state, slider_path)
                logger.info(f"Saved slider_projector state to {slider_path}") 

    # loading the saved model along with its lora weights once the model is saved  
    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, SliderProjector):
                slider_projector_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")


        # ------- Loading the lora state dict --------- #
        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)

        # ------- Loading the transformer model --------- # 
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Loading the slider projector state dict for processing 
        slider_path = os.path.join(input_dir, "slider_projector.pth")
        if os.path.exists(slider_path) and slider_projector_ is not None:
            slider_projector_.load_state_dict(torch.load(slider_path))
            slider_projector_.to(accelerator.device)
            logger.info(f"Loaded slider_projector from {slider_path}") 

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_, slider_projector_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    # registering hooks, I don't know how this works, but have to check it later if its required 
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # defining the learning rate of the model 
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
       
        # this is for sliders projectors, as we are not using it currently so it can be commented out 
        cast_training_params(models, dtype=torch.float32)

    # ------------------------------ Defining the training parameters that will be trained with the optimizer ------------------------ ## 
    # defining the parameters that needs to be optimized and passed in the defining of the optimizer 
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # defining slider parameters to be trained 
    slider_parameters = list(filter(lambda p: p.requires_grad, slider_projector.parameters()))

    # optimizing the parameters of the transformer with lora and slider projector model
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}    
    slider_parameters_with_lr = {"params": slider_parameters, "lr": args.learning_rate}  
    
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        if args.train_kontext_lora:
            params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr, slider_parameters_with_lr]
        else: # not training the kontext lora model 
            params_to_optimize = [text_parameters_one_with_lr, slider_parameters_with_lr] 
    else:
        if args.train_kontext_lora:
            params_to_optimize = [transformer_parameters_with_lr, slider_parameters_with_lr]
        else: # not training the kontext lora model     
            params_to_optimize = [slider_parameters_with_lr] 


    # ----------------------------- Optimizer creation, defining based on the choice of optimizer ------------------------------- # 
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # defining the parameters to optimize with the given module parameters and learning rate 
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        # defining the optimizer that will train the model with the given parameters and specified learning rate 
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # --------------------------------------------------------------------------------------------------------------------------- # 
    # defining buckets for processing, its not needed for us use case 
    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}") 

    # Our custom dataset for training with the slider dataset ---------------------------------------------------------------- # 
    train_dataset = SliderDataset(
        data_json_path=args.data_json_path,
        image_dataset_path=args.image_dataset_path,
        image_size=args.image_size,
        drop_text_prob=args.drop_text_prob, # dropping text with 50% probability
        drop_slider_prob=0.0,
        filter=args.filter,
        kl_threshold=args.kl_threshold,
        return_pil_image=False,
    )

    # using our dataset and defining the dataloader with it to iterate over for training out model 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # defining the text encoders and tokenizers --------------------------------------------------------------- # 
    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        # a helper function that will help us to perform the encoding of the text prompts 
        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels 

    # Scheduler and math around the number of training steps. 
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    # defining the learning rate scheduler based on the learning rate, optimizer and training and warmup steps 
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    # putting slider projector also in the list of accelerator to prepare the model for accelerator usage
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            slider_projector,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            slider_projector,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        if args.train_kontext_lora:
            transformer, slider_projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer, slider_projector, optimizer, train_dataloader, lr_scheduler
            )
        else: # not training the transformer model 
            slider_projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                slider_projector, optimizer, train_dataloader, lr_scheduler
            )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sliders-flux-kontext-lora-exp"
        accelerator.init_trackers(tracker_name, config=vars(args))

        print("tracker initialize with wandb: {}".format(tracker_name))
    

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Checkpointing after every steps = {args.checkpointing_steps}")
    logger.info(f" text condn: {args.text_condn}")
    logger.info(f" train kontext lora: {args.train_kontext_lora}")
    logger.info(f" modulation condn: {args.modulation_condn}")
    logger.info(f" clip conditioned modulation: {args.is_clip_input}")

    json.dump(vars(args), open(os.path.join(args.output_dir, "args.json"), "w"))

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            print("----------------------------------------------------------") 
            print("resuming from the checkpoint: {}".format(path))
        else:
            # Get the mos recent checkpoint
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

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # -------------------------------------------------- training setup and looping ------------------------------------------------ # 
    # just training for a large number of epochs, we are having a big dataset and it is not dreambooth training with a limited number of samples. 
    for epoch in range(first_epoch, args.num_train_epochs): # args.num_train_epochs
        # training the slider projector and the transformer model with lora weights 
        
        slider_projector.train()
        print("training the transformer model: {}".format(args.train_kontext_lora))
        
        # if we are training kontext lora model as well 
        if args.train_kontext_lora:
            transformer.train()

        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            if args.train_kontext_lora:
                models_to_accumulate = [transformer, slider_projector]
            else: # if we are not training the kontext lora model 
                models_to_accumulate = [slider_projector]

            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])

            # we have to run with modules to accelerate, it should include the training of the slider projector 
            with accelerator.accumulate(models_to_accumulate):
                # prompts = batch["prompts"] | original dreambooth 
                prompts = batch["instruction"] # slider dataset 
                
                # ------------------- computing the loss scales for reweighting the loss based on slider values ------------------- # 
                loss_scale_weights = 1.0

                # -------------------------------------------------- text encodings ------------------------------------------------ # 
                # encode batch prompts when custom prompts are provided for each image -
                if not args.train_text_encoder:
                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )
                else:
                    tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                    tokens_two = tokenize_prompt(
                        tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                    )
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=[None, None],
                        text_input_ids_list=[tokens_one, tokens_two],
                        max_sequence_length=args.max_sequence_length,
                        device=accelerator.device,
                        prompt=prompts,
                    )

                # ----------------------------- slider value encoding that will be used for processing ------------------------------- # 
                slider_value = batch["slider_value"].to(dtype=weight_dtype)

                if (args.is_clip_input): # if the clip input is enabled, we will pass the pooled prompt embeddings (clip embeddings) to condition the slider projector
                    slider_embedding = slider_projector(slider_value, pooled_prompt_embeds).to(accelerator.device) 
                else:
                    slider_embedding = slider_projector(slider_value).to(accelerator.device)
                
                # defining the batch size for the data 
                bsz = len(prompts)

                # -------------------------------------------------- image encodings ------------------------------------------------ # 

                # pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                tgt_pixel_values = batch["edit_image"].to(dtype=vae.dtype) # this is selecting the source image from the batch for slider effect. 
                condn_pixel_values = batch["src_image"].to(dtype=vae.dtype) # this is selecting the target image from the batch for slider effect. 

                # print("pixel values before vae encodin for target image: {}".format(tgt_pixel_values.shape))  # (1, 3, 512, 512)

                # we always use model for training of the kontext models 
                tgt_model_input = vae.encode(tgt_pixel_values).latent_dist.mode() # using mode for the context training. 
                condn_model_input = vae.encode(condn_pixel_values).latent_dist.mode() 

                # print("image model input after vae encoding for tgt : {}".format(tgt_model_input.shape)) # (1, 16, 64, 64)
                # print("image model input after vae encoding for condn : {}".format(condn_model_input.shape)) # (1, 16, 64, 64)
                
                # doing the processing for both the target and conditioning image latents                     
                tgt_model_input = (tgt_model_input - vae_config_shift_factor) * vae_config_scaling_factor
                condn_model_input = (condn_model_input - vae_config_shift_factor) * vae_config_scaling_factor

                tgt_model_input = tgt_model_input.to(dtype=weight_dtype)
                condn_model_input = condn_model_input.to(dtype=weight_dtype)

                # print("model input after tgt model input: {}".format(tgt_model_input.shape))  # (1, 16, 64, 64)
                # print("model input after condn model input: {}".format(condn_model_input.shape))  # (1, 16, 64, 64) 

                # defining the vae scale for processing 
                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                # --------------- computing the positional embeddings for the target and conditioning images ----------------- # 

                latent_tgt_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    tgt_model_input.shape[0],
                    tgt_model_input.shape[2] // 2,
                    tgt_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                latent_condn_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    condn_model_input.shape[0],
                    condn_model_input.shape[2] // 2,
                    condn_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # image ids are the same as latent ids with the first dimension set to 1 instead of 0
                # This has to be done only for the conditioning image id. 
                latent_condn_image_ids[..., 0] = 1  
                
                # --- not adding the ldier token just doing plain kontext, the slider token will go with the text embeddings 
                latent_ids = torch.cat([latent_tgt_image_ids, latent_condn_image_ids]) 
                # print("latent ids shape: {}".format(latent_ids.shape)) # (2048, 3) 

                # -------------------------------- defining the latents for processing --------------------------------- # 
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(tgt_model_input)

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=tgt_model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=tgt_model_input.ndim, dtype=tgt_model_input.dtype)
                # adding noise only to the target edit latents, later this will be combined with the source latents before passing to transformer
                noisy_model_input = (1.0 - sigmas) * tgt_model_input + sigmas * noise

                # packing latents for the noisy target image to effectively pass them into the transformer model for generation
                packed_noisy_model_input = FluxKontextPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=tgt_model_input.shape[0],
                    num_channels_latents=tgt_model_input.shape[1],
                    height=tgt_model_input.shape[2],
                    width=tgt_model_input.shape[3],
                )

                # packing the conditioning latents that will be clean and will be combined with the noisy target latents next 
                packed_clean_condn_input = FluxKontextPipeline._pack_latents(
                    condn_model_input,
                    batch_size=condn_model_input.shape[0],
                    num_channels_latents=condn_model_input.shape[1],
                    height=condn_model_input.shape[2],
                    width=condn_model_input.shape[3],
                )

                # combining both the set of latents into a single tensor that will be passed for processing in model ---------------------------------------- # 
                packed_combined_model_input = torch.cat([packed_noisy_model_input, packed_clean_condn_input], dim=1)

                # handle guidance | passing the distilled guidance scalar to the model 
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(noisy_model_input.shape[0]) # using the shape of the clean latents to obtain the guidance scalar vector 
                else:
                    guidance = None

                # TODO: here we have to add the slider token conditioning to the model pred 
                # print("latent ids just before passing to the transformer: {}".format(latent_ids.shape)) # (4096, 3)
                # Predict the noise residual
                # print("repeat slider token shape before passing to the transformer: {}".format(repeated_slider_token.shape)) # (1, 128, 4096) 
                
                model_pred_extended = transformer(
                    hidden_states=packed_combined_model_input, # combined latent codes 
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds, # passing the extended prompt embeddings to the transformer 
                    txt_ids=text_ids, # extented text ids for conditioning on the slider tokens 
                    img_ids=latent_ids, # combined latent ids 
                    return_dict=False,
                    # passing the slider value encoded in an embedding via the slider projector for adjusting the modulation parameters 
                    modulation_embeddings=slider_embedding, 
                )[0]

                # dividing the image into two components and only taking the noisy target image for computing the loss 
                model_pred = model_pred_extended[:, :packed_noisy_model_input.size(1)] 
                # print("model pred shape after splitting: {}".format(model_pred.shape)) # (1, 1024, 64)
                
                model_pred = FluxKontextPipeline._unpack_latents(
                    model_pred,
                    height=tgt_model_input.shape[2] * vae_scale_factor,
                    width=tgt_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - tgt_model_input 

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean() 

                accelerator.backward(loss)

                ### Checking whether the gradients are being computed for the slider projector network. 
                # slider_param = slider_projector.parameters().__next__() 

                # # --------------- checking the gradients for the slider projector network --------------- #
                # if slider_param.grad is not None:
                #     # Check if the gradients are non-zero
                #     grad_sum = torch.sum(slider_param.grad).item()
                #     print(f"✅ Slider projector has gradients! Sum of grads: {grad_sum}")
                #     if grad_sum == 0.0:
                #         print("⚠️ Warning: Gradients are all zero.")
                # else:
                #     print("❌ ERROR: Slider projector has NO gradients. It is not being trained.")

                # syncing the gradients via the accelerator 
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters(), slider_projector.parameters())
                        if args.train_text_encoder
                        else itertools.chain(transformer.parameters(), slider_projector.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # we can skip this part as saving the model checkpoints with accelerator may not be the best way 
                if accelerator.is_main_process:
                    # ------------------------------- saving the models at intermediate checkpoints --------------------------- # 
                    if global_step % args.checkpointing_steps == 0:
                        
                        # # we can save with this logic also but have to keep in mind the sliders have to be saved separately, then the saving logic will not fail. 
                        # # ---------------------------- This is accelerator based saving of checkpoints, we are doing a simpler version of this ------------------ # 
                        # # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None: # checking if we have to remove the earlier checkpoint from training or not 
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # this will store all the modules that are prepared with accelerate and also save the optimization state or optimization dict. 
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}") 
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            monotonic_log(accelerator, logs, global_step)  

            if global_step >= args.max_train_steps:
                break

            # ----------------- Shifting the logging inside the training loop if each of the epoch is for the full dataset --------------- # 
            # After training the model, this will be called to validate the training of the model | this is for validation not for saving the model 
            if accelerator.is_main_process and accelerator.sync_gradients: 
                
                # if args.validation_prompt is not None and epoch % 20 == 0: # for the case of performing overfitting based on the single stack and n iteratiosn 
                if args.validation_prompt is not None and (step//args.gradient_accumulation_steps) % 500 == 1:  # validated at every x/4 index as the number of interations are reduced by the number of gradients number steps. 
                    if not args.train_text_encoder:
                        text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                        text_encoder_one.to(weight_dtype)
                        text_encoder_two.to(weight_dtype) 

                    # new pipeline that takes care of the slider model as an additional input for the final model training 
                    # Note: as the pipeline expects the model to be loaded from saved state dict, thats why passing the model components as unwrapped version of the same. 
                    # original pipeline of flux-kontext that does not take the slider token embeddings as input 
                    # defining the new pipeline that can do slider based inferences for the input edit instead of using the kontext model 
                    pipeline = FluxKontextSliderPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        vae=vae,
                        transformer=unwrap_model(transformer), 
                        slider_projector=unwrap_model(slider_projector), 
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    ) 

                    pipeline_args = {"prompt": args.validation_prompt}
                    images = log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                        torch_dtype=weight_dtype,
                    )
                    if not args.train_text_encoder:
                        del text_encoder_one, text_encoder_two
                        free_memory()

                    images = None
                    free_memory()


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # ------------------------------- saving the model components --------------------------- # 
        # this can be integrated with the previous part where you wish to save the model components after every set of validation steps. 
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None
        
        # adding a component to save the slider projector network along with the other model components 
        # modules_to_save["slider_projector"] = slider_projector

        FluxKontextPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # saving the slider projector network along with the components of the kontext model. 
        slider_projector = unwrap_model(slider_projector)
        slider_projector_path = os.path.join(args.output_dir, "slider_projector.pth")
        torch.save(slider_projector.state_dict(), slider_projector_path)
        logger.info(f"Saved slider projector to {slider_projector_path}")

        # Final inference
        # Load previous pipeline
        transformer = FluxTransformer2DModelwithSliderConditioning.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        ) 

        logger.info("checking if the slider projector needs the clip text embedding: {}".format(args.is_clip_input))
        slider_projector = SliderProjector(
            out_dim=args.slider_projector_out_dim,
            pe_dim=2,
            n_layers=args.slider_projector_n_layers,
            is_clip_input=args.is_clip_input
        )
        
        slider_projector.load_state_dict(torch.load(slider_projector_path))
        slider_projector.to(accelerator.device, dtype=weight_dtype)
        logger.info(f"Loaded slider projector from {slider_projector_path}")
        # ------------------------------- --------------------- --------------------------- # 

        # original pipeline of flux-kontext that does not take the slider token embeddings as input 
        pipeline = FluxKontextPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer, 
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )  
        
        pipeline.load_lora_weights(args.output_dir)
        logger.info(f"Loaded kontext pipeline weights from {args.output_dir}") 

        images = None

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
