import os
import gc
from typing import List, Tuple, Dict
import json

import torch
import gradio as gr
from PIL import Image

from diffusers import FluxTransformer2DModel
from sliders_model import SliderProjector, SliderProjector_wo_clip
from sliders_pipeline import FluxKontextSliderPipeline


# -----------------------------
# Environment & device
# -----------------------------
# Avoid meta-tensor init from environment leftovers
os.environ.pop("ACCELERATE_INIT_EMPTY_WEIGHTS", None)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# -----------------------------
# Model / pipeline loading
# -----------------------------
def load_pipeline_single_gpu(device_str: str) -> FluxKontextSliderPipeline:
    pretrained = "black-forest-labs/FLUX.1-Kontext-dev"

    n_slider_layers = 4
    slider_projector_out_dim = 6144
    trained_models_path = "./model_weights/"
    is_clip_input = True

    # Load transformer fully on CPU; avoid meta tensors
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained,
        subfolder="transformer",
        device_map=None,
        low_cpu_mem_usage=False,
    )
    transformer.eval()
    weight_dtype = transformer.dtype  # keep checkpoint dtype

    # Slider projector
    if is_clip_input:
        slider_projector = SliderProjector(
            out_dim=slider_projector_out_dim, pe_dim=2, n_layers=n_slider_layers, is_clip_input=True
        )
    else:
        slider_projector = SliderProjector_wo_clip(
            out_dim=slider_projector_out_dim, pe_dim=2, n_layers=n_slider_layers
        )

    # putting both the models to infer 
    transformer.eval()
    slider_projector.eval()

    # Load projector weights on CPU
    slider_projector_path = os.path.join(trained_models_path, "slider_projector.pth")
    state_dict = torch.load(slider_projector_path)
    print("state_dict keys: {}".format(state_dict.keys()))

    slider_projector.load_state_dict(state_dict)
    print(f"loaded slider_projector from {slider_projector_path}")
    # ------------------------------- --------------------- --------------------------- # 

    # Build full pipeline on CPU; no device_map sharding
    pipeline = FluxKontextSliderPipeline.from_pretrained(
        pretrained,
        transformer=transformer,
        slider_projector=slider_projector,
        torch_dtype=weight_dtype,
        device_map=None,
        low_cpu_mem_usage=False,
    )

    print("loading the pipeline lora weights from: {}".format(trained_models_path))

    pipeline.load_lora_weights(trained_models_path)
    print("loaded the pipeline with lora weights from: {}".format(trained_models_path)) 
    
    # Move everything to the single device
    pipeline.to(device_str)
    return pipeline


PIPELINE = load_pipeline_single_gpu(DEVICE)
print(f"[init] Pipeline loaded on {DEVICE}")


# -----------------------------
# Sample Images & Precomputed Results
# -----------------------------

def create_sample_entry(name, image_filename, prompt, result_folder, num_results=5, result_pattern="image_{i}.png", precomputed_base="./sample_images/precomputed"):
    """
    Helper function to create a sample entry with subfolder organization.
    
    Args:
        name: Display name in dropdown
        image_filename: Filename in ./sample_images/
        prompt: Editing instruction 
        result_folder: Subfolder name in precomputed directory
        num_results: Number of precomputed results (default 5)
        result_pattern: Filename pattern, {i} will be replaced with 0,1,2,3,4 (default "image_{i}.png")
        precomputed_base: Base path for precomputed results (default "./sample_images/precomputed")
    """
    return {
        "name": name,
        "image_path": f"./sample_images/{image_filename}",
        "prompt": prompt,
        "precomputed_results": [f"{precomputed_base}/{result_folder}/{result_pattern.format(i=i)}" for i in range(num_results)]
    }

def load_samples_from_config(config_file="sample_config.json"):
    """Load sample data from a JSON configuration file."""
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sample config: {e}")
    return []

def discover_samples_automatically(sample_dir="./sample_images", precomputed_dir="./sample_images/precomputed"):
    """Automatically discover samples based on directory structure with subfolders."""
    discovered_samples = []
    
    if not os.path.exists(sample_dir) or not os.path.exists(precomputed_dir):
        return discovered_samples
    
    # Look for subfolders in precomputed directory
    for subfolder in os.listdir(precomputed_dir):
        subfolder_path = os.path.join(precomputed_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Look for sequential result files in subfolder
            precomputed_files = []
            for i in range(0, 15):  # Check for up to 15 results starting from 0
                # Try different patterns
                for pattern in [f"image_{i}.png", f"image_{i}.jpg", f"{i}.jpg", f"{i}.png", f"result_{i}.jpg", f"output_{i}.png"]:
                    result_path = os.path.join(subfolder_path, pattern)
                    if os.path.exists(result_path):
                        precomputed_files.append(result_path)
                        break
                else:
                    # If no file with this index found, stop looking (but continue if we found at least one)
                    if i == 0 and not precomputed_files:
                        continue  # Keep trying from index 0
                    elif not precomputed_files:
                        break  # No files found at all
                    else:
                        break  # Found some files but this index is missing, stop here
            
            if precomputed_files:
                # Try to find corresponding source image
                img_path = None
                # Common naming patterns for source images
                base_name = subfolder.split('_')[0] # e.g., "portrait" from "portrait_smile"
                for ext in ['.jpg', '.jpeg', '.png']:
                    candidate = os.path.join(sample_dir, f"{base_name}{ext}")
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                
                if img_path:
                    sample = {
                        "name": f"{subfolder.replace('_', ' ').title()} - Auto-discovered",
                        "image_path": img_path,
                        "prompt": f"Edit: {subfolder.replace('_', ' ')}",  # Default prompt
                        "precomputed_results": precomputed_files
                    }
                    discovered_samples.append(sample)
    
    return discovered_samples

# Main sample data - using your actual folder structure
SAMPLE_DATA = [
    create_sample_entry("Stylization", "aesthetic_model2.png", "Transform the image into a Van Gogh Style painting", "aesthetic_model2_vangogh", 11),
    create_sample_entry("Weather Change", "enfield3.png", "Transform the scene into winter season with heavy snowfall", "enfield3_winter_snow", 11),
    create_sample_entry("Illumination Change", "lamp6.png", "Turn on the lamp with blue lighting", "light_lamp_blue_side", 11),
    create_sample_entry("Appearance Change", "jackson2.png", "Transform his jacket into a blue fluffy fur jacket", "jackson_fluffy", 11),
    create_sample_entry("Scene Edit", "venice1.png", "Grow ivy on the walls of the buildings on the side", "venice1_grow_ivy", 11)
]

# Add more samples using the helper function
# Modify these examples or add your own:

ADDITIONAL_SAMPLES = [
    # Add your own samples here following your folder structure:
    # 
    # For your structure (./sample_images/precomputed/folder_name/image_0.png, image_1.png, etc.):
    # create_sample_entry("Display Name", "your_image.png", "editing prompt", "folder_name", 12),
    #
    # Examples based on your pattern:
    # create_sample_entry("New Sample", "new_image.png", "apply some effect", "new_folder", 12),
    # create_sample_entry("Another Edit", "source.png", "different editing instruction", "another_folder", 10),
    
    # Note: 
    # - Images should be in ./sample_images/
    # - Precomputed results should be in ./sample_images/precomputed/folder_name/
    # - Default pattern is image_0.png, image_1.png, etc.
    # - Adjust the number (12) to match how many results you have
]

# Extend the main sample data with additional samples
SAMPLE_DATA.extend(ADDITIONAL_SAMPLES)

# Optional: Auto-discover additional samples from directories
# Uncomment to automatically find additional samples beyond the manual ones above:
# AUTO_DISCOVERED = discover_samples_automatically()
# if AUTO_DISCOVERED:
#     print(f"Auto-discovered {len(AUTO_DISCOVERED)} additional samples:")
#     for sample in AUTO_DISCOVERED:
#         print(f"  - {sample['name']}")
#     SAMPLE_DATA.extend(AUTO_DISCOVERED)

# Optional: Load samples from external JSON config
# CONFIG_SAMPLES = load_samples_from_config("sample_config.json")
# SAMPLE_DATA.extend(CONFIG_SAMPLES)

def load_sample_image(image_path: str) -> Image.Image:
    """Load a sample image, with fallback to a placeholder if file doesn't exist."""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            # Create a placeholder image if sample doesn't exist
            placeholder = Image.new('RGB', (512, 512), color=(200, 200, 200))
            return placeholder
    except Exception as e:
        print(f"Error loading sample image {image_path}: {e}")
        # Return a placeholder image
        placeholder = Image.new('RGB', (512, 512), color=(200, 200, 200))
        return placeholder

def load_precomputed_results(result_paths: List[str]) -> List[Image.Image]:
    """Load precomputed result images, with fallbacks for missing files."""
    results = []
    for path in result_paths:
        try:
            if os.path.exists(path):
                results.append(Image.open(path))
            else:
                # Create placeholder result
                placeholder = Image.new('RGB', (512, 512), color=(150, 150, 150))
                results.append(placeholder)
        except Exception as e:
            print(f"Error loading precomputed result {path}: {e}")
            placeholder = Image.new('RGB', (512, 512), color=(150, 150, 150))
            results.append(placeholder)
    return results


# -----------------------------
# Helpers
# -----------------------------
def resize_image(img: Image.Image, target: int = 512) -> Image.Image:
    """Resize shortest side to target, then center-crop to target x target."""
    w, h = img.size
    try:
        resample = Image.Resampling.BICUBIC  # PIL >= 10
    except Exception:
        resample = Image.BICUBIC

    if h > w:
        new_w, new_h = target, int(target * h / w)
    elif h < w:
        new_w, new_h = int(target * w / h), target
    else:
        new_w, new_h = target, target

    # resizing the image to a fixed lower dimension size of 512  
    img = img.resize((new_w, new_h), resample)
    return img


def _encode_prompt(prompt: str):
    with torch.no_grad():
        pe, ppe, _ = PIPELINE.encode_prompt(prompt, prompt_2=prompt)
    return pe, ppe


# -----------------------------
# Inference functions
# -----------------------------
def generate_image_stack_edits(text_prompt, n_edits, input_image):
    """
    Compute n_edits images on a single GPU for slider values in (0,1],
    return (list_of_images, first_image) so the UI shows immediately.
    """
    if not input_image or not text_prompt or text_prompt.startswith("Please select"):
        return [], None

    n = int(n_edits) if n_edits is not None else 1
    n = max(1, n)
    slider_values = [(i + 1) / float(n) for i in range(n)]  # (0,1] inclusive

    img = resize_image(input_image, 512)
    pe, ppe = _encode_prompt(text_prompt)

    results: List[Image.Image] = []
    gen_base = 64  # deterministic seed base

    # not using batching for now just a simple forward loop 
    # batch_size = 2
    # n_batches = n // batch_size
    # batched_slider_values = [[slider_values[i*batch_size: (i+1)*batch_size]] for i in range(n_batches)]
    # print(f"batched_slider_values: {batched_slider_values}")

    for i, sv in enumerate(slider_values):
        gen = torch.Generator(device=DEVICE if DEVICE != "cpu" else "cpu").manual_seed(gen_base + i)
        with torch.no_grad():
            # replicating based on the number of examples in the batch size         

            out = PIPELINE(
                image=img,
                height=img.height,
                width=img.width,
                num_inference_steps=28,
                prompt_embeds=pe,
                pooled_prompt_embeds=ppe,
                generator=gen,
                text_condn=False,
                modulation_condn=True,
                slider_value=torch.tensor(sv, device=DEVICE if DEVICE != "cpu" else "cpu").reshape(1, 1),
                is_clip_input=True,
            )
            results.append(out.images[0])

        if DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    first = results[0] if results else None
    return results, first


def generate_single_image(text_prompt, slider_value, input_image):
    if not input_image or not text_prompt or text_prompt.startswith("Please select"):
        return None

    img = resize_image(input_image, 512)
    sv = float(slider_value)
    pe, ppe = _encode_prompt(text_prompt)

    gen = torch.Generator(device=DEVICE if DEVICE != "cpu" else "cpu").manual_seed(64)
    with torch.no_grad():
        out = PIPELINE(
            image=img,
            height=img.height,
            width=img.width,
            num_inference_steps=28,
            prompt_embeds=pe,
            pooled_prompt_embeds=ppe,
            generator=gen,
            text_condn=False,
            modulation_condn=True,
            slider_value=torch.tensor(sv, device=DEVICE if DEVICE != "cpu" else "cpu").reshape(1, 1),
            is_clip_input=True,
        )
        result = out.images[0]

    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return result


# -----------------------------
# Sample Loading Functions
# -----------------------------
def get_sample_by_name(sample_name: str):
    """Get sample data by name."""
    for sample in SAMPLE_DATA:
        if sample["name"] == sample_name:
            return sample
    return None

def load_sample_to_main_interface(sample_name: str):
    """Load selected sample to main interface with precomputed results."""
    if not sample_name:
        return (
            None, 
            "Please select a sample above to see the editing instruction", 
            [], 
            None, 
            gr.update(minimum=0, maximum=0, step=1, value=0, label="Edit Strength Level")
        )
    
    sample = get_sample_by_name(sample_name)
    if not sample:
        return (
            None, 
            "Sample not found", 
            [], 
            None, 
            gr.update(minimum=0, maximum=0, step=1, value=0, label="Edit Strength Level")
        )
    
    # Load sample image
    sample_image = load_sample_image(sample["image_path"])
    prompt = sample["prompt"]
    
    # Load precomputed results
    precomputed_images = load_precomputed_results(sample["precomputed_results"])
    first_result = precomputed_images[0] if precomputed_images else None
    
    # Update slider range for precomputed results
    n_results = len(precomputed_images)
    slider_update = gr.update(
        minimum=0, 
        maximum=max(0, n_results-1), 
        step=1, 
        value=0,
        label=f"Edit Strength Level (0-{n_results-1}) - Precomputed"
    )
    
    return sample_image, prompt, precomputed_images, first_result, slider_update


# -----------------------------
# Helpers
# -----------------------------
def update_slider_range(n_edits):
    """Update the slider range based on number of edits."""
    return gr.update(
        minimum=0,
        maximum=max(0, int(n_edits)-1),
        step=1,
        value=0,
        label=f"Edit Strength Level (0-{int(n_edits)-1})"
    )


def display_selected_image(slider_index: int, images_list: List[Image.Image]) -> Image.Image:
    """
    Display the image corresponding to the slider index from the generated images list.
    
    Args:
        slider_index: Current slider position (0-based index)
        images_list: List of generated/precomputed images
        
    Returns:
        Selected image or None if invalid index/empty list
    """
    if not images_list or len(images_list) == 0:
        return None
        
    # Clamp index to valid range
    idx = max(0, min(int(slider_index), len(images_list) - 1))
    return images_list[idx]

# -----------------------------
# Gradio UI
# -----------------------------
# Add new helper function for user uploads
def process_user_upload(uploaded_image, user_prompt, n_edits_val):
    """Handle user uploaded images and custom prompts."""
    if uploaded_image is None:
        return None, [], None, gr.update(minimum=0, maximum=0, step=1, value=0, label="Edit Strength Level")
    
    # Resize uploaded image
    processed_image = resize_image(uploaded_image, 512)
    
    # Generate edits
    generated_list, first_result = generate_image_stack_edits(user_prompt, n_edits_val, processed_image)
    
    # Update slider range
    slider_update = gr.update(
        minimum=0,
        maximum=max(0, len(generated_list)),
        step=1,
        value=0,
        label=f"Edit Strength Level (0-{len(generated_list)-1})"
    )
    
    return processed_image, generated_list, first_result, slider_update

with gr.Blocks() as demo:
    gr.Markdown("# Kontinuous Kontext - Continuous Strength Control for Instruction-based Image Editing")
    
    # Add description section
    gr.Markdown("""
    ## About
    ### Kontinuous Kontext allows you to edit a given image with a freeform text instruction and a slider strength value.
    ### The slider strength enables precise control for the extent of the applied edit and generates smooth transitions between different editing levels.

    ### You can either:
    1. Choose from our sample images with predefined edit instructions
    2. Upload your own image and specify custom editing instructions

    Checkout the [paper](https://arxiv.org/pdf/2510.08532v1) and the [project page](https://snap-research.github.io/kontinuouskontext) for more details.
    """)

    # Add custom CSS for tabs
    gr.Markdown("""
    <style>
    .tabs.svelte-710i53 {
        margin-top: 2em !important;
        margin-bottom: 2em !important;
    }
    .tabs.svelte-710i53 button {
        font-size: 1.2em !important;
        padding: 0.5em 2em !important;
        min-width: 200px !important;
    }
    #sample_image, #sample_output, #upload_image, #upload_output {
        min-height: 512px !important;
        max-height: 512px !important;
    }
    </style>
    """)

    with gr.Tabs() as tabs:
        # Common style parameters for images
        IMAGE_WIDTH = 512
        IMAGE_HEIGHT = 512
        
        with gr.TabItem("📸 Examples") as tab1:  # Added emoji and changed tab name
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    sample_dropdown = gr.Dropdown(
                        choices=[sample["name"] for sample in SAMPLE_DATA],
                        label="Select Sample Image & Prompt",
                        value=None
                    )
                    sample_text = gr.Textbox(lines=1, show_label=False, placeholder="Please select a sample above", interactive=False)
                    sample_n_edits = gr.Number(value=5, minimum=1, maximum=20, step=1, label="Number of Edits", precision=0)
                    sample_image = gr.Image(
                        type="pil", 
                        label="Source Image", 
                        width=IMAGE_WIDTH, 
                        height=IMAGE_HEIGHT, 
                        interactive=False,
                        elem_id="sample_image"
                    )
                    sample_button = gr.Button("Display Edits")  # Added back
                with gr.Column(scale=1):
                    with gr.Row():
                        sample_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.1, 
                            value=0, 
                            label="Edit Strength",
                            scale=1,
                            min_width=100
                        )
                    sample_output = gr.Image(
                        type="pil", 
                        label="Edited Output", 
                        width=IMAGE_WIDTH, 
                        height=IMAGE_HEIGHT,
                        elem_id="sample_output"
                    )

        with gr.TabItem("⬆️ Upload Your Image") as tab2:  # Added emoji and changed tab name
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    upload_text = gr.Textbox(lines=1, label="Enter Editing Prompt", placeholder="Describe the edit you want...")
                    upload_n_edits = gr.Number(value=5, minimum=1, maximum=20, step=1, label="Number of Edits", precision=0)
                    upload_image = gr.Image(
                        type="pil", 
                        label="Upload Image", 
                        width=IMAGE_WIDTH, 
                        height=IMAGE_HEIGHT,
                        elem_id="upload_image"
                    )
                    upload_button = gr.Button("Generate Edits")  # Kept consistent with sample tab
                with gr.Column(scale=1):
                    with gr.Row():
                        upload_slider = gr.Slider(
                            minimum=0, 
                            maximum=1, 
                            step=0.1, 
                            value=0, 
                            label="Edit Strength Level",
                            scale=1,
                            min_width=100
                        )
                    upload_output = gr.Image(
                        type="pil", 
                        label="Edited Output", 
                        width=IMAGE_WIDTH, 
                        height=IMAGE_HEIGHT,
                        elem_id="upload_output"
                    )

    # States for both tabs
    sample_generated_images = gr.State([])
    upload_generated_images = gr.State([])

    # Sample tab logic
    sample_dropdown.change(
        load_sample_to_main_interface,
        inputs=[sample_dropdown],
        outputs=[sample_image, sample_text, sample_generated_images, sample_output, sample_slider]
    )

    sample_button.click(
        generate_image_stack_edits,
        inputs=[sample_text, sample_n_edits, sample_image],
        outputs=[sample_generated_images, sample_output],
    ).then(
        update_slider_range,
        inputs=[sample_n_edits],
        outputs=[sample_slider],
    )

    sample_slider.change(
        display_selected_image,
        inputs=[sample_slider, sample_generated_images],
        outputs=[sample_output],
    )

    # Upload tab logic - Remove duplicate click handler and combine the logic
    upload_button.click(
        generate_image_stack_edits,  # Generate images first
        inputs=[upload_text, upload_n_edits, upload_image],
        outputs=[upload_generated_images, upload_output],
    ).then(
        update_slider_range,  # Then update slider range
        inputs=[upload_n_edits],
        outputs=[upload_slider],
    )

    # Update slider when n_edits changes
    upload_n_edits.change(
        update_slider_range,
        inputs=[upload_n_edits],
        outputs=[upload_slider],
    )

    upload_slider.change(
        display_selected_image,
        inputs=[upload_slider, upload_generated_images],
        outputs=[upload_output],
    )

    # Add citation section at the bottom
    gr.Markdown("""
    ---
    ### If you find this work useful, please cite:
    ```bibtex
    @article{kontinuous_kontext_2025,
            title={Kontinuous Kontext: Continuous Strength Control for Instruction-based Image Editing},
            author={R Parihar, O Patashnik, D Ostashev, R Venkatesh Babu, D Cohen-Or, and J Wang},
            journal={Arxiv},
            year={2025}
    }
    ```
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)