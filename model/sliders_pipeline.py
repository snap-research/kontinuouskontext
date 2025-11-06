# kontext_sliders_pipeline.py
import torch
from diffusers import FluxKontextPipeline  # Base pipeline from Diffusers
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
import numpy as np
from diffusers.pipelines.flux.pipeline_flux_kontext import *

# custom import for transformer models 
from model.transformer_flux import FluxTransformer2DModelwithSliderConditioning


from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


# defining the custom pipeline allowing for inference with the pretrained slider projector and the flux-kontext model. 
class FluxKontextSliderPipeline(FluxKontextPipeline):
    """
    Custom pipeline extending FluxKontextPipeline with slider conditioning.
    Minimal changes: Override __init__ to load slider_projector, and __call__ for slider-aware inference.
    """

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModelwithSliderConditioning,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        slider_projector=None,  # the slider projector model loaded with the weights 
        text_condn: bool = False, 
    ):
        # Calling the parent __init__ with the base arguments that are passed in the pipeline 
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        ) 

        device = self._execution_device
        # Minimal addition: Load your custom slider_projector
        self.slider_projector = slider_projector

        self.text_condn = text_condn # whether we are conditioning in the text space or the modulation space 
        self.slider_projector.eval()  # Set to eval mode for inference 
    
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        # slider values as additional input for the pipeline ------------- #  
        # slider projector is already initialized in the parent call so that we can call it to obtain the embeddings for the sliders 
        text_condn: bool = False,
        modulation_condn: bool = False,
        slider_value: Optional[torch.FloatTensor] = None,
        is_clip_input: bool = False, # This is to check whether the slider projector takes the clip text embedding as input for modulating 
    ):
        # small modification to keep all the values on the same device, and the device is passed along with the pipeline to the model. 

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        # print("vae scale factor: {}".format(self.vae_scale_factor)) 
        # print("default sample size: {}".format(self.default_sample_size)) 
        # print("default sample size: height: {}, width: {}".format(height, width)) 

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of 
        # print("after width and height quantized: height: {}, width: {}".format(height, width)) 


        # not checking for the height and width are matching to the predefined dimensions for inferences. 
        # if height != original_height or width != original_width:
            # print("height and width are not matching the original dimensions ..")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        device = self._execution_device
        # print("execution device: {}".format(device)) 

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 3. Preprocess image ---------- this is the older preprocessing function that is forcing the images to be of the size 1024x1024, but we are training with 512x512 so changing the output to be of the same dimensions. 
        # if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        #     img = image[0] if isinstance(image, list) else image
        #     image_height, image_width = self.image_processor.get_default_height_width(img)
        #     aspect_ratio = image_width / image_height
        #     if _auto_resize:
        #         # Kontext is trained on specific resolutions, using one of them is recommended
        #         _, image_width, image_height = min(
        #             (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        #         )
        #     image_width = image_width // multiple_of * multiple_of
        #     image_height = image_height // multiple_of * multiple_of
        #     image = self.image_processor.resize(image, image_height, image_width)
        #     image = self.image_processor.preprocess(image, image_height, image_width)

        # 3.1 Custom image preprocessing module that will reshape the images to the given input dimensions 
        # overriding the height and the width for the original model components as we have a fixed size for images in our dataset. 
        height = original_height 
        width = original_width

        image = self.image_processor.resize(image, height, width)
        image = self.image_processor.preprocess(image, height, width) 
        # print("image shape after preprocessing: {}".format(image.shape)) 

        # 3. -------------------------------------- Preparing the slider values -------------------------------------- # 
        # This is a correct way to check the device for a tensor and a model in PyTorch.
        # print(f"slider_value device: {slider_value.device}")  # tensor device
        # print(f"slider_projector device: {next(self.slider_projector.parameters()).device}")  # model device
        
        # if clip input is enabled, we will compute the embeddings using both the slider values and the pooled prompt embeddings 
        if (is_clip_input):
            # TODO: This may not work with larget batch size then 1, please validate this once then run the output. 
            # this takes vector as input as the slider_value is also a list 
            # Ensure pooled_prompt_embeds is a tensor of shape [1, 1, ...] (one higher dimension)
            pooled_prompt_embeds_tensor = torch.tensor(pooled_prompt_embeds).unsqueeze(0).to(device)
            slider_value = slider_value.to(device)
            
            self.slider_projector = self.slider_projector.to(device)
            # print("pooled prompt device: {}".format(pooled_prompt_embeds_tensor.device))
            # print("slider value device: {}".format(slider_value.device))
            # print("slider projector device: {}".format(next(self.slider_projector.parameters()).device))

            slider_embeddings = self.slider_projector(slider_value, pooled_prompt_embeds_tensor).to(device) 
        else:
            slider_embeddings = self.slider_projector(slider_value).to(device)


        # print("slider embeddings device: {}".format(slider_embeddings.device))  
        # multiplying the slider embeddings with a random value to check whether there is any effect of changing the slider in the input         
        # slider_embeddings = slider_embeddings * (np.random.rand() * 4 - 2)

        # print("slider embeddings norm: {}".format(slider_embeddings.norm())) 
        # print("slider value inside the pipeline: {}".format(slider_value))
        # print("slider embeddings: {}".format(slider_embeddings.shape)) # (1, 1, 64)
        slider_id = torch.tensor([0,0,2]).reshape(1,3).to(device)

        # replicating the same slider embeddings with n_repeat times 
        n_repeats = 1
        repeated_slider_token = slider_embeddings.repeat(1, n_repeats, 1)
        repeated_slider_id = slider_id.repeat(n_repeats, 1)

        # ------------------------------- concatenating the slider embeddings with the text embeddings --------------- # 
        # if we are conditioning in the text space then will concatenate the slider tokens to the conditioning

        if text_condn:      
            print("using text conditioning ...")  
            extended_text_ids = torch.cat([text_ids, repeated_slider_id], dim=0)
            extended_prompt_embeds = torch.cat([prompt_embeds, repeated_slider_token], dim=1)   
        else:
            extended_text_ids = text_ids
            extended_prompt_embeds = prompt_embeds 

        if modulation_condn:
            modulation_embeddings = repeated_slider_token
        else:
            modulation_embeddings = None 

        # print("concatenated text ids shape: {}".format(extended_text_ids.shape)) # (640, 3)
        # print("concatenated prompt embeds shape: {}".format(extended_prompt_embeds.shape)) # (1, 640, 4096) 

        # print("slider id: {}".format(slider_id.shape)) # (1, 3) 
        #--------------------- defined the slider components that I will use along with the other inputs to perform the forward pass of the model. ---------------------# 

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if image_ids is not None:
            # latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension 
            # TODO: Verify the shapes here, adding the slider id along with the ids for the input and target images 
            # print("original latent ids: {}".format(latent_ids.shape)) 
            ## --- not using the slider id along with the visual tokens, we are adding them along with the text tokens --- ## 
            # latent_ids = torch.cat([latent_ids, image_ids, slider_id], dim=0)  

            # --- using the standard image and text latent conditioning and not adding the slider ids in the model --- ## 
            latent_ids = torch.cat([latent_ids, image_ids], dim=0) 

            # print("latent ids after concatenation: {}".format(latent_ids.shape)) 

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # -------------- Logic for ip adapter, we can remove this ----------------------- # 
        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                # stacking the latents for the generated latent and the input image latent 
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                
                # print("latent model shape after concatenation: {}".format(latent_model_input.shape)) 
                timestep = t.expand(latents.shape[0]).to(latents.dtype) 

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=extended_prompt_embeds,
                    txt_ids=extended_text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ## adding the modulation token, if we are working with modulation space conditioning 
                    modulation_embeddings=modulation_embeddings,  # passing the modulation embeddings that will be defined based on whether the modulation inference is enabled or not 
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        modulation_embeddings=modulation_embeddings,  
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step() 

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
