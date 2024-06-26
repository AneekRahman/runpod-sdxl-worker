'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler
)

from compel import Compel, ReturnedEmbeddingsType

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #

# Define Constants
BASE_MODEL = "RunDiffusion/Juggernaut-X-v10"
REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
VAE_AUTOENCODER = "madebyollin/sdxl-vae-fp16-fix"

# Define ModelHandler Class
class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_loras(self, loras: list):
        # List of adapter_name and adapter_weights of loras
        adapter_names = []
        adapter_weights = []
      
        # Loop through each loraName (Hugginface lora name)
        for loraObject in loras:
            lora_name = loraObject["lora_name"]
            lora_weight = loraObject.get("lora_weight", 1.0)
            
            # Append the name and weight to list
            adapter_names.append(lora_name)
            adapter_weights.append(lora_weight)
            
            # Load the loras into BASE pipeline
            self.base.load_lora_weights(
                lora_name, 
                weight_name=loraObject["file_name"], 
                adapter_name=lora_name)
                        
            print("Loaded lora: ", lora_name)
        
        # Set the weight of loras
        self.base.set_adapters(adapter_names, adapter_weights=adapter_weights)

    # Load base SDXL model
    def load_base(self):
        # Get the floating point fix VAE
        vae = AutoencoderKL.from_pretrained(
            VAE_AUTOENCODER, 
            torch_dtype=torch.float16
        )

        # Get the SDXL model
        base_pipe = DiffusionPipeline.from_pretrained(
            BASE_MODEL,
            vae=vae,
            torch_dtype=torch.float16, 
            # variant="fp16", 
            # use_safetensors=True, 
            add_watermarker=False,
            safety_checker=None # No safety checking
        )

        # Use GPU for faster inference
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)

        # Enable Xformers for memory efficieny. BUT SLOWER!!
        # base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        # Get the floating point fix VAE
        vae = AutoencoderKL.from_pretrained(
            VAE_AUTOENCODER, 
            torch_dtype=torch.float16
        )
        
        # Get the SDXL Refiner model
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            REFINER_MODEL, 
            torch_dtype=torch.float16,
            vae=vae,
            variant="fp16", 
            use_safetensors=True, 
            add_watermarker=False,
            safety_checker=None # No safety checking
        )

        # Use GPU for faster inference
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)

        # Enable Xformers for memory efficieny. BUT SLOWER!!
        # refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    # Load models into memory
    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)

            self.base = future_base.result()
            self.refiner = future_refiner.result()


# Initiate ModelHandler
MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #

# Turn prompt text into prompt embeddings
def get_prompt_embeddings(base, text:str):
    compel = Compel(
        tokenizer=[base.tokenizer, base.tokenizer_2] ,
        text_encoder=[base.text_encoder, base.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True])
    
    return compel(text)

# Upload images to cloudflare
def _save_and_upload_images(images, job_id):
    # Make temp folder to save the image 
    os.makedirs(f"/{job_id}", exist_ok=True)
    
    image_urls = []
    
    # Go through images
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        # Try to upload image to Cloudflare R2
        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            # Else, return the image data in base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


# Define KarrasDPM
class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)

# Map the schedulers
def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "KarrasDPM": KarrasDPM.from_config(config),
        "DPM++SDE": DPMSolverSDEScheduler.from_config(config),
    }[name]


# Main Entry function
@torch.inference_mode()
def generate_image(job):
    # ------------------ PREPARE INPUT ------------------
    
    # Retrieve inputs
    job_input = job["input"]

    # Validate input against rp_shemas.py
    validated_input = validate(job_input, INPUT_SCHEMA)

    # Return error is found
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    
    # Return structed/default input after validation
    job_input = validated_input['validated_input']

    # Get prompt_image_url
    prompt_image_url = job_input['image_url']

    # Get seed
    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")
        
    # Load the lora if available
    if job_input['loras']:
        MODELS.load_loras(job_input['loras'])
    else:
        MODELS.base.set_adapters([], adapter_weights=[])
        MODELS.base.disable_lora()
        
    # ------------------ GENERATE ------------------

    # Create diffusers generator using seed
    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    # Set the appropriate scheduler
    MODELS.base.scheduler = make_scheduler(job_input['scheduler'], MODELS.base.scheduler.config)
    
    # Get prompt embeddings from prompt text
    prompt_conditioning, prompt_pooled = get_prompt_embeddings(MODELS.base, job_input['prompt'])
    negative_conditioning, negative_prompt_pooled = get_prompt_embeddings(MODELS.base, job_input['negative_prompt'])

    if prompt_image_url:  
        # ---=== Use image + prompt ===---
        
        # Load image from URL
        init_image = load_image(prompt_image_url).convert("RGB")

        # Run using refiner
        output = MODELS.refiner(
            prompt_embeds=prompt_conditioning, 
            pooled_prompt_embeds=prompt_pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_prompt_pooled,
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['refiner_strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        # ---=== Ensemble of expert denoisers ===---
        
        # Generate image (end at step = num_inference_steps * denoising_end_start)
        output = MODELS.base(
            prompt_embeds=prompt_conditioning, 
            pooled_prompt_embeds=prompt_pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_prompt_pooled,
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            denoising_end=job_input['denoising_end_start'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        # Refine image (start from step = num_inference_steps * denoising_end_start)
        try:
            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['refiner_strength'],
                image=output,
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
                "refresh_worker": True
            }
        
    # ------------------ GENERATE END ------------------

    # Upload images to cloudflare
    image_urls = _save_and_upload_images(output, job['id'])

    # Response object
    results = {
        "images": image_urls,
        "input": job["input"]
    }

    # Return results to client
    return results


# TODO Uncomment below line in production
# runpod.serverless.start({"handler": generate_image})

# TODO Remove below ALL lines in production
thisdict = {
    "id": "test_id",
    "input": {
        "prompt": "Portrait of a kind of a young man, twenty years with brown hair, childish, droopy eyes, unconventional beauty, moody atmosphere by Saul Leiter",
        "seed" : 3913886038,
        "negative_prompt" : "CGI, Unreal, Airbrushed, Digital",
        "loras" : [{
            "lora_name" : "CiroN2022/toy-face",
            "file_name" : "toy_face_sdxl.safetensors",
            "lora_weight" : 1.0,
        }]
    }
}
res = generate_image(thisdict)
print(res)