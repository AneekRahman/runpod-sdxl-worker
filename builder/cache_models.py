# builder/model_fetcher.py

from rp_handler import BASE_MODEL, REFINER_MODEL, VAE_AUTOENCODER
import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# Cache model function
def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    # How many times to retry caching model
    max_retries = 3
    
    for attempt in range(max_retries):
      try:
          return model_class.from_pretrained(model_name, **kwargs)
      except OSError as err:
          if attempt < max_retries - 1:
              print(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
          else:
              raise

# Cache Base and Refiner model
def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    base_args = {
        "torch_dtype": torch.float16,
        # "variant": "fp16",
        # "use_safetensors": True
    }
    
    refiner_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    pipe = fetch_pretrained_model(
        DiffusionPipeline,
        BASE_MODEL, 
        **base_args)
    
    vae = fetch_pretrained_model( 
        AutoencoderKL, 
        VAE_AUTOENCODER, 
        **{"torch_dtype": torch.float16})
    
    print("Loaded VAE")
    refiner = fetch_pretrained_model(
        StableDiffusionXLImg2ImgPipeline, 
        REFINER_MODEL, 
        **refiner_args)

    return pipe, refiner, vae


if __name__ == "__main__":
    get_diffusion_pipelines()
