INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': False,
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'scheduler': {
        'type': str,
        'required': False,
        'default': 'DPM++SDE'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 5
    },
    'refiner_inference_steps': {
        'type': int,
        'required': False,
        'default': 5
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 2
    },
    'refiner_strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda val: val == 1 or val == 4
    },
    'denoising_end_start': {
        'type': float,
        'required': False,
        'default': None
    },
    'loras': {
        'type': list,
        'required': False,
        'default': None
    },
}
