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
        'default': 'DPMSolverMultistep'
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 35
    },
    'refiner_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.0
    },
    'strength': {
        'type': float,
        'required': False,
        'default': 0.3
    },
    'image_url': {
        'type': str,
        'required': False,
        'default': None
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda val: val == 1 or val == 4
    },
    'high_noise_frac': {
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
