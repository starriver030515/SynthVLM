from diffusers import DiffusionPipeline
import torch
import os
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Function to generate images using Stable Diffusion XL.
def gen_img_from_sdxl(args):
    config = load_config(args.diffusion_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(args.synth_caption_path, "r") as f:
        captions = json.load(f)
    
    # Load the pre-trained Stable Diffusion model and set the computation precision to float16
    base = DiffusionPipeline.from_pretrained(
        config['base_name_or_path'],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    refiner = DiffusionPipeline.from_pretrained(
        config['refiner_name_or_path'],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    for caption in captions:
        prompt = ""
        for conversation in caption['conversations']:
            if conversation['from'] == 'gpt':
                prompt += conversation['value']
                
        # Generate an image using the prompt provided in the caption
        image = base(
            prompt=prompt,
            negative_prompt = config['negative_prompt'],
            num_inference_steps=config['n_steps'],
            denoising_end=config['high_noise_frac'],
            output_type="latent",
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
        ).images

        image = refiner(
            prompt=prompt,
            negative_prompt = config['negative_prompt'],
            num_inference_steps=config['n_steps'],
            denoising_end=config['high_noise_frac'],
            image=image,
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
        ).images[0]
        
        image.resize((args.width, args.height))
        subdir = caption["image"].rsplit("/", 1)[0]
        if not os.path.exists(os.path.join(args.image_dir, subdir)):
            os.makedirs(os.path.join(args.image_dir, subdir))
        image_path = os.path.join(args.image_dir, caption["image"])
        image.save(image_path)
    


