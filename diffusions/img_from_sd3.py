from diffusers import StableDiffusion3Pipeline
import torch
import os
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

# Function to generate images using Stable Diffusion 3.
def gen_img_from_sd3(args):
    config = load_config(args.diffusion_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(args.synth_caption_path, "r") as f:
        captions = json.load(f)
    
    # Load the pre-trained Stable Diffusion model and set the computation precision to float16
    pipe = StableDiffusion3Pipeline.from_pretrained(config['model_name_or_path'], torch_dtype=torch.float16)
    pipe.to(device)

    for caption in captions:
        prompt = ""
        for conversation in caption['conversations']:
            if conversation['from'] == 'gpt':
                prompt += conversation['value']

        # Generate an image using the prompt provided in the caption
        image = pipe(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=config['negative_prompt'],
            num_inference_steps=config['n_steps'],
            height=config['height'],
            width=config['width'],
            guidance_scale=config['guidance_scale'],
            max_sequence_length=config['max_sequence_length'],
        ).images[0]

        image = image.resize((args.width, args.height))
        subdir = caption["image"].rsplit("/", 1)[0]
        if not os.path.exists(os.path.join(args.image_dir, subdir)):
            os.makedirs(os.path.join(args.image_dir, subdir))
        image_path = os.path.join(args.image_dir, caption["image"])
        image.save(image_path)