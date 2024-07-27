import os
import json
import random

# List of prompts for llava, used to randomly select a prompt to describe images
llava_prompt_for_image = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image."
    "Give a short and clear explanation of the subsequent image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
]

# Function to transform the format of JSON data from one structure to standard structure
def gen_synth_dataset(args):
    with open(args.init_caption_path, "r") as f:
        captions = json.load(f)
    
    new_annotations = []
    for index, annotation in enumerate(captions):
        # Calculate the subfolder index based on the number of annotations
        folder_index = (index // 10000)
        target_subfolder = f"{folder_index:05d}"
        # format of the image name: 00000xxxx.jpg
        target_image_name = f"{folder_index:05d}{index % 10000:04d}.jpg"

        # Randomly select a prompt to be used for describing the image
        random_prompt = random.choice(llava_prompt_for_image)
        for repeat_time in range(args.repeat):
            image_dir = f'part_{repeat_time}'
            new_annotation = {
                "id": f"{target_subfolder}{index % 10000:04d}",
                "image": f"{image_dir}/{target_subfolder}/{target_image_name}",
                "conversations": [
                    {"from": "human", "value": f"{random_prompt}\n<image>"},
                    {"from": "gpt", "value": annotation["caption"]},
                ],
            }
            new_annotations.append(new_annotation)

    with open(args.synth_caption_path, "w") as json_file:
        json.dump(new_annotations, json_file, indent=4)