import argparse

from utils.gen_synth_dataset import gen_synth_dataset
from utils.select_with_clipscore import select_with_clipscore

from diffusions.diffusion_dict import diffusion_dict

from clipscore.clipscore import calc_clipscore

def args_parser():
    parser = argparse.ArgumentParser(description="Change format of a file.")
    parser.add_argument('--init_caption_path', type=str, help='Path to the input JSON file')
    parser.add_argument('--synth_caption_path', type=str, help='Directory to save the output')
    parser.add_argument('--diffusion_model', type=str, help='')
    parser.add_argument('--diffusion_config', type=str, help='')
    parser.add_argument('--image_dir', type=str, help='')
    parser.add_argument('--width', type=int, help='')
    parser.add_argument('--height', type=int, help='')
    parser.add_argument('--clipscore_path', type=str, help='')
    parser.add_argument('--filtered_clipscore_path', type=str, help='')
    parser.add_argument('--select_radio', type=float, help='')
    parser.add_argument('--repeat', type=int, help='')
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    gen_synth_dataset(args)
    diffusion_dict[args.diffusion_model](args)
    calc_clipscore(args)
    select_with_clipscore(args)
    
    