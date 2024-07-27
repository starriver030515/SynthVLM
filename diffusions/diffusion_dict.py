from diffusions.img_from_sd3 import gen_img_from_sd3
from diffusions.img_from_sdxl import gen_img_from_sdxl

diffusion_dict = {
    'stable_diffusion_3': gen_img_from_sd3,
    'stable_diffusion_xl': gen_img_from_sdxl
}