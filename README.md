# SynthVLM: High-Efficiency and High-Quality Synthetic Data for Vision Language Models

---

🚀🚀🚀 Official implementation of **SynthVLM: High-Efficiency and High-Quality Synthetic Data for Vision Language Models**.

<p align="center">
  <img src="demo/imagecmp.pdf">
</p>


- **Authors**: Zheng Liu*, Hao Liang*, Wentao Xiong, Chong Chen, Conghui He, Bin Cui, Wentao Zhang

## Quick Usage

## Data Preparation

Prepare your captions in JSON format. Here’s an example of how your JSON should look:

```json
[
    {
        "id": 1,
        "caption": "The bus in the image is white and red. The back of the bus features an advertisement. The bus is driving down the street, which is crowded with people and other vehicles."
    },
    {
        "id": 2,
        "caption": "The dog in the image is brown with a red collar. It sits behind a window, looking out longingly, which gives it a sense of longing for the outdoors or something it sees."
    },
]
```

## Data Generation Instructions

To generate images, run the `run.sh` script. The settings can be adjusted as follows:

- **Resolution**: Default is set to 1024x1024 pixels. Modify `width` and `height` in the script to customize.
- **Repetitions**: By default, each caption generates one image. Adjust the `repeat` parameter to increase the number of images per caption, selecting the best quality image for each.

### Model Selection

The process supports two diffusion models:

- **Stable Diffusion 3**: Default model, optimized for a balance between speed and quality.
- **Stable Diffusion XL**: Use this model for faster image generation.

## License

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
