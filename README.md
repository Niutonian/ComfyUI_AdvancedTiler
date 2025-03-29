

# LTX Hackaton ComfyUI_AdvancedTiler

This node was created during a feverish coding night, it is very experimental, doesn't do any of the thing written below, but it is fun to use, and after all, that is what matters


A collection of custom nodes for ComfyUI that provide advanced noise tiling capabilities for image and video generation. These nodes are designed to generate noise tensors that can be used in diffusion models to create seamlessly tiling outputs, with support for both spatial (X, Y) and temporal (T) tiling.

## Features

- **TiledNoise**: Generates noise for images with seamless tiling on the X and/or Y axes.
- **TiledNoiseVideo**: Generates noise for videos with seamless tiling on the X, Y, and/or T (temporal) axes using a cropping method.
- **TiledNoiseVideoRotate**: Generates noise for videos with rotational tiling on the X, Y, and/or T axes, where rotations are applied only when tiling is enabled for the corresponding axis.
- **TiledNoiseVideoRotateAlways**: Similar to `TiledNoiseVideoRotate`, but rotations are applied regardless of the tiling toggles.

## Installation

### Prerequisites
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) must be installed and running.
- Python 3.8+ and PyTorch (as required by ComfyUI).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/niutonian/ComfyUI_AdvancedTiler.git
