# Import from your noise tiling node file
from .nodes_noise_tiling import NODE_CLASS_MAPPINGS as TiledNoise_MAPPINGS
from .nodes_noise_tiling import NODE_DISPLAY_NAME_MAPPINGS as TiledNoise_DISPLAY_MAPPINGS

# Combine mappings
NODE_CLASS_MAPPINGS = {
    **TiledNoise_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **TiledNoise_DISPLAY_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("❇️ Custom Nodes: All nodes loaded from ComfyUI_AdvancedTiler.")