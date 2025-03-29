# nodes_latent_tiling.py

import torch

class LatentTiler:
    """
    A node to prepare a latent tensor for seamless tiling on X and/or Y axes
    by rolling the tensor content. Place this node before the sampler.
    Works for both image (B, C, H, W) and video (B, C, F, H, W) latents.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", ),
                "tile_x": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                "tile_y": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "tile_latent"
    CATEGORY = "latent/tiling" # Or choose a category you prefer

    def tile_latent(self, latent, tile_x, tile_y):
        samples = latent["samples"].clone() # Clone to avoid modifying the original tensor in-place
        ndim = samples.ndim

        # Determine spatial dimensions (Height and Width)
        # Image: (B, C, H, W) -> H is -2, W is -1
        # Video: (B, C, F, H, W) -> H is -2, W is -1
        height_dim = -2
        width_dim = -1

        dims_to_roll = []
        shifts = []

        if tile_y:
            height = samples.shape[height_dim]
            shift_y = height // 2
            if shift_y > 0:
                dims_to_roll.append(height_dim)
                shifts.append(shift_y)
                print(f"[LatentTiler] Rolling Y-axis (dim {height_dim}) by {shift_y} pixels.")

        if tile_x:
            width = samples.shape[width_dim]
            shift_x = width // 2
            if shift_x > 0:
                dims_to_roll.append(width_dim)
                shifts.append(shift_x)
                print(f"[LatentTiler] Rolling X-axis (dim {width_dim}) by {shift_x} pixels.")

        if dims_to_roll:
            rolled_samples = torch.roll(samples, shifts=tuple(shifts), dims=tuple(dims_to_roll))
        else:
            # No tiling requested, return original samples
            rolled_samples = samples
            print("[LatentTiler] No tiling enabled.")

        # Create a new latent dictionary with the modified samples
        # Copy other potential keys from the original latent dict
        new_latent = latent.copy()
        new_latent["samples"] = rolled_samples

        return (new_latent,)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LatentTiler": LatentTiler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentTiler": "Latent Tiler (for Seamless Output)" # Changed name slightly for clarity
}

print(">>> Custom Node: LatentTiler loaded.")