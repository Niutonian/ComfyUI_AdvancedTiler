import torch
import comfy.sample

# --- NOISE PROVIDER CLASS ---
class NoiseProvider:
    def __init__(self, seed, noise_tensor):
        self.seed = seed
        self.noise = noise_tensor
        print(f"[NoiseProvider] Initialized with seed {seed} and noise tensor shape {noise_tensor.shape}")

    def generate_noise(self, latent_ignored):
        # Debug: Log the shape of the latent tensor passed by the sampler
        if isinstance(latent_ignored, dict) and "samples" in latent_ignored:
            if isinstance(latent_ignored.get('samples'), torch.Tensor):
                print(f"[NoiseProvider] generate_noise received latent_ignored with 'samples' shape: {latent_ignored['samples'].shape}")
            else:
                print(f"[NoiseProvider] generate_noise received latent_ignored, but 'samples' is not a Tensor (type: {type(latent_ignored.get('samples'))}).")
        elif isinstance(latent_ignored, torch.Tensor):
            print(f"[NoiseProvider] generate_noise received latent_ignored TENSOR shape: {latent_ignored.shape}")
        else:
            print(f"[NoiseProvider] generate_noise received latent_ignored type: {type(latent_ignored)}")

        print(f"[NoiseProvider] Returning noise of shape {self.noise.shape}")
        return self.noise

# --- TILED NOISE FOR IMAGES ---
class TiledNoise:
    """
    Generates noise suitable for seamless tiling on X and/or Y axes for images.
    The noise tensor matches the shape of the provided latent tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),  # Required to determine shape and channels
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tile_x": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                "tile_y": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
            }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "generate_tiled_noise"
    CATEGORY = "noise/tiling"

    def generate_tiled_noise(self, latent, seed, tile_x, tile_y):
        # Extract the latent tensor from the input
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("[TiledNoise] Invalid latent input. Expected a dict with 'samples' key.")
        
        latent_tensor = latent["samples"]
        if not isinstance(latent_tensor, torch.Tensor):
            raise ValueError("[TiledNoise] Latent 'samples' must be a torch.Tensor.")

        # Latent shape: (batch_size, channels, height // 8, width // 8)
        batch_size, channels, latent_height, latent_width = latent_tensor.shape
        print(f"[TiledNoise] Generating noise with shape: {latent_tensor.shape}")

        # For seamless tiling, generate a larger noise tensor and crop it
        # If tiling is enabled, double the size of the dimension to create overlap
        noise_height = latent_height * 2 if tile_y else latent_height
        noise_width = latent_width * 2 if tile_x else latent_width
        noise_shape = (batch_size, channels, noise_height, noise_width)

        # Generate the larger noise tensor
        noise = comfy.sample.prepare_noise(torch.zeros(noise_shape, device='cpu'), seed)
        print(f"[TiledNoise] Generated larger noise tensor with shape: {noise.shape}")

        # Crop the noise tensor to the desired size, ensuring seamless boundaries
        if tile_y:
            # Take the middle section of the height to ensure the top and bottom match
            start_h = latent_height // 2
            noise = noise[:, :, start_h:start_h + latent_height, :]
        if tile_x:
            # Take the middle section of the width to ensure the left and right match
            start_w = latent_width // 2
            noise = noise[:, :, :, start_w:start_w + latent_width]

        print(f"[TiledNoise] Cropped noise tensor to shape: {noise.shape}")

        # Verify the final shape matches the latent tensor
        if noise.shape != latent_tensor.shape:
            raise ValueError(f"[TiledNoise] Final noise shape {noise.shape} does not match latent shape {latent_tensor.shape}")

        # Return a NoiseProvider instance
        provider = NoiseProvider(seed, noise)
        return (provider,)

# --- TILED NOISE FOR VIDEOS (ORIGINAL) ---
class TiledNoiseVideo(TiledNoise):
    """
    Generates noise suitable for seamless tiling on X, Y, and/or T axes for videos.
    The noise tensor matches the shape of the provided latent tensor.
    """
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["tile_t"] = ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"})
        cls.CATEGORY = "noise/tiling/video"
        return types

    def generate_tiled_noise(self, latent, seed, tile_x, tile_y, tile_t):
        # Extract the latent tensor from the input
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("[TiledNoiseVideo] Invalid latent input. Expected a dict with 'samples' key.")
        
        latent_tensor = latent["samples"]
        if not isinstance(latent_tensor, torch.Tensor):
            raise ValueError("[TiledNoiseVideo] Latent 'samples' must be a torch.Tensor.")

        # Latent shape for video: (batch_size, channels, length, height // 8, width // 8)
        batch_size, channels, latent_length, latent_height, latent_width = latent_tensor.shape
        print(f"[TiledNoiseVideo] Generating noise with shape: {latent_tensor.shape}")

        # For seamless tiling, generate a larger noise tensor and crop it
        # If tiling is enabled, double the size of the dimension to create overlap
        noise_length = latent_length * 2 if tile_t else latent_length
        noise_height = latent_height * 2 if tile_y else latent_height
        noise_width = latent_width * 2 if tile_x else latent_width
        noise_shape = (batch_size, channels, noise_length, noise_height, noise_width)

        # Generate the larger noise tensor
        noise = comfy.sample.prepare_noise(torch.zeros(noise_shape, device='cpu'), seed)
        print(f"[TiledNoiseVideo] Generated larger noise tensor with shape: {noise.shape}")

        # Crop the noise tensor to the desired size, ensuring seamless boundaries
        if tile_t:
            # Take the middle section of the length to ensure the start and end frames match
            start_l = latent_length // 2
            noise = noise[:, :, start_l:start_l + latent_length, :, :]
        if tile_y:
            # Take the middle section of the height to ensure the top and bottom match
            start_h = latent_height // 2
            noise = noise[:, :, :, start_h:start_h + latent_height, :]
        if tile_x:
            # Take the middle section of the width to ensure the left and right match
            start_w = latent_width // 2
            noise = noise[:, :, :, :, start_w:start_w + latent_width]

        print(f"[TiledNoiseVideo] Cropped noise tensor to shape: {noise.shape}")

        # Verify the final shape matches the latent tensor
        if noise.shape != latent_tensor.shape:
            raise ValueError(f"[TiledNoiseVideo] Final noise shape {noise.shape} does not match latent shape {latent_tensor.shape}")

        # Return a NoiseProvider instance
        provider = NoiseProvider(seed, noise)
        return (provider,)

# --- TILED NOISE FOR VIDEOS WITH ROTATION (ORIGINAL BEHAVIOR) ---
class TiledNoiseVideoRotate(TiledNoise):
    """
    Generates noise for videos with rotational tiling on X, Y, and/or T axes.
    The noise tensor matches the shape of the provided latent tensor.
    Rotations are applied only when the corresponding tile_[x/y/t] is enabled.
    """
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["tile_t"] = ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"})
        # Add rotation direction and amount for each axis
        types["required"]["rotate_x_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_x_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})  # Fraction of axis size
        types["required"]["rotate_y_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_y_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        types["required"]["rotate_t_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_t_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        cls.CATEGORY = "noise/tiling/video"
        return types

    def generate_tiled_noise(self, latent, seed, tile_x, tile_y, tile_t,
                             rotate_x_direction, rotate_x_amount,
                             rotate_y_direction, rotate_y_amount,
                             rotate_t_direction, rotate_t_amount):
        # Extract the latent tensor from the input
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("[TiledNoiseVideoRotate] Invalid latent input. Expected a dict with 'samples' key.")
        
        latent_tensor = latent["samples"]
        if not isinstance(latent_tensor, torch.Tensor):
            raise ValueError("[TiledNoiseVideoRotate] Latent 'samples' must be a torch.Tensor.")

        # Latent shape for video: (batch_size, channels, length, height // 8, width // 8)
        batch_size, channels, latent_length, latent_height, latent_width = latent_tensor.shape
        print(f"[TiledNoiseVideoRotate] Generating noise with shape: {latent_tensor.shape}")

        # Generate the noise tensor with the same shape as the latent tensor
        noise_shape = (batch_size, channels, latent_length, latent_height, latent_width)
        noise = comfy.sample.prepare_noise(torch.zeros(noise_shape, device='cpu'), seed)
        print(f"[TiledNoiseVideoRotate] Generated noise tensor with shape: {noise.shape}")

        # Apply rotational tiling using torch.roll
        dims_to_roll = []
        shifts = []

        if tile_t:
            # Calculate the shift amount for the T-axis (length)
            t_size = latent_length
            shift_t = int(t_size * rotate_t_amount)  # Convert fraction to number of units
            if rotate_t_direction == "counterclockwise":
                shift_t = -shift_t  # Negative shift for counterclockwise
            if shift_t != 0:
                dims_to_roll.append(-3)  # Length dimension
                shifts.append(shift_t)
                print(f"[TiledNoiseVideoRotate] Rotating T-axis (dim -3) by {shift_t} units ({rotate_t_direction})")

        if tile_y:
            # Calculate the shift amount for the Y-axis (height)
            h_size = latent_height
            shift_y = int(h_size * rotate_y_amount)
            if rotate_y_direction == "counterclockwise":
                shift_y = -shift_y
            if shift_y != 0:
                dims_to_roll.append(-2)  # Height dimension
                shifts.append(shift_y)
                print(f"[TiledNoiseVideoRotate] Rotating Y-axis (dim -2) by {shift_y} units ({rotate_y_direction})")

        if tile_x:
            # Calculate the shift amount for the X-axis (width)
            w_size = latent_width
            shift_x = int(w_size * rotate_x_amount)
            if rotate_x_direction == "counterclockwise":
                shift_x = -shift_x
            if shift_x != 0:
                dims_to_roll.append(-1)  # Width dimension
                shifts.append(shift_x)
                print(f"[TiledNoiseVideoRotate] Rotating X-axis (dim -1) by {shift_x} units ({rotate_x_direction})")

        # Apply the rotations if any
        if dims_to_roll:
            noise = torch.roll(noise, shifts=tuple(shifts), dims=tuple(dims_to_roll))
        else:
            print("[TiledNoiseVideoRotate] No rotations applied (all shifts are 0 or tiling disabled).")

        print(f"[TiledNoiseVideoRotate] Final noise tensor shape: {noise.shape}")

        # Verify the final shape matches the latent tensor
        if noise.shape != latent_tensor.shape:
            raise ValueError(f"[TiledNoiseVideoRotate] Final noise shape {noise.shape} does not match latent shape {latent_tensor.shape}")

        # Return a NoiseProvider instance
        provider = NoiseProvider(seed, noise)
        return (provider,)

# --- TILED NOISE FOR VIDEOS WITH ROTATION (ALWAYS APPLIED) ---
class TiledNoiseVideoRotateAlways(TiledNoise):
    """
    Generates noise for videos with rotational tiling on X, Y, and/or T axes.
    The noise tensor matches the shape of the provided latent tensor.
    Rotations are always applied based on amount and direction, regardless of tiling toggles.
    """
    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["tile_t"] = ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"})
        # Add rotation direction and amount for each axis
        types["required"]["rotate_x_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_x_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})  # Fraction of axis size
        types["required"]["rotate_y_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_y_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        types["required"]["rotate_t_direction"] = (["clockwise", "counterclockwise"], {"default": "clockwise"})
        types["required"]["rotate_t_amount"] = ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01})
        cls.CATEGORY = "noise/tiling/video"
        return types

    def generate_tiled_noise(self, latent, seed, tile_x, tile_y, tile_t,
                             rotate_x_direction, rotate_x_amount,
                             rotate_y_direction, rotate_y_amount,
                             rotate_t_direction, rotate_t_amount):
        # Extract the latent tensor from the input
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("[TiledNoiseVideoRotateAlways] Invalid latent input. Expected a dict with 'samples' key.")
        
        latent_tensor = latent["samples"]
        if not isinstance(latent_tensor, torch.Tensor):
            raise ValueError("[TiledNoiseVideoRotateAlways] Latent 'samples' must be a torch.Tensor.")

        # Latent shape for video: (batch_size, channels, length, height // 8, width // 8)
        batch_size, channels, latent_length, latent_height, latent_width = latent_tensor.shape
        print(f"[TiledNoiseVideoRotateAlways] Generating noise with shape: {latent_tensor.shape}")

        # Generate the noise tensor with the same shape as the latent tensor
        noise_shape = (batch_size, channels, latent_length, latent_height, latent_width)
        noise = comfy.sample.prepare_noise(torch.zeros(noise_shape, device='cpu'), seed)
        print(f"[TiledNoiseVideoRotateAlways] Generated noise tensor with shape: {noise.shape}")

        # Apply rotational tiling using torch.roll, regardless of tile_[x/y/t]
        dims_to_roll = []
        shifts = []

        # T-axis (length) rotation
        t_size = latent_length
        shift_t = int(t_size * rotate_t_amount)  # Convert fraction to number of units
        if rotate_t_direction == "counterclockwise":
            shift_t = -shift_t  # Negative shift for counterclockwise
        if shift_t != 0:
            dims_to_roll.append(-3)  # Length dimension
            shifts.append(shift_t)
            print(f"[TiledNoiseVideoRotateAlways] Rotating T-axis (dim -3) by {shift_t} units ({rotate_t_direction})")

        # Y-axis (height) rotation
        h_size = latent_height
        shift_y = int(h_size * rotate_y_amount)
        if rotate_y_direction == "counterclockwise":
            shift_y = -shift_y
        if shift_y != 0:
            dims_to_roll.append(-2)  # Height dimension
            shifts.append(shift_y)
            print(f"[TiledNoiseVideoRotateAlways] Rotating Y-axis (dim -2) by {shift_y} units ({rotate_y_direction})")

        # X-axis (width) rotation
        w_size = latent_width
        shift_x = int(w_size * rotate_x_amount)
        if rotate_x_direction == "counterclockwise":
            shift_x = -shift_x
        if shift_x != 0:
            dims_to_roll.append(-1)  # Width dimension
            shifts.append(shift_x)
            print(f"[TiledNoiseVideoRotateAlways] Rotating X-axis (dim -1) by {shift_x} units ({rotate_x_direction})")

        # Apply the rotations if any
        if dims_to_roll:
            noise = torch.roll(noise, shifts=tuple(shifts), dims=tuple(dims_to_roll))
        else:
            print("[TiledNoiseVideoRotateAlways] No rotations applied (all shifts are 0).")

        # Optional: If tile_[x/y/t] is enabled, we can add additional tiling logic here in the future
        if tile_t:
            print("[TiledNoiseVideoRotateAlways] T-axis tiling enabled (no additional logic applied yet).")
        if tile_y:
            print("[TiledNoiseVideoRotateAlways] Y-axis tiling enabled (no additional logic applied yet).")
        if tile_x:
            print("[TiledNoiseVideoRotateAlways] X-axis tiling enabled (no additional logic applied yet).")

        print(f"[TiledNoiseVideoRotateAlways] Final noise tensor shape: {noise.shape}")

        # Verify the final shape matches the latent tensor
        if noise.shape != latent_tensor.shape:
            raise ValueError(f"[TiledNoiseVideoRotateAlways] Final noise shape {noise.shape} does not match latent shape {latent_tensor.shape}")

        # Return a NoiseProvider instance
        provider = NoiseProvider(seed, noise)
        return (provider,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TiledNoise": TiledNoise,
    "TiledNoiseVideo": TiledNoiseVideo,
    "TiledNoiseVideoRotate": TiledNoiseVideoRotate,
    "TiledNoiseVideoRotateAlways": TiledNoiseVideoRotateAlways,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TiledNoise": "Tiled Noise (Image)",
    "TiledNoiseVideo": "Tiled Noise (Video)",
    "TiledNoiseVideoRotate": "Tiled Noise Rotate (Video)",
    "TiledNoiseVideoRotateAlways": "Tiled Noise Rotate Always (Video)",
}

print(">>> Custom Node: TiledNoise loaded.")