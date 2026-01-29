"""
Color Utilities
Color ramp and recoloring tools for PBR maps
"""

import torch


class ColorRamp:
    """
    Color Ramp node - Map grayscale values to colors
    Similar to Blender's ColorRamp node with visual gradient preview
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["custom", "grayscale", "heat", "rainbow", "gold_metal", "rust", "copper", "blue_metal"], {
                    "default": "custom",
                    "tooltip": "Load a preset gradient or use custom colors"
                }),
                "interpolation": (["linear", "ease_in", "ease_out", "constant"], {
                    "default": "linear",
                    "tooltip": "Interpolation mode between color stops"
                }),
                
                # Color stops stored as JSON string (managed by widget)
                "color_stops": ("STRING", {
                    "default": '[{"pos":0.0,"r":0.0,"g":0.0,"b":0.0},{"pos":1.0,"r":1.0,"g":1.0,"b":1.0}]',
                    "multiline": False,
                    "tooltip": "Color stops data (managed by visual widget)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_color_ramp"
    CATEGORY = "Texture Alchemist/Color"
    
    def get_preset_stops(self, preset):
        """Get preset color stops"""
        presets = {
            "grayscale": [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
            ],
            "heat": [
                (0.0, 0.0, 0.0, 0.0),    # Black
                (0.33, 0.5, 0.0, 0.0),   # Dark red
                (0.66, 1.0, 0.5, 0.0),   # Orange
                (1.0, 1.0, 1.0, 0.5),    # Yellow-white
            ],
            "rainbow": [
                (0.0, 0.5, 0.0, 0.5),    # Purple
                (0.33, 0.0, 0.0, 1.0),   # Blue
                (0.5, 0.0, 1.0, 0.0),    # Green
                (0.66, 1.0, 1.0, 0.0),   # Yellow
                (1.0, 1.0, 0.0, 0.0),    # Red
            ],
            "gold_metal": [
                (0.0, 0.2, 0.15, 0.0),   # Dark bronze
                (0.5, 0.8, 0.6, 0.2),    # Gold
                (1.0, 1.0, 0.95, 0.7),   # Bright gold
            ],
            "rust": [
                (0.0, 0.2, 0.1, 0.05),   # Dark rust
                (0.5, 0.6, 0.3, 0.1),    # Orange rust
                (1.0, 0.8, 0.5, 0.3),    # Light rust
            ],
            "copper": [
                (0.0, 0.2, 0.1, 0.05),   # Dark copper
                (0.5, 0.7, 0.4, 0.2),    # Copper
                (1.0, 0.9, 0.7, 0.5),    # Bright copper
            ],
            "blue_metal": [
                (0.0, 0.05, 0.1, 0.15),  # Dark blue metal
                (0.5, 0.3, 0.4, 0.6),    # Blue metal
                (1.0, 0.7, 0.8, 0.95),   # Bright blue metal
            ],
        }
        return presets.get(preset, None)
    
    def convert_to_grayscale(self, image):
        """Convert image to grayscale for value mapping"""
        if image.shape[-1] == 1:
            return image
        
        # Luminance weights
        weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                               device=image.device, dtype=image.dtype)
        gray = torch.sum(image * weights, dim=-1, keepdim=True)
        return gray
    
    def _parse_color_stops(self, color_stops_json, device, dtype):
        """Parse color stops from JSON string"""
        import json
        try:
            stops_data = json.loads(color_stops_json)
            stops = []
            for stop in stops_data:
                pos = stop.get('pos', 0.0)
                r = stop.get('r', 0.0)
                g = stop.get('g', 0.0)
                b = stop.get('b', 0.0)
                stops.append((pos, torch.tensor([r, g, b], device=device, dtype=dtype)))
            return stops
        except:
            # Fallback to default gradient
            return [
                (0.0, torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)),
                (1.0, torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype))
            ]
    
    def interpolate_color(self, value, pos1, color1, pos2, color2, mode="linear"):
        """Interpolate between two colors based on value"""
        # Clamp value between positions
        if value <= pos1:
            return color1
        if value >= pos2:
            return color2
        
        # Normalize value between positions
        t = (value - pos1) / (pos2 - pos1)
        
        # Apply interpolation mode
        if mode == "ease_in":
            t = t * t  # Quadratic ease in
        elif mode == "ease_out":
            t = 1 - (1 - t) * (1 - t)  # Quadratic ease out
        elif mode == "constant":
            t = 0.0  # Step function (use first color)
        # linear mode: t remains as is
        
        # Linear interpolation between colors
        color = color1 * (1 - t) + color2 * t
        return color
    
    def apply_color_ramp(self, image, preset, interpolation, color_stops):
        """Apply color ramp to image"""
        
        print("\n" + "="*60)
        print("Color Ramp")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Preset: {preset}")
        print(f"Interpolation: {interpolation}")
        
        # Convert to grayscale to get values
        gray = self.convert_to_grayscale(image)
        
        # Build list of color stops
        if preset != "custom":
            # Load preset
            preset_stops = self.get_preset_stops(preset)
            if preset_stops:
                stops = []
                for ps in preset_stops:
                    stops.append((ps[0], torch.tensor([ps[1], ps[2], ps[3]], device=image.device, dtype=image.dtype)))
                print(f"✓ Loaded preset: {preset}")
            else:
                # Fallback to custom if preset not found
                stops = self._parse_color_stops(color_stops, image.device, image.dtype)
        else:
            # Use custom stops
            stops = self._parse_color_stops(color_stops, image.device, image.dtype)
        
        # Sort stops by position
        stops.sort(key=lambda x: x[0])
        
        print(f"Active color stops: {len(stops)}")
        for i, (pos, color) in enumerate(stops):
            print(f"  Stop {i+1}: pos={pos:.2f}, color=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
        
        # Create output image
        batch, height, width, channels = image.shape
        output = torch.zeros((batch, height, width, 3), device=image.device, dtype=image.dtype)
        
        # Vectorized color ramp application (much faster!)
        # Extract positions and colors as tensors
        positions = torch.tensor([s[0] for s in stops], device=image.device, dtype=image.dtype)
        colors = torch.stack([s[1] for s in stops], dim=0)  # Shape: (num_stops, 3)
        
        # For each pixel, find which stop interval it falls into
        # searchsorted finds the index of the right stop for each value
        gray_flat = gray.reshape(-1)
        
        # Find the right stop index for each pixel
        # searchsorted returns indices where values would be inserted to maintain order
        right_indices = torch.searchsorted(positions, gray_flat, right=False)
        right_indices = torch.clamp(right_indices, 1, len(stops) - 1)
        left_indices = right_indices - 1
        
        # Get the stop positions and colors for interpolation
        pos1 = positions[left_indices]  # Shape: (num_pixels,)
        pos2 = positions[right_indices]
        color1 = colors[left_indices]   # Shape: (num_pixels, 3)
        color2 = colors[right_indices]
        
        # Calculate interpolation factor
        t = (gray_flat - pos1) / (pos2 - pos1 + 1e-8)
        t = torch.clamp(t, 0.0, 1.0).unsqueeze(1)  # Shape: (num_pixels, 1)
        
        # Apply interpolation mode
        if interpolation == "ease_in":
            t = t * t  # Quadratic ease in
        elif interpolation == "ease_out":
            t = 1 - (1 - t) * (1 - t)  # Quadratic ease out
        elif interpolation == "constant":
            t = torch.zeros_like(t)  # Step function (use first color)
        # linear mode: t remains as is
        
        # Interpolate colors
        output_flat = color1 * (1 - t) + color2 * t
        
        output = output_flat.reshape(batch, height, width, 3)
        
        print(f"✓ Color ramp applied")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        print("="*60 + "\n")
        
        return (output,)


class SimpleRecolor:
    """
    Simple two-color recolor utility
    Maps dark values to one color, light values to another
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "dark_color_r": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "dark_color_g": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "dark_color_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "light_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "light_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "light_color_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "blend_mode": (["linear", "smooth"], {
                    "default": "smooth",
                    "tooltip": "How to blend between colors"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "recolor"
    CATEGORY = "Texture Alchemist/Color"
    
    def recolor(self, image, dark_color_r, dark_color_g, dark_color_b,
                light_color_r, light_color_g, light_color_b, blend_mode):
        """Recolor image with two-color gradient"""
        
        print("\n" + "="*60)
        print("Simple Recolor")
        print("="*60)
        
        # Convert to grayscale to get values
        if image.shape[-1] == 1:
            gray = image
        else:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            gray = torch.sum(image * weights, dim=-1, keepdim=True)
        
        # Create color vectors
        dark_color = torch.tensor([dark_color_r, dark_color_g, dark_color_b],
                                  device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        light_color = torch.tensor([light_color_r, light_color_g, light_color_b],
                                   device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        
        # Apply blend mode
        t = gray  # Blend factor (0-1)
        
        if blend_mode == "smooth":
            # Smoothstep interpolation
            t = t * t * (3.0 - 2.0 * t)
        
        # Interpolate between dark and light colors
        output = dark_color * (1.0 - t) + light_color * t
        
        print(f"Dark color: ({dark_color_r:.2f}, {dark_color_g:.2f}, {dark_color_b:.2f})")
        print(f"Light color: ({light_color_r:.2f}, {light_color_g:.2f}, {light_color_b:.2f})")
        print(f"Blend mode: {blend_mode}")
        print(f"✓ Recolored")
        print("="*60 + "\n")
        
        return (output,)


# Node registration
class HSVAdjuster:
    """
    Adjust Hue, Saturation, and Value of textures
    Better color control than RGB adjustments
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue_shift": ("FLOAT", {
                    "default": 0.0,
                    "min": -0.5,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Hue shift (-0.5 to 0.5, wraps around color wheel)"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Saturation multiplier (0=grayscale, 1=normal, >1=vivid)"
                }),
                "value": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Brightness multiplier"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist/Color"
    
    def adjust(self, image, hue_shift, saturation, value):
        """Adjust HSV values"""
        
        print("\n" + "="*60)
        print("HSV Adjuster")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Hue shift: {hue_shift:+.3f}")
        print(f"Saturation: {saturation:.3f}x")
        print(f"Value: {value:.3f}x")
        
        # Ensure RGB
        if image.shape[-1] == 1:
            image = image.repeat(1, 1, 1, 3)
        
        # Convert RGB to HSV
        hsv = self._rgb_to_hsv(image)
        
        # Adjust H, S, V
        hsv[:, :, :, 0] = (hsv[:, :, :, 0] + hue_shift) % 1.0  # Hue wraps
        hsv[:, :, :, 1] = torch.clamp(hsv[:, :, :, 1] * saturation, 0.0, 1.0)
        hsv[:, :, :, 2] = torch.clamp(hsv[:, :, :, 2] * value, 0.0, 1.0)
        
        # Convert back to RGB
        result = self._hsv_to_rgb(hsv)
        
        print(f"✓ HSV adjusted")
        print("="*60 + "\n")
        
        return (result,)
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_rgb, argmax_rgb = rgb.max(dim=-1)
        min_rgb = rgb.min(dim=-1)[0]
        diff = max_rgb - min_rgb
        
        # Hue
        h = torch.zeros_like(max_rgb)
        
        mask_r = (argmax_rgb == 0)
        mask_g = (argmax_rgb == 1)
        mask_b = (argmax_rgb == 2)
        
        h[mask_r] = ((g - b) / (diff + 1e-8))[mask_r] % 6.0
        h[mask_g] = (((b - r) / (diff + 1e-8)) + 2.0)[mask_g]
        h[mask_b] = (((r - g) / (diff + 1e-8)) + 4.0)[mask_b]
        
        h = h / 6.0  # Normalize to [0, 1]
        
        # Saturation
        s = diff / (max_rgb + 1e-8)
        
        # Value
        v = max_rgb
        
        return torch.stack([h, s, v], dim=-1)
    
    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        h, s, v = hsv[:, :, :, 0], hsv[:, :, :, 1], hsv[:, :, :, 2]
        
        h = h * 6.0  # Scale to [0, 6]
        
        c = v * s
        x = c * (1.0 - torch.abs((h % 2.0) - 1.0))
        m = v - c
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        mask0 = (h >= 0) & (h < 1)
        mask1 = (h >= 1) & (h < 2)
        mask2 = (h >= 2) & (h < 3)
        mask3 = (h >= 3) & (h < 4)
        mask4 = (h >= 4) & (h < 5)
        mask5 = (h >= 5) & (h < 6)
        
        r[mask0] = c[mask0]
        g[mask0] = x[mask0]
        
        r[mask1] = x[mask1]
        g[mask1] = c[mask1]
        
        g[mask2] = c[mask2]
        b[mask2] = x[mask2]
        
        g[mask3] = x[mask3]
        b[mask3] = c[mask3]
        
        r[mask4] = x[mask4]
        b[mask4] = c[mask4]
        
        r[mask5] = c[mask5]
        b[mask5] = x[mask5]
        
        r = r + m
        g = g + m
        b = b + m
        
        return torch.stack([r, g, b], dim=-1)


class ColorImage:
    """
    Generate solid color image with interactive color picker
    Choose any color using color wheel or hex input
    Supports alpha channel for transparency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 512,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 16,
                    "max": 8192,
                    "step": 8,
                    "display": "number",
                    "tooltip": "Image height in pixels"
                }),
                "color": ("STRING", {
                    "default": "#FF0000",
                    "multiline": False,
                    "tooltip": "Color in hex format (e.g., #FF0000 for red)"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of images to generate"
                }),
            },
            "hidden": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_color_image"
    CATEGORY = "Texture Alchemist/Color"
    
    def generate_color_image(self, width, height, color, batch_size, alpha=1.0):
        """
        Generate a solid color RGBA image
        """
        
        print("\n" + "="*60)
        print("Color Image Generator")
        print("="*60)
        print(f"Dimensions: {width}×{height}")
        print(f"Color: {color}")
        print(f"Alpha: {alpha:.2f}")
        print(f"Batch size: {batch_size}")
        
        # Parse hex color
        color = color.strip()
        if not color.startswith('#'):
            color = '#' + color
        
        # Remove # and parse
        try:
            hex_color = color.lstrip('#')
            
            # Support both #RGB and #RRGGBB formats
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            
            if len(hex_color) != 6:
                print(f"⚠️ Invalid color format: {color}, using red")
                r, g, b = 1.0, 0.0, 0.0
            else:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                
            print(f"✓ RGBA: ({r:.3f}, {g:.3f}, {b:.3f}, {alpha:.3f})")
            
        except ValueError as e:
            print(f"⚠️ Error parsing color: {e}, using red")
            r, g, b = 1.0, 0.0, 0.0
        
        # Create RGBA image tensor (B, H, W, 4)
        image = torch.zeros(batch_size, height, width, 4, dtype=torch.float32)
        image[:, :, :, 0] = r
        image[:, :, :, 1] = g
        image[:, :, :, 2] = b
        image[:, :, :, 3] = alpha
        
        print(f"✓ Generated RGBA image: {image.shape}")
        print("="*60 + "\n")
        
        return (image,)


class ColorCode:
    """
    Color picker that outputs color codes instead of images
    Get hex, RGB, and RGBA values with interactive picker and eyedropper
    Supports alpha channel for transparency
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("STRING", {
                    "default": "#FF0000",
                    "multiline": False,
                    "tooltip": "Color in hex format (e.g., #FF0000 for red)"
                }),
            },
            "hidden": {
                "alpha": ("FLOAT", {
                    "default": 1.0,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("hex", "rgb", "rgba", "alpha")
    FUNCTION = "get_color_values"
    CATEGORY = "Texture Alchemist/Color"
    
    def get_color_values(self, color, alpha=1.0):
        """
        Parse color and return hex, RGB, RGBA format outputs with alpha
        """
        
        print("\n" + "="*60)
        print("Color Code")
        print("="*60)
        print(f"Input color: {color}")
        print(f"Alpha: {alpha:.2f}")
        
        # Parse hex color
        color = color.strip()
        if not color.startswith('#'):
            color = '#' + color
        
        # Remove # and parse
        try:
            hex_color = color.lstrip('#')
            
            # Support both #RGB and #RRGGBB formats
            if len(hex_color) == 3:
                hex_color = ''.join([c*2 for c in hex_color])
            
            if len(hex_color) != 6:
                print(f"⚠️ Invalid color format: {color}, using red")
                r, g, b = 255, 0, 0
            else:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            
            # Create output strings
            hex_out = color.upper()
            rgb_out = f"rgb({r}, {g}, {b})"
            rgba_out = f"rgba({r}, {g}, {b}, {alpha:.2f})"
            
            print(f"✓ Hex: {hex_out}")
            print(f"✓ RGB: {rgb_out}")
            print(f"✓ RGBA: {rgba_out}")
            print(f"✓ Alpha: {alpha:.2f}")
            
        except ValueError as e:
            print(f"⚠️ Error parsing color: {e}, using red")
            hex_out = "#FF0000"
            rgb_out = "rgb(255, 0, 0)"
            rgba_out = f"rgba(255, 0, 0, {alpha:.2f})"
        
        print("="*60 + "\n")
        
        return (hex_out, rgb_out, rgba_out, alpha)


NODE_CLASS_MAPPINGS = {
    "ColorRamp": ColorRamp,
    "SimpleRecolor": SimpleRecolor,
    "HSVAdjuster": HSVAdjuster,
    "ColorImage": ColorImage,
    "ColorCode": ColorCode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorRamp": "Color Ramp",
    "SimpleRecolor": "Simple Recolor",
    "HSVAdjuster": "HSV Adjuster",
    "ColorImage": "Color Image 🎨",
    "ColorCode": "Color Code 🎨",
}

