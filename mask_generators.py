"""
Mask Generation Utilities for TextureAlchemy
Advanced mask creation for weathering, selection, and compositing
"""

import torch
import torch.nn.functional as F
import math


class EdgeWearMaskGenerator:
    """
    Generate edge wear masks from normal maps and curvature
    Perfect for realistic paint chipping and edge damage
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "edge_width": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Width of edge wear (0.0-1.0)"
                }),
                "edge_sharpness": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Edge falloff sharpness"
                }),
                "noise_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Variation noise scale"
                }),
                "noise_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Amount of edge variation"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "normal_map": ("IMAGE",),
                "curvature_map": ("IMAGE",),
                "ao_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Masks"
    
    def generate(self, width, height, edge_width, edge_sharpness, noise_scale, noise_strength, seed,
                 normal_map=None, curvature_map=None, ao_map=None):
        """Generate edge wear mask"""
        
        print(f"\n{'='*60}")
        print(f"Edge Wear Mask Generator")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        # Start with procedural edge detection or use provided maps
        if curvature_map is not None:
            # Use curvature directly
            mask = curvature_map.clone()
            if mask.shape[1:3] != (height, width):
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
        elif normal_map is not None:
            # Derive curvature from normal map
            mask = self._normal_to_curvature(normal_map, height, width)
        else:
            # Create procedural edge pattern
            mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        
        # Add noise variation
        if noise_strength > 0.0:
            noise = self._generate_noise(height, width, noise_scale, seed, device, dtype)
            mask = mask + noise * noise_strength
        
        # Apply edge width and sharpness
        mask = torch.pow(torch.clamp(mask, 0.0, 1.0), 1.0 / edge_sharpness)
        mask = torch.clamp(mask / edge_width, 0.0, 1.0)
        
        # Optionally modulate by AO (wear accumulates in crevices)
        if ao_map is not None:
            ao = ao_map.clone()
            if ao.shape[1:3] != (height, width):
                ao = F.interpolate(
                    ao.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            if ao.shape[3] > 1:
                ao = ao.mean(dim=3, keepdim=True)
            
            # More wear in exposed areas (high AO)
            mask = mask * (1.0 - ao * 0.3)
        
        # Convert to 3-channel for compatibility
        mask = mask.mean(dim=3, keepdim=True).repeat(1, 1, 1, 3)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        print(f"✓ Edge wear mask generated")
        print(f"  Resolution: {width}×{height}")
        print(f"  Edge width: {edge_width:.2f}")
        print(f"  Noise: {noise_strength:.2f}")
        print(f"{'='*60}\n")
        
        return (mask,)
    
    def _normal_to_curvature(self, normal_map, height, width):
        """Derive curvature from normal map"""
        if normal_map.shape[1:3] != (height, width):
            normal_map = F.interpolate(
                normal_map.permute(0, 3, 1, 2),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Simple curvature: how much does normal change
        dx = torch.abs(normal_map[:, :, 1:, :] - normal_map[:, :, :-1, :])
        dy = torch.abs(normal_map[:, 1:, :, :] - normal_map[:, :-1, :, :])
        
        dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')
        
        curv = (dx + dy).mean(dim=3, keepdim=True)
        return curv
    
    def _generate_noise(self, height, width, scale, seed, device, dtype):
        """Generate Perlin-like noise"""
        torch.manual_seed(seed)
        
        # Simple multi-octave noise
        noise = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        
        for octave in range(3):
            freq = scale * (2 ** octave)
            amp = 1.0 / (2 ** octave)
            
            grid_h = int(height / freq) + 2
            grid_w = int(width / freq) + 2
            
            grid = torch.rand((1, grid_h, grid_w, 1), device=device, dtype=dtype)
            upsampled = F.interpolate(
                grid.permute(0, 3, 1, 2),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            noise = noise + upsampled * amp
        
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
        return noise


class DirtGrimeMaskGenerator:
    """
    Generate procedural dirt and grime accumulation masks
    Based on AO and crevice detection
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "density": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Overall dirt coverage"
                }),
                "scale": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Size of dirt patches"
                }),
                "crevice_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Dirt accumulation in crevices"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Dirt pattern contrast"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ao_map": ("IMAGE",),
                "height_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Masks"
    
    def generate(self, width, height, density, scale, crevice_strength, contrast, seed,
                 ao_map=None, height_map=None):
        """Generate dirt/grime mask"""
        
        print(f"\n{'='*60}")
        print(f"Dirt/Grime Mask Generator")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        torch.manual_seed(seed)
        
        # Generate base noise pattern
        mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        
        # Multi-scale noise for natural dirt patterns
        for octave in range(4):
            freq = scale * (2 ** octave)
            amp = 1.0 / (2 ** (octave + 1))
            
            grid_h = max(int(height / freq), 2) + 2
            grid_w = max(int(width / freq), 2) + 2
            
            grid = torch.rand((1, grid_h, grid_w, 1), device=device, dtype=dtype)
            upsampled = F.interpolate(
                grid.permute(0, 3, 1, 2),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            mask = mask + upsampled * amp
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-7)
        
        # Apply contrast
        mask = torch.pow(mask, 1.0 / contrast)
        
        # Apply density threshold
        mask = torch.clamp((mask - (1.0 - density)) / density, 0.0, 1.0)
        
        # Modulate by AO (dirt accumulates in crevices)
        if ao_map is not None:
            ao = ao_map.clone()
            if ao.shape[1:3] != (height, width):
                ao = F.interpolate(
                    ao.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            if ao.shape[3] > 1:
                ao = ao.mean(dim=3, keepdim=True)
            
            # More dirt in dark areas (low AO = crevices)
            ao_factor = 1.0 - ao
            mask = mask + ao_factor * crevice_strength
        
        # Modulate by height (dirt in low areas)
        if height_map is not None:
            hm = height_map.clone()
            if hm.shape[1:3] != (height, width):
                hm = F.interpolate(
                    hm.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            if hm.shape[3] > 1:
                hm = hm.mean(dim=3, keepdim=True)
            
            # More dirt in low areas
            height_factor = 1.0 - hm
            mask = mask + height_factor * crevice_strength * 0.5
        
        # Convert to 3-channel
        mask = mask.repeat(1, 1, 1, 3)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        print(f"✓ Dirt/grime mask generated")
        print(f"  Resolution: {width}×{height}")
        print(f"  Density: {density:.2f}")
        print(f"  Scale: {scale:.1f}")
        print(f"{'='*60}\n")
        
        return (mask,)


class GradientMaskGenerator:
    """
    Generate gradient masks with multiple styles and noise overlay
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "gradient_type": (["linear", "radial", "angle", "diamond", "square"],),
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Gradient angle (degrees)"
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Center X position (0-1)"
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Center Y position (0-1)"
                }),
                "noise_amount": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Add noise variation"
                }),
                "noise_scale": ("FLOAT", {
                    "default": 20.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "invert": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Masks"
    
    def generate(self, width, height, gradient_type, angle, center_x, center_y,
                 noise_amount, noise_scale, invert, seed):
        """Generate gradient mask"""
        
        print(f"\n{'='*60}")
        print(f"Gradient Mask Generator - {gradient_type.upper()}")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        cx = center_x
        cy = center_y
        
        if gradient_type == "linear":
            # Linear gradient based on angle
            rad = math.radians(angle)
            mask = x_coords * math.cos(rad) + y_coords * math.sin(rad)
        
        elif gradient_type == "radial":
            # Radial from center
            dx = x_coords - cx
            dy = y_coords - cy
            mask = torch.sqrt(dx * dx + dy * dy)
        
        elif gradient_type == "angle":
            # Angular/conical gradient
            dx = x_coords - cx
            dy = y_coords - cy
            mask = (torch.atan2(dy, dx) + math.pi) / (2 * math.pi)
            # Rotate by angle
            mask = (mask + angle / 360.0) % 1.0
        
        elif gradient_type == "diamond":
            # Diamond shape
            dx = torch.abs(x_coords - cx)
            dy = torch.abs(y_coords - cy)
            mask = dx + dy
        
        else:  # square
            # Square shape
            dx = torch.abs(x_coords - cx)
            dy = torch.abs(y_coords - cy)
            mask = torch.max(dx, dy)
        
        # Normalize to 0-1
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-7)
        
        # Add noise if requested
        if noise_amount > 0.0:
            torch.manual_seed(seed)
            noise = torch.rand((height, width), device=device, dtype=dtype)
            
            # Multi-scale noise
            for octave in range(2):
                freq = noise_scale * (2 ** octave)
                amp = 1.0 / (2 ** octave)
                
                grid_h = max(int(height / freq), 2) + 2
                grid_w = max(int(width / freq), 2) + 2
                
                grid = torch.rand((grid_h, grid_w), device=device, dtype=dtype)
                upsampled = F.interpolate(
                    grid.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                
                noise = noise + upsampled * amp
            
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
            mask = mask * (1.0 - noise_amount) + noise * noise_amount
        
        # Invert if requested
        if invert:
            mask = 1.0 - mask
        
        # Convert to image format
        mask = mask.unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, 3)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        print(f"✓ Gradient mask generated")
        print(f"  Type: {gradient_type}")
        print(f"  Resolution: {width}×{height}")
        print(f"  Noise: {noise_amount:.2f}")
        print(f"{'='*60}\n")
        
        return (mask,)


class ColorSelectionMask:
    """
    Create masks by selecting color ranges (like Photoshop's Color Range)
    Perfect for isolating specific colors for editing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {
                    "default": "#FF0000",
                    "tooltip": "Target color in hex format (#RRGGBB)"
                }),
                "tolerance": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Color matching tolerance"
                }),
                "feather": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge softness"
                }),
                "mode": (["rgb", "hsv", "luminance"],),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select_color"
    CATEGORY = "Texture Alchemist/Masks"
    
    def select_color(self, image, target_color, tolerance, feather, mode, invert):
        """Create mask from color selection"""
        
        print(f"\n{'='*60}")
        print(f"Color Selection Mask")
        print(f"{'='*60}")
        print(f"  Target color: {target_color}")
        print(f"  Mode: {mode}")
        print(f"  Tolerance: {tolerance:.2f}")
        
        # Parse hex color
        target_color = target_color.lstrip('#')
        r = int(target_color[0:2], 16) / 255.0
        g = int(target_color[2:4], 16) / 255.0
        b = int(target_color[4:6], 16) / 255.0
        
        target = torch.tensor([r, g, b], device=image.device, dtype=image.dtype)
        
        if mode == "rgb":
            # RGB distance
            diff = image - target.view(1, 1, 1, 3)
            distance = torch.sqrt((diff * diff).sum(dim=3, keepdim=True))
            max_dist = math.sqrt(3.0)  # Maximum RGB distance
            distance = distance / max_dist
        
        elif mode == "hsv":
            # HSV distance (more perceptual)
            hsv_image = self._rgb_to_hsv(image)
            hsv_target = self._rgb_to_hsv(target.view(1, 1, 1, 3))
            
            # Hue is circular, handle wrap-around
            h_diff = torch.abs(hsv_image[:, :, :, 0:1] - hsv_target[:, :, :, 0:1])
            h_diff = torch.min(h_diff, 1.0 - h_diff) * 2.0  # Wrap-around
            
            s_diff = torch.abs(hsv_image[:, :, :, 1:2] - hsv_target[:, :, :, 1:2])
            v_diff = torch.abs(hsv_image[:, :, :, 2:3] - hsv_target[:, :, :, 2:3])
            
            distance = torch.sqrt(h_diff * h_diff + s_diff * s_diff + v_diff * v_diff)
            distance = distance / math.sqrt(3.0)
        
        else:  # luminance
            # Luminance distance
            lum_image = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
            lum_target = 0.299 * r + 0.587 * g + 0.114 * b
            distance = torch.abs(lum_image - lum_target)
        
        # Convert distance to mask with tolerance and feather
        mask = 1.0 - torch.clamp(distance / (tolerance + 1e-7), 0.0, 1.0)
        
        # Apply feather (smooth edges)
        if feather > 0.0:
            feather_range = feather
            mask = torch.where(
                mask > (1.0 - feather_range),
                mask,
                torch.where(
                    mask < feather_range,
                    torch.zeros_like(mask),
                    (mask - feather_range) / (1.0 - 2.0 * feather_range)
                )
            )
        
        # Invert if requested
        if invert:
            mask = 1.0 - mask
        
        # Convert to 3-channel
        mask = mask.repeat(1, 1, 1, 3)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        print(f"✓ Color selection mask created")
        print(f"  Pixels selected: {(mask.mean() * 100):.1f}%")
        print(f"{'='*60}\n")
        
        return (mask,)
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r = rgb[:, :, :, 0]
        g = rgb[:, :, :, 1]
        b = rgb[:, :, :, 2]
        
        max_c = torch.max(torch.max(r, g), b)
        min_c = torch.min(torch.min(r, g), b)
        diff = max_c - min_c
        
        # Hue
        h = torch.zeros_like(max_c)
        mask = diff != 0
        
        h = torch.where((max_c == r) & mask, ((g - b) / (diff + 1e-7)) % 6.0, h)
        h = torch.where((max_c == g) & mask, ((b - r) / (diff + 1e-7)) + 2.0, h)
        h = torch.where((max_c == b) & mask, ((r - g) / (diff + 1e-7)) + 4.0, h)
        h = h / 6.0
        
        # Saturation
        s = torch.where(max_c != 0, diff / (max_c + 1e-7), torch.zeros_like(max_c))
        
        # Value
        v = max_c
        
        return torch.stack([h, s, v], dim=3)


# Node registration
NODE_CLASS_MAPPINGS = {
    "EdgeWearMaskGenerator": EdgeWearMaskGenerator,
    "DirtGrimeMaskGenerator": DirtGrimeMaskGenerator,
    "GradientMaskGenerator": GradientMaskGenerator,
    "ColorSelectionMask": ColorSelectionMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeWearMaskGenerator": "Edge Wear Mask Generator",
    "DirtGrimeMaskGenerator": "Dirt/Grime Mask Generator",
    "GradientMaskGenerator": "Gradient Mask Generator",
    "ColorSelectionMask": "Color Selection Mask",
}

