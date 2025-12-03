"""
Advanced Material Generation for TextureAlchemy
SSS, Anisotropy, Translucency, Emission maps
"""

import torch
import torch.nn.functional as F


class SSSMapGenerator:
    """
    Generate Subsurface Scattering maps from albedo and thickness
    For skin, wax, jade, marble materials
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE",),
                "scatter_color": ("STRING", {
                    "default": "#FFCCAA",
                    "tooltip": "SSS tint color (e.g., #FFCCAA for skin)"
                }),
                "scatter_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "depth_influence": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "How much darker areas scatter more"
                }),
            },
            "optional": {
                "thickness": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sss_map",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Materials"
    
    def generate(self, albedo, scatter_color, scatter_strength, depth_influence, thickness=None):
        """Generate SSS map"""
        
        print(f"\n{'='*60}")
        print(f"SSS Map Generator")
        print(f"{'='*60}")
        
        # Parse scatter color
        scatter_color = scatter_color.lstrip('#')
        r = int(scatter_color[0:2], 16) / 255.0
        g = int(scatter_color[2:4], 16) / 255.0
        b = int(scatter_color[4:6], 16) / 255.0
        
        color_tint = torch.tensor([r, g, b], device=albedo.device, dtype=albedo.dtype)
        
        # Base SSS from albedo luminosity
        lum = 0.299 * albedo[:, :, :, 0] + 0.587 * albedo[:, :, :, 1] + 0.114 * albedo[:, :, :, 2]
        
        # Darker areas scatter more
        sss_strength = (1.0 - lum) * depth_influence + (1.0 - depth_influence)
        sss_strength = sss_strength.unsqueeze(3)
        
        # Apply thickness if provided
        if thickness is not None:
            if thickness.shape[1:3] != albedo.shape[1:3]:
                thickness = F.interpolate(
                    thickness.permute(0, 3, 1, 2),
                    size=(albedo.shape[1], albedo.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            thick_val = thickness.mean(dim=3, keepdim=True)
            sss_strength = sss_strength * thick_val
        
        # Apply scatter color
        sss_map = color_tint.view(1, 1, 1, 3) * sss_strength * scatter_strength
        
        # Blend with albedo
        result = albedo * (1.0 - scatter_strength) + sss_map
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ SSS map generated")
        print(f"  Scatter color: #{scatter_color}")
        print(f"  Strength: {scatter_strength:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


class AnisotropyMapGenerator:
    """
    Generate anisotropy maps for brushed metal, hair, fabric
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "pattern_type": (["brushed_horizontal", "brushed_vertical", "brushed_circular", "hair_flow", "custom_angle"],),
                "angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Brush/flow direction in degrees"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "variation": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Random variation in direction"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anisotropy",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Materials"
    
    def generate(self, width, height, pattern_type, angle, strength, variation, seed):
        """Generate anisotropy map"""
        
        print(f"\n{'='*60}")
        print(f"Anisotropy Map Generator - {pattern_type.upper()}")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        torch.manual_seed(seed)
        
        # Create coordinate grid
        y_coords = torch.linspace(-1, 1, height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.linspace(-1, 1, width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        # Generate direction map based on pattern
        if pattern_type == "brushed_horizontal":
            direction = torch.zeros_like(x_coords)
        elif pattern_type == "brushed_vertical":
            direction = torch.ones_like(x_coords) * 90.0
        elif pattern_type == "brushed_circular":
            direction = torch.atan2(y_coords, x_coords) * 180.0 / 3.14159
        elif pattern_type == "hair_flow":
            # Wavy flow pattern
            direction = x_coords * 45.0 + torch.sin(y_coords * 5.0) * 15.0
        else:  # custom_angle
            direction = torch.ones_like(x_coords) * angle
        
        # Add variation
        if variation > 0.0:
            noise = torch.rand((height, width), device=device, dtype=dtype) * 2.0 - 1.0
            direction = direction + noise * variation * 45.0
        
        # Convert direction to tangent vectors (normalized)
        # Store as RGB where RG = tangent XY, B = strength
        import math
        dir_rad = direction * math.pi / 180.0
        
        tangent_x = torch.cos(dir_rad) * strength
        tangent_y = torch.sin(dir_rad) * strength
        
        # Normalize to 0-1 range
        tangent_x = (tangent_x + 1.0) * 0.5
        tangent_y = (tangent_y + 1.0) * 0.5
        
        aniso_map = torch.stack([tangent_x, tangent_y, torch.ones_like(tangent_x) * strength], dim=2)
        aniso_map = aniso_map.unsqueeze(0)
        aniso_map = torch.clamp(aniso_map, 0.0, 1.0)
        
        print(f"✓ Anisotropy map generated")
        print(f"  Pattern: {pattern_type}")
        print(f"  Angle: {angle:.1f}°")
        print(f"  Strength: {strength:.2f}")
        print(f"{'='*60}\n")
        
        return (aniso_map,)


class TranslucencyMapGenerator:
    """
    Generate translucency maps for leaves, paper, thin fabrics
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE",),
                "translucency_color": ("STRING", {
                    "default": "#88FF88",
                    "tooltip": "Color when backlit (e.g., green for leaves)"
                }),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "thickness": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("translucency",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Materials"
    
    def generate(self, albedo, translucency_color, strength, thickness=None):
        """Generate translucency map"""
        
        print(f"\n{'='*60}")
        print(f"Translucency Map Generator")
        print(f"{'='*60}")
        
        # Parse color
        translucency_color = translucency_color.lstrip('#')
        r = int(translucency_color[0:2], 16) / 255.0
        g = int(translucency_color[2:4], 16) / 255.0
        b = int(translucency_color[4:6], 16) / 255.0
        
        trans_color = torch.tensor([r, g, b], device=albedo.device, dtype=albedo.dtype)
        
        # Base translucency from albedo brightness
        lum = 0.299 * albedo[:, :, :, 0] + 0.587 * albedo[:, :, :, 1] + 0.114 * albedo[:, :, :, 2]
        trans_strength = lum.unsqueeze(3) * strength
        
        # Apply thickness if provided
        if thickness is not None:
            if thickness.shape[1:3] != albedo.shape[1:3]:
                thickness = F.interpolate(
                    thickness.permute(0, 3, 1, 2),
                    size=(albedo.shape[1], albedo.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            thick_val = 1.0 - thickness.mean(dim=3, keepdim=True)  # Thinner = more translucent
            trans_strength = trans_strength * thick_val
        
        # Apply translucency color
        result = trans_color.view(1, 1, 1, 3) * trans_strength
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Translucency map generated")
        print(f"  Color: #{translucency_color}")
        print(f"  Strength: {strength:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


class EmissionMaskGenerator:
    """
    Generate emission masks from brightness threshold or patterns
    For glowing elements, lights, screens, magic effects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["brightness_threshold", "color_select", "procedural"],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Emission intensity"
                }),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "emission_color": ("STRING", {"default": "#FFFFFF"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "pattern_scale": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0, "display": "number"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("emission",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Materials"
    
    def generate(self, mode, strength, source_image=None, threshold=0.7, emission_color="#FFFFFF",
                 width=512, height=512, pattern_scale=20.0, seed=0):
        """Generate emission mask"""
        
        print(f"\n{'='*60}")
        print(f"Emission Mask Generator - {mode.upper()}")
        print(f"{'='*60}")
        
        if mode == "brightness_threshold" and source_image is not None:
            # Create emission from bright areas
            lum = 0.299 * source_image[:, :, :, 0] + 0.587 * source_image[:, :, :, 1] + 0.114 * source_image[:, :, :, 2]
            mask = torch.clamp((lum - threshold) / (1.0 - threshold), 0.0, 1.0)
            emission = source_image * mask.unsqueeze(3) * strength
        
        elif mode == "color_select" and source_image is not None:
            # Use specific color as emission
            color = emission_color.lstrip('#')
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            
            target = torch.tensor([r, g, b], device=source_image.device, dtype=source_image.dtype)
            
            # Color distance
            diff = source_image - target.view(1, 1, 1, 3)
            dist = torch.sqrt((diff * diff).sum(dim=3, keepdim=True))
            
            # Create mask from similar colors
            mask = torch.clamp(1.0 - dist / threshold, 0.0, 1.0)
            emission = source_image * mask * strength
        
        else:  # procedural
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32
            
            torch.manual_seed(seed)
            
            # Generate procedural emission pattern
            noise = torch.rand((1, height, width, 1), device=device, dtype=dtype)
            
            # Multi-scale
            for octave in range(3):
                freq = pattern_scale * (2 ** octave)
                amp = 1.0 / (2 ** octave)
                
                grid_h = max(int(height / freq), 2) + 2
                grid_w = max(int(width / freq), 2) + 2
                
                grid = torch.rand((1, grid_h, grid_w, 1), device=device, dtype=dtype)
                upsampled = F.interpolate(
                    grid.permute(0, 3, 1, 2),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
                
                noise = noise + upsampled * amp
            
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
            
            # Apply color
            color = emission_color.lstrip('#')
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            
            color_vec = torch.tensor([r, g, b], device=device, dtype=dtype)
            
            emission = noise * color_vec.view(1, 1, 1, 3) * strength
        
        emission = torch.clamp(emission, 0.0, 1.0)
        
        print(f"✓ Emission mask generated")
        print(f"  Mode: {mode}")
        print(f"  Strength: {strength:.2f}")
        print(f"{'='*60}\n")
        
        return (emission,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SSSMapGenerator": SSSMapGenerator,
    "AnisotropyMapGenerator": AnisotropyMapGenerator,
    "TranslucencyMapGenerator": TranslucencyMapGenerator,
    "EmissionMaskGenerator": EmissionMaskGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSSMapGenerator": "SSS Map Generator",
    "AnisotropyMapGenerator": "Anisotropy Map Generator",
    "TranslucencyMapGenerator": "Translucency Map Generator",
    "EmissionMaskGenerator": "Emission Mask Generator",
}

