"""
Advanced Height/Displacement Tools for TextureAlchemy
Height amplifier, combiner, displacement to vector
"""

import torch
import torch.nn.functional as F


class HeightAmplifier:
    """
    Amplify or compress height range
    Preserves mid-tones while increasing/decreasing contrast
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_map": ("IMAGE",),
                "amount": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Amplification amount (1.0 = no change, >1 = amplify, <1 = compress)"
                }),
                "center_point": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Midpoint to preserve"
                }),
                "method": (["power", "linear", "smooth"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("height",)
    FUNCTION = "amplify"
    CATEGORY = "Texture Alchemist/Height"
    
    def amplify(self, height_map, amount, center_point, method):
        """Amplify height map"""
        
        print(f"\n{'='*60}")
        print(f"Height Amplifier - {method.upper()}")
        print(f"{'='*60}")
        
        # Convert to grayscale if needed
        if height_map.shape[3] > 1:
            height = height_map.mean(dim=3, keepdim=True)
        else:
            height = height_map
        
        if method == "power":
            # Power curve
            centered = height - center_point
            amplified = torch.sign(centered) * torch.pow(torch.abs(centered), 1.0 / amount)
            result = amplified + center_point
        
        elif method == "linear":
            # Linear amplification around center
            result = (height - center_point) * amount + center_point
        
        else:  # smooth
            # Smooth S-curve
            centered = height - center_point
            amplified = centered * amount
            # Apply smooth step
            result = amplified / torch.sqrt(1.0 + amplified * amplified) + center_point
        
        # Convert back to 3-channel
        result = result.repeat(1, 1, 1, 3)
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Height amplified")
        print(f"  Amount: {amount:.2f}x")
        print(f"  Method: {method}")
        print(f"  Center: {center_point:.2f}")
        print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"{'='*60}\n")
        
        return (result,)


class HeightCombiner:
    """
    Combine multiple height maps intelligently
    Layer macro and micro detail
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
            },
            "optional": {
                "layer1": ("IMAGE",),
                "layer1_mode": (["add", "max", "min", "multiply", "screen", "overlay"], {"default": "add"}),
                "layer1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
                
                "layer2": ("IMAGE",),
                "layer2_mode": (["add", "max", "min", "multiply", "screen", "overlay"], {"default": "overlay"}),
                "layer2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "number"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("height",)
    FUNCTION = "combine"
    CATEGORY = "Texture Alchemist/Height"
    
    def combine(self, base, **kwargs):
        """Combine height maps"""
        
        print(f"\n{'='*60}")
        print(f"Height Combiner")
        print(f"{'='*60}")
        
        result = base.clone()
        if result.shape[3] > 1:
            result = result.mean(dim=3, keepdim=True)
        
        layers_applied = 0
        
        for i in range(1, 3):
            layer_key = f"layer{i}"
            mode_key = f"layer{i}_mode"
            strength_key = f"layer{i}_strength"
            
            if layer_key in kwargs and kwargs[layer_key] is not None:
                layer = kwargs[layer_key]
                mode = kwargs.get(mode_key, "add")
                strength = kwargs.get(strength_key, 1.0)
                
                # Resize if needed
                if layer.shape[1:3] != result.shape[1:3]:
                    layer = F.interpolate(
                        layer.permute(0, 3, 1, 2),
                        size=(result.shape[1], result.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Convert to grayscale
                if layer.shape[3] > 1:
                    layer = layer.mean(dim=3, keepdim=True)
                
                # Apply blend mode
                if mode == "add":
                    blended = result + layer - 0.5
                elif mode == "max":
                    blended = torch.max(result, layer)
                elif mode == "min":
                    blended = torch.min(result, layer)
                elif mode == "multiply":
                    blended = result * layer
                elif mode == "screen":
                    blended = 1.0 - (1.0 - result) * (1.0 - layer)
                else:  # overlay
                    blended = torch.where(
                        result < 0.5,
                        2.0 * result * layer,
                        1.0 - 2.0 * (1.0 - result) * (1.0 - layer)
                    )
                
                # Apply strength
                result = result * (1.0 - strength) + blended * strength
                layers_applied += 1
                
                print(f"  Layer {i}: {mode} (strength {strength:.2f})")
        
        # Convert to 3-channel
        result = result.repeat(1, 1, 1, 3)
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"\n✓ Combined {layers_applied} height layers")
        print(f"{'='*60}\n")
        
        return (result,)


class DisplacementToVector:
    """
    Convert height map to XYZ displacement vector map
    For advanced displacement mapping
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height_map": ("IMAGE",),
                "magnitude": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Displacement magnitude"
                }),
                "direction_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "direction_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "direction_z": ("FLOAT", {
                    "default": 1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Default is pure vertical displacement"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vector_displacement",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Height"
    
    def convert(self, height_map, magnitude, direction_x, direction_y, direction_z):
        """Convert to vector displacement"""
        
        print(f"\n{'='*60}")
        print(f"Displacement to Vector")
        print(f"{'='*60}")
        
        # Convert to grayscale
        if height_map.shape[3] > 1:
            height = height_map.mean(dim=3, keepdim=True)
        else:
            height = height_map
        
        # Normalize direction vector
        length = torch.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        if length < 1e-7:
            direction_x, direction_y, direction_z = 0.0, 0.0, 1.0
            length = 1.0
        
        dir_x = direction_x / length
        dir_y = direction_y / length
        dir_z = direction_z / length
        
        # Create vector displacement
        displacement = (height - 0.5) * magnitude  # Center around 0
        
        vec_x = displacement * dir_x
        vec_y = displacement * dir_y
        vec_z = displacement * dir_z
        
        # Normalize to 0-1 range for output
        vec_x = (vec_x + magnitude/2) / magnitude
        vec_y = (vec_y + magnitude/2) / magnitude
        vec_z = (vec_z + magnitude/2) / magnitude
        
        result = torch.cat([vec_x, vec_y, vec_z], dim=3)
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Vector displacement created")
        print(f"  Magnitude: {magnitude:.2f}")
        print(f"  Direction: ({dir_x:.2f}, {dir_y:.2f}, {dir_z:.2f})")
        print(f"{'='*60}\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "HeightAmplifier": HeightAmplifier,
    "HeightCombiner": HeightCombiner,
    "DisplacementToVector": DisplacementToVector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeightAmplifier": "Height Amplifier",
    "HeightCombiner": "Height Combiner",
    "DisplacementToVector": "Displacement to Vector",
}

