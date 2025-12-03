"""
Map Utilities
Height and AO map processing
"""

import torch


class LotusHeightProcessor:
    """
    Process Lotus depth/height map output
    Connect: LotusSampler → VAEDecode → This node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_depth": ("IMAGE",),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert height values"
                }),
                "bit_depth": (["8-bit", "16-bit", "32-bit"], {
                    "default": "16-bit",
                    "tooltip": "Output bit depth (save as EXR for 16/32-bit)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("height",)
    FUNCTION = "process"
    CATEGORY = "Texture Alchemist/Maps"
    
    def process(self, lotus_depth, invert, bit_depth):
        """Process height map"""
        
        height = lotus_depth.clone()
        
        # Convert to grayscale
        if height.shape[-1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                   device=height.device, dtype=height.dtype)
            height = torch.sum(height * weights, dim=-1, keepdim=True)
            height = height.repeat(1, 1, 1, 3)
        
        # Normalize to 0-1
        h_min = height.min()
        h_max = height.max()
        if h_max - h_min > 1e-6:
            height = (height - h_min) / (h_max - h_min)
        
        # Invert if requested
        if invert:
            height = 1.0 - height
        
        print(f"✓ Height processed ({bit_depth})")
        if bit_depth in ["16-bit", "32-bit"]:
            print(f"  Note: Save as OpenEXR to preserve {bit_depth} precision")
        
        return (height,)


class AOApproximator:
    """
    Generate Ambient Occlusion map from height and normal maps
    Uses height sampling to detect concave areas that would be occluded
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Sampling radius in pixels (larger = broader occlusion)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "AO intensity multiplier"
                }),
                "samples": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of sampling directions (more = better quality, slower)"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Contrast adjustment for AO"
                }),
                "use_normal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use normal map to improve AO quality (if provided)"
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate_ao"
    CATEGORY = "Texture Alchemist/Maps"
    
    def generate_ao(self, height, radius, strength, samples, contrast, use_normal, normal=None):
        """Generate AO from height map"""
        
        print("\n" + "="*60)
        print("AO Approximator")
        print("="*60)
        print(f"Height shape: {height.shape}")
        if normal is not None:
            print(f"Normal shape: {normal.shape}")
        print(f"Radius: {radius}, Samples: {samples}, Strength: {strength}, Contrast: {contrast}")
        print(f"Use normal: {use_normal}")
        
        # Convert height to grayscale if needed
        height_map = height.clone()
        if height_map.shape[-1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                   device=height_map.device, dtype=height_map.dtype)
            height_map = torch.sum(height_map * weights, dim=-1, keepdim=True)
        elif height_map.shape[-1] != 1:
            height_map = height_map[..., 0:1]
        
        batch, h, w, channels = height.shape
        
        # Initialize AO map (start fully lit)
        ao = torch.ones((batch, h, w, 1), device=height.device, dtype=height.dtype)
        
        # Generate sampling angles
        import math
        angles = [2.0 * math.pi * i / samples for i in range(samples)]
        
        print(f"\n⚙ Computing AO with {samples} samples at radius {radius}...")
        
        # For each sampling direction
        occlusion_sum = torch.zeros_like(ao)
        
        for angle in angles:
            # Calculate offset
            dx = int(round(math.cos(angle) * radius))
            dy = int(round(math.sin(angle) * radius))
            
            # Skip if no offset
            if dx == 0 and dy == 0:
                continue
            
            # Sample height at offset position (with boundary handling)
            # Pad the height map to handle boundaries
            pad_h = abs(dy) if dy != 0 else 0
            pad_w = abs(dx) if dx != 0 else 0
            
            # Use reflection padding
            height_padded = torch.nn.functional.pad(
                height_map.permute(0, 3, 1, 2),  # BHWC -> BCHW
                (pad_w, pad_w, pad_h, pad_h),
                mode='replicate'
            ).permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Calculate the sampling positions
            y_start = max(0, dy) + pad_h
            y_end = y_start + h
            x_start = max(0, dx) + pad_w
            x_end = x_start + w
            
            # Center position
            center_y = pad_h
            center_x = pad_w
            
            # Sample neighbor and center
            neighbor_height = height_padded[:, y_start:y_end, x_start:x_end, :]
            center_height = height_padded[:, center_y:center_y+h, center_x:center_x+w, :]
            
            # Calculate height difference (positive if neighbor is higher = occlusion)
            height_diff = neighbor_height - center_height
            
            # Convert to occlusion (only positive differences contribute)
            occlusion = torch.clamp(height_diff, 0.0, 1.0)
            
            occlusion_sum += occlusion
        
        # Average the occlusion
        if samples > 0:
            occlusion_avg = occlusion_sum / samples
        else:
            occlusion_avg = occlusion_sum
        
        # Apply strength
        occlusion_avg = occlusion_avg * strength
        
        # Convert to AO (1.0 = no occlusion, 0.0 = full occlusion)
        ao = 1.0 - torch.clamp(occlusion_avg, 0.0, 1.0)
        
        # Apply contrast
        if contrast != 1.0:
            ao = (ao - 0.5) * contrast + 0.5
            ao = torch.clamp(ao, 0.0, 1.0)
        
        # Optional: Use normal map to bias AO
        if use_normal and normal is not None:
            print("✓ Applying normal-based bias")
            
            # Convert normal to grayscale or use blue channel (up direction)
            if normal.shape[-1] >= 3:
                # Use blue channel (Z/up direction)
                # Areas facing up get less AO, areas facing down/sideways get more
                normal_z = normal[..., 2:3]
                
                # Resize to match AO if needed
                if normal_z.shape[1:3] != ao.shape[1:3]:
                    normal_z = torch.nn.functional.interpolate(
                        normal_z.permute(0, 3, 1, 2),
                        size=ao.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Blend AO with normal bias
                # Faces pointing up (normal_z close to 1.0) = less AO
                # Faces pointing sideways/down (normal_z close to 0.5 or less) = more AO
                normal_bias = torch.clamp((1.0 - normal_z) * 0.5, 0.0, 0.5)
                ao = ao * (1.0 - normal_bias)
        
        # Convert to RGB
        ao_rgb = ao.repeat(1, 1, 1, 3)
        
        print(f"\n✓ AO generated")
        print(f"  Range: [{ao_rgb.min():.3f}, {ao_rgb.max():.3f}]")
        print("\n" + "="*60)
        print("✓ AO Approximation Complete")
        print("="*60 + "\n")
        
        return (ao_rgb,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LotusHeightProcessor": LotusHeightProcessor,
    "AOApproximator": AOApproximator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LotusHeightProcessor": "Height Processor (Lotus)",
    "AOApproximator": "AO Approximator",
}

