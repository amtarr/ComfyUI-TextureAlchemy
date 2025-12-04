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
    Generate Ambient Occlusion map from height and/or normal maps
    - With height: Uses height sampling to detect concave areas that would be occluded
    - Without height: Uses normal map orientation to approximate basic AO
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "height": ("IMAGE",),
                "normal": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate_ao"
    CATEGORY = "Texture Alchemist/Maps"
    
    def generate_ao(self, radius, strength, samples, contrast, use_normal, height=None, normal=None):
        """Generate AO from height and/or normal map"""
        
        print("\n" + "="*60)
        print("AO Approximator")
        print("="*60)
        
        # Check inputs
        if height is None and normal is None:
            print("⚠ Warning: No height or normal map provided, returning neutral AO")
            # Create a neutral gray AO map (50% occlusion)
            ao_rgb = torch.ones((1, 512, 512, 3), dtype=torch.float32) * 0.7
            print("="*60 + "\n")
            return (ao_rgb,)
        
        if height is not None:
            print(f"Height shape: {height.shape}")
        if normal is not None:
            print(f"Normal shape: {normal.shape}")
        print(f"Radius: {radius}, Samples: {samples}, Strength: {strength}, Contrast: {contrast}")
        print(f"Use normal: {use_normal}")
        
        # Determine reference shape and device
        if height is not None:
            reference = height
            batch, h, w, channels = height.shape
        else:
            reference = normal
            batch, h, w, channels = normal.shape
        
        # Convert height to grayscale if needed
        if height is not None:
            height_map = height.clone()
            if height_map.shape[-1] == 3:
                weights = torch.tensor([0.299, 0.587, 0.114], 
                                       device=height_map.device, dtype=height_map.dtype)
                height_map = torch.sum(height_map * weights, dim=-1, keepdim=True)
            elif height_map.shape[-1] != 1:
                height_map = height_map[..., 0:1]
        else:
            height_map = None
        
        # Initialize AO map (start fully lit)
        ao = torch.ones((batch, h, w, 1), device=reference.device, dtype=reference.dtype)
        
        # If we have a height map, use height-based AO
        if height_map is not None:
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
        else:
            # No height map - generate basic AO from normal orientation if available
            print("\n⚙ No height map - generating basic AO from normal orientation...")
            occlusion_avg = torch.zeros((batch, h, w, 1), device=reference.device, dtype=reference.dtype)
        
        # Apply strength
        occlusion_avg = occlusion_avg * strength
        
        # Convert to AO (1.0 = no occlusion, 0.0 = full occlusion)
        ao = 1.0 - torch.clamp(occlusion_avg, 0.0, 1.0)
        
        # If we don't have height but have normal, generate AO from normal map
        if height_map is None and normal is not None:
            print("✓ Generating AO from normal map orientation")
            
            if normal.shape[-1] >= 3:
                # Use blue channel (Z/up direction) and curvature
                normal_z = normal[..., 2:3]
                
                # Resize to match AO if needed
                if normal_z.shape[1:3] != ao.shape[1:3]:
                    normal_z = torch.nn.functional.interpolate(
                        normal_z.permute(0, 3, 1, 2),
                        size=ao.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Generate AO based on surface orientation
                # Faces pointing up (normal_z ~1.0) = bright (little AO)
                # Faces pointing sideways/down (normal_z ~0.5 or less) = dark (more AO)
                # Map range: 1.0 (up) -> 0.9 AO, 0.5 (sideways) -> 0.6 AO, 0.0 (down) -> 0.3 AO
                ao = normal_z * 0.6 + 0.3
                ao = torch.clamp(ao, 0.0, 1.0)
        
        # Apply contrast
        if contrast != 1.0:
            ao = (ao - 0.5) * contrast + 0.5
            ao = torch.clamp(ao, 0.0, 1.0)
        
        # If we have both height-based AO and normal, use normal to enhance/bias
        if height_map is not None and use_normal and normal is not None:
            print("✓ Applying normal-based bias to height AO")
            
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


class GammaAdjust:
    """
    Apply gamma correction to images
    Gamma < 1.0 brightens, Gamma > 1.0 darkens
    Common values: 0.45 (linear to sRGB), 2.2 (sRGB to linear)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma value (< 1.0 = brighten, > 1.0 = darken, 1.0 = no change)"
                }),
                "per_channel": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply gamma to each channel independently (useful for color correction)"
                }),
            },
            "optional": {
                "gamma_red": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma for red channel (only used if per_channel is enabled)"
                }),
                "gamma_green": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma for green channel (only used if per_channel is enabled)"
                }),
                "gamma_blue": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma for blue channel (only used if per_channel is enabled)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust_gamma"
    CATEGORY = "Texture Alchemist/Adjustment"
    
    def adjust_gamma(self, image, gamma, per_channel, gamma_red=1.0, gamma_green=1.0, gamma_blue=1.0):
        """Apply gamma correction to image"""
        
        print("\n" + "="*60)
        print("Gamma Adjust")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Gamma: {gamma}")
        print(f"Per channel: {per_channel}")
        
        # Check if we need to do anything
        if not per_channel and gamma == 1.0:
            print("✓ Gamma is 1.0, no adjustment needed")
            print("="*60 + "\n")
            return (image,)
        
        if per_channel and gamma_red == 1.0 and gamma_green == 1.0 and gamma_blue == 1.0:
            print("✓ All channel gammas are 1.0, no adjustment needed")
            print("="*60 + "\n")
            return (image,)
        
        # Clamp input to valid range [0, 1]
        image_clamped = torch.clamp(image, 0.0, 1.0)
        
        if per_channel and image.shape[-1] >= 3:
            # Apply different gamma to each channel
            print(f"  Red gamma: {gamma_red}")
            print(f"  Green gamma: {gamma_green}")
            print(f"  Blue gamma: {gamma_blue}")
            
            # Split channels
            r = image_clamped[..., 0:1]
            g = image_clamped[..., 1:2]
            b = image_clamped[..., 2:3]
            
            # Apply gamma to each channel
            r_adjusted = torch.pow(r, gamma_red)
            g_adjusted = torch.pow(g, gamma_green)
            b_adjusted = torch.pow(b, gamma_blue)
            
            # Recombine
            result = torch.cat([r_adjusted, g_adjusted, b_adjusted], dim=-1)
            
            # If there was an alpha channel, preserve it
            if image.shape[-1] == 4:
                alpha = image_clamped[..., 3:4]
                result = torch.cat([result, alpha], dim=-1)
                print("  ✓ Alpha channel preserved")
            elif image.shape[-1] > 4:
                # Preserve extra channels
                extra = image_clamped[..., 3:]
                result = torch.cat([result, extra], dim=-1)
        else:
            # Apply same gamma to all channels
            result = torch.pow(image_clamped, gamma)
        
        print(f"\n✓ Gamma correction applied")
        print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print("="*60 + "\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "LotusHeightProcessor": LotusHeightProcessor,
    "AOApproximator": AOApproximator,
    "GammaAdjust": GammaAdjust,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LotusHeightProcessor": "Height Processor (Lotus)",
    "AOApproximator": "AO Approximator",
    "GammaAdjust": "Gamma Adjust",
}

