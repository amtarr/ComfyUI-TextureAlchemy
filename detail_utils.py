"""
Detail Control Utilities for TextureAlchemy
Frequency separation, clarity, smart blur, detail enhancement
"""

import torch
import torch.nn.functional as F
import math


class FrequencySeparation:
    """
    Separate texture into high and low frequency components
    Professional texture artist's secret weapon!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "Separation radius (low-pass filter size)"
                }),
                "method": (["gaussian", "median", "bilateral"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("low_frequency", "high_frequency", "preview_split")
    FUNCTION = "separate"
    CATEGORY = "Texture Alchemist/Detail"
    
    def separate(self, image, radius, method):
        """Separate into low and high frequency"""
        
        print(f"\n{'='*60}")
        print(f"Frequency Separation - {method.upper()}")
        print(f"{'='*60}")
        
        # Low frequency = blur
        if method == "gaussian":
            low_freq = self._gaussian_blur(image, radius)
        elif method == "median":
            low_freq = self._median_blur(image, radius)
        else:  # bilateral
            low_freq = self._bilateral_blur(image, radius)
        
        # High frequency = original - low
        high_freq = image - low_freq + 0.5  # Add 0.5 to center at mid-gray
        high_freq = torch.clamp(high_freq, 0.0, 1.0)
        
        # Create preview showing split
        batch, h, w, c = image.shape
        preview = torch.cat([low_freq[:, :, :w//2, :], high_freq[:, :, w//2:, :]], dim=2)
        
        print(f"✓ Frequency separation complete")
        print(f"  Method: {method}")
        print(f"  Radius: {radius:.1f}")
        print(f"  Low freq range: [{low_freq.min():.3f}, {low_freq.max():.3f}]")
        print(f"  High freq range: [{high_freq.min():.3f}, {high_freq.max():.3f}]")
        print(f"{'='*60}\n")
        
        return (low_freq, high_freq, preview)
    
    def _gaussian_blur(self, image, radius):
        """Gaussian blur"""
        sigma = radius / 3.0
        kernel_size = int(radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 99))
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Separable convolution
        image_t = image.permute(0, 3, 1, 2)
        
        # Horizontal
        kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(image_t.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.pad(image_t, (kernel_size//2, kernel_size//2, 0, 0), mode='replicate'),
                          kernel_h, groups=image_t.shape[1])
        
        # Vertical
        kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(blurred.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.pad(blurred, (0, 0, kernel_size//2, kernel_size//2), mode='replicate'),
                          kernel_v, groups=blurred.shape[1])
        
        return blurred.permute(0, 2, 3, 1)
    
    def _median_blur(self, image, radius):
        """Median blur (approximated)"""
        # PyTorch doesn't have native median blur, use max pooling as approximation
        kernel_size = max(3, int(radius))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        image_t = image.permute(0, 3, 1, 2)
        
        # Use average pooling as approximation
        blurred = F.avg_pool2d(
            F.pad(image_t, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate'),
            kernel_size=kernel_size,
            stride=1
        )
        
        return blurred.permute(0, 2, 3, 1)
    
    def _bilateral_blur(self, image, radius):
        """Bilateral blur (edge-preserving, simplified)"""
        # Simplified bilateral filter
        sigma_space = radius / 3.0
        sigma_color = 0.1
        
        # Use Gaussian blur with edge detection
        gaussian = self._gaussian_blur(image, radius)
        
        # Detect edges
        gray = image.mean(dim=3, keepdim=True)
        dx = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        dy = torch.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
        
        dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')
        
        edges = (dx + dy).repeat(1, 1, 1, image.shape[3])
        
        # Preserve edges
        blend = torch.exp(-edges / sigma_color)
        result = image * (1.0 - blend) + gaussian * blend
        
        return result


class FrequencyRecombine:
    """
    Recombine low and high frequency components with adjustable balance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "low_frequency": ("IMAGE",),
                "high_frequency": ("IMAGE",),
                "high_freq_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "High frequency detail strength"
                }),
                "low_freq_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Low frequency base strength"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "recombine"
    CATEGORY = "Texture Alchemist/Detail"
    
    def recombine(self, low_frequency, high_frequency, high_freq_strength, low_freq_strength):
        """Recombine frequencies"""
        
        print(f"\n{'='*60}")
        print(f"Frequency Recombine")
        print(f"{'='*60}")
        
        # Recombine: low + (high - 0.5) * strength
        high_centered = (high_frequency - 0.5) * high_freq_strength
        result = low_frequency * low_freq_strength + high_centered
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Frequencies recombined")
        print(f"  High strength: {high_freq_strength:.2f}")
        print(f"  Low strength: {low_freq_strength:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


class ClarityEnhancer:
    """
    Enhance mid-tone contrast (like Lightroom Clarity)
    Makes textures "pop" without changing overall brightness
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clarity": ("FLOAT", {
                    "default": 0.5,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Clarity amount (0=no change, negative=soften)"
                }),
                "radius": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Effect radius"
                }),
                "protect_shadows": ("BOOLEAN", {"default": True}),
                "protect_highlights": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "enhance"
    CATEGORY = "Texture Alchemist/Detail"
    
    def enhance(self, image, clarity, radius, protect_shadows, protect_highlights):
        """Apply clarity enhancement"""
        
        print(f"\n{'='*60}")
        print(f"Clarity Enhancer")
        print(f"{'='*60}")
        
        # Create luminosity mask
        lum = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
        
        # Create blurred version
        sigma = radius / 3.0
        kernel_size = int(radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 99))
        
        # Gaussian blur
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        image_t = image.permute(0, 3, 1, 2)
        
        # Horizontal blur
        kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(image_t.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.pad(image_t, (kernel_size//2, kernel_size//2, 0, 0), mode='replicate'),
                          kernel_h, groups=image_t.shape[1])
        
        # Vertical blur
        kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(blurred.shape[1], 1, 1, 1)
        blurred = F.conv2d(F.pad(blurred, (0, 0, kernel_size//2, kernel_size//2), mode='replicate'),
                          kernel_v, groups=blurred.shape[1])
        
        blurred = blurred.permute(0, 2, 3, 1)
        
        # High pass for mid-tone contrast
        high_pass = image - blurred
        
        # Apply clarity with protection
        mask = torch.ones_like(lum)
        
        if protect_shadows:
            # Reduce effect in shadows
            shadow_mask = torch.clamp(lum * 2.0, 0.0, 1.0)
            mask = mask * shadow_mask
        
        if protect_highlights:
            # Reduce effect in highlights
            highlight_mask = torch.clamp((1.0 - lum) * 2.0, 0.0, 1.0)
            mask = mask * highlight_mask
        
        mask = mask.repeat(1, 1, 1, image.shape[3])
        
        # Apply clarity
        result = image + high_pass * clarity * mask
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Clarity applied")
        print(f"  Amount: {clarity:.2f}")
        print(f"  Radius: {radius:.1f}")
        print(f"  Protect shadows: {protect_shadows}")
        print(f"  Protect highlights: {protect_highlights}")
        print(f"{'='*60}\n")
        
        return (result,)


class SmartBlur:
    """
    Edge-preserving blur for noise reduction
    Blurs while maintaining sharp edges
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "Blur radius"
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge detection sensitivity"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of blur passes"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "blur"
    CATEGORY = "Texture Alchemist/Detail"
    
    def blur(self, image, radius, edge_threshold, iterations):
        """Apply edge-preserving blur"""
        
        print(f"\n{'='*60}")
        print(f"Smart Blur")
        print(f"{'='*60}")
        
        result = image.clone()
        
        for i in range(iterations):
            # Detect edges
            gray = result.mean(dim=3, keepdim=True)
            
            # Sobel edge detection
            sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            gray_t = gray.permute(0, 3, 1, 2)
            dx = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), sobel_x)
            dy = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), sobel_y)
            
            edges = torch.sqrt(dx * dx + dy * dy).permute(0, 2, 3, 1)
            edges = edges.repeat(1, 1, 1, result.shape[3])
            
            # Create edge mask
            edge_mask = torch.sigmoid((edges - edge_threshold) * 20.0)
            
            # Gaussian blur
            sigma = radius / 3.0
            kernel_size = int(radius * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, 49))
            
            x = torch.arange(kernel_size, dtype=result.dtype, device=result.device) - kernel_size // 2
            kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            result_t = result.permute(0, 3, 1, 2)
            
            # Horizontal blur
            kernel_h = kernel_1d.view(1, 1, 1, -1).repeat(result_t.shape[1], 1, 1, 1)
            blurred = F.conv2d(F.pad(result_t, (kernel_size//2, kernel_size//2, 0, 0), mode='replicate'),
                              kernel_h, groups=result_t.shape[1])
            
            # Vertical blur
            kernel_v = kernel_1d.view(1, 1, -1, 1).repeat(blurred.shape[1], 1, 1, 1)
            blurred = F.conv2d(F.pad(blurred, (0, 0, kernel_size//2, kernel_size//2), mode='replicate'),
                              kernel_v, groups=blurred.shape[1])
            
            blurred = blurred.permute(0, 2, 3, 1)
            
            # Blend based on edges (preserve edges)
            result = result * edge_mask + blurred * (1.0 - edge_mask)
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Smart blur applied")
        print(f"  Radius: {radius:.1f}")
        print(f"  Edge threshold: {edge_threshold:.2f}")
        print(f"  Iterations: {iterations}")
        print(f"{'='*60}\n")
        
        return (result,)


class MicroDetailOverlay:
    """
    Add high-frequency detail layer to textures
    Perfect for adding surface texture to smooth materials
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
                "detail": ("IMAGE",),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Detail map scale/tiling"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Detail strength"
                }),
                "mode": (["overlay", "linear_light", "soft_light", "add"],),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "overlay"
    CATEGORY = "Texture Alchemist/Detail"
    
    def overlay(self, base, detail, scale, intensity, mode, mask=None):
        """Overlay micro detail"""
        
        print(f"\n{'='*60}")
        print(f"Micro Detail Overlay - {mode.upper()}")
        print(f"{'='*60}")
        
        batch, h, w, c = base.shape
        
        # Scale detail if needed
        if scale != 1.0:
            new_h = int(h * scale)
            new_w = int(w * scale)
            detail = F.interpolate(
                detail.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            
            # Tile if scaled up
            if scale > 1.0:
                detail = detail[:, :h, :w, :]
            else:
                # Tile if scaled down
                detail = detail.repeat(1, int(1/scale)+1, int(1/scale)+1, 1)[:, :h, :w, :]
        
        # Resize to match base
        if detail.shape[1:3] != (h, w):
            detail = F.interpolate(
                detail.permute(0, 3, 1, 2),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Apply blend mode
        if mode == "overlay":
            result = torch.where(
                base < 0.5,
                2.0 * base * detail,
                1.0 - 2.0 * (1.0 - base) * (1.0 - detail)
            )
        elif mode == "linear_light":
            result = base + 2.0 * detail - 1.0
        elif mode == "soft_light":
            result = torch.where(
                detail < 0.5,
                2.0 * base * detail + base * base * (1.0 - 2.0 * detail),
                2.0 * base * (1.0 - detail) + torch.sqrt(base) * (2.0 * detail - 1.0)
            )
        else:  # add
            result = base + detail - 0.5
        
        # Apply intensity
        result = base * (1.0 - intensity) + result * intensity
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape[1:3] != (h, w):
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            if mask.shape[3] > 1:
                mask = mask.mean(dim=3, keepdim=True).repeat(1, 1, 1, c)
            
            result = base * (1.0 - mask) + result * mask
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Micro detail applied")
        print(f"  Mode: {mode}")
        print(f"  Scale: {scale:.2f}")
        print(f"  Intensity: {intensity:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "FrequencySeparation": FrequencySeparation,
    "FrequencyRecombine": FrequencyRecombine,
    "ClarityEnhancer": ClarityEnhancer,
    "SmartBlur": SmartBlur,
    "MicroDetailOverlay": MicroDetailOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrequencySeparation": "Frequency Separation",
    "FrequencyRecombine": "Frequency Recombine",
    "ClarityEnhancer": "Clarity Enhancer",
    "SmartBlur": "Smart Blur (Edge-Preserving)",
    "MicroDetailOverlay": "Micro Detail Overlay",
}

