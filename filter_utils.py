"""
Filter Utilities for TextureAlchemy
Denoise, edge detection, enhancement filters
"""

import torch
import torch.nn.functional as F


class DenoiseFilter:
    """
    Noise reduction with detail preservation
    Multiple denoising algorithms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "method": (["bilateral", "non_local_means", "gaussian", "median"],),
                "preserve_edges": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised",)
    FUNCTION = "denoise"
    CATEGORY = "Texture Alchemist/Filters"
    
    def denoise(self, image, strength, method, preserve_edges):
        """Apply denoising"""
        
        print(f"\n{'='*60}")
        print(f"Denoise Filter - {method.upper()}")
        print(f"{'='*60}")
        
        if method == "bilateral":
            result = self._bilateral_filter(image, strength, preserve_edges)
        elif method == "non_local_means":
            result = self._nlm_filter(image, strength)
        elif method == "gaussian":
            result = self._gaussian_denoise(image, strength)
        else:  # median
            result = self._median_filter(image, strength)
        
        # Blend with original based on strength
        result = image * (1.0 - strength) + result * strength
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Denoising applied")
        print(f"  Method: {method}")
        print(f"  Strength: {strength:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)
    
    def _bilateral_filter(self, image, strength, preserve_edges):
        """Bilateral filter (edge-preserving blur)"""
        sigma_space = 5.0 * strength
        sigma_color = 0.1
        
        kernel_size = max(3, int(sigma_space * 2) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Simple bilateral approximation
        gaussian = self._gaussian_denoise(image, strength)
        
        if preserve_edges:
            # Detect edges
            gray = image.mean(dim=3, keepdim=True)
            dx = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
            dy = torch.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
            
            dx = F.pad(dx, (0, 0, 0, 1), mode='replicate')
            dy = F.pad(dy, (0, 0, 0, 0, 0, 1), mode='replicate')
            
            edges = (dx + dy).repeat(1, 1, 1, image.shape[3])
            
            # Preserve edges
            blend = torch.exp(-edges / sigma_color)
            return image * (1.0 - blend) + gaussian * blend
        
        return gaussian
    
    def _nlm_filter(self, image, strength):
        """Non-local means (simplified)"""
        # Simplified patch-based denoising
        # Use Gaussian blur as approximation for performance
        return self._gaussian_denoise(image, strength * 1.5)
    
    def _gaussian_denoise(self, image, strength):
        """Gaussian blur denoising"""
        sigma = 2.0 * strength
        kernel_size = max(3, int(sigma * 4) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = min(kernel_size, 25)
        
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
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
    
    def _median_filter(self, image, strength):
        """Median filter (approximated with average pooling)"""
        kernel_size = max(3, int(strength * 10) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        image_t = image.permute(0, 3, 1, 2)
        
        filtered = F.avg_pool2d(
            F.pad(image_t, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate'),
            kernel_size=kernel_size,
            stride=1
        )
        
        return filtered.permute(0, 2, 3, 1)


class EdgeDetection:
    """
    Detect edges in images
    Multiple edge detection algorithms
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["sobel", "scharr", "prewitt", "canny_approx", "laplacian"],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge threshold (0=all, 1=strong only)"
                }),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edges",)
    FUNCTION = "detect"
    CATEGORY = "Texture Alchemist/Filters"
    
    def detect(self, image, method, strength, threshold, invert):
        """Detect edges"""
        
        print(f"\n{'='*60}")
        print(f"Edge Detection - {method.upper()}")
        print(f"{'='*60}")
        
        # Convert to grayscale
        gray = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
        gray_t = gray.permute(0, 3, 1, 2)
        
        if method == "sobel":
            sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            dx = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), sobel_x)
            dy = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), sobel_y)
            
        elif method == "scharr":
            scharr_x = torch.tensor([[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]], 
                                    dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            scharr_y = torch.tensor([[[-3, -10, -3], [0, 0, 0], [3, 10, 3]]], 
                                    dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            dx = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), scharr_x)
            dy = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), scharr_y)
            
        elif method == "prewitt":
            prewitt_x = torch.tensor([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]], 
                                     dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            prewitt_y = torch.tensor([[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]], 
                                     dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            dx = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), prewitt_x)
            dy = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), prewitt_y)
            
        elif method == "laplacian":
            laplacian = torch.tensor([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]], 
                                     dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            edges = F.conv2d(F.pad(gray_t, (1, 1, 1, 1), mode='replicate'), laplacian)
            edges = torch.abs(edges)
            
        else:  # canny_approx
            # Simplified Canny (Gaussian + Sobel)
            # Gaussian blur first
            sigma = 1.0
            kernel_size = 5
            x = torch.arange(kernel_size, dtype=gray.dtype, device=gray.device) - kernel_size // 2
            kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            kernel_h = kernel_1d.view(1, 1, 1, -1)
            blurred = F.conv2d(F.pad(gray_t, (2, 2, 0, 0), mode='replicate'), kernel_h)
            
            kernel_v = kernel_1d.view(1, 1, -1, 1)
            blurred = F.conv2d(F.pad(blurred, (0, 0, 2, 2), mode='replicate'), kernel_v)
            
            # Sobel
            sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                                   dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
            
            dx = F.conv2d(F.pad(blurred, (1, 1, 1, 1), mode='replicate'), sobel_x)
            dy = F.conv2d(F.pad(blurred, (1, 1, 1, 1), mode='replicate'), sobel_y)
        
        # Calculate edge magnitude (except for laplacian)
        if method != "laplacian":
            edges = torch.sqrt(dx * dx + dy * dy)
        
        edges = edges.permute(0, 2, 3, 1)
        
        # Normalize
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-7)
        
        # Apply strength and threshold
        edges = edges * strength
        edges = torch.clamp((edges - threshold) / (1.0 - threshold + 1e-7), 0.0, 1.0)
        
        # Invert if requested
        if invert:
            edges = 1.0 - edges
        
        # Convert to 3-channel
        edges = edges.repeat(1, 1, 1, 3)
        
        print(f"✓ Edges detected")
        print(f"  Method: {method}")
        print(f"  Strength: {strength:.2f}")
        print(f"  Threshold: {threshold:.2f}")
        print(f"{'='*60}\n")
        
        return (edges,)


class ImageEnhancement:
    """
    General image enhancement
    Sharpen, contrast, vibrance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "vibrance": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("enhanced",)
    FUNCTION = "enhance"
    CATEGORY = "Texture Alchemist/Filters"
    
    def enhance(self, image, sharpen, contrast, vibrance):
        """Apply enhancements"""
        
        print(f"\n{'='*60}")
        print(f"Image Enhancement")
        print(f"{'='*60}")
        
        result = image.clone()
        
        # Sharpen
        if sharpen > 0.0:
            kernel = torch.tensor([[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]], 
                                  dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
            kernel = kernel / 9.0  # Normalize
            
            image_t = result.permute(0, 3, 1, 2)
            
            sharpened = torch.zeros_like(image_t)
            for c in range(image_t.shape[1]):
                sharpened[:, c:c+1, :, :] = F.conv2d(
                    F.pad(image_t[:, c:c+1, :, :], (1, 1, 1, 1), mode='replicate'),
                    kernel
                )
            
            sharpened = sharpened.permute(0, 2, 3, 1)
            result = result + (sharpened - result) * sharpen
        
        # Contrast
        if contrast != 1.0:
            mean = result.mean()
            result = (result - mean) * contrast + mean
        
        # Vibrance (smart saturation)
        if vibrance != 0.0:
            # Convert to HSV
            r, g, b = result[:, :, :, 0], result[:, :, :, 1], result[:, :, :, 2]
            
            max_c = torch.max(torch.max(r, g), b)
            min_c = torch.min(torch.min(r, g), b)
            
            # Saturation
            s = torch.where(max_c != 0, (max_c - min_c) / (max_c + 1e-7), torch.zeros_like(max_c))
            
            # Vibrance affects less saturated colors more
            vibrance_mask = (1.0 - s) * torch.abs(vibrance)
            
            # Adjust saturation
            new_s = s + vibrance_mask * torch.sign(vibrance)
            new_s = torch.clamp(new_s, 0.0, 1.0)
            
            # Convert back (simplified)
            scale = new_s / (s + 1e-7)
            result = torch.stack([
                min_c + (r - min_c) * scale,
                min_c + (g - min_c) * scale,
                min_c + (b - min_c) * scale
            ], dim=3)
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Enhancement applied")
        print(f"  Sharpen: {sharpen:.2f}")
        print(f"  Contrast: {contrast:.2f}")
        print(f"  Vibrance: {vibrance:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DenoiseFilter": DenoiseFilter,
    "EdgeDetection": EdgeDetection,
    "ImageEnhancement": ImageEnhancement,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DenoiseFilter": "Denoise Filter",
    "EdgeDetection": "Edge Detection",
    "ImageEnhancement": "Image Enhancement",
}

