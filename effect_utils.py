"""
Effect Utilities
Curvature, wear, detail blending, and masking tools
"""

import torch
import torch.nn.functional as F


class CurvatureGenerator:
    """
    Generate curvature map from normal map or height map
    Detects edges and crevices for wear/dirt masks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_map": ("IMAGE",),
                "input_type": (["normal", "height"], {
                    "default": "normal",
                    "tooltip": "Type of input map"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Curvature detection strength"
                }),
                "blur_radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Smoothing radius"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("curvature",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Effects"
    
    def generate(self, input_map, input_type, strength, blur_radius):
        """Generate curvature map"""
        
        print("\n" + "="*60)
        print("Curvature Generator")
        print("="*60)
        print(f"Input shape: {input_map.shape}")
        print(f"Input type: {input_type}")
        print(f"Strength: {strength}")
        
        # Convert to grayscale
        if input_map.shape[-1] > 1:
            if input_type == "normal":
                # For normals, use Z channel (blue)
                gray = input_map[:, :, :, 2:3]
            else:
                # For height, use luminance
                weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                       device=input_map.device, dtype=input_map.dtype)
                gray = torch.sum(input_map * weights, dim=-1, keepdim=True)
        else:
            gray = input_map
        
        # Compute second derivative (curvature)
        gray_bchw = gray.permute(0, 3, 1, 2)
        
        # Laplacian kernel (detects edges and curvature)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                 device=input_map.device, dtype=input_map.dtype).view(1, 1, 3, 3) / 4.0
        
        curvature = F.conv2d(gray_bchw, laplacian, padding=1)
        
        # Apply strength
        curvature = curvature * strength
        
        # Normalize and enhance
        curvature = torch.abs(curvature)
        
        # Apply blur if requested
        if blur_radius > 0:
            kernel_size = int(blur_radius * 4) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = blur_radius
            
            # Gaussian blur
            x = torch.arange(kernel_size, device=input_map.device, dtype=input_map.dtype)
            kernel = torch.exp(-0.5 * ((x - kernel_size // 2) / sigma) ** 2)
            kernel = kernel / kernel.sum()
            
            kernel_h = kernel.view(1, 1, 1, -1)
            kernel_v = kernel.view(1, 1, -1, 1)
            
            padding = kernel_size // 2
            curvature = F.pad(curvature, (padding, padding, 0, 0), mode='replicate')
            curvature = F.conv2d(curvature, kernel_h)
            curvature = F.pad(curvature, (0, 0, padding, padding), mode='replicate')
            curvature = F.conv2d(curvature, kernel_v)
        
        # Normalize to 0-1
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
        
        # Convert to RGB
        curvature = curvature.permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
        
        print(f"✓ Curvature map generated")
        print(f"  Output shape: {curvature.shape}")
        print("="*60 + "\n")
        
        return (curvature,)


class DetailMapBlender:
    """
    Blend detail maps onto base maps without washing out
    Uses overlay/detail blending for normals and roughness
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
                "detail": ("IMAGE",),
                "map_type": (["normal", "roughness", "generic"], {
                    "default": "generic",
                    "tooltip": "Map type determines blend method"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Detail strength (0=no detail, 1=full)"
                }),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended",)
    FUNCTION = "blend"
    CATEGORY = "Texture Alchemist/Effects"
    
    def blend(self, base, detail, map_type, strength, mask=None):
        """Blend detail map onto base"""
        
        print("\n" + "="*60)
        print("Detail Map Blender")
        print("="*60)
        print(f"Base shape: {base.shape}")
        print(f"Detail shape: {detail.shape}")
        print(f"Map type: {map_type}")
        print(f"Strength: {strength}")
        
        # Resize detail to match base if needed
        if detail.shape[1:3] != base.shape[1:3]:
            detail = self._resize_to_match(detail, base)
        
        # Apply blending based on map type
        if map_type == "normal":
            # RNM (Reoriented Normal Mapping) for normals
            result = self._blend_normals_rnm(base, detail, strength)
        elif map_type == "roughness":
            # Multiply/overlay for roughness
            result = base * (1.0 + (detail - 0.5) * strength)
        else:
            # Generic overlay blend
            result = base + (detail - 0.5) * strength
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape[1:3] != base.shape[1:3]:
                mask = self._resize_to_match(mask, base)
            if mask.shape[-1] != 1:
                mask_weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                            device=mask.device, dtype=mask.dtype)
                mask = torch.sum(mask * mask_weights, dim=-1, keepdim=True)
            
            result = base * (1.0 - mask) + result * mask
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Detail blended")
        print("="*60 + "\n")
        
        return (result,)
    
    def _blend_normals_rnm(self, base, detail, strength):
        """Reoriented Normal Mapping blend"""
        # Convert from [0,1] to [-1,1]
        base_n = base * 2.0 - 1.0
        detail_n = detail * 2.0 - 1.0
        
        # Apply strength to detail
        detail_n = detail_n * strength
        
        # RNM blend
        result_n = torch.zeros_like(base_n)
        result_n[:, :, :, 0] = base_n[:, :, :, 0] + detail_n[:, :, :, 0]
        result_n[:, :, :, 1] = base_n[:, :, :, 1] + detail_n[:, :, :, 1]
        if base_n.shape[-1] >= 3:
            result_n[:, :, :, 2] = base_n[:, :, :, 2] * detail_n[:, :, :, 2]
        
        # Normalize
        length = torch.sqrt(torch.sum(result_n ** 2, dim=-1, keepdim=True)) + 1e-8
        result_n = result_n / length
        
        # Convert back to [0,1]
        result = (result_n + 1.0) / 2.0
        
        return result
    
    def _resize_to_match(self, source, target):
        """Resize source to match target"""
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        resized = F.interpolate(source_bchw, size=(target_h, target_w), 
                               mode='bilinear', align_corners=False)
        return resized.permute(0, 2, 3, 1)


class WearGenerator:
    """
    Generate wear and edge damage for materials
    Uses curvature and AO for realistic wear patterns
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE",),
                "wear_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Overall wear intensity"
                }),
                "edge_wear": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Wear on edges (needs curvature or height)"
                }),
                "dirt_strength": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Dirt accumulation in crevices"
                }),
            },
            "optional": {
                "curvature": ("IMAGE",),
                "ao": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("worn_albedo", "wear_mask")
    FUNCTION = "generate_wear"
    CATEGORY = "Texture Alchemist/Effects"
    
    def generate_wear(self, albedo, wear_strength, edge_wear, dirt_strength, 
                      curvature=None, ao=None):
        """Generate wear effects"""
        
        print("\n" + "="*60)
        print("Wear Generator")
        print("="*60)
        print(f"Albedo shape: {albedo.shape}")
        print(f"Wear strength: {wear_strength}")
        
        batch, height, width, channels = albedo.shape
        device = albedo.device
        dtype = albedo.dtype
        
        # Create wear mask
        wear_mask = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Add edge wear from curvature
        if curvature is not None and edge_wear > 0:
            curv = self._to_grayscale(curvature)
            if curv.shape[1:3] != albedo.shape[1:3]:
                curv = self._resize_to_match(curv, albedo)
            wear_mask = wear_mask + curv * edge_wear
        
        # Add dirt from AO (inverted)
        if ao is not None and dirt_strength > 0:
            ao_gray = self._to_grayscale(ao)
            if ao_gray.shape[1:3] != albedo.shape[1:3]:
                ao_gray = self._resize_to_match(ao_gray, albedo)
            dirt = (1.0 - ao_gray) * dirt_strength
            wear_mask = wear_mask + dirt
        
        # Normalize and apply strength
        wear_mask = torch.clamp(wear_mask * wear_strength, 0.0, 1.0)
        
        # Apply wear to albedo (darken)
        worn_albedo = albedo * (1.0 - wear_mask * 0.5)
        
        # Convert mask to RGB for preview
        wear_mask_rgb = wear_mask.repeat(1, 1, 1, 3)
        
        print(f"✓ Wear generated")
        print(f"  Worn albedo: {worn_albedo.shape}")
        print(f"  Wear mask: {wear_mask_rgb.shape}")
        print("="*60 + "\n")
        
        return (worn_albedo, wear_mask_rgb)
    
    def _to_grayscale(self, image):
        """Convert to grayscale"""
        if image.shape[-1] == 1:
            return image
        weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                               device=image.device, dtype=image.dtype)
        return torch.sum(image * weights, dim=-1, keepdim=True)
    
    def _resize_to_match(self, source, target):
        """Resize source to match target"""
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        resized = F.interpolate(source_bchw, size=(target_h, target_w), 
                               mode='bilinear', align_corners=False)
        return resized.permute(0, 2, 3, 1)


class GradientMap:
    """
    Create masks from gradients (like Color Ramp but outputs grayscale masks)
    Useful for creating selection masks from height/AO/curvature
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_range_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Values below this become black"
                }),
                "input_range_max": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Values above this become white"
                }),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask"
                }),
                "smoothness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge smoothness (0=hard, >0=soft)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "create_mask"
    CATEGORY = "Texture Alchemist/Effects"
    
    def create_mask(self, image, input_range_min, input_range_max, invert, smoothness):
        """Create gradient mask"""
        
        print("\n" + "="*60)
        print("Gradient Map")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Range: [{input_range_min:.2f}, {input_range_max:.2f}]")
        
        # Convert to grayscale
        if image.shape[-1] > 1:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            gray = torch.sum(image * weights, dim=-1, keepdim=True)
        else:
            gray = image
        
        # Remap range
        mask = (gray - input_range_min) / (input_range_max - input_range_min + 1e-8)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Apply smoothness (ease in/out)
        if smoothness > 0:
            # Smoothstep function
            t = mask
            mask = t * t * (3.0 - 2.0 * t)
            # Additional smoothing passes
            smooth_passes = int(smoothness * 10)
            for _ in range(smooth_passes):
                mask = mask * mask * (3.0 - 2.0 * mask)
        
        # Invert if requested
        if invert:
            mask = 1.0 - mask
        
        # Convert to RGB
        mask_rgb = mask.repeat(1, 1, 1, 3)
        
        print(f"✓ Gradient mask created")
        print(f"  Output range: [{mask_rgb.min():.3f}, {mask_rgb.max():.3f}]")
        print("="*60 + "\n")
        
        return (mask_rgb,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "CurvatureGenerator": CurvatureGenerator,
    "DetailMapBlender": DetailMapBlender,
    "WearGenerator": WearGenerator,
    "GradientMap": GradientMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CurvatureGenerator": "Curvature Map Generator",
    "DetailMapBlender": "Detail Map Blender",
    "WearGenerator": "Wear & Edge Damage Generator",
    "GradientMap": "Gradient Map (Mask)",
}

