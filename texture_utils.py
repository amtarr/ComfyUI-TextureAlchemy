"""
Texture Utilities
Tiling, scaling, and projection tools for PBR textures
"""

import torch
import torch.nn.functional as F


class SeamlessTiling:
    """
    Make textures tileable using various methods
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["mirror", "blend_edges", "offset"], {
                    "default": "blend_edges",
                    "tooltip": "Tiling method: mirror (fastest), blend_edges (best), offset (simple)"
                }),
                "blend_width": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge blend width (0-0.5, fraction of image)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "edge_mask")
    FUNCTION = "make_seamless"
    CATEGORY = "Texture Alchemist/Texture"
    
    def make_seamless(self, image, method, blend_width):
        """Make texture seamlessly tileable and generate edge mask"""
        
        print("\n" + "="*60)
        print("Seamless Tiling Maker")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Method: {method}")
        
        if method == "mirror":
            result, edge_mask = self._mirror_tiling(image, blend_width)
        elif method == "blend_edges":
            result, edge_mask = self._blend_edges(image, blend_width)
        else:  # offset
            result, edge_mask = self._offset_tiling(image, blend_width)
        
        print(f"✓ Seamless texture created")
        print(f"✓ Edge mask generated")
        print(f"  Mask range: [{edge_mask.min():.3f}, {edge_mask.max():.3f}]")
        print("="*60 + "\n")
        
        return (result, edge_mask)
    
    def _mirror_tiling(self, image, blend_width):
        """Mirror method - flip and blend"""
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Create mirrored versions
        img_lr = torch.flip(image, [2])  # Left-right mirror
        img_tb = torch.flip(image, [1])  # Top-bottom mirror
        img_both = torch.flip(image, [1, 2])  # Both
        
        # Blend with 50/50
        result = (image + img_lr + img_tb + img_both) / 4.0
        
        # Create edge mask - white at all edges
        edge_mask = self._create_edge_mask(batch, height, width, blend_width, device, dtype)
        
        return result, edge_mask
    
    def _blend_edges(self, image, blend_width):
        """Blend edges method - smooth transition"""
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Calculate blend region size
        blend_h = int(height * blend_width)
        blend_w = int(width * blend_width)
        
        if blend_h == 0 or blend_w == 0:
            # No blending, return image and empty mask
            edge_mask = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
            return image, edge_mask.repeat(1, 1, 1, 3)
        
        # Create blend masks
        mask_h = torch.linspace(0, 1, blend_h, device=device, dtype=dtype)
        mask_w = torch.linspace(0, 1, blend_w, device=device, dtype=dtype)
        
        result = image.clone()
        
        # Blend top-bottom
        top = result[:, :blend_h, :, :]
        bottom = result[:, -blend_h:, :, :]
        mask_h_reshaped = mask_h.view(1, -1, 1, 1)
        blended_tb = top * mask_h_reshaped + bottom * (1 - mask_h_reshaped)
        result[:, :blend_h, :, :] = blended_tb
        result[:, -blend_h:, :, :] = torch.flip(blended_tb, [1])
        
        # Blend left-right
        left = result[:, :, :blend_w, :]
        right = result[:, :, -blend_w:, :]
        mask_w_reshaped = mask_w.view(1, 1, -1, 1)
        blended_lr = left * mask_w_reshaped + right * (1 - mask_w_reshaped)
        result[:, :, :blend_w, :] = blended_lr
        result[:, :, -blend_w:, :] = torch.flip(blended_lr, [2])
        
        # Create edge mask - white at edges, fading to black toward center
        edge_mask = self._create_edge_mask(batch, height, width, blend_width, device, dtype)
        
        return result, edge_mask
    
    def _offset_tiling(self, image, blend_width):
        """Offset method - shift by half"""
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Roll by half in both dimensions
        result = torch.roll(image, shifts=(height // 2, width // 2), dims=(1, 2))
        
        # Create edge mask - highlight center cross where seams now appear
        edge_mask = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Calculate edge width
        edge_h = max(1, int(height * blend_width))
        edge_w = max(1, int(width * blend_width))
        
        # Mark vertical center seam
        center_w = width // 2
        edge_mask[:, :, center_w - edge_w:center_w + edge_w, :] = 1.0
        
        # Mark horizontal center seam
        center_h = height // 2
        edge_mask[:, center_h - edge_h:center_h + edge_h, :, :] = 1.0
        
        # Convert to RGB
        edge_mask = edge_mask.repeat(1, 1, 1, 3)
        
        return result, edge_mask
    
    def _create_edge_mask(self, batch, height, width, blend_width, device, dtype):
        """Create edge mask showing where seams/edges are"""
        # Calculate edge region size
        edge_h = max(1, int(height * blend_width))
        edge_w = max(1, int(width * blend_width))
        
        # Create mask (1.0 at edges, 0.0 in center)
        edge_mask = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Create gradient from edge to center
        mask_h = torch.linspace(1, 0, edge_h, device=device, dtype=dtype)
        mask_w = torch.linspace(1, 0, edge_w, device=device, dtype=dtype)
        
        # Top edge
        edge_mask[:, :edge_h, :, :] = torch.max(
            edge_mask[:, :edge_h, :, :],
            mask_h.view(1, -1, 1, 1)
        )
        
        # Bottom edge
        edge_mask[:, -edge_h:, :, :] = torch.max(
            edge_mask[:, -edge_h:, :, :],
            torch.flip(mask_h, [0]).view(1, -1, 1, 1)
        )
        
        # Left edge
        edge_mask[:, :, :edge_w, :] = torch.max(
            edge_mask[:, :, :edge_w, :],
            mask_w.view(1, 1, -1, 1)
        )
        
        # Right edge
        edge_mask[:, :, -edge_w:, :] = torch.max(
            edge_mask[:, :, -edge_w:, :],
            torch.flip(mask_w, [0]).view(1, 1, -1, 1)
        )
        
        # Convert to RGB for display
        edge_mask_rgb = edge_mask.repeat(1, 1, 1, 3)
        
        return edge_mask_rgb


class TextureScaler:
    """
    Smart texture resolution scaling with multiple methods
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.125,
                    "max": 8.0,
                    "step": 0.125,
                    "display": "number",
                    "tooltip": "Scale multiplier (0.5 = half size, 2.0 = double size)"
                }),
                "method": (["nearest", "bilinear", "bicubic", "lanczos"], {
                    "default": "bicubic",
                    "tooltip": "Scaling method: nearest (pixel art), bilinear (fast), bicubic (quality), lanczos (best)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale"
    CATEGORY = "Texture Alchemist/Texture"
    
    def scale(self, image, scale_factor, method):
        """Scale texture resolution"""
        
        print("\n" + "="*60)
        print("Texture Scaler")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Scale factor: {scale_factor}x")
        print(f"Method: {method}")
        
        batch, height, width, channels = image.shape
        
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Convert to BCHW for interpolation
        image_bchw = image.permute(0, 3, 1, 2)
        
        # Map method names to PyTorch modes
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic"  # PyTorch doesn't have lanczos, use bicubic
        }
        
        mode = mode_map[method]
        antialias = (method == "lanczos")
        
        # Scale
        scaled = F.interpolate(
            image_bchw,
            size=(new_height, new_width),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
            antialias=antialias
        )
        
        # Convert back to BHWC
        result = scaled.permute(0, 2, 3, 1)
        
        print(f"✓ Scaled to {new_width}x{new_height}")
        print(f"  Output shape: {result.shape}")
        print("="*60 + "\n")
        
        return (result,)


class TriplanarProjection:
    """
    Apply triplanar projection to remove UV seams
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_sharpness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Sharpness of projection blend (higher = sharper transitions)"
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Texture tiling scale"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_triplanar"
    CATEGORY = "Texture Alchemist/Texture"
    
    def apply_triplanar(self, image, blend_sharpness, scale):
        """Apply triplanar projection blending"""
        
        print("\n" + "="*60)
        print("Triplanar Projection")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Blend sharpness: {blend_sharpness}")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Create simple XYZ projections by rotating the image
        # This is a simplified version - real triplanar needs 3D coordinates
        
        # X projection (side view) - original
        proj_x = image
        
        # Y projection (top view) - rotate 90 degrees
        proj_y = torch.rot90(image, k=1, dims=[1, 2])
        
        # Z projection (front view) - rotate 180 degrees  
        proj_z = torch.rot90(image, k=2, dims=[1, 2])
        
        # Create blend weights based on position
        # Simplified - use gradient-based weights
        y_pos = torch.linspace(0, 1, height, device=device, dtype=dtype)
        x_pos = torch.linspace(0, 1, width, device=device, dtype=dtype)
        
        y_grid = y_pos.view(1, -1, 1, 1).repeat(batch, 1, width, 1)
        x_grid = x_pos.view(1, 1, -1, 1).repeat(batch, height, 1, 1)
        
        # Calculate blend weights
        weight_x = torch.pow(torch.abs(x_grid - 0.5) * 2, blend_sharpness)
        weight_y = torch.pow(torch.abs(y_grid - 0.5) * 2, blend_sharpness)
        weight_z = torch.pow(1.0 - torch.abs(x_grid + y_grid - 1.0), blend_sharpness)
        
        # Normalize weights
        total_weight = weight_x + weight_y + weight_z + 1e-8
        weight_x = weight_x / total_weight
        weight_y = weight_y / total_weight
        weight_z = weight_z / total_weight
        
        # Blend projections
        result = proj_x * weight_x + proj_y * weight_y + proj_z * weight_z
        
        print(f"✓ Triplanar projection applied")
        print("="*60 + "\n")
        
        return (result,)


class TextureOffset:
    """
    Offset texture boundaries with X/Y shifting, rotation, and wrapping
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "offset_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Horizontal offset (-1.0 to 1.0, fraction of width)"
                }),
                "offset_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Vertical offset (-1.0 to 1.0, fraction of height)"
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -360.0,
                    "max": 360.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Rotation in degrees (0-360)"
                }),
                "wrap_mode": (["repeat", "clamp", "mirror"], {
                    "default": "repeat",
                    "tooltip": "Wrapping mode for edges: repeat (tile), clamp (extend), mirror (reflect)"
                }),
                "edge_mask_width": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Edge mask width (0-0.5, fraction of image) - shows affected areas"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "edge_mask")
    FUNCTION = "offset_texture"
    CATEGORY = "Texture Alchemist/Texture"
    
    def offset_texture(self, image, offset_x, offset_y, rotation, wrap_mode, edge_mask_width):
        """Offset and rotate texture with wrapping and edge mask"""
        import math
        
        print("\n" + "="*60)
        print("Texture Offset")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Offset: X={offset_x:.2f}, Y={offset_y:.2f}")
        print(f"Rotation: {rotation}°")
        print(f"Wrap mode: {wrap_mode}")
        print(f"Edge mask width: {edge_mask_width:.2f}")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Convert image to BHWC -> BCHW for PyTorch operations
        image_t = image.permute(0, 3, 1, 2)
        
        # Create edge mask BEFORE transformations (in original coordinate space)
        edge_mask_original = self._create_original_edge_mask(
            batch, height, width, edge_mask_width, device, dtype
        )
        # Convert to BCHW for transformation
        edge_mask_t = edge_mask_original.permute(0, 3, 1, 2)
        
        # Step 1: Apply offset (roll/shift) to BOTH image and mask
        if offset_x != 0.0 or offset_y != 0.0:
            shift_x = int(width * offset_x)
            shift_y = int(height * offset_y)
            
            # Use torch.roll for repeat mode (supports true circular wrapping)
            if wrap_mode == "repeat":
                image_t = torch.roll(image_t, shifts=(shift_y, shift_x), dims=(2, 3))
                edge_mask_t = torch.roll(edge_mask_t, shifts=(shift_y, shift_x), dims=(2, 3))
            else:
                # For non-repeat modes, use affine transform with grid_sample
                padding_mode = "border" if wrap_mode == "clamp" else "reflection"
                
                theta = torch.tensor([
                    [1, 0, -2.0 * offset_x],
                    [0, 1, -2.0 * offset_y]
                ], dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
                
                grid = F.affine_grid(theta, image_t.size(), align_corners=False)
                image_t = F.grid_sample(image_t, grid, mode='bilinear', 
                                       padding_mode=padding_mode, align_corners=False)
                
                # Apply same transformation to mask
                grid_mask = F.affine_grid(theta, edge_mask_t.size(), align_corners=False)
                edge_mask_t = F.grid_sample(edge_mask_t, grid_mask, mode='bilinear', 
                                           padding_mode=padding_mode, align_corners=False)
        
        # Step 2: Apply rotation to BOTH image and mask
        if rotation != 0.0:
            angle_rad = math.radians(rotation)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            
            # For repeat mode with rotation, tile the image to avoid edge artifacts
            if wrap_mode == "repeat":
                # Tile 3x3, rotate center, crop back
                tiled = image_t.repeat(1, 1, 3, 3)
                tiled_mask = edge_mask_t.repeat(1, 1, 3, 3)
                
                # Rotation matrix
                theta = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0]
                ], dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
                
                grid = F.affine_grid(theta, tiled.size(), align_corners=False)
                rotated = F.grid_sample(tiled, grid, mode='bilinear', 
                                       padding_mode='zeros', align_corners=False)
                
                # Rotate mask too
                grid_mask = F.affine_grid(theta, tiled_mask.size(), align_corners=False)
                rotated_mask = F.grid_sample(tiled_mask, grid_mask, mode='bilinear', 
                                            padding_mode='zeros', align_corners=False)
                
                # Crop center tile back to original size
                image_t = rotated[:, :, height:height*2, width:width*2]
                edge_mask_t = rotated_mask[:, :, height:height*2, width:width*2]
            else:
                # For clamp/mirror modes, use grid_sample directly
                padding_mode = "border" if wrap_mode == "clamp" else "reflection"
                
                theta = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0]
                ], dtype=dtype, device=device).unsqueeze(0).repeat(batch, 1, 1)
                
                grid = F.affine_grid(theta, image_t.size(), align_corners=False)
                image_t = F.grid_sample(image_t, grid, mode='bilinear', 
                                       padding_mode=padding_mode, align_corners=False)
                
                # Apply same rotation to mask
                grid_mask = F.affine_grid(theta, edge_mask_t.size(), align_corners=False)
                edge_mask_t = F.grid_sample(edge_mask_t, grid_mask, mode='bilinear', 
                                           padding_mode=padding_mode, align_corners=False)
        
        # Convert back to BCHW -> BHWC
        result = image_t.permute(0, 2, 3, 1)
        edge_mask = edge_mask_t.permute(0, 2, 3, 1)
        
        print(f"✓ Texture offset applied")
        print(f"  Output shape: {result.shape}")
        print(f"✓ Edge mask generated and transformed")
        print(f"  Mask range: [{edge_mask.min():.3f}, {edge_mask.max():.3f}]")
        print("="*60 + "\n")
        
        return (result, edge_mask)
    
    def _create_original_edge_mask(self, batch, height, width, edge_width, device, dtype):
        """Create edge mask in original coordinate space (before transformations)"""
        edge_mask = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Calculate edge region size
        edge_h = max(1, int(height * edge_width))
        edge_w = max(1, int(width * edge_width))
        
        # Create gradient from edge to center
        mask_h = torch.linspace(1, 0, edge_h, device=device, dtype=dtype)
        mask_w = torch.linspace(1, 0, edge_w, device=device, dtype=dtype)
        
        # Mark all four edges (they will be transformed along with the image)
        # Top edge
        edge_mask[:, :edge_h, :, :] = torch.max(
            edge_mask[:, :edge_h, :, :],
            mask_h.view(1, -1, 1, 1)
        )
        
        # Bottom edge
        edge_mask[:, -edge_h:, :, :] = torch.max(
            edge_mask[:, -edge_h:, :, :],
            torch.flip(mask_h, [0]).view(1, -1, 1, 1)
        )
        
        # Left edge
        edge_mask[:, :, :edge_w, :] = torch.max(
            edge_mask[:, :, :edge_w, :],
            mask_w.view(1, 1, -1, 1)
        )
        
        # Right edge
        edge_mask[:, :, -edge_w:, :] = torch.max(
            edge_mask[:, :, -edge_w:, :],
            torch.flip(mask_w, [0]).view(1, 1, -1, 1)
        )
        
        # Convert to RGB for display
        edge_mask_rgb = edge_mask.repeat(1, 1, 1, 3)
        
        return edge_mask_rgb


class TextureTiler:
    """
    Create tiled grid of texture (e.g., 2x2, 3x3)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_x": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of horizontal tiles"
                }),
                "tile_y": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of vertical tiles"
                }),
                "scale_to_input": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, scales output back to input size (tiles appear smaller). If False, output size increases."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "tile_texture"
    CATEGORY = "Texture Alchemist/Texture"
    
    def tile_texture(self, image, tile_x, tile_y, scale_to_input):
        """Create tiled grid of texture"""
        
        print("\n" + "="*60)
        print("Texture Tiler")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Tiles: {tile_x}x{tile_y}")
        print(f"Scale to input: {scale_to_input}")
        
        batch, height, width, channels = image.shape
        device = image.device
        
        # Repeat horizontally
        tiled_h = image.repeat(1, 1, tile_x, 1)
        
        # Repeat vertically
        tiled = tiled_h.repeat(1, tile_y, 1, 1)
        
        # If scale_to_input, resize back to original dimensions
        if scale_to_input and (tile_x > 1 or tile_y > 1):
            # Convert BHWC -> BCHW for interpolate
            tiled_t = tiled.permute(0, 3, 1, 2)
            
            # Resize to original input size
            tiled_t = F.interpolate(
                tiled_t, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            
            # Convert back BCHW -> BHWC
            tiled = tiled_t.permute(0, 2, 3, 1)
            
            print(f"✓ Tiled texture created and scaled to input size")
            print(f"  Output shape: {tiled.shape} (same as input)")
            print(f"  Each tile: {width//tile_x}x{height//tile_y} (approximate)")
        else:
            print(f"✓ Tiled texture created")
            print(f"  Output shape: {tiled.shape}")
            print(f"  Resolution: {tiled.shape[2]}x{tiled.shape[1]} ({tile_x*width}x{tile_y*height})")
        
        print("="*60 + "\n")
        
        return (tiled,)


class SmartTextureResizer:
    """
    Intelligently resize textures to optimal resolutions
    Ensures dimensions are multiples of specified value (e.g., 32, 64)
    Perfect for GPU optimization and game engines
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.25,
                    "max": 16.0,
                    "step": 0.25,
                    "display": "number",
                    "tooltip": "Target size in megapixels (e.g., 2.0 = 2 million pixels)"
                }),
                "multiple_of": ([4, 8, 16, 32, 64, 128, 256], {
                    "default": 32,
                    "tooltip": "Ensure width/height are multiples of this value"
                }),
                "resize_mode": (["fit_within", "fit_exact", "no_upscale"], {
                    "default": "fit_within",
                    "tooltip": "fit_within (stay under target), fit_exact (closest match), no_upscale (never increase size)"
                }),
                "scaling_method": (["bicubic", "bilinear", "lanczos", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Resampling algorithm for quality"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_smart"
    CATEGORY = "Texture Alchemist/Texture"
    
    def resize_smart(self, image, target_megapixels, multiple_of, resize_mode, scaling_method):
        """Smart resize to optimal resolution"""
        import math
        
        print("\n" + "="*60)
        print("Smart Texture Resizer")
        print("="*60)
        
        batch, height, width, channels = image.shape
        device = image.device
        
        current_megapixels = (width * height) / 1_000_000
        
        print(f"Input: {width}×{height} ({current_megapixels:.2f} MP)")
        print(f"Target: {target_megapixels:.2f} MP")
        print(f"Multiple of: {multiple_of}")
        print(f"Mode: {resize_mode}")
        
        # Calculate optimal dimensions
        new_width, new_height = self._calculate_optimal_dimensions(
            width, height, target_megapixels, multiple_of, resize_mode
        )
        
        # Check if resize is needed
        if new_width == width and new_height == height:
            print("✓ Already at optimal resolution, no resize needed")
            print("="*60 + "\n")
            return (image,)
        
        # Perform resize
        image_t = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Map scaling method to interpolate mode
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic"  # PyTorch doesn't have lanczos, use bicubic
        }
        
        resized = F.interpolate(
            image_t,
            size=(new_height, new_width),
            mode=mode_map.get(scaling_method, "bicubic"),
            align_corners=False if mode_map.get(scaling_method) != "nearest" else None
        )
        
        result = resized.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        final_megapixels = (new_width * new_height) / 1_000_000
        
        print(f"✓ Resized to optimal resolution")
        print(f"  Output: {new_width}×{new_height} ({final_megapixels:.2f} MP)")
        print(f"  Width multiple: {new_width // multiple_of} × {multiple_of}")
        print(f"  Height multiple: {new_height // multiple_of} × {multiple_of}")
        print(f"  GPU optimized: {'✓' if new_width % 32 == 0 and new_height % 32 == 0 else '✗'}")
        print("="*60 + "\n")
        
        return (result,)
    
    def _calculate_optimal_dimensions(self, width, height, target_mp, multiple, mode):
        """Calculate optimal width and height"""
        import math
        
        current_pixels = width * height
        target_pixels = target_mp * 1_000_000
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Handle different resize modes
        if mode == "no_upscale" and current_pixels <= target_pixels:
            # Don't upscale, just round to nearest multiple
            new_width = self._round_to_multiple(width, multiple)
            new_height = self._round_to_multiple(height, multiple)
            return new_width, new_height
        
        # Calculate scale factor to reach target
        scale = math.sqrt(target_pixels / current_pixels)
        
        if mode == "fit_within":
            # Stay under target (scale down a bit to ensure we don't exceed)
            scale *= 0.95
        
        # Calculate new dimensions maintaining aspect ratio
        new_width = width * scale
        new_height = height * scale
        
        # Round to nearest multiple
        new_width = self._round_to_multiple(int(new_width), multiple)
        new_height = self._round_to_multiple(int(new_height), multiple)
        
        # Ensure we have valid dimensions
        new_width = max(multiple, new_width)
        new_height = max(multiple, new_height)
        
        # For fit_exact, try to match target more closely
        if mode == "fit_exact":
            # Adjust if we're too far from target
            actual_pixels = new_width * new_height
            if abs(actual_pixels - target_pixels) > target_pixels * 0.2:
                # Recalculate with adjusted scale
                scale_adjust = math.sqrt(target_pixels / actual_pixels)
                new_width = self._round_to_multiple(int(new_width * scale_adjust), multiple)
                new_height = self._round_to_multiple(int(new_height * scale_adjust), multiple)
        
        # Final validation
        new_width = max(multiple, new_width)
        new_height = max(multiple, new_height)
        
        return new_width, new_height
    
    def _round_to_multiple(self, value, multiple):
        """Round value to nearest multiple"""
        return max(multiple, round(value / multiple) * multiple)


class SquareMaker:
    """
    Convert images to square by scaling or cropping
    Perfect for textures, AI models, and game engines
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["crop", "scale"], {
                    "default": "crop",
                    "tooltip": "crop (maintain aspect, remove edges) or scale (stretch to square)"
                }),
                "square_size": (["shortest_edge", "longest_edge", "custom"], {
                    "default": "shortest_edge",
                    "tooltip": "Base square size on shortest edge, longest edge, or custom size"
                }),
                "custom_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Custom square size (only used when square_size=custom)"
                }),
                "crop_position": (["top_left", "top_center", "top_right", 
                                   "middle_left", "center", "middle_right",
                                   "bottom_left", "bottom_center", "bottom_right"], {
                    "default": "center",
                    "tooltip": "Where to crop from (only used when method=crop)"
                }),
                "scaling_method": (["bicubic", "bilinear", "lanczos", "nearest"], {
                    "default": "lanczos",
                    "tooltip": "Resampling algorithm for quality"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make_square"
    CATEGORY = "Texture Alchemist/Texture"
    
    def make_square(self, image, method, square_size, custom_size, crop_position, scaling_method):
        """Convert image to square"""
        
        print("\n" + "="*60)
        print("Square Maker")
        print("="*60)
        
        batch, height, width, channels = image.shape
        device = image.device
        
        print(f"Input: {width}×{height}")
        print(f"Method: {method}")
        print(f"Square size mode: {square_size}")
        
        # Determine target square size
        if square_size == "shortest_edge":
            target_size = min(width, height)
        elif square_size == "longest_edge":
            target_size = max(width, height)
        else:  # custom
            target_size = custom_size
        
        print(f"Target square size: {target_size}×{target_size}")
        
        if method == "crop":
            result = self._crop_to_square(image, target_size, crop_position)
        else:  # scale
            result = self._scale_to_square(image, target_size, scaling_method)
        
        print(f"✓ Square image created: {target_size}×{target_size}")
        print("="*60 + "\n")
        
        return (result,)
    
    def _crop_to_square(self, image, target_size, position):
        """Crop image to square at specified position"""
        batch, height, width, channels = image.shape
        device = image.device
        
        # If already square and correct size, return as-is
        if width == height == target_size:
            return image
        
        # If need to scale first (image is smaller than target or not matching)
        if width < target_size or height < target_size or (width != target_size and height != target_size):
            # Scale so that the smallest dimension matches target_size
            if width < height:
                # Width is smaller, scale so width = target_size
                scale_factor = target_size / width
                new_width = target_size
                new_height = int(height * scale_factor)
            else:
                # Height is smaller or equal, scale so height = target_size
                scale_factor = target_size / height
                new_height = target_size
                new_width = int(width * scale_factor)
            
            # Perform scaling
            image_t = image.permute(0, 3, 1, 2)
            scaled = F.interpolate(
                image_t,
                size=(new_height, new_width),
                mode='bicubic',
                align_corners=False
            )
            image = scaled.permute(0, 2, 3, 1)
            height, width = new_height, new_width
        
        # Calculate crop coordinates based on position
        crop_x, crop_y = self._get_crop_coordinates(width, height, target_size, position)
        
        # Perform crop
        cropped = image[:, crop_y:crop_y+target_size, crop_x:crop_x+target_size, :]
        
        print(f"  Cropped from position: {position}")
        print(f"  Crop coordinates: ({crop_x}, {crop_y})")
        
        return cropped
    
    def _scale_to_square(self, image, target_size, method):
        """Scale (stretch) image to square"""
        batch, height, width, channels = image.shape
        
        # If already correct size, return as-is
        if width == height == target_size:
            return image
        
        image_t = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Map scaling method
        mode_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic"
        }
        
        scaled = F.interpolate(
            image_t,
            size=(target_size, target_size),
            mode=mode_map.get(method, "bicubic"),
            align_corners=False if mode_map.get(method) != "nearest" else None
        )
        
        result = scaled.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        print(f"  Scaled from {width}×{height} to {target_size}×{target_size}")
        
        return result
    
    def _get_crop_coordinates(self, width, height, target_size, position):
        """Calculate crop coordinates based on position"""
        # Calculate maximum offsets
        max_x = max(0, width - target_size)
        max_y = max(0, height - target_size)
        
        # Position mapping
        positions = {
            "top_left": (0, 0),
            "top_center": (max_x // 2, 0),
            "top_right": (max_x, 0),
            "middle_left": (0, max_y // 2),
            "center": (max_x // 2, max_y // 2),
            "middle_right": (max_x, max_y // 2),
            "bottom_left": (0, max_y),
            "bottom_center": (max_x // 2, max_y),
            "bottom_right": (max_x, max_y),
        }
        
        return positions.get(position, (max_x // 2, max_y // 2))


class TextureEqualizer:
    """
    Equalize textures by removing uneven lighting and shadows
    Based on the High Pass + Linear Light technique
    Perfect for cleaning up diffuse/albedo and height maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("FLOAT", {
                    "default": 100.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "High pass radius - controls detail preservation (50-150 typical)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Effect strength (1.0 = full correction, 0.0 = no change)"
                }),
                "preserve_color": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve original color hue (recommended for albedo)"
                }),
                "method": (["overlay", "soft_light", "linear_light"], {
                    "default": "overlay",
                    "tooltip": "Blend method: overlay (Photoshop standard), soft_light (subtle), linear_light (strong)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "average_color")
    FUNCTION = "equalize"
    CATEGORY = "Texture Alchemist/Texture"
    
    def equalize(self, image, radius, strength, preserve_color, method):
        """Equalize texture lighting using High Pass + blend mode technique"""
        
        print("\n" + "="*60)
        print("Texture Equalizer")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Radius: {radius}")
        print(f"Strength: {strength}")
        print(f"Preserve color: {preserve_color}")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Convert to BCHW for processing
        image_t = image.permute(0, 3, 1, 2)
        
        # Step 1: Calculate average color (simulates Photoshop's Average Blur)
        average_color_value = torch.mean(image_t, dim=[2, 3], keepdim=True)
        
        print(f"Average color: {average_color_value.squeeze().tolist()}")
        
        # Step 2: Create Gaussian blur for High Pass filter
        # High Pass = Original - Gaussian Blur
        # Radius in pixels -> sigma for Gaussian
        sigma = radius / 3.0  # Approximate conversion
        
        # Create Gaussian kernel
        kernel_size = int(radius * 2) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 201))  # Limit kernel size
        
        print(f"Gaussian blur: kernel_size={kernel_size}, sigma={sigma:.2f}")
        
        # Gaussian blur using separable convolution for efficiency
        blurred = self._gaussian_blur(image_t, kernel_size, sigma)
        
        print(f"Blurred range: [{blurred.min():.3f}, {blurred.max():.3f}]")
        
        # Step 3: High Pass filter
        high_pass = image_t - blurred + 0.5  # Add 0.5 to center around mid-gray
        high_pass = torch.clamp(high_pass, 0.0, 1.0)  # Clamp high pass to valid range
        
        # Step 4: Apply blend mode based on method
        print(f"High Pass range: [{high_pass.min():.3f}, {high_pass.max():.3f}]")
        print(f"Average color: {average_color_value.squeeze().tolist()}")
        print(f"Blend method: {method}")
        
        if method == "linear_light":
            # Linear Light: base + 2 * (blend - 0.5) = base + 2 * blend - 1
            result = average_color_value + 2.0 * high_pass - 1.0
        elif method == "overlay":
            # Overlay blend mode (Photoshop standard for equalization)
            # Inverted high pass with overlay blend
            # Invert the high pass first
            inverted_hp = 1.0 - high_pass
            # Overlay formula
            result = torch.where(
                average_color_value < 0.5,
                2.0 * average_color_value * inverted_hp,
                1.0 - 2.0 * (1.0 - average_color_value) * (1.0 - inverted_hp)
            )
        else:  # soft_light
            # Soft Light blend mode (gentler version of overlay)
            inverted_hp = 1.0 - high_pass
            result = torch.where(
                inverted_hp < 0.5,
                2.0 * average_color_value * inverted_hp + average_color_value * average_color_value * (1.0 - 2.0 * inverted_hp),
                2.0 * average_color_value * (1.0 - inverted_hp) + torch.sqrt(average_color_value) * (2.0 * inverted_hp - 1.0)
            )
        
        print(f"Result BEFORE clamp: [{result.min():.3f}, {result.max():.3f}]")
        
        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)
        print(f"Result AFTER clamp: [{result.min():.3f}, {result.max():.3f}]")
        
        # Step 5: Blend with original based on strength
        # This replaces the "layer opacity" control in Photoshop
        if strength < 1.0:
            result = image_t * (1.0 - strength) + result * strength
        
        print(f"Final result (after strength blend): [{result.min():.3f}, {result.max():.3f}]")
        
        # Step 6: Preserve color if requested
        if preserve_color and channels >= 3:
            # Convert to HSV, keep only V (luminance) from result, H and S from original
            result = self._preserve_color_hue(image_t, result)
        
        # Convert back to BHWC
        result = result.permute(0, 2, 3, 1)
        
        # Create average color output (expand to full image size for visualization)
        average_color_expanded = average_color_value.expand(-1, -1, height, width).permute(0, 2, 3, 1)
        
        print(f"✓ Texture equalized")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Average color range: [{average_color_expanded.min():.3f}, {average_color_expanded.max():.3f}]")
        print("="*60 + "\n")
        
        return (result, average_color_expanded)
    
    def _gaussian_blur(self, image, kernel_size, sigma):
        """Apply Gaussian blur using separable convolution"""
        import math
        
        # Create 1D Gaussian kernel
        kernel_range = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        kernel_range = kernel_range - (kernel_size - 1) / 2.0
        
        kernel_1d = torch.exp(-0.5 * (kernel_range / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Reshape for conv2d
        kernel_h = kernel_1d.view(1, 1, kernel_size, 1).repeat(image.shape[1], 1, 1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, kernel_size).repeat(image.shape[1], 1, 1, 1)
        
        # Apply separable convolution (horizontal then vertical)
        padding = kernel_size // 2
        
        # Horizontal blur
        blurred = F.conv2d(image, kernel_w, padding=(0, padding), groups=image.shape[1])
        # Vertical blur
        blurred = F.conv2d(blurred, kernel_h, padding=(padding, 0), groups=image.shape[1])
        
        return blurred
    
    def _preserve_color_hue(self, original, equalized):
        """Preserve hue and saturation from original, take luminance from equalized"""
        # Convert RGB to HSV
        original_hsv = self._rgb_to_hsv(original)
        equalized_hsv = self._rgb_to_hsv(equalized)
        
        # Take H and S from original, V from equalized
        result_hsv = torch.cat([
            original_hsv[:, 0:1, :, :],  # Hue from original
            original_hsv[:, 1:2, :, :],  # Saturation from original
            equalized_hsv[:, 2:3, :, :]  # Value from equalized
        ], dim=1)
        
        # Convert back to RGB
        result_rgb = self._hsv_to_rgb(result_hsv)
        
        return result_rgb
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        
        max_val = torch.max(torch.max(r, g), b)
        min_val = torch.min(torch.min(r, g), b)
        delta = max_val - min_val
        
        # Hue
        hue = torch.zeros_like(max_val)
        mask = delta > 1e-6
        
        r_max = (max_val == r) & mask
        g_max = (max_val == g) & mask
        b_max = (max_val == b) & mask
        
        hue[r_max] = ((g - b) / delta)[r_max] % 6.0
        hue[g_max] = ((b - r) / delta + 2.0)[g_max]
        hue[b_max] = ((r - g) / delta + 4.0)[b_max]
        hue = hue / 6.0
        
        # Saturation
        sat = torch.where(max_val > 1e-6, delta / max_val, torch.zeros_like(max_val))
        
        # Value
        val = max_val
        
        return torch.cat([hue, sat, val], dim=1)
    
    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        h, s, v = hsv[:, 0:1, :, :], hsv[:, 1:2, :, :], hsv[:, 2:3, :, :]
        
        h = h * 6.0
        i = torch.floor(h)
        f = h - i
        
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        
        i = i.long() % 6
        
        # Create output tensor
        rgb = torch.zeros_like(hsv)
        
        # Assign values based on hue sector
        for sector in range(6):
            mask = (i == sector)
            if sector == 0:
                rgb[:, 0:1, :, :] = torch.where(mask, v, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, t, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, p, rgb[:, 2:3, :, :])
            elif sector == 1:
                rgb[:, 0:1, :, :] = torch.where(mask, q, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, v, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, p, rgb[:, 2:3, :, :])
            elif sector == 2:
                rgb[:, 0:1, :, :] = torch.where(mask, p, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, v, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, t, rgb[:, 2:3, :, :])
            elif sector == 3:
                rgb[:, 0:1, :, :] = torch.where(mask, p, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, q, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, v, rgb[:, 2:3, :, :])
            elif sector == 4:
                rgb[:, 0:1, :, :] = torch.where(mask, t, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, p, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, v, rgb[:, 2:3, :, :])
            else:  # sector == 5
                rgb[:, 0:1, :, :] = torch.where(mask, v, rgb[:, 0:1, :, :])
                rgb[:, 1:2, :, :] = torch.where(mask, p, rgb[:, 1:2, :, :])
                rgb[:, 2:3, :, :] = torch.where(mask, q, rgb[:, 2:3, :, :])
        
        return rgb


class UpscaleCalculator:
    """
    Calculate correct scale factors for multi-pass upscaling
    Accounts for cumulative effect when chaining multiple upscalers
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 0.125,
                    "max": 16.0,
                    "step": 0.125,
                    "display": "number",
                    "tooltip": "Desired final scale multiplier (e.g., 4.0 for 512→2048)"
                }),
                "upscaler_multiplier": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "The multiplier of each upscaler (e.g., 4.0 for 4× upscaler)"
                }),
                "number_of_passes": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many upscalers will be chained"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "scale_per_pass", "target_width", "target_height", "info")
    FUNCTION = "calculate_upscale"
    CATEGORY = "Texture Alchemist/Texture"
    
    def calculate_upscale(self, image, target_scale, upscaler_multiplier, number_of_passes):
        """Calculate scale factors for multi-pass upscaling"""
        import math
        
        print("\n" + "="*60)
        print("Upscale Calculator")
        print("="*60)
        
        batch, height, width, channels = image.shape
        
        print(f"Input: {width}×{height}")
        print(f"Target scale: {target_scale}×")
        print(f"Upscaler multiplier: {upscaler_multiplier}×")
        print(f"Number of passes: {number_of_passes}")
        
        # Calculate target dimensions
        target_width = int(width * target_scale)
        target_height = int(height * target_scale)
        
        print(f"\n📐 TARGET DIMENSIONS:")
        print(f"  {width}×{height} → {target_width}×{target_height}")
        print(f"  Scale: {target_scale}×")
        
        # Calculate the scale factor to apply after each upscaler pass
        # Formula: scale_per_pass = (target_scale / upscaler^passes)^(1/passes)
        # Simplified: scale_per_pass = target_scale^(1/passes) / upscaler
        
        if number_of_passes == 1:
            # Simple case: only one pass
            scale_per_pass = target_scale / upscaler_multiplier
        else:
            # Multi-pass: need to account for cumulative effect
            # S^N * U^N = D, where S is scale per pass, U is upscaler, D is desired, N is passes
            # S = (D / U^N)^(1/N) = D^(1/N) / U
            scale_per_pass = math.pow(target_scale, 1.0 / number_of_passes) / upscaler_multiplier
        
        print(f"\n🔢 CALCULATION:")
        print(f"  Formula: scale_per_pass = target_scale^(1/passes) / upscaler")
        print(f"  scale_per_pass = {target_scale}^(1/{number_of_passes}) / {upscaler_multiplier}")
        print(f"  scale_per_pass = {math.pow(target_scale, 1.0/number_of_passes):.4f} / {upscaler_multiplier}")
        print(f"  scale_per_pass = {scale_per_pass:.6f}")
        
        # Verify the calculation
        print(f"\n✓ VERIFICATION:")
        current_size = width
        for i in range(number_of_passes):
            current_size = current_size * upscaler_multiplier * scale_per_pass
            print(f"  After pass {i+1}: {current_size:.1f}×{current_size * height / width:.1f}")
        
        final_width = width * math.pow(upscaler_multiplier * scale_per_pass, number_of_passes)
        final_height = height * math.pow(upscaler_multiplier * scale_per_pass, number_of_passes)
        
        print(f"\n  Expected final: {target_width}×{target_height}")
        print(f"  Calculated final: {final_width:.1f}×{final_height:.1f}")
        
        accuracy = abs(final_width - target_width) / target_width * 100
        if accuracy < 0.1:
            print(f"  ✓ Perfect match!")
        elif accuracy < 1:
            print(f"  ✓ Very close (within 1%)")
        else:
            print(f"  ⚠ Deviation: {accuracy:.2f}%")
        
        # Create summary
        info_lines = [
            f"Input: {width}×{height}",
            f"Target: {target_width}×{target_height} ({target_scale}×)",
            f"Scale per pass: {scale_per_pass:.6f}",
            f"Passes: {number_of_passes}",
            f"Final: {final_width:.0f}×{final_height:.0f}"
        ]
        info_string = " | ".join(info_lines)
        
        print(f"\n💡 USAGE:")
        print(f"  Connect this image to your first upscaler")
        print(f"  Set EACH upscaler's scale factor to: {scale_per_pass:.6f}")
        print(f"  Chain {number_of_passes} upscalers together")
        print(f"  Result: {target_width}×{target_height} ✓")
        
        print("="*60 + "\n")
        
        return (image, scale_per_pass, target_width, target_height, info_string)


class UpscaleToResolution:
    """
    Calculate correct scale factors for multi-pass upscaling to a target resolution
    Specify exact output dimensions instead of scale multiplier
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 16384,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Desired final width in pixels"
                }),
                "target_height": ("INT", {
                    "default": 2048,
                    "min": 64,
                    "max": 16384,
                    "step": 64,
                    "display": "number",
                    "tooltip": "Desired final height in pixels"
                }),
                "upscaler_multiplier": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "The multiplier of each upscaler (e.g., 4.0 for 4× upscaler)"
                }),
                "number_of_passes": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number",
                    "tooltip": "How many upscalers will be chained"
                }),
                "maintain_aspect": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Maintain original aspect ratio (scale to fit within target dimensions)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("image", "scale_per_pass", "final_width", "final_height", "total_scale", "info")
    FUNCTION = "calculate_upscale"
    CATEGORY = "Texture Alchemist/Texture"
    
    def calculate_upscale(self, image, target_width, target_height, upscaler_multiplier, 
                         number_of_passes, maintain_aspect):
        """Calculate scale factors for multi-pass upscaling to target resolution"""
        import math
        
        print("\n" + "="*60)
        print("Upscale to Resolution Calculator")
        print("="*60)
        
        batch, height, width, channels = image.shape
        
        print(f"Input: {width}×{height}")
        print(f"Target: {target_width}×{target_height}")
        print(f"Upscaler multiplier: {upscaler_multiplier}×")
        print(f"Number of passes: {number_of_passes}")
        print(f"Maintain aspect: {maintain_aspect}")
        
        # Calculate required scale based on target dimensions
        if maintain_aspect:
            # Scale to fit within target dimensions (preserve aspect ratio)
            scale_w = target_width / width
            scale_h = target_height / height
            
            # Use the smaller scale to ensure it fits within target
            target_scale = min(scale_w, scale_h)
            
            # Calculate actual final dimensions
            final_width = int(width * target_scale)
            final_height = int(height * target_scale)
            
            print(f"\n📐 ASPECT RATIO PRESERVED:")
            print(f"  Scale width: {scale_w:.4f}×")
            print(f"  Scale height: {scale_h:.4f}×")
            print(f"  Using scale: {target_scale:.4f}× (smaller to fit within target)")
            print(f"  Actual output: {final_width}×{final_height}")
            
            if final_width != target_width or final_height != target_height:
                print(f"  ⚠ Note: Output will be smaller than target to preserve aspect ratio")
        else:
            # Non-uniform scaling (may distort image)
            scale_w = target_width / width
            scale_h = target_height / height
            
            if abs(scale_w - scale_h) > 0.01:
                print(f"\n⚠ WARNING: Non-uniform scaling will distort the image!")
                print(f"  Width scale: {scale_w:.4f}×")
                print(f"  Height scale: {scale_h:.4f}×")
                print(f"  Difference: {abs(scale_w - scale_h):.4f}×")
            
            # For simplicity, use average scale
            target_scale = (scale_w + scale_h) / 2.0
            final_width = target_width
            final_height = target_height
            
            print(f"\n📐 NON-UNIFORM SCALING:")
            print(f"  Using average scale: {target_scale:.4f}×")
        
        print(f"\n🎯 TARGET SCALE:")
        print(f"  {width}×{height} → {final_width}×{final_height}")
        print(f"  Scale factor: {target_scale:.4f}×")
        
        # Check if target is achievable
        max_possible = width * math.pow(upscaler_multiplier, number_of_passes)
        if final_width > max_possible:
            print(f"\n⚠ WARNING: Target may not be achievable!")
            print(f"  Maximum possible: {int(max_possible)}×{int(max_possible * height / width)}")
            print(f"  You requested: {final_width}×{final_height}")
            print(f"  Consider adding more passes or using a higher multiplier upscaler")
        
        # Calculate the scale factor to apply after each upscaler pass
        if number_of_passes == 1:
            scale_per_pass = target_scale / upscaler_multiplier
        else:
            scale_per_pass = math.pow(target_scale, 1.0 / number_of_passes) / upscaler_multiplier
        
        print(f"\n🔢 CALCULATION:")
        print(f"  Formula: scale_per_pass = target_scale^(1/passes) / upscaler")
        print(f"  scale_per_pass = {target_scale:.4f}^(1/{number_of_passes}) / {upscaler_multiplier}")
        print(f"  scale_per_pass = {math.pow(target_scale, 1.0/number_of_passes):.6f} / {upscaler_multiplier}")
        print(f"  scale_per_pass = {scale_per_pass:.6f}")
        
        # Verify the calculation
        print(f"\n✓ VERIFICATION:")
        current_w = width
        current_h = height
        for i in range(number_of_passes):
            current_w = current_w * upscaler_multiplier * scale_per_pass
            current_h = current_h * upscaler_multiplier * scale_per_pass
            print(f"  After pass {i+1}: {current_w:.1f}×{current_h:.1f}")
        
        print(f"\n  Target: {final_width}×{final_height}")
        print(f"  Result: {current_w:.1f}×{current_h:.1f}")
        
        accuracy_w = abs(current_w - final_width) / final_width * 100
        accuracy_h = abs(current_h - final_height) / final_height * 100
        max_accuracy = max(accuracy_w, accuracy_h)
        
        if max_accuracy < 0.1:
            print(f"  ✓ Perfect match!")
        elif max_accuracy < 1:
            print(f"  ✓ Very close (within 1%)")
        else:
            print(f"  ⚠ Deviation: {max_accuracy:.2f}%")
        
        # Create summary
        info_lines = [
            f"Input: {width}×{height}",
            f"Target: {final_width}×{final_height}",
            f"Scale: {target_scale:.4f}×",
            f"Scale per pass: {scale_per_pass:.6f}",
            f"Passes: {number_of_passes}"
        ]
        info_string = " | ".join(info_lines)
        
        print(f"\n💡 USAGE:")
        print(f"  Connect this image to your first upscaler")
        print(f"  Set EACH upscaler's scale factor to: {scale_per_pass:.6f}")
        print(f"  Chain {number_of_passes} upscalers together")
        print(f"  Result: {final_width}×{final_height} ✓")
        
        print("="*60 + "\n")
        
        return (image, scale_per_pass, final_width, final_height, target_scale, info_string)


class PaddingCalculator:
    """
    Calculate exact padding dimensions for percentage-based padding
    Always rounds to whole numbers for perfect cropping
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Original image width in pixels"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 16384,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Original image height in pixels"
                }),
                "pad_percentage": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Padding percentage (will be rounded to whole pixels)"
                }),
                "rounding_mode": (["round", "floor", "ceil"], {
                    "default": "round",
                    "tooltip": "How to round fractional pixels: round (nearest), floor (down), ceil (up)"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("new_width", "new_height", "pad_pixels_width", "pad_pixels_height", "info")
    FUNCTION = "calculate_padding"
    CATEGORY = "Texture Alchemist/Texture"
    
    def calculate_padding(self, width, height, pad_percentage, rounding_mode):
        """
        Calculate padding that rounds to whole numbers for perfect cropping
        Pads evenly on both sides
        """
        
        print("\n" + "="*60)
        print("Padding Calculator")
        print("="*60)
        print(f"Original dimensions: {width} × {height}")
        print(f"Requested padding: {pad_percentage}%")
        print(f"Rounding mode: {rounding_mode}")
        
        # Calculate raw padding per side
        raw_pad_width = (width * pad_percentage) / 100.0
        raw_pad_height = (height * pad_percentage) / 100.0
        
        print(f"\nRaw padding calculation:")
        print(f"  Width: {raw_pad_width:.2f} pixels per side")
        print(f"  Height: {raw_pad_height:.2f} pixels per side")
        
        # Round according to mode
        import math
        
        if rounding_mode == "round":
            pad_pixels_width = round(raw_pad_width)
            pad_pixels_height = round(raw_pad_height)
        elif rounding_mode == "floor":
            pad_pixels_width = math.floor(raw_pad_width)
            pad_pixels_height = math.floor(raw_pad_height)
        else:  # ceil
            pad_pixels_width = math.ceil(raw_pad_width)
            pad_pixels_height = math.ceil(raw_pad_height)
        
        # Calculate new dimensions (original + padding on both sides)
        new_width = width + (pad_pixels_width * 2)
        new_height = height + (pad_pixels_height * 2)
        
        # Calculate actual percentage applied
        actual_percent_width = (pad_pixels_width / width) * 100.0 if width > 0 else 0
        actual_percent_height = (pad_pixels_height / height) * 100.0 if height > 0 else 0
        
        print(f"\n✓ Rounded padding:")
        print(f"  Width: {pad_pixels_width} pixels per side")
        print(f"  Height: {pad_pixels_height} pixels per side")
        print(f"\n✓ New dimensions:")
        print(f"  {width} × {height} → {new_width} × {new_height}")
        print(f"\n✓ Actual padding percentage:")
        print(f"  Width: {actual_percent_width:.2f}%")
        print(f"  Height: {actual_percent_height:.2f}%")
        
        # Create info string
        info_lines = [
            f"Original: {width} × {height}",
            f"Requested: {pad_percentage}% padding",
            f"Rounded: {pad_pixels_width}px (W), {pad_pixels_height}px (H) per side",
            f"New size: {new_width} × {new_height}",
            f"Actual: {actual_percent_width:.2f}% (W), {actual_percent_height:.2f}% (H)"
        ]
        info_string = " | ".join(info_lines)
        
        print(f"\n💡 Usage:")
        print(f"  1. Pad image by {pad_pixels_width}px (W) and {pad_pixels_height}px (H)")
        print(f"  2. Process the {new_width} × {new_height} image")
        print(f"  3. Crop back using the same padding values")
        print("="*60 + "\n")
        
        return (new_width, new_height, pad_pixels_width, pad_pixels_height, info_string)


class InpaintCropExtractor:
    """
    Extract masked region as a cropped image for processing
    Finds bounding box of mask and crops image + mask to that region
    Perfect for efficient inpainting workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Extra padding around mask bounding box (pixels)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask before processing (e.g., for white background masks)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT", "INT", "INT", "STRING", "BBOX_DATA")
    RETURN_NAMES = ("cropped_image", "masked_crop", "cropped_mask", "bbox_x", "bbox_y", "bbox_width", "bbox_height", "info", "bbox_data")
    FUNCTION = "extract_crop"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def extract_crop(self, image, mask, padding, invert_mask):
        """
        Find mask bounding box and extract cropped region
        """
        
        print("\n" + "="*60)
        print("Inpaint Crop Extractor")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Padding: {padding}px")
        print(f"Invert mask: {invert_mask}")
        
        batch, height, width, channels = image.shape
        
        # Ensure mask matches image dimensions
        if len(mask.shape) == 3:  # (B, H, W)
            mask_2d = mask[0]  # Take first in batch
        else:  # (B, H, W, C)
            mask_2d = mask[0, :, :, 0]
        
        # Resize mask if needed
        if mask_2d.shape != (height, width):
            mask_reshaped = mask.unsqueeze(1) if len(mask.shape) == 3 else mask.permute(0, 3, 1, 2)
            mask_resized = F.interpolate(
                mask_reshaped,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            if len(mask.shape) == 3:
                mask = mask_resized.squeeze(1)
                mask_2d = mask[0]
            else:
                mask = mask_resized.permute(0, 2, 3, 1)
                mask_2d = mask[0, :, :, 0]
        
        # Apply inversion if requested
        if invert_mask:
            mask_2d = 1.0 - mask_2d
            print("✓ Mask inverted for processing.")
        
        # Debugging mask analysis
        print("\n🔍 Analyzing mask:")
        print(f"  Mask shape: {mask_2d.shape}")
        print(f"  Mask range: [{mask_2d.min():.3f}, {mask_2d.max():.3f}]")
        print(f"  Pixels > 0.01: {(mask_2d > 0.01).sum().item()}")
        print(f"  Pixels > 0.1: {(mask_2d > 0.1).sum().item()}")
        print(f"  Pixels > 0.5: {(mask_2d > 0.5).sum().item()}")
        print(f"  Pixels > 0.9: {(mask_2d > 0.9).sum().item()}")
        
        # Debugging edge analysis
        print("\n🔍 Edge analysis:")
        if mask_2d.shape[0] > 0 and mask_2d.shape[1] > 0:
            print(f"  Top-left corner (0,0): {mask_2d[0, 0].item():.3f}")
            print(f"  Top-right corner (0,-1): {mask_2d[0, -1].item():.3f}")
            print(f"  Bottom-left corner (-1,0): {mask_2d[-1, 0].item():.3f}")
            print(f"  Bottom-right corner (-1,-1): {mask_2d[-1, -1].item():.3f}")
            print(f"  Top row max: {mask_2d[0, :].max().item():.3f}")
            print(f"  Bottom row max: {mask_2d[-1, :].max().item():.3f}")
            print(f"  Left column max: {mask_2d[:, 0].max().item():.3f}")
            print(f"  Right column max: {mask_2d[:, -1].max().item():.3f}")
        
        # Find bounding box of mask (where mask > 0.01)
        mask_binary = (mask_2d > 0.01).float()
        
        # Find non-zero coordinates
        nonzero = torch.nonzero(mask_binary, as_tuple=False)
        
        print(f"  Non-zero count: {nonzero.shape[0]}")
        
        if nonzero.shape[0] == 0:
            # Empty mask - return full image
            print("\n⚠️ Empty mask detected (after thresholding) - returning full image")
            bbox_x, bbox_y = 0, 0
            bbox_width, bbox_height = width, height
        else:
            # Get bounding box
            y_min = nonzero[:, 0].min().item()
            y_max = nonzero[:, 0].max().item()
            x_min = nonzero[:, 1].min().item()
            x_max = nonzero[:, 1].max().item()
            
            # Add padding (expand bounding box within original image bounds)
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            bbox_x = x_min
            bbox_y = y_min
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            
            print(f"\n✓ Bounding box found:")
            print(f"  Position: ({bbox_x}, {bbox_y})")
            print(f"  Size: {bbox_width} × {bbox_height}")
            print(f"  Original canvas: {width} × {height}")
            print(f"  Coverage: {(bbox_width * bbox_height) / (width * height) * 100:.1f}% of image")
        
        print("\n✂️ Cropping:")
        
        # Crop image
        cropped_image = image[:, bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width, :]
        
        # Crop mask (use the processed mask_2d, then reconstruct full mask if needed)
        # If we inverted, we need to work with inverted data
        cropped_mask_2d = mask_2d[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
        
        # Expand to batch if needed
        if batch > 1:
            cropped_mask = cropped_mask_2d.unsqueeze(0).repeat(batch, 1, 1)
        else:
            cropped_mask = cropped_mask_2d.unsqueeze(0)
        
        print(f"  Cropped image: {cropped_image.shape}")
        print(f"  Cropped mask: {cropped_mask.shape}")
        print(f"  Expected: {batch, bbox_height, bbox_width}")
        
        # Create masked crop (image with mask applied)
        # Expand mask to match image channels
        mask_expanded = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, channels)
        masked_crop = cropped_image * mask_expanded
        
        # Create info string
        info_lines = [
            f"BBox: ({bbox_x}, {bbox_y})",
            f"Size: {bbox_width}×{bbox_height}",
            f"Original: {width}×{height}",
            f"Padding: {padding}px"
        ]
        if invert_mask:
            info_lines.append("Mask inverted")
        info_string = " | ".join(info_lines)
        
        print(f"\n✓ Outputs:")
        print(f"  Cropped Image: {cropped_image.shape}")
        print(f"  Masked Crop: {masked_crop.shape}")
        print(f"  Cropped Mask: {cropped_mask.shape}")
        print(f"  BBox: x={bbox_x}, y={bbox_y}, w={bbox_width}, h={bbox_height}")
        
        # Print mask statistics
        mask_min = cropped_mask.min().item()
        mask_max = cropped_mask.max().item()
        mask_mean = cropped_mask.mean().item()
        coverage = (cropped_mask > 0.01).float().mean().item() * 100
        
        print(f"\n📊 Cropped Mask Statistics:")
        print(f"  Range: [{mask_min:.3f}, {mask_max:.3f}]")
        print(f"  Mean: {mask_mean:.3f}")
        print(f"  Coverage: {coverage:.1f}% of crop")
        print("="*60 + "\n")
        
        # Create bbox data bundle for simplified workflow
        bbox_data = {
            "x": bbox_x,
            "y": bbox_y,
            "width": bbox_width,
            "height": bbox_height,
            "original_width": width,
            "original_height": height
        }
        
        return (cropped_image, masked_crop, cropped_mask, bbox_x, bbox_y, bbox_width, bbox_height, info_string, bbox_data)


class InpaintStitcher:
    """
    Stitch processed crop back onto original image
    Uses bounding box coordinates from InpaintCropExtractor
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_crop": ("IMAGE",),
                "bbox_x": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "display": "number"
                }),
                "bbox_y": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16384,
                    "display": "number"
                }),
                "bbox_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 16384,
                    "display": "number"
                }),
                "bbox_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 16384,
                    "display": "number"
                }),
                "blend_mode": (["replace", "blend"], {
                    "default": "replace",
                    "tooltip": "Replace: hard composite, Blend: soft edges"
                }),
                "feather": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Feather/blend edges (pixels, only for blend mode)"
                }),
            },
            "optional": {
                "blend_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched_image",)
    FUNCTION = "stitch"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def stitch(self, original_image, processed_crop, bbox_x, bbox_y, bbox_width, bbox_height, 
               blend_mode, feather, blend_mask=None):
        """
        Composite processed crop back onto original image
        """
        
        print("\n" + "="*60)
        print("Inpaint Stitcher")
        print("="*60)
        print(f"Original image: {original_image.shape}")
        print(f"Processed crop: {processed_crop.shape}")
        print(f"BBox: ({bbox_x}, {bbox_y}) size {bbox_width}×{bbox_height}")
        print(f"Blend mode: {blend_mode}")
        
        batch, height, width, channels = original_image.shape
        
        # Create result as copy of original
        result = original_image.clone()
        
        # Resize processed crop if it doesn't match bbox dimensions
        crop_batch, crop_h, crop_w, crop_c = processed_crop.shape
        
        # Handle channel mismatch (e.g., RGB vs RGBA)
        if crop_c != channels:
            print(f"  Channel mismatch detected: crop={crop_c}, original={channels}")
            if crop_c == 3 and channels == 4:
                # Crop is RGB, original is RGBA - add alpha channel to crop
                alpha_channel = torch.ones((crop_batch, crop_h, crop_w, 1), 
                                          device=processed_crop.device, 
                                          dtype=processed_crop.dtype)
                processed_crop = torch.cat([processed_crop, alpha_channel], dim=-1)
                crop_c = 4
                print(f"  Added alpha channel to crop")
            elif crop_c == 4 and channels == 3:
                # Crop is RGBA, original is RGB - drop alpha channel
                processed_crop = processed_crop[:, :, :, :3]
                crop_c = 3
                print(f"  Dropped alpha channel from crop")
        
        if (crop_h, crop_w) != (bbox_height, bbox_width):
            print(f"  Resizing crop from {crop_w}×{crop_h} to {bbox_width}×{bbox_height}")
            processed_crop = F.interpolate(
                processed_crop.permute(0, 3, 1, 2),
                size=(bbox_height, bbox_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Handle negative bbox coordinates (from padding)
        # Calculate the visible region
        crop_x_start = max(0, -bbox_x)  # Offset into crop if bbox_x is negative
        crop_y_start = max(0, -bbox_y)  # Offset into crop if bbox_y is negative
        
        paste_x = max(0, bbox_x)  # Where to paste in original (0 if bbox_x negative)
        paste_y = max(0, bbox_y)  # Where to paste in original (0 if bbox_y negative)
        
        # Calculate the actual region that fits
        paste_x_end = min(paste_x + (bbox_width - crop_x_start), width)
        paste_y_end = min(paste_y + (bbox_height - crop_y_start), height)
        
        actual_width = paste_x_end - paste_x
        actual_height = paste_y_end - paste_y
        
        # Extract the visible portion of the crop
        crop_x_end = crop_x_start + actual_width
        crop_y_end = crop_y_start + actual_height
        
        if crop_x_start > 0 or crop_y_start > 0 or crop_x_end < bbox_width or crop_y_end < bbox_height:
            print(f"  Cropping to visible region: crop[{crop_y_start}:{crop_y_end}, {crop_x_start}:{crop_x_end}] → paste[{paste_y}:{paste_y_end}, {paste_x}:{paste_x_end}]")
            processed_crop = processed_crop[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]
        
        if blend_mode == "replace":
            # Hard composite - direct replacement
            result[:, paste_y:paste_y_end, paste_x:paste_x_end, :] = processed_crop
            print(f"✓ Hard composite applied")
            
        else:  # blend
            # Create blend mask
            if blend_mask is not None:
                # Use provided mask
                print(f"  Using provided blend mask")
                if len(blend_mask.shape) == 3:  # (B, H, W)
                    mask_2d = blend_mask[0]
                else:  # (B, H, W, C)
                    mask_2d = blend_mask[0, :, :, 0]
                
                # Extract the same region from the mask as we did from the crop
                if mask_2d.shape[0] == bbox_height and mask_2d.shape[1] == bbox_width:
                    # Mask matches original bbox size, extract the visible portion
                    mask_2d = mask_2d[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    blend_alpha = mask_2d
                    print(f"  Extracted mask region: {blend_alpha.shape}")
                elif mask_2d.shape != (actual_height, actual_width):
                    # Resize to actual dimensions
                    mask_reshaped = blend_mask.unsqueeze(1) if len(blend_mask.shape) == 3 else blend_mask.permute(0, 3, 1, 2)
                    mask_resized = F.interpolate(
                        mask_reshaped,
                        size=(actual_height, actual_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    if len(blend_mask.shape) == 3:
                        blend_alpha = mask_resized[0, 0]
                    else:
                        blend_alpha = mask_resized[0, 0, :, :]
                    print(f"  Resized mask: {blend_alpha.shape}")
                else:
                    blend_alpha = mask_2d[:actual_height, :actual_width]
            elif feather > 0:
                # Create feathered edge mask
                blend_alpha = torch.ones((actual_height, actual_width), 
                                        device=original_image.device, 
                                        dtype=original_image.dtype)
                
                # Apply feather to edges
                feather_pixels = min(feather, actual_width // 2, actual_height // 2)
                if feather_pixels > 0:
                    for i in range(feather_pixels):
                        alpha_value = i / feather_pixels
                        # Top
                        blend_alpha[i, :] = alpha_value
                        # Bottom
                        blend_alpha[actual_height - 1 - i, :] = alpha_value
                        # Left
                        blend_alpha[:, i] = torch.minimum(blend_alpha[:, i], 
                                                         torch.tensor(alpha_value, device=blend_alpha.device))
                        # Right
                        blend_alpha[:, actual_width - 1 - i] = torch.minimum(
                            blend_alpha[:, actual_width - 1 - i],
                            torch.tensor(alpha_value, device=blend_alpha.device))
                
                print(f"  Applied {feather_pixels}px feather")
            else:
                # No feather, just alpha blend
                blend_alpha = torch.ones((actual_height, actual_width),
                                        device=original_image.device,
                                        dtype=original_image.dtype)
            
            # Expand alpha to match channels
            blend_alpha = blend_alpha.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, 1, channels)
            
            # Alpha blend
            original_region = result[:, paste_y:paste_y_end, paste_x:paste_x_end, :]
            result[:, paste_y:paste_y_end, paste_x:paste_x_end, :] = (
                processed_crop * blend_alpha + original_region * (1.0 - blend_alpha)
            )
            print(f"✓ Soft blend applied")
        
        print(f"\n✓ Stitched result: {result.shape}")
        print("="*60 + "\n")
        
        return (result,)


class SimpleInpaintCrop:
    """
    SIMPLIFIED INPAINTING - CROP
    One-click crop with bundled bbox data for easy stitching
    Perfect for streamlined inpainting workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Extra padding around mask (pixels)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask before processing"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BBOX_DATA", "STRING")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "masked_composite", "bbox_data", "info")
    FUNCTION = "crop_for_inpaint"
    CATEGORY = "Texture Alchemist/Inpainting/Simple"
    
    def crop_for_inpaint(self, image, mask, padding, invert_mask):
        """
        Extract masked region for inpainting with bundled bbox data
        """
        
        print("\n" + "="*60)
        print("Simple Inpaint Crop")
        print("="*60)
        print(f"Image: {image.shape}")
        print(f"Mask: {mask.shape}")
        print(f"Padding: {padding}px")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Ensure mask matches image dimensions
        if len(mask.shape) == 3:  # (B, H, W)
            mask_2d = mask[0]
        else:  # (B, H, W, C)
            mask_2d = mask[0, :, :, 0]
        
        # Resize mask if needed
        if mask_2d.shape != (height, width):
            mask_reshaped = mask.unsqueeze(1) if len(mask.shape) == 3 else mask.permute(0, 3, 1, 2)
            mask_resized = F.interpolate(
                mask_reshaped,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            if len(mask.shape) == 3:
                mask = mask_resized.squeeze(1)
                mask_2d = mask[0]
            else:
                mask = mask_resized.permute(0, 2, 3, 1)
                mask_2d = mask[0, :, :, 0]
        
        # Apply inversion if requested
        if invert_mask:
            mask_2d = 1.0 - mask_2d
            print("✓ Mask inverted")
        
        # Find bounding box
        mask_binary = (mask_2d > 0.01).float()
        nonzero = torch.nonzero(mask_binary, as_tuple=False)
        
        if nonzero.shape[0] == 0:
            # Empty mask - return full image
            print("⚠️ Empty mask - using full image")
            bbox_x, bbox_y = 0, 0
            bbox_width, bbox_height = width, height
        else:
            y_min = nonzero[:, 0].min().item()
            y_max = nonzero[:, 0].max().item()
            x_min = nonzero[:, 1].min().item()
            x_max = nonzero[:, 1].max().item()
            
            # Add padding
            y_min = max(0, y_min - padding)
            y_max = min(height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(width - 1, x_max + padding)
            
            bbox_x = x_min
            bbox_y = y_min
            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1
            
            print(f"✓ BBox: ({bbox_x}, {bbox_y}) {bbox_width}×{bbox_height}")
        
        # Crop image and mask
        cropped_image = image[:, bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width, :]
        cropped_mask_2d = mask_2d[bbox_y:bbox_y+bbox_height, bbox_x:bbox_x+bbox_width]
        
        # Expand mask to batch
        if batch > 1:
            cropped_mask = cropped_mask_2d.unsqueeze(0).repeat(batch, 1, 1)
        else:
            cropped_mask = cropped_mask_2d.unsqueeze(0)
        
        # Create masked composite (image with mask applied)
        mask_expanded = cropped_mask.unsqueeze(-1).repeat(1, 1, 1, channels)
        masked_composite = cropped_image * mask_expanded
        
        # Create bbox data bundle
        bbox_data = {
            "x": bbox_x,
            "y": bbox_y,
            "width": bbox_width,
            "height": bbox_height,
            "original_width": width,
            "original_height": height
        }
        
        # Info string
        info = f"Crop: {bbox_width}×{bbox_height} at ({bbox_x},{bbox_y}) | Orig: {width}×{height}"
        
        print(f"✓ Outputs: Crop {bbox_width}×{bbox_height}")
        print("="*60 + "\n")
        
        return (cropped_image, cropped_mask, masked_composite, bbox_data, info)


class SimpleInpaintStitch:
    """
    SIMPLIFIED INPAINTING - STITCH
    One-click stitch using bundled bbox data from SimpleInpaintCrop
    Perfect for streamlined inpainting workflows
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_crop": ("IMAGE",),
                "bbox_data": ("BBOX_DATA",),
                "blend_mode": (["replace", "blend"], {
                    "default": "replace",
                    "tooltip": "Replace: hard edge, Blend: soft edge"
                }),
                "feather": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Edge feathering in pixels (blend mode only)"
                }),
            },
            "optional": {
                "blend_mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "stitch_inpaint"
    CATEGORY = "Texture Alchemist/Inpainting/Simple"
    
    def stitch_inpaint(self, original_image, processed_crop, bbox_data, blend_mode, feather, blend_mask=None):
        """
        Stitch processed crop back onto original using bundled bbox data
        """
        
        print("\n" + "="*60)
        print("Simple Inpaint Stitch")
        print("="*60)
        
        # Extract bbox data from bundle
        bbox_x = bbox_data["x"]
        bbox_y = bbox_data["y"]
        bbox_width = bbox_data["width"]
        bbox_height = bbox_data["height"]
        
        print(f"Original: {original_image.shape}")
        print(f"Processed crop: {processed_crop.shape}")
        print(f"BBox: ({bbox_x}, {bbox_y}) {bbox_width}×{bbox_height}")
        print(f"Mode: {blend_mode}")
        
        batch, height, width, channels = original_image.shape
        
        # Create result as copy of original
        result = original_image.clone()
        
        # Resize processed crop if needed
        crop_batch, crop_h, crop_w, crop_c = processed_crop.shape
        
        # Handle channel mismatch (e.g., RGB vs RGBA)
        if crop_c != channels:
            print(f"  Channel mismatch detected: crop={crop_c}, original={channels}")
            if crop_c == 3 and channels == 4:
                # Crop is RGB, original is RGBA - add alpha channel to crop
                alpha_channel = torch.ones((crop_batch, crop_h, crop_w, 1), 
                                          device=processed_crop.device, 
                                          dtype=processed_crop.dtype)
                processed_crop = torch.cat([processed_crop, alpha_channel], dim=-1)
                crop_c = 4
                print(f"  Added alpha channel to crop")
            elif crop_c == 4 and channels == 3:
                # Crop is RGBA, original is RGB - drop alpha channel
                processed_crop = processed_crop[:, :, :, :3]
                crop_c = 3
                print(f"  Dropped alpha channel from crop")
        
        if (crop_h, crop_w) != (bbox_height, bbox_width):
            print(f"  Resizing crop: {crop_w}×{crop_h} → {bbox_width}×{bbox_height}")
            processed_crop = F.interpolate(
                processed_crop.permute(0, 3, 1, 2),
                size=(bbox_height, bbox_width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Handle negative bbox coordinates (from padding)
        # Calculate the visible region
        crop_x_start = max(0, -bbox_x)  # Offset into crop if bbox_x is negative
        crop_y_start = max(0, -bbox_y)  # Offset into crop if bbox_y is negative
        
        paste_x = max(0, bbox_x)  # Where to paste in original (0 if bbox_x negative)
        paste_y = max(0, bbox_y)  # Where to paste in original (0 if bbox_y negative)
        
        # Calculate the actual region that fits
        paste_x_end = min(paste_x + (bbox_width - crop_x_start), width)
        paste_y_end = min(paste_y + (bbox_height - crop_y_start), height)
        
        actual_width = paste_x_end - paste_x
        actual_height = paste_y_end - paste_y
        
        # Extract the visible portion of the crop
        crop_x_end = crop_x_start + actual_width
        crop_y_end = crop_y_start + actual_height
        
        if crop_x_start > 0 or crop_y_start > 0 or crop_x_end < bbox_width or crop_y_end < bbox_height:
            print(f"  Cropping to visible region: crop[{crop_y_start}:{crop_y_end}, {crop_x_start}:{crop_x_end}] → paste[{paste_y}:{paste_y_end}, {paste_x}:{paste_x_end}]")
            processed_crop = processed_crop[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]
        
        if blend_mode == "replace":
            # Hard composite
            result[:, paste_y:paste_y_end, paste_x:paste_x_end, :] = processed_crop
            print("✓ Hard composite")
        else:  # blend
            # Create blend mask
            if blend_mask is not None:
                # Use provided mask
                print("  Using custom blend mask")
                if len(blend_mask.shape) == 3:
                    mask_2d = blend_mask[0]
                else:
                    mask_2d = blend_mask[0, :, :, 0]
                
                # Extract the same region from the mask as we did from the crop
                if mask_2d.shape[0] == bbox_height and mask_2d.shape[1] == bbox_width:
                    # Mask matches original bbox size, extract the visible portion
                    mask_2d = mask_2d[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                    blend_alpha = mask_2d
                    print(f"  Extracted mask region: {blend_alpha.shape}")
                elif mask_2d.shape != (actual_height, actual_width):
                    # Resize to actual dimensions
                    mask_reshaped = blend_mask.unsqueeze(1) if len(blend_mask.shape) == 3 else blend_mask.permute(0, 3, 1, 2)
                    mask_resized = F.interpolate(
                        mask_reshaped,
                        size=(actual_height, actual_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    blend_alpha = mask_resized[0, 0] if len(blend_mask.shape) == 3 else mask_resized[0, 0, :, :]
                    print(f"  Resized mask: {blend_alpha.shape}")
                else:
                    blend_alpha = mask_2d[:actual_height, :actual_width]
            elif feather > 0:
                # Create feathered edge mask
                blend_alpha = torch.ones((actual_height, actual_width), 
                                        device=original_image.device, 
                                        dtype=original_image.dtype)
                
                feather_pixels = min(feather, actual_width // 2, actual_height // 2)
                if feather_pixels > 0:
                    for i in range(feather_pixels):
                        alpha_value = i / feather_pixels
                        # Feather all edges
                        blend_alpha[i, :] = alpha_value
                        blend_alpha[actual_height - 1 - i, :] = alpha_value
                        blend_alpha[:, i] = torch.minimum(blend_alpha[:, i], 
                                                         torch.tensor(alpha_value, device=blend_alpha.device))
                        blend_alpha[:, actual_width - 1 - i] = torch.minimum(blend_alpha[:, actual_width - 1 - i],
                                                                             torch.tensor(alpha_value, device=blend_alpha.device))
                print(f"✓ Feathered blend ({feather_pixels}px)")
            else:
                # No feathering
                blend_alpha = torch.ones((actual_height, actual_width), 
                                        device=original_image.device, 
                                        dtype=original_image.dtype)
                print("✓ Soft blend (no feather)")
            
            # Apply blend
            blend_alpha_expanded = blend_alpha.unsqueeze(0).unsqueeze(-1).repeat(batch, 1, 1, channels)
            original_region = result[:, paste_y:paste_y_end, paste_x:paste_x_end, :]
            result[:, paste_y:paste_y_end, paste_x:paste_x_end, :] = (
                processed_crop * blend_alpha_expanded + 
                original_region * (1 - blend_alpha_expanded)
            )
        
        print(f"✓ Result: {result.shape}")
        print("="*60 + "\n")
        
        return (result,)


class QwenImagePrep:
    """
    Prepare images for Qwen vision encoders
    Fixes dimension/shape errors by resizing to compatible resolutions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (["224x224", "280x280", "336x336", "448x448", "560x560", "672x672", "custom"], {
                    "default": "448x448",
                    "tooltip": "Target resolution (must be multiple of 14 for Qwen)"
                }),
                "resize_mode": (["stretch", "crop", "pad"], {
                    "default": "crop",
                    "tooltip": "How to fit image: stretch (distort), crop (maintain aspect), pad (add borders)"
                }),
                "interpolation": (["bilinear", "bicubic", "lanczos", "nearest"], {
                    "default": "bicubic",
                    "tooltip": "Resampling method"
                }),
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 448,
                    "min": 14,
                    "max": 4096,
                    "step": 14,
                    "display": "number",
                    "tooltip": "Custom width (must be multiple of 14)"
                }),
                "custom_height": ("INT", {
                    "default": 448,
                    "min": 14,
                    "max": 4096,
                    "step": 14,
                    "display": "number",
                    "tooltip": "Custom height (must be multiple of 14)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "info")
    FUNCTION = "prep_image"
    CATEGORY = "Texture Alchemist/Texture"
    
    def prep_image(self, image, resolution, resize_mode, interpolation, 
                   custom_width=448, custom_height=448):
        """
        Resize and format image for Qwen encoder compatibility
        """
        
        print("\n" + "="*60)
        print("Qwen Image Prep")
        print("="*60)
        print(f"Input shape: {image.shape}")
        print(f"Target resolution: {resolution}")
        print(f"Resize mode: {resize_mode}")
        
        batch, in_height, in_width, channels = image.shape
        
        # Parse target resolution
        if resolution == "custom":
            # Ensure custom dimensions are multiples of 14
            target_width = (custom_width // 14) * 14
            target_height = (custom_height // 14) * 14
            if target_width != custom_width or target_height != custom_height:
                print(f"⚠️ Adjusted custom size to nearest multiple of 14:")
                print(f"  {custom_width}×{custom_height} → {target_width}×{target_height}")
        else:
            # Parse preset resolution
            target_width, target_height = map(int, resolution.split('x'))
        
        print(f"Target dimensions: {target_width}×{target_height}")
        print(f"Input dimensions: {in_width}×{in_height}")
        
        # Determine interpolation mode
        if interpolation == "bicubic":
            mode = "bicubic"
        elif interpolation == "lanczos":
            mode = "bicubic"  # PyTorch doesn't have lanczos, use bicubic
        elif interpolation == "nearest":
            mode = "nearest"
        else:  # bilinear
            mode = "bilinear"
        
        # Process based on resize mode
        if resize_mode == "stretch":
            # Simple resize - may distort aspect ratio
            result = self._resize_stretch(image, target_height, target_width, mode)
            
        elif resize_mode == "crop":
            # Resize and center crop - maintains aspect ratio
            result = self._resize_crop(image, target_height, target_width, mode)
            
        else:  # pad
            # Resize and pad - maintains aspect ratio, no cropping
            result = self._resize_pad(image, target_height, target_width, mode)
        
        # Ensure proper channel count (RGB)
        if result.shape[-1] == 1:
            # Grayscale to RGB
            result = result.repeat(1, 1, 1, 3)
            print("  Converted grayscale to RGB")
        elif result.shape[-1] == 4:
            # RGBA to RGB (drop alpha)
            result = result[:, :, :, :3]
            print("  Converted RGBA to RGB")
        elif result.shape[-1] != 3:
            # Other channel counts - take first 3 or pad
            if result.shape[-1] > 3:
                result = result[:, :, :, :3]
            else:
                # Pad to 3 channels
                pad_channels = 3 - result.shape[-1]
                padding = torch.zeros((batch, target_height, target_width, pad_channels),
                                    device=result.device, dtype=result.dtype)
                result = torch.cat([result, padding], dim=-1)
            print(f"  Adjusted channels to 3")
        
        # Clamp to valid range
        result = torch.clamp(result, 0.0, 1.0)
        
        # Create info string
        aspect_in = in_width / in_height
        aspect_out = target_width / target_height
        info_lines = [
            f"Input: {in_width}×{in_height}",
            f"Output: {target_width}×{target_height}",
            f"Mode: {resize_mode}",
            f"Aspect: {aspect_in:.2f} → {aspect_out:.2f}"
        ]
        info_string = " | ".join(info_lines)
        
        print(f"\n✓ Output shape: {result.shape}")
        print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"  Channels: {result.shape[-1]}")
        print(f"  Qwen-ready: ✓")
        print("="*60 + "\n")
        
        return (result, target_width, target_height, info_string)
    
    def _resize_stretch(self, image, target_h, target_w, mode):
        """Simple stretch resize"""
        # Convert to BCHW for interpolate
        image_bchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            image_bchw,
            size=(target_h, target_w),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        return resized.permute(0, 2, 3, 1)
    
    def _resize_crop(self, image, target_h, target_w, mode):
        """Resize and center crop to maintain aspect ratio"""
        batch, in_h, in_w, channels = image.shape
        
        # Calculate scale to fill target while maintaining aspect
        scale_h = target_h / in_h
        scale_w = target_w / in_w
        scale = max(scale_h, scale_w)  # Scale to fill
        
        # Resize to intermediate size
        inter_h = int(in_h * scale)
        inter_w = int(in_w * scale)
        
        image_bchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            image_bchw,
            size=(inter_h, inter_w),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        resized = resized.permute(0, 2, 3, 1)
        
        # Center crop
        crop_y = (inter_h - target_h) // 2
        crop_x = (inter_w - target_w) // 2
        cropped = resized[:, crop_y:crop_y+target_h, crop_x:crop_x+target_w, :]
        
        return cropped
    
    def _resize_pad(self, image, target_h, target_w, mode):
        """Resize and pad to maintain aspect ratio"""
        batch, in_h, in_w, channels = image.shape
        
        # Calculate scale to fit within target
        scale_h = target_h / in_h
        scale_w = target_w / in_w
        scale = min(scale_h, scale_w)  # Scale to fit
        
        # Resize to intermediate size
        inter_h = int(in_h * scale)
        inter_w = int(in_w * scale)
        
        image_bchw = image.permute(0, 3, 1, 2)
        resized = F.interpolate(
            image_bchw,
            size=(inter_h, inter_w),
            mode=mode,
            align_corners=False if mode != "nearest" else None
        )
        resized = resized.permute(0, 2, 3, 1)
        
        # Create padded image (black borders)
        result = torch.zeros((batch, target_h, target_w, channels),
                           device=image.device, dtype=image.dtype)
        
        # Center the resized image
        pad_y = (target_h - inter_h) // 2
        pad_x = (target_w - inter_w) // 2
        result[:, pad_y:pad_y+inter_h, pad_x:pad_x+inter_w, :] = resized
        
        return result


class CropToMaskWithPadding:
    """
    Crop image to mask bounds and expand canvas with padding
    Padding adds new pixels (expands canvas), not just includes more source
    Option to pad to square for uniform dimensions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Padding in pixels (expands canvas)"
                }),
                "pad_to_square": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Expand to square canvas (uses longest dimension)"
                }),
                "padding_color": (["black", "white", "mirror", "edge_extend", "average", "custom"], {
                    "default": "black",
                    "tooltip": "How to fill padding area"
                }),
                "custom_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "Hex color for custom padding (e.g., #FF5733)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask (swap black/white)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT", "BBOX_DATA", "STRING")
    RETURN_NAMES = ("image", "mask", "original_width", "original_height", "crop_x", "crop_y", "bbox_data", "info")
    FUNCTION = "crop_and_pad"
    CATEGORY = "Texture Alchemist/Texture"
    
    def crop_and_pad(self, image, mask, padding, pad_to_square, padding_color, custom_color, invert_mask):
        """
        Crop to mask bounds and expand canvas with padding
        """
        
        print("\n" + "="*60)
        print("Crop to Mask with Padding")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Padding: {padding}px")
        print(f"Pad to square: {pad_to_square}")
        print(f"Padding color: {padding_color}")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Ensure mask is 2D for this image
        if len(mask.shape) == 3:  # (B, H, W)
            mask_2d = mask[0]
        else:  # (H, W)
            mask_2d = mask
        
        # Resize mask if needed
        if mask_2d.shape != (height, width):
            mask_reshaped = mask.unsqueeze(0).unsqueeze(0) if len(mask.shape) == 2 else mask.unsqueeze(1)
            mask_resized = F.interpolate(
                mask_reshaped,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            mask_2d = mask_resized[0, 0] if len(mask.shape) == 2 else mask_resized[0, 0]
        
        # Invert mask if requested
        if invert_mask:
            mask_2d = 1.0 - mask_2d
            print(f"✓ Mask inverted")
        
        # Find bounding box of mask
        print(f"\n🔍 Analyzing mask:")
        print(f"  Mask shape: {mask_2d.shape}")
        print(f"  Mask range: [{mask_2d.min():.3f}, {mask_2d.max():.3f}]")
        print(f"  Pixels > 0.01: {(mask_2d > 0.01).sum().item()}")
        print(f"  Pixels > 0.1: {(mask_2d > 0.1).sum().item()}")
        print(f"  Pixels > 0.5: {(mask_2d > 0.5).sum().item()}")
        print(f"  Pixels > 0.9: {(mask_2d > 0.9).sum().item()}")
        
        # Check edges for scattered pixels
        print(f"\n🔍 Edge analysis:")
        print(f"  Top-left corner (0,0): {mask_2d[0, 0]:.3f}")
        print(f"  Top-right corner (0,-1): {mask_2d[0, -1]:.3f}")
        print(f"  Bottom-left corner (-1,0): {mask_2d[-1, 0]:.3f}")
        print(f"  Bottom-right corner (-1,-1): {mask_2d[-1, -1]:.3f}")
        print(f"  Top row max: {mask_2d[0, :].max():.3f}")
        print(f"  Bottom row max: {mask_2d[-1, :].max():.3f}")
        print(f"  Left column max: {mask_2d[:, 0].max():.3f}")
        print(f"  Right column max: {mask_2d[:, -1].max():.3f}")
        
        # Use higher threshold to avoid noise
        mask_binary = (mask_2d > 0.95).float()  # Even higher threshold!
        nonzero = torch.nonzero(mask_binary, as_tuple=False)
        
        print(f"  Non-zero count: {nonzero.shape[0]}")
        
        if nonzero.shape[0] == 0:
            # Empty mask - return original image
            print("\n⚠️ Empty mask detected - returning original")
            bbox_data = {
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "original_width": width,
                "original_height": height
            }
            return (image, mask, width, height, 0, 0, bbox_data, "Empty mask")
        
        # Get bounding box
        y_min = nonzero[:, 0].min().item()
        y_max = nonzero[:, 0].max().item()
        x_min = nonzero[:, 1].min().item()
        x_max = nonzero[:, 1].max().item()
        
        # Calculate crop dimensions
        crop_width = x_max - x_min + 1
        crop_height = y_max - y_min + 1
        
        print(f"\n✓ Mask bounds found:")
        print(f"  Position: ({x_min}, {y_min})")
        print(f"  Size: {crop_width} × {crop_height}")
        print(f"  Original canvas: {width} × {height}")
        print(f"  Crop reduces canvas by: {((1 - (crop_width * crop_height) / (width * height)) * 100):.1f}%")
        
        # Crop image and mask to bounds
        cropped_image = image[:, y_min:y_max+1, x_min:x_max+1, :]
        cropped_mask = mask_2d[y_min:y_max+1, x_min:x_max+1]
        
        # If we inverted for detection, invert back for output
        if invert_mask:
            cropped_mask = 1.0 - cropped_mask
        
        print(f"\n✂️ Cropping:")
        print(f"  Cropped image: {cropped_image.shape}")
        print(f"  Cropped mask: {cropped_mask.shape}")
        print(f"  Expected: [{batch}, {crop_height}, {crop_width}, {channels}]")
        
        # Add padding (expand canvas)
        if padding > 0 or pad_to_square:
            # Calculate final dimensions
            if pad_to_square:
                max_dim = max(crop_width, crop_height)
                final_width = max_dim + (padding * 2)
                final_height = max_dim + (padding * 2)
            else:
                final_width = crop_width + (padding * 2)
                final_height = crop_height + (padding * 2)
            
            print(f"\n✓ Expanding canvas:")
            print(f"  From: {crop_width} × {crop_height}")
            print(f"  To: {final_width} × {final_height}")
            
            # Create padded canvas
            if padding_color == "black":
                padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                          device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            elif padding_color == "white":
                padded_image = torch.ones((batch, final_height, final_width, channels), 
                                         device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            elif padding_color == "average":
                # Calculate average color of the cropped image
                avg_color = cropped_image.mean(dim=[1, 2], keepdim=True)  # Average across H,W
                padded_image = avg_color.expand(batch, final_height, final_width, channels).clone()
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
                print(f"✓ Using average color: R={avg_color[0,0,0,0]:.3f}, G={avg_color[0,0,0,1]:.3f}, B={avg_color[0,0,0,2]:.3f}")
            elif padding_color == "custom":
                # Parse hex color
                hex_color = custom_color.strip().lstrip('#')
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    # Create color tensor [1, 1, 1, channels]
                    if channels == 4:
                        color = torch.tensor([r, g, b, 1.0], device=device, dtype=dtype).view(1, 1, 1, 4)
                    else:
                        color = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
                    padded_image = color.expand(batch, final_height, final_width, channels).clone()
                    padded_mask = torch.zeros((final_height, final_width), 
                                             device=device, dtype=dtype)
                    print(f"✓ Using custom color #{hex_color}: R={r:.3f}, G={g:.3f}, B={b:.3f}")
                else:
                    print(f"⚠️ Invalid hex color '{custom_color}', using black")
                    padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                              device=device, dtype=dtype)
                    padded_mask = torch.zeros((final_height, final_width), 
                                             device=device, dtype=dtype)
            else:
                # For mirror and edge_extend, start with zeros
                padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                          device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            
            # Calculate center position
            y_offset = (final_height - crop_height) // 2
            x_offset = (final_width - crop_width) // 2
            
            # Place cropped content in center
            padded_image[:, y_offset:y_offset+crop_height, x_offset:x_offset+crop_width, :] = cropped_image
            padded_mask[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width] = cropped_mask
            
            # Apply padding style
            if padding_color == "mirror":
                padded_image = self._apply_mirror_padding(
                    padded_image, cropped_image, 
                    y_offset, x_offset, crop_height, crop_width
                )
            elif padding_color == "edge_extend":
                padded_image = self._apply_edge_extend(
                    padded_image, cropped_image,
                    y_offset, x_offset, crop_height, crop_width
                )
            
            result_image = padded_image
            result_mask = padded_mask.unsqueeze(0)  # Add batch dim
        else:
            result_image = cropped_image
            result_mask = cropped_mask.unsqueeze(0)
            final_width = crop_width
            final_height = crop_height
        
        # Create bundled bbox data (for SimpleInpaintStitch compatibility)
        bbox_data = {
            "x": x_min,
            "y": y_min,
            "width": final_width,
            "height": final_height,
            "original_width": width,
            "original_height": height
        }
        
        # Create info string
        info_lines = [
            f"Original: {width}×{height}",
            f"Crop: {crop_width}×{crop_height}",
            f"Final: {final_width}×{final_height}",
            f"Position: ({x_min},{y_min})"
        ]
        info_string = " | ".join(info_lines)
        
        print(f"\n✓ Output:")
        print(f"  Result image: {result_image.shape}")
        print(f"  Result mask: {result_mask.shape}")
        print(f"  Original dimensions: {width}×{height}")
        print(f"  Final dimensions: {final_width}×{final_height}")
        print(f"  Crop position: ({x_min}, {y_min})")
        print(f"  Canvas reduced: {width}×{height} → {final_width}×{final_height}")
        print("="*60 + "\n")
        
        return (result_image, result_mask, width, height, x_min, y_min, bbox_data, info_string)
    
    def _apply_mirror_padding(self, padded, cropped, y_off, x_off, h, w):
        """Mirror edges for padding"""
        batch = padded.shape[0]
        
        # Mirror top
        if y_off > 0:
            mirror_h = min(y_off, h)
            padded[:, y_off-mirror_h:y_off, x_off:x_off+w, :] = torch.flip(
                cropped[:, :mirror_h, :, :], [1]
            )
        
        # Mirror bottom
        if y_off + h < padded.shape[1]:
            mirror_h = min(padded.shape[1] - (y_off + h), h)
            padded[:, y_off+h:y_off+h+mirror_h, x_off:x_off+w, :] = torch.flip(
                cropped[:, h-mirror_h:h, :, :], [1]
            )
        
        # Mirror left
        if x_off > 0:
            mirror_w = min(x_off, w)
            padded[:, y_off:y_off+h, x_off-mirror_w:x_off, :] = torch.flip(
                cropped[:, :, :mirror_w, :], [2]
            )
        
        # Mirror right
        if x_off + w < padded.shape[2]:
            mirror_w = min(padded.shape[2] - (x_off + w), w)
            padded[:, y_off:y_off+h, x_off+w:x_off+w+mirror_w, :] = torch.flip(
                cropped[:, :, w-mirror_w:w, :], [2]
            )
        
        return padded
    
    def _apply_edge_extend(self, padded, cropped, y_off, x_off, h, w):
        """Extend edges for padding"""
        batch = padded.shape[0]
        
        # Extend top
        if y_off > 0:
            padded[:, :y_off, x_off:x_off+w, :] = cropped[:, 0:1, :, :].repeat(1, y_off, 1, 1)
        
        # Extend bottom
        if y_off + h < padded.shape[1]:
            padded[:, y_off+h:, x_off:x_off+w, :] = cropped[:, -1:, :, :].repeat(
                1, padded.shape[1] - (y_off + h), 1, 1
            )
        
        # Extend left
        if x_off > 0:
            padded[:, y_off:y_off+h, :x_off, :] = cropped[:, :, 0:1, :].repeat(1, 1, x_off, 1)
        
        # Extend right
        if x_off + w < padded.shape[2]:
            padded[:, y_off:y_off+h, x_off+w:, :] = cropped[:, :, -1:, :].repeat(
                1, 1, padded.shape[2] - (x_off + w), 1
            )
        
        return padded


class SimpleCropToMaskWithPadding:
    """
    SIMPLE CROP TO MASK WITH PADDING
    Simplified crop-to-mask with canvas padding
    Compatible with SimpleInpaintStitch workflow
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Padding in pixels (expands canvas)"
                }),
                "pad_to_square": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Expand to square canvas (uses longest dimension)"
                }),
                "padding_color": (["black", "white", "mirror", "edge_extend", "average", "custom"], {
                    "default": "black",
                    "tooltip": "How to fill padding area"
                }),
                "custom_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "tooltip": "Hex color for custom padding (e.g., #FF5733)"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert mask (swap black/white)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "BBOX_DATA", "STRING")
    RETURN_NAMES = ("cropped_image", "cropped_mask", "masked_composite", "bbox_data", "info")
    FUNCTION = "crop_and_pad"
    CATEGORY = "Texture Alchemist/Inpainting/Simple"
    
    def crop_and_pad(self, image, mask, padding, pad_to_square, padding_color, custom_color, invert_mask):
        """
        Crop to mask bounds, expand canvas with padding, output simplified
        """
        
        print("\n" + "="*60)
        print("Simple Crop to Mask with Padding")
        print("="*60)
        print(f"Image: {image.shape}")
        print(f"Mask: {mask.shape}")
        print(f"Padding: {padding}px")
        print(f"Pad to square: {pad_to_square}")
        print(f"Padding: {padding_color}")
        
        batch, height, width, channels = image.shape
        device = image.device
        dtype = image.dtype
        
        # Ensure mask is 2D for this image
        if len(mask.shape) == 3:  # (B, H, W)
            mask_2d = mask[0]
        else:  # (H, W)
            mask_2d = mask
        
        # Resize mask if needed
        if mask_2d.shape != (height, width):
            mask_reshaped = mask.unsqueeze(0).unsqueeze(0) if len(mask.shape) == 2 else mask.unsqueeze(1)
            mask_resized = F.interpolate(
                mask_reshaped,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
            mask_2d = mask_resized[0, 0] if len(mask.shape) == 2 else mask_resized[0, 0]
        
        # Invert mask if requested
        if invert_mask:
            mask_2d = 1.0 - mask_2d
            print("✓ Mask inverted")
        
        # Find bounding box
        mask_binary = (mask_2d > 0.95).float()
        nonzero = torch.nonzero(mask_binary, as_tuple=False)
        
        if nonzero.shape[0] == 0:
            # Empty mask - return full image
            print("⚠️ Empty mask - using full image")
            bbox_x, bbox_y = 0, 0
            bbox_width, bbox_height = width, height
            cropped_image = image
            cropped_mask_2d = mask_2d
        else:
            y_min = nonzero[:, 0].min().item()
            y_max = nonzero[:, 0].max().item()
            x_min = nonzero[:, 1].min().item()
            x_max = nonzero[:, 1].max().item()
            
            # Calculate crop dimensions
            crop_width = x_max - x_min + 1
            crop_height = y_max - y_min + 1
            
            print(f"✓ Mask bounds: ({x_min}, {y_min}) {crop_width}×{crop_height}")
            
            # Crop image and mask to bounds
            cropped_image = image[:, y_min:y_max+1, x_min:x_max+1, :]
            cropped_mask_2d = mask_2d[y_min:y_max+1, x_min:x_max+1]
            
            bbox_x = x_min
            bbox_y = y_min
            bbox_width = crop_width
            bbox_height = crop_height
        
        # Add padding (expand canvas)
        if padding > 0 or pad_to_square:
            # Calculate final dimensions
            if pad_to_square:
                max_dim = max(bbox_width, bbox_height)
                final_width = max_dim + (padding * 2)
                final_height = max_dim + (padding * 2)
            else:
                final_width = bbox_width + (padding * 2)
                final_height = bbox_height + (padding * 2)
            
            print(f"✓ Expanding: {bbox_width}×{bbox_height} → {final_width}×{final_height}")
            
            # Create padded canvas
            if padding_color == "black":
                padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                          device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            elif padding_color == "white":
                padded_image = torch.ones((batch, final_height, final_width, channels), 
                                         device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            elif padding_color == "average":
                # Calculate average color of the cropped image
                avg_color = cropped_image.mean(dim=[1, 2], keepdim=True)  # Average across H,W
                padded_image = avg_color.expand(batch, final_height, final_width, channels).clone()
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
                print(f"✓ Using average color: R={avg_color[0,0,0,0]:.3f}, G={avg_color[0,0,0,1]:.3f}, B={avg_color[0,0,0,2]:.3f}")
            elif padding_color == "custom":
                # Parse hex color
                hex_color = custom_color.strip().lstrip('#')
                if len(hex_color) == 6:
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    # Create color tensor [1, 1, 1, channels]
                    if channels == 4:
                        color = torch.tensor([r, g, b, 1.0], device=device, dtype=dtype).view(1, 1, 1, 4)
                    else:
                        color = torch.tensor([r, g, b], device=device, dtype=dtype).view(1, 1, 1, 3)
                    padded_image = color.expand(batch, final_height, final_width, channels).clone()
                    padded_mask = torch.zeros((final_height, final_width), 
                                             device=device, dtype=dtype)
                    print(f"✓ Using custom color #{hex_color}: R={r:.3f}, G={g:.3f}, B={b:.3f}")
                else:
                    print(f"⚠️ Invalid hex color '{custom_color}', using black")
                    padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                              device=device, dtype=dtype)
                    padded_mask = torch.zeros((final_height, final_width), 
                                             device=device, dtype=dtype)
            else:
                # For mirror and edge_extend, start with zeros
                padded_image = torch.zeros((batch, final_height, final_width, channels), 
                                          device=device, dtype=dtype)
                padded_mask = torch.zeros((final_height, final_width), 
                                         device=device, dtype=dtype)
            
            # Calculate center position (where original content will be placed)
            y_offset = (final_height - bbox_height) // 2
            x_offset = (final_width - bbox_width) // 2
            
            # Place cropped content in center
            padded_image[:, y_offset:y_offset+bbox_height, x_offset:x_offset+bbox_width, :] = cropped_image
            padded_mask[y_offset:y_offset+bbox_height, x_offset:x_offset+bbox_width] = cropped_mask_2d
            
            # Apply padding style
            if padding_color == "mirror":
                padded_image = self._apply_mirror_padding(
                    padded_image, cropped_image, 
                    y_offset, x_offset, bbox_height, bbox_width
                )
            elif padding_color == "edge_extend":
                padded_image = self._apply_edge_extend(
                    padded_image, cropped_image,
                    y_offset, x_offset, bbox_height, bbox_width
                )
            
            result_image = padded_image
            result_mask = padded_mask.unsqueeze(0)  # Add batch dim
            
            # CRITICAL: Adjust bbox position to account for padding offset
            # The original content is now at (x_offset, y_offset) within the padded crop
            # So we need to shift the bbox position back by this offset for proper alignment
            bbox_x = bbox_x - x_offset
            bbox_y = bbox_y - y_offset
            print(f"✓ Adjusted bbox for padding: ({bbox_x}, {bbox_y})")
        else:
            result_image = cropped_image
            result_mask = cropped_mask_2d.unsqueeze(0)
            final_width = bbox_width
            final_height = bbox_height
        
        # Create masked composite (image with mask applied)
        mask_expanded = result_mask.unsqueeze(-1).repeat(1, 1, 1, channels)
        masked_composite = result_image * mask_expanded
        
        # Create bbox data bundle (for SimpleInpaintStitch)
        bbox_data = {
            "x": bbox_x,
            "y": bbox_y,
            "width": final_width,
            "height": final_height,
            "original_width": width,
            "original_height": height
        }
        
        # Info string
        info = f"Crop: {final_width}×{final_height} at ({bbox_x},{bbox_y}) | Orig: {width}×{height}"
        
        print(f"✓ Output: {final_width}×{final_height}")
        print("="*60 + "\n")
        
        return (result_image, result_mask, masked_composite, bbox_data, info)
    
    def _apply_mirror_padding(self, padded, cropped, y_off, x_off, h, w):
        """Mirror edges for padding"""
        batch = padded.shape[0]
        
        # Mirror top
        if y_off > 0:
            mirror_h = min(y_off, h)
            padded[:, y_off-mirror_h:y_off, x_off:x_off+w, :] = torch.flip(
                cropped[:, :mirror_h, :, :], [1]
            )
        
        # Mirror bottom
        if y_off + h < padded.shape[1]:
            mirror_h = min(padded.shape[1] - (y_off + h), h)
            padded[:, y_off+h:y_off+h+mirror_h, x_off:x_off+w, :] = torch.flip(
                cropped[:, h-mirror_h:h, :, :], [1]
            )
        
        # Mirror left
        if x_off > 0:
            mirror_w = min(x_off, w)
            padded[:, y_off:y_off+h, x_off-mirror_w:x_off, :] = torch.flip(
                cropped[:, :, :mirror_w, :], [2]
            )
        
        # Mirror right
        if x_off + w < padded.shape[2]:
            mirror_w = min(padded.shape[2] - (x_off + w), w)
            padded[:, y_off:y_off+h, x_off+w:x_off+w+mirror_w, :] = torch.flip(
                cropped[:, :, w-mirror_w:w, :], [2]
            )
        
        return padded
    
    def _apply_edge_extend(self, padded, cropped, y_off, x_off, h, w):
        """Extend edges for padding"""
        batch = padded.shape[0]
        
        # Extend top
        if y_off > 0:
            padded[:, :y_off, x_off:x_off+w, :] = cropped[:, 0:1, :, :].repeat(1, y_off, 1, 1)
        
        # Extend bottom
        if y_off + h < padded.shape[1]:
            padded[:, y_off+h:, x_off:x_off+w, :] = cropped[:, -1:, :, :].repeat(
                1, padded.shape[1] - (y_off + h), 1, 1
            )
        
        # Extend left
        if x_off > 0:
            padded[:, y_off:y_off+h, :x_off, :] = cropped[:, :, 0:1, :].repeat(1, 1, x_off, 1)
        
        # Extend right
        if x_off + w < padded.shape[2]:
            padded[:, y_off:y_off+h, x_off+w:, :] = cropped[:, :, -1:, :].repeat(
                1, 1, padded.shape[2] - (x_off + w), 1
            )
        
        return padded


# Node registration
class BBoxReposition:
    """
    Reposition an image on a base canvas using bbox data
    Supports 9 anchor positions (corners, edges, center)
    Outputs repositioned image with updated bbox data
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_image": ("IMAGE",),
                "bbox_data": ("BBOX_DATA",),
                "position": (["top_left", "top_center", "top_right", 
                             "center_left", "center", "center_right",
                             "bottom_left", "bottom_center", "bottom_right"], {
                    "default": "center",
                    "tooltip": "Where to position the image on the base canvas"
                }),
                "blend_mode": (["replace", "blend"], {
                    "default": "replace",
                    "tooltip": "Replace or alpha blend with base"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "BBOX_DATA", "STRING")
    RETURN_NAMES = ("image", "bbox_data", "info")
    FUNCTION = "reposition"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def reposition(self, image, base_image, bbox_data, position, blend_mode):
        """
        Reposition image on base canvas according to anchor position
        """
        
        print("\n" + "="*60)
        print("BBox Reposition")
        print("="*60)
        print(f"Image: {image.shape}")
        print(f"Base: {base_image.shape}")
        print(f"Position: {position}")
        print(f"BBox Data: {bbox_data}")
        
        batch, img_h, img_w, channels = image.shape
        base_batch, base_h, base_w, base_channels = base_image.shape
        
        # Match channels if needed
        if channels != base_channels:
            print(f"  Channel mismatch: image={channels}, base={base_channels}")
            if channels == 3 and base_channels == 4:
                # Add alpha to image
                alpha = torch.ones((batch, img_h, img_w, 1), device=image.device, dtype=image.dtype)
                image = torch.cat([image, alpha], dim=-1)
                channels = 4
                print("  Added alpha channel to image")
            elif channels == 4 and base_channels == 3:
                # Drop alpha from image
                image = image[:, :, :, :3]
                channels = 3
                print("  Dropped alpha channel from image")
        
        # Calculate position on base canvas
        if position == "top_left":
            new_x, new_y = 0, 0
        elif position == "top_center":
            new_x, new_y = (base_w - img_w) // 2, 0
        elif position == "top_right":
            new_x, new_y = base_w - img_w, 0
        elif position == "center_left":
            new_x, new_y = 0, (base_h - img_h) // 2
        elif position == "center":
            new_x, new_y = (base_w - img_w) // 2, (base_h - img_h) // 2
        elif position == "center_right":
            new_x, new_y = base_w - img_w, (base_h - img_h) // 2
        elif position == "bottom_left":
            new_x, new_y = 0, base_h - img_h
        elif position == "bottom_center":
            new_x, new_y = (base_w - img_w) // 2, base_h - img_h
        elif position == "bottom_right":
            new_x, new_y = base_w - img_w, base_h - img_h
        
        # Clamp to boundaries
        new_x = max(0, min(new_x, base_w - 1))
        new_y = max(0, min(new_y, base_h - 1))
        
        # Calculate actual paste region
        paste_w = min(img_w, base_w - new_x)
        paste_h = min(img_h, base_h - new_y)
        
        print(f"✓ New position: ({new_x}, {new_y})")
        print(f"  Paste region: {paste_w}×{paste_h}")
        
        # Create result (copy of base)
        result = base_image.clone()
        
        # Paste image
        if blend_mode == "replace":
            result[:, new_y:new_y+paste_h, new_x:new_x+paste_w, :] = image[:, :paste_h, :paste_w, :]
        else:  # blend
            # Simple alpha blend
            result[:, new_y:new_y+paste_h, new_x:new_x+paste_w, :] = (
                result[:, new_y:new_y+paste_h, new_x:new_x+paste_w, :] * 0.5 +
                image[:, :paste_h, :paste_w, :] * 0.5
            )
        
        # Create updated bbox data
        new_bbox_data = {
            "x": new_x,
            "y": new_y,
            "width": img_w,
            "height": img_h,
            "original_width": base_w,
            "original_height": base_h
        }
        
        info = f"Repositioned to {position}: ({new_x}, {new_y}) on {base_w}×{base_h} canvas"
        
        print(f"✓ Updated BBox: {new_bbox_data}")
        print("="*60 + "\n")
        
        return (result, new_bbox_data, info)


class PatchUpscale:
    """
    Upscale a cropped patch to target megapixels for high-quality processing
    Outputs scale data for PatchFit to restore original dimensions
    Perfect for inpainting workflows: crop → upscale → process → downscale → stitch
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_megapixels": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 16.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Target resolution in megapixels (e.g., 2.0 = 2MP)"
                }),
                "max_dimension": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 64,
                    "tooltip": "Maximum width/height (prevents extreme upscaling)"
                }),
                "upscale_method": (["bicubic", "bilinear", "lanczos", "nearest"], {
                    "default": "bicubic",
                    "tooltip": "Interpolation method for upscaling"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to upscale with image"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "PATCH_SCALE_DATA", "STRING")
    RETURN_NAMES = ("upscaled_image", "upscaled_mask", "scale_data", "info")
    FUNCTION = "upscale_patch"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def upscale_patch(self, image, target_megapixels, max_dimension, upscale_method, mask=None):
        """
        Upscale patch to target megapixels while preserving aspect ratio
        """
        
        print("\n" + "="*60)
        print("Patch Upscale")
        print("="*60)
        
        batch, height, width, channels = image.shape
        original_mp = (width * height) / 1_000_000
        aspect_ratio = width / height
        
        print(f"Original: {width}×{height} ({original_mp:.2f}MP)")
        print(f"Aspect ratio: {aspect_ratio:.3f}")
        print(f"Target: {target_megapixels}MP")
        
        # Calculate target dimensions
        # total_pixels = target_megapixels * 1_000_000
        # width * height = total_pixels
        # width / height = aspect_ratio
        # width = aspect_ratio * height
        # (aspect_ratio * height) * height = total_pixels
        # height^2 = total_pixels / aspect_ratio
        
        target_pixels = target_megapixels * 1_000_000
        new_height = int((target_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        
        # Clamp to max dimension
        if new_width > max_dimension or new_height > max_dimension:
            if new_width > new_height:
                scale = max_dimension / new_width
            else:
                scale = max_dimension / new_height
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            print(f"  Clamped to max dimension: {max_dimension}")
        
        # Don't downscale (only upscale or keep same)
        if new_width < width or new_height < height:
            new_width = width
            new_height = height
            print(f"  Already at or above target MP - keeping original size")
        
        final_mp = (new_width * new_height) / 1_000_000
        scale_factor = new_width / width
        
        print(f"✓ New dimensions: {new_width}×{new_height} ({final_mp:.2f}MP)")
        print(f"  Scale factor: {scale_factor:.3f}x")
        
        # Perform upscale
        if new_width != width or new_height != height:
            # Map method names to torch interpolation modes
            mode_map = {
                "bicubic": "bicubic",
                "bilinear": "bilinear",
                "lanczos": "bicubic",  # torch doesn't have lanczos, use bicubic
                "nearest": "nearest"
            }
            mode = mode_map.get(upscale_method, "bicubic")
            
            upscaled = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(new_height, new_width),
                mode=mode,
                align_corners=False if mode != "nearest" else None
            ).permute(0, 2, 3, 1)
            
            print(f"  Method: {upscale_method}")
        else:
            upscaled = image
            print(f"  No upscaling needed")
        
        # Upscale mask if provided
        if mask is not None:
            print(f"  Upscaling mask: {mask.shape}")
            
            # Ensure mask is in correct format
            if len(mask.shape) == 3:  # (B, H, W)
                mask_to_scale = mask.unsqueeze(1)  # (B, 1, H, W)
            elif len(mask.shape) == 4:  # (B, H, W, C)
                mask_to_scale = mask.permute(0, 3, 1, 2)  # (B, C, H, W)
            else:
                mask_to_scale = mask
            
            if new_width != width or new_height != height:
                upscaled_mask = F.interpolate(
                    mask_to_scale,
                    size=(new_height, new_width),
                    mode='bilinear',  # Always use bilinear for masks
                    align_corners=False
                )
            else:
                upscaled_mask = mask_to_scale
            
            # Convert back to (B, H, W) format
            if len(mask.shape) == 3:
                upscaled_mask = upscaled_mask.squeeze(1)
            elif len(mask.shape) == 4:
                upscaled_mask = upscaled_mask.permute(0, 2, 3, 1)
            
            print(f"  Upscaled mask: {upscaled_mask.shape}")
        else:
            # Create empty mask if none provided
            upscaled_mask = torch.zeros((batch, new_height, new_width), 
                                       device=image.device, dtype=image.dtype)
            print(f"  No mask provided")
        
        # Create scale data for PatchFit
        scale_data = {
            "original_width": width,
            "original_height": height,
            "upscaled_width": new_width,
            "upscaled_height": new_height,
            "scale_factor": scale_factor,
            "target_megapixels": target_megapixels,
            "actual_megapixels": final_mp
        }
        
        info = f"Upscaled {width}×{height} → {new_width}×{new_height} ({scale_factor:.2f}x, {final_mp:.2f}MP)"
        
        print(f"✓ Scale data saved for PatchFit")
        print("="*60 + "\n")
        
        return (upscaled, upscaled_mask, scale_data, info)


class PatchFit:
    """
    Downscale processed patch back to original dimensions
    Uses scale data from PatchUpscale to restore exact original size
    Completes the workflow: crop → upscale → process → fit → stitch
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_data": ("PATCH_SCALE_DATA",),
                "downscale_method": (["bicubic", "bilinear", "lanczos", "area"], {
                    "default": "area",
                    "tooltip": "Interpolation method for downscaling (area is best for downscaling)"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional mask to downscale with image"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("fitted_image", "fitted_mask", "info")
    FUNCTION = "fit_patch"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def fit_patch(self, image, scale_data, downscale_method, mask=None):
        """
        Downscale processed patch back to original dimensions
        """
        
        print("\n" + "="*60)
        print("Patch Fit")
        print("="*60)
        
        batch, height, width, channels = image.shape
        
        target_width = scale_data["original_width"]
        target_height = scale_data["original_height"]
        original_scale = scale_data["scale_factor"]
        
        print(f"Current: {width}×{height}")
        print(f"Target: {target_width}×{target_height}")
        print(f"Original scale factor: {original_scale:.3f}x")
        
        # Check if dimensions match expected upscaled size
        expected_width = scale_data["upscaled_width"]
        expected_height = scale_data["upscaled_height"]
        
        if width != expected_width or height != expected_height:
            print(f"⚠️  Warning: Image size doesn't match expected upscaled size")
            print(f"   Expected: {expected_width}×{expected_height}")
            print(f"   Actual: {width}×{height}")
            print(f"   Continuing with downscale anyway...")
        
        # Perform downscale
        if width != target_width or height != target_height:
            # Map method names to torch interpolation modes
            mode_map = {
                "bicubic": "bicubic",
                "bilinear": "bilinear",
                "lanczos": "bicubic",  # torch doesn't have lanczos, use bicubic
                "area": "area",
                "nearest": "nearest"
            }
            mode = mode_map.get(downscale_method, "area")
            
            fitted = F.interpolate(
                image.permute(0, 3, 1, 2),
                size=(target_height, target_width),
                mode=mode,
                align_corners=False if mode not in ["nearest", "area"] else None
            ).permute(0, 2, 3, 1)
            
            actual_scale = target_width / width
            print(f"✓ Downscaled: {width}×{height} → {target_width}×{target_height}")
            print(f"  Scale factor: {actual_scale:.3f}x")
            print(f"  Method: {downscale_method}")
        else:
            fitted = image
            print(f"✓ Already at target size - no downscaling needed")
        
        # Downscale mask if provided
        if mask is not None:
            print(f"  Downscaling mask: {mask.shape}")
            
            # Ensure mask is in correct format
            if len(mask.shape) == 3:  # (B, H, W)
                mask_to_scale = mask.unsqueeze(1)  # (B, 1, H, W)
            elif len(mask.shape) == 4:  # (B, H, W, C)
                mask_to_scale = mask.permute(0, 3, 1, 2)  # (B, C, H, W)
            else:
                mask_to_scale = mask
            
            if width != target_width or height != target_height:
                fitted_mask = F.interpolate(
                    mask_to_scale,
                    size=(target_height, target_width),
                    mode='bilinear',  # Always use bilinear for masks
                    align_corners=False
                )
            else:
                fitted_mask = mask_to_scale
            
            # Convert back to original format
            if len(mask.shape) == 3:
                fitted_mask = fitted_mask.squeeze(1)
            elif len(mask.shape) == 4:
                fitted_mask = fitted_mask.permute(0, 2, 3, 1)
            
            print(f"  Downscaled mask: {fitted_mask.shape}")
        else:
            # Create empty mask if none provided
            fitted_mask = torch.zeros((batch, target_height, target_width), 
                                     device=image.device, dtype=image.dtype)
            print(f"  No mask provided")
        
        info = f"Fitted {width}×{height} → {target_width}×{target_height} (ready for stitching)"
        
        print(f"✓ Ready for stitching back into composition")
        print("="*60 + "\n")
        
        return (fitted, fitted_mask, info)


class BBoxEditor:
    """
    Manually edit/override bbox data values
    Useful for fine-tuning or correcting bbox coordinates
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_data": ("BBOX_DATA",),
                "override_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override X position (-1 = keep original)"
                }),
                "override_y": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override Y position (-1 = keep original)"
                }),
                "override_width": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override width (-1 = keep original)"
                }),
                "override_height": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override height (-1 = keep original)"
                }),
                "override_original_width": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override original canvas width (-1 = keep original)"
                }),
                "override_original_height": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 10000,
                    "tooltip": "Override original canvas height (-1 = keep original)"
                }),
            }
        }
    
    RETURN_TYPES = ("BBOX_DATA", "STRING")
    RETURN_NAMES = ("bbox_data", "info")
    FUNCTION = "edit_bbox"
    CATEGORY = "Texture Alchemist/Inpainting"
    
    def edit_bbox(self, bbox_data, override_x, override_y, override_width, override_height, 
                  override_original_width, override_original_height):
        """
        Override bbox data values with user-specified values
        """
        
        print("\n" + "="*60)
        print("BBox Editor")
        print("="*60)
        print(f"Original BBox: {bbox_data}")
        
        # Create new bbox data with overrides
        new_bbox_data = {
            "x": bbox_data.get("x", 0) if override_x == -1 else override_x,
            "y": bbox_data.get("y", 0) if override_y == -1 else override_y,
            "width": bbox_data.get("width", 0) if override_width == -1 else override_width,
            "height": bbox_data.get("height", 0) if override_height == -1 else override_height,
            "original_width": bbox_data.get("original_width", 0) if override_original_width == -1 else override_original_width,
            "original_height": bbox_data.get("original_height", 0) if override_original_height == -1 else override_original_height
        }
        
        # Build info string showing what changed
        changes = []
        if override_x != -1:
            changes.append(f"x: {bbox_data.get('x')} → {new_bbox_data['x']}")
        if override_y != -1:
            changes.append(f"y: {bbox_data.get('y')} → {new_bbox_data['y']}")
        if override_width != -1:
            changes.append(f"width: {bbox_data.get('width')} → {new_bbox_data['width']}")
        if override_height != -1:
            changes.append(f"height: {bbox_data.get('height')} → {new_bbox_data['height']}")
        if override_original_width != -1:
            changes.append(f"orig_w: {bbox_data.get('original_width')} → {new_bbox_data['original_width']}")
        if override_original_height != -1:
            changes.append(f"orig_h: {bbox_data.get('original_height')} → {new_bbox_data['original_height']}")
        
        if changes:
            info = "Modified: " + ", ".join(changes)
            print(f"✓ Changes: {len(changes)}")
            for change in changes:
                print(f"  - {change}")
        else:
            info = "No changes (all overrides set to -1)"
            print("  No changes made")
        
        print(f"✓ New BBox: {new_bbox_data}")
        print("="*60 + "\n")
        
        return (new_bbox_data, info)


NODE_CLASS_MAPPINGS = {
    "SeamlessTiling": SeamlessTiling,
    "TextureScaler": TextureScaler,
    "TriplanarProjection": TriplanarProjection,
    "TextureOffset": TextureOffset,
    "TextureTiler": TextureTiler,
    "SmartTextureResizer": SmartTextureResizer,
    "SquareMaker": SquareMaker,
    "TextureEqualizer": TextureEqualizer,
    "UpscaleCalculator": UpscaleCalculator,
    "UpscaleToResolution": UpscaleToResolution,
    "PaddingCalculator": PaddingCalculator,
    "InpaintCropExtractor": InpaintCropExtractor,
    "InpaintStitcher": InpaintStitcher,
    "SimpleInpaintCrop": SimpleInpaintCrop,
    "SimpleInpaintStitch": SimpleInpaintStitch,
    "QwenImagePrep": QwenImagePrep,
    "CropToMaskWithPadding": CropToMaskWithPadding,
    "SimpleCropToMaskWithPadding": SimpleCropToMaskWithPadding,
    "BBoxReposition": BBoxReposition,
    "BBoxEditor": BBoxEditor,
    "PatchUpscale": PatchUpscale,
    "PatchFit": PatchFit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeamlessTiling": "Seamless Tiling Maker",
    "TextureScaler": "Texture Scaler",
    "TriplanarProjection": "Triplanar Projection",
    "TextureOffset": "Texture Offset",
    "TextureTiler": "Texture Tiler",
    "SmartTextureResizer": "Smart Texture Resizer",
    "SquareMaker": "Square Maker",
    "TextureEqualizer": "Texture Equalizer",
    "UpscaleCalculator": "Upscale Calculator (Multi-Pass)",
    "UpscaleToResolution": "Upscale to Resolution (Multi-Pass)",
    "PaddingCalculator": "Padding Calculator",
    "InpaintCropExtractor": "Inpaint Crop Extractor (Advanced)",
    "InpaintStitcher": "Inpaint Stitcher (Advanced)",
    "SimpleInpaintCrop": "Simple Inpaint Crop ⚡",
    "SimpleInpaintStitch": "Simple Inpaint Stitch ⚡",
    "QwenImagePrep": "Qwen Image Prep",
    "CropToMaskWithPadding": "Crop to Mask (with Padding)",
    "SimpleCropToMaskWithPadding": "Simple Crop to Mask (with Padding) ⚡",
    "BBoxReposition": "BBox Reposition 📐",
    "BBoxEditor": "BBox Editor ✏️",
    "PatchUpscale": "Patch Upscale 🔍",
    "PatchFit": "Patch Fit 📦",
}

