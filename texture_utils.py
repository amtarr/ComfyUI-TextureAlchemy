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


# Node registration
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
}

