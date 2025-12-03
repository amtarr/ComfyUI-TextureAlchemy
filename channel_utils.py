"""
Channel Utilities
Pack and unpack multiple maps into RGB channels
"""

import torch


class ChannelPacker:
    """
    Pack 3 grayscale maps into RGB channels
    Common: Pack Roughness (R), Metallic (G), AO (B)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "red_channel": ("IMAGE",),
            },
            "optional": {
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
                "preset": (["custom", "orm_unity", "orm_unreal", "rma"], {
                    "default": "custom",
                    "tooltip": "Preset channel layouts: ORM (AO/Roughness/Metallic), RMA (Roughness/Metallic/AO)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("packed",)
    FUNCTION = "pack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def pack(self, red_channel, green_channel=None, blue_channel=None, preset="custom"):
        """Pack grayscale images into RGB channels"""
        
        print("\n" + "="*60)
        print("Channel Packer")
        print("="*60)
        print(f"Preset: {preset}")
        
        # Get dimensions from red channel
        batch, height, width, channels = red_channel.shape
        device = red_channel.device
        dtype = red_channel.dtype
        
        # Convert to grayscale if needed
        r = self._to_grayscale(red_channel)
        g = self._to_grayscale(green_channel) if green_channel is not None else torch.ones_like(r)
        b = self._to_grayscale(blue_channel) if blue_channel is not None else torch.ones_like(r)
        
        # Resize to match if sizes differ
        if green_channel is not None:
            g = self._resize_to_match(g, r)
        if blue_channel is not None:
            b = self._resize_to_match(b, r)
        
        # Pack into RGB
        packed = torch.cat([r, g, b], dim=-1)
        
        print(f"✓ Channels packed")
        print(f"  R: {r.shape}")
        print(f"  G: {g.shape}")
        print(f"  B: {b.shape}")
        print(f"  Output: {packed.shape}")
        print("="*60 + "\n")
        
        return (packed,)
    
    def _to_grayscale(self, image):
        """Convert to single channel grayscale"""
        if image.shape[-1] == 1:
            return image
        elif image.shape[-1] == 3:
            # Luminance
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            return torch.sum(image * weights, dim=-1, keepdim=True)
        else:
            # Just take first channel
            return image[:, :, :, 0:1]
    
    def _resize_to_match(self, source, target):
        """Resize source to match target dimensions"""
        if source.shape[1:3] == target.shape[1:3]:
            return source
        
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        
        resized = torch.nn.functional.interpolate(
            source_bchw,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized.permute(0, 2, 3, 1)


class ChannelUnpacker:
    """
    Unpack RGB channels into separate grayscale maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "packed": ("IMAGE",),
                "output_channels": (["all", "r_only", "g_only", "b_only", "rg", "rb", "gb"], {
                    "default": "all",
                    "tooltip": "Which channels to extract"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red", "green", "blue")
    FUNCTION = "unpack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def unpack(self, packed, output_channels="all"):
        """Unpack RGB channels into separate images"""
        
        print("\n" + "="*60)
        print("Channel Unpacker")
        print("="*60)
        print(f"Input shape: {packed.shape}")
        print(f"Output channels: {output_channels}")
        
        batch, height, width, channels = packed.shape
        device = packed.device
        dtype = packed.dtype
        
        # Extract channels
        if channels >= 3:
            r = packed[:, :, :, 0:1]
            g = packed[:, :, :, 1:2]
            b = packed[:, :, :, 2:3]
        elif channels == 2:
            r = packed[:, :, :, 0:1]
            g = packed[:, :, :, 1:2]
            b = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        else:
            r = packed
            g = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
            b = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Convert to RGB for display
        r_rgb = r.repeat(1, 1, 1, 3)
        g_rgb = g.repeat(1, 1, 1, 3)
        b_rgb = b.repeat(1, 1, 1, 3)
        
        print(f"✓ Channels unpacked")
        print(f"  Red: {r_rgb.shape}")
        print(f"  Green: {g_rgb.shape}")
        print(f"  Blue: {b_rgb.shape}")
        print("="*60 + "\n")
        
        return (r_rgb, g_rgb, b_rgb)


class GrayscaleToColor:
    """
    Convert grayscale image to RGB by repeating the channel
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "grayscale": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("color",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Channel"
    
    def convert(self, grayscale):
        """Convert grayscale to RGB"""
        
        print("\n" + "="*60)
        print("Grayscale to Color")
        print("="*60)
        print(f"Input shape: {grayscale.shape}")
        
        batch, height, width, channels = grayscale.shape
        
        # If already RGB or more, return as-is
        if channels >= 3:
            print("✓ Already color (RGB+), returning as-is")
            print("="*60 + "\n")
            return (grayscale,)
        
        # Convert to grayscale first if needed
        if channels == 1:
            gray = grayscale
        else:
            # Average multiple channels
            gray = torch.mean(grayscale, dim=-1, keepdim=True)
        
        # Repeat to create RGB
        color = gray.repeat(1, 1, 1, 3)
        
        print(f"✓ Converted to color")
        print(f"  Output shape: {color.shape}")
        print("="*60 + "\n")
        
        return (color,)


class ColorToGrayscale:
    """
    Convert color (RGB) image to grayscale using luminance
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color": ("IMAGE",),
                "method": (["luminance", "average", "lightness", "red_only", "green_only", "blue_only"], {
                    "default": "luminance",
                    "tooltip": "Conversion method: luminance (perceptual), average, lightness (min+max/2), or single channel"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grayscale",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Channel"
    
    def convert(self, color, method="luminance"):
        """Convert color to grayscale"""
        
        print("\n" + "="*60)
        print("Color to Grayscale")
        print("="*60)
        print(f"Input shape: {color.shape}")
        print(f"Method: {method}")
        
        batch, height, width, channels = color.shape
        device = color.device
        dtype = color.dtype
        
        # If already grayscale, return as-is (as RGB for display)
        if channels == 1:
            print("✓ Already grayscale, returning as RGB")
            gray = color.repeat(1, 1, 1, 3)
            print("="*60 + "\n")
            return (gray,)
        
        # Extract RGB channels (handle RGBA by ignoring alpha)
        r = color[:, :, :, 0:1]
        g = color[:, :, :, 1:2] if channels >= 2 else r
        b = color[:, :, :, 2:3] if channels >= 3 else r
        
        # Apply conversion method
        if method == "luminance":
            # Perceptual luminance weights (Rec. 709)
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)
            gray_1ch = r * weights[0] + g * weights[1] + b * weights[2]
        elif method == "average":
            # Simple average
            gray_1ch = (r + g + b) / 3.0
        elif method == "lightness":
            # (max + min) / 2
            rgb_stack = torch.cat([r, g, b], dim=-1)
            gray_1ch = (torch.max(rgb_stack, dim=-1, keepdim=True)[0] + 
                       torch.min(rgb_stack, dim=-1, keepdim=True)[0]) / 2.0
        elif method == "red_only":
            gray_1ch = r
        elif method == "green_only":
            gray_1ch = g
        elif method == "blue_only":
            gray_1ch = b
        else:
            gray_1ch = r * 0.2126 + g * 0.7152 + b * 0.0722
        
        # Convert to RGB for display
        grayscale = gray_1ch.repeat(1, 1, 1, 3)
        
        print(f"✓ Converted to grayscale")
        print(f"  Output shape: {grayscale.shape}")
        print("="*60 + "\n")
        
        return (grayscale,)


class ChannelPackerORMA:
    """
    Pack 4 maps into RGBA: Occlusion (R), Roughness (G), Metallic (B), Alpha (A)
    Advanced ORM with alpha channel support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "occlusion": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
            },
            "optional": {
                "alpha": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("orma_packed",)
    FUNCTION = "pack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def pack(self, occlusion, roughness, metallic, alpha=None):
        """Pack ORMA channels"""
        
        print("\n" + "="*60)
        print("Channel Packer (ORMA)")
        print("="*60)
        
        # Get dimensions from occlusion
        batch, height, width, channels = occlusion.shape
        device = occlusion.device
        dtype = occlusion.dtype
        
        # Convert to grayscale if needed
        o = self._to_grayscale(occlusion)
        r = self._to_grayscale(roughness)
        m = self._to_grayscale(metallic)
        
        # Resize to match if sizes differ
        r = self._resize_to_match(r, o)
        m = self._resize_to_match(m, o)
        
        # Handle alpha channel
        if alpha is not None:
            a = self._to_grayscale(alpha)
            a = self._resize_to_match(a, o)
        else:
            # Default to fully opaque
            a = torch.ones_like(o)
        
        # Pack into RGBA
        packed = torch.cat([o, r, m, a], dim=-1)
        
        print(f"✓ ORMA packed")
        print(f"  Occlusion (R): {o.shape}")
        print(f"  Roughness (G): {r.shape}")
        print(f"  Metallic (B): {m.shape}")
        print(f"  Alpha (A): {a.shape}")
        print(f"  Output (RGBA): {packed.shape}")
        print("="*60 + "\n")
        
        return (packed,)
    
    def _to_grayscale(self, image):
        """Convert to single channel grayscale"""
        if image.shape[-1] == 1:
            return image
        elif image.shape[-1] >= 3:
            # Luminance
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            rgb = image[:, :, :, 0:3]
            return torch.sum(rgb * weights, dim=-1, keepdim=True)
        else:
            # Just take first channel
            return image[:, :, :, 0:1]
    
    def _resize_to_match(self, source, target):
        """Resize source to match target dimensions"""
        if source.shape[1:3] == target.shape[1:3]:
            return source
        
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        
        resized = torch.nn.functional.interpolate(
            source_bchw,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized.permute(0, 2, 3, 1)


class ChannelPackerRMA:
    """
    Pack RMA: Roughness (R), Metallic (G), AO (B)
    Alternative to ORM format
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "ao": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rma_packed",)
    FUNCTION = "pack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def pack(self, roughness, metallic, ao):
        """Pack RMA channels"""
        
        print("\n" + "="*60)
        print("Channel Packer (RMA)")
        print("="*60)
        
        # Get dimensions from roughness
        batch, height, width, channels = roughness.shape
        device = roughness.device
        dtype = roughness.dtype
        
        # Convert to grayscale if needed
        r = self._to_grayscale(roughness)
        m = self._to_grayscale(metallic)
        a = self._to_grayscale(ao)
        
        # Resize to match if sizes differ
        m = self._resize_to_match(m, r)
        a = self._resize_to_match(a, r)
        
        # Pack into RGB
        packed = torch.cat([r, m, a], dim=-1)
        
        print(f"✓ RMA packed")
        print(f"  Roughness (R): {r.shape}")
        print(f"  Metallic (G): {m.shape}")
        print(f"  AO (B): {a.shape}")
        print(f"  Output (RGB): {packed.shape}")
        print("="*60 + "\n")
        
        return (packed,)
    
    def _to_grayscale(self, image):
        """Convert to single channel grayscale"""
        if image.shape[-1] == 1:
            return image
        elif image.shape[-1] >= 3:
            # Luminance
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            rgb = image[:, :, :, 0:3]
            return torch.sum(rgb * weights, dim=-1, keepdim=True)
        else:
            # Just take first channel
            return image[:, :, :, 0:1]
    
    def _resize_to_match(self, source, target):
        """Resize source to match target dimensions"""
        if source.shape[1:3] == target.shape[1:3]:
            return source
        
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        
        resized = torch.nn.functional.interpolate(
            source_bchw,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized.permute(0, 2, 3, 1)


class ChannelPackerRMAA:
    """
    Pack RMAA: Roughness (R), Metallic (G), AO (B), Alpha (A)
    RMA format with alpha channel support
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "ao": ("IMAGE",),
            },
            "optional": {
                "alpha": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rmaa_packed",)
    FUNCTION = "pack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def pack(self, roughness, metallic, ao, alpha=None):
        """Pack RMAA channels"""
        
        print("\n" + "="*60)
        print("Channel Packer (RMAA)")
        print("="*60)
        
        # Get dimensions from roughness
        batch, height, width, channels = roughness.shape
        device = roughness.device
        dtype = roughness.dtype
        
        # Convert to grayscale if needed
        r = self._to_grayscale(roughness)
        m = self._to_grayscale(metallic)
        a = self._to_grayscale(ao)
        
        # Resize to match if sizes differ
        m = self._resize_to_match(m, r)
        a = self._resize_to_match(a, r)
        
        # Handle alpha channel
        if alpha is not None:
            alpha_ch = self._to_grayscale(alpha)
            alpha_ch = self._resize_to_match(alpha_ch, r)
        else:
            # Default to fully opaque
            alpha_ch = torch.ones_like(r)
        
        # Pack into RGBA
        packed = torch.cat([r, m, a, alpha_ch], dim=-1)
        
        print(f"✓ RMAA packed")
        print(f"  Roughness (R): {r.shape}")
        print(f"  Metallic (G): {m.shape}")
        print(f"  AO (B): {a.shape}")
        print(f"  Alpha (A): {alpha_ch.shape}")
        print(f"  Output (RGBA): {packed.shape}")
        print("="*60 + "\n")
        
        return (packed,)
    
    def _to_grayscale(self, image):
        """Convert to single channel grayscale"""
        if image.shape[-1] == 1:
            return image
        elif image.shape[-1] >= 3:
            # Luminance
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=image.device, dtype=image.dtype)
            rgb = image[:, :, :, 0:3]
            return torch.sum(rgb * weights, dim=-1, keepdim=True)
        else:
            # Just take first channel
            return image[:, :, :, 0:1]
    
    def _resize_to_match(self, source, target):
        """Resize source to match target dimensions"""
        if source.shape[1:3] == target.shape[1:3]:
            return source
        
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        
        resized = torch.nn.functional.interpolate(
            source_bchw,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False
        )
        
        return resized.permute(0, 2, 3, 1)


class ChannelUnpackerRMA:
    """
    Unpack RMA: Extract Roughness, Metallic, AO from packed RGB
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rma_packed": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("roughness", "metallic", "ao")
    FUNCTION = "unpack"
    CATEGORY = "Texture Alchemist/Channel"
    
    def unpack(self, rma_packed):
        """Unpack RMA channels"""
        
        print("\n" + "="*60)
        print("Channel Unpacker (RMA)")
        print("="*60)
        print(f"Input shape: {rma_packed.shape}")
        
        batch, height, width, channels = rma_packed.shape
        device = rma_packed.device
        dtype = rma_packed.dtype
        
        # Extract channels
        if channels >= 3:
            roughness = rma_packed[:, :, :, 0:1]
            metallic = rma_packed[:, :, :, 1:2]
            ao = rma_packed[:, :, :, 2:3]
        elif channels == 2:
            roughness = rma_packed[:, :, :, 0:1]
            metallic = rma_packed[:, :, :, 1:2]
            ao = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        else:
            roughness = rma_packed
            metallic = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
            ao = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Convert to RGB for display
        roughness_rgb = roughness.repeat(1, 1, 1, 3)
        metallic_rgb = metallic.repeat(1, 1, 1, 3)
        ao_rgb = ao.repeat(1, 1, 1, 3)
        
        print(f"✓ RMA unpacked")
        print(f"  Roughness: {roughness_rgb.shape}")
        print(f"  Metallic: {metallic_rgb.shape}")
        print(f"  AO: {ao_rgb.shape}")
        print("="*60 + "\n")
        
        return (roughness_rgb, metallic_rgb, ao_rgb)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ChannelPacker": ChannelPacker,
    "ChannelUnpacker": ChannelUnpacker,
    "GrayscaleToColor": GrayscaleToColor,
    "ColorToGrayscale": ColorToGrayscale,
    "ChannelPackerORMA": ChannelPackerORMA,
    "ChannelPackerRMA": ChannelPackerRMA,
    "ChannelPackerRMAA": ChannelPackerRMAA,
    "ChannelUnpackerRMA": ChannelUnpackerRMA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChannelPacker": "Channel Packer (RGB)",
    "ChannelUnpacker": "Channel Unpacker (RGB)",
    "GrayscaleToColor": "Grayscale to Color",
    "ColorToGrayscale": "Color to Grayscale",
    "ChannelPackerORMA": "Channel Packer (ORMA)",
    "ChannelPackerRMA": "Channel Packer (RMA)",
    "ChannelPackerRMAA": "Channel Packer (RMAA)",
    "ChannelUnpackerRMA": "Channel Unpacker (RMA)",
}

