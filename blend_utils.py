"""
Blending and Compositing Utilities for TextureAlchemy
Advanced image blending with multiple modes and layer support
"""

import torch
import torch.nn.functional as F


class BlendModeUtility:
    """
    Apply Photoshop-style blend modes between two images
    Supports all major blend modes for texture compositing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE",),
                "blend": ("IMAGE",),
                "mode": ([
                    "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
                    "color_dodge", "color_burn", "linear_dodge", "linear_burn",
                    "vivid_light", "linear_light", "pin_light", "hard_mix",
                    "difference", "exclusion", "subtract", "divide",
                    "hue", "saturation", "color", "luminosity",
                    "lighter", "darker", "lighten", "darken"
                ],),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Layer opacity (0.0 = fully transparent, 1.0 = fully opaque)"
                }),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "apply_blend"
    CATEGORY = "Texture Alchemist/Blending"
    
    def apply_blend(self, base, blend, mode, opacity, mask=None):
        """Apply blend mode between two images"""
        
        print(f"\n{'='*60}")
        print(f"Blend Mode Utility - {mode.upper()}")
        print(f"{'='*60}")
        print(f"Base: {base.shape}")
        print(f"Blend: {blend.shape}")
        
        # Match channel count first
        base_channels = base.shape[3]
        blend_channels = blend.shape[3]
        
        if base_channels != blend_channels:
            print(f"  Channel mismatch: base={base_channels}, blend={blend_channels}")
            if base_channels == 3 and blend_channels == 4:
                # Base is RGB, blend is RGBA - drop alpha from blend
                blend = blend[:, :, :, :3]
                print(f"  Dropped alpha channel from blend layer")
            elif base_channels == 4 and blend_channels == 3:
                # Base is RGBA, blend is RGB - add alpha to blend
                alpha_channel = torch.ones((blend.shape[0], blend.shape[1], blend.shape[2], 1), 
                                          device=blend.device, dtype=blend.dtype)
                blend = torch.cat([blend, alpha_channel], dim=-1)
                print(f"  Added alpha channel to blend layer")
            else:
                print(f"  Warning: Unhandled channel mismatch")
        
        # Ensure same dimensions
        if base.shape[1:3] != blend.shape[1:3]:
            print(f"  Resizing blend: {blend.shape[2]}×{blend.shape[1]} → {base.shape[2]}×{base.shape[1]}")
            blend = F.interpolate(
                blend.permute(0, 3, 1, 2),
                size=(base.shape[1], base.shape[2]),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Apply blend mode
        result = self._blend(base, blend, mode)
        
        # Apply opacity
        if opacity < 1.0:
            result = base * (1.0 - opacity) + result * opacity
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != base.shape:
                mask = F.interpolate(
                    mask.permute(0, 3, 1, 2),
                    size=(base.shape[1], base.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Convert to grayscale if needed
            if mask.shape[3] > 1:
                mask = mask.mean(dim=3, keepdim=True)
            
            result = base * (1.0 - mask) + result * mask
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Blend applied: {mode}")
        print(f"  Opacity: {opacity:.2f}")
        print(f"  Mask: {'Yes' if mask is not None else 'No'}")
        print(f"{'='*60}\n")
        
        return (result,)
    
    def _blend(self, base, blend, mode):
        """Apply specific blend mode"""
        
        # Basic blend modes
        if mode == "normal":
            return blend
        
        elif mode == "multiply":
            return base * blend
        
        elif mode == "screen":
            return 1.0 - (1.0 - base) * (1.0 - blend)
        
        elif mode == "overlay":
            return torch.where(
                base < 0.5,
                2.0 * base * blend,
                1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
            )
        
        elif mode == "soft_light":
            return torch.where(
                blend < 0.5,
                2.0 * base * blend + base * base * (1.0 - 2.0 * blend),
                2.0 * base * (1.0 - blend) + torch.sqrt(base) * (2.0 * blend - 1.0)
            )
        
        elif mode == "hard_light":
            return torch.where(
                blend < 0.5,
                2.0 * base * blend,
                1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
            )
        
        elif mode == "color_dodge":
            return torch.clamp(base / (1.0 - blend + 1e-7), 0.0, 1.0)
        
        elif mode == "color_burn":
            return torch.clamp(1.0 - (1.0 - base) / (blend + 1e-7), 0.0, 1.0)
        
        elif mode == "linear_dodge":
            return torch.clamp(base + blend, 0.0, 1.0)
        
        elif mode == "linear_burn":
            return torch.clamp(base + blend - 1.0, 0.0, 1.0)
        
        elif mode == "vivid_light":
            return torch.where(
                blend < 0.5,
                torch.clamp(1.0 - (1.0 - base) / (2.0 * blend + 1e-7), 0.0, 1.0),
                torch.clamp(base / (2.0 * (1.0 - blend) + 1e-7), 0.0, 1.0)
            )
        
        elif mode == "linear_light":
            return torch.clamp(base + 2.0 * blend - 1.0, 0.0, 1.0)
        
        elif mode == "pin_light":
            return torch.where(
                blend < 0.5,
                torch.min(base, 2.0 * blend),
                torch.max(base, 2.0 * (blend - 0.5))
            )
        
        elif mode == "hard_mix":
            vl = self._blend(base, blend, "vivid_light")
            return torch.where(vl < 0.5, torch.zeros_like(vl), torch.ones_like(vl))
        
        elif mode == "difference":
            return torch.abs(base - blend)
        
        elif mode == "exclusion":
            return base + blend - 2.0 * base * blend
        
        elif mode == "subtract":
            return torch.clamp(base - blend, 0.0, 1.0)
        
        elif mode == "divide":
            return torch.clamp(base / (blend + 1e-7), 0.0, 1.0)
        
        elif mode == "lighter":
            return torch.max(base, blend)
        
        elif mode == "darker":
            return torch.min(base, blend)
        
        elif mode == "lighten":
            return torch.max(base, blend)
        
        elif mode == "darken":
            return torch.min(base, blend)
        
        # HSL blend modes
        elif mode in ["hue", "saturation", "color", "luminosity"]:
            return self._blend_hsl(base, blend, mode)
        
        return blend
    
    def _blend_hsl(self, base, blend, mode):
        """HSL-based blend modes"""
        base_hsv = self._rgb_to_hsv(base)
        blend_hsv = self._rgb_to_hsv(blend)
        
        if mode == "hue":
            result_hsv = torch.cat([
                blend_hsv[:, :, :, 0:1],
                base_hsv[:, :, :, 1:2],
                base_hsv[:, :, :, 2:3]
            ], dim=3)
        elif mode == "saturation":
            result_hsv = torch.cat([
                base_hsv[:, :, :, 0:1],
                blend_hsv[:, :, :, 1:2],
                base_hsv[:, :, :, 2:3]
            ], dim=3)
        elif mode == "color":
            result_hsv = torch.cat([
                blend_hsv[:, :, :, 0:1],
                blend_hsv[:, :, :, 1:2],
                base_hsv[:, :, :, 2:3]
            ], dim=3)
        else:  # luminosity
            result_hsv = torch.cat([
                base_hsv[:, :, :, 0:1],
                base_hsv[:, :, :, 1:2],
                blend_hsv[:, :, :, 2:3]
            ], dim=3)
        
        return self._hsv_to_rgb(result_hsv)
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_c = torch.max(torch.max(r, g), b)
        min_c = torch.min(torch.min(r, g), b)
        diff = max_c - min_c
        
        # Hue
        h = torch.zeros_like(max_c)
        mask = diff != 0
        
        h = torch.where(
            (max_c == r) & mask,
            ((g - b) / (diff + 1e-7)) % 6.0,
            h
        )
        h = torch.where(
            (max_c == g) & mask,
            ((b - r) / (diff + 1e-7)) + 2.0,
            h
        )
        h = torch.where(
            (max_c == b) & mask,
            ((r - g) / (diff + 1e-7)) + 4.0,
            h
        )
        h = h / 6.0
        
        # Saturation
        s = torch.where(max_c != 0, diff / (max_c + 1e-7), torch.zeros_like(max_c))
        
        # Value
        v = max_c
        
        return torch.stack([h, s, v], dim=3)
    
    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        h, s, v = hsv[:, :, :, 0] * 6.0, hsv[:, :, :, 1], hsv[:, :, :, 2]
        
        c = v * s
        x = c * (1.0 - torch.abs(h % 2.0 - 1.0))
        m = v - c
        
        h_floor = torch.floor(h) % 6
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        mask0 = h_floor == 0
        mask1 = h_floor == 1
        mask2 = h_floor == 2
        mask3 = h_floor == 3
        mask4 = h_floor == 4
        mask5 = h_floor == 5
        
        r = torch.where(mask0, c, r)
        r = torch.where(mask1, x, r)
        r = torch.where(mask2 | mask3, torch.zeros_like(r), r)
        r = torch.where(mask4, x, r)
        r = torch.where(mask5, c, r)
        
        g = torch.where(mask0, x, g)
        g = torch.where(mask1, c, g)
        g = torch.where(mask2, c, g)
        g = torch.where(mask3, x, g)
        g = torch.where(mask4 | mask5, torch.zeros_like(g), g)
        
        b = torch.where(mask0 | mask1, torch.zeros_like(b), b)
        b = torch.where(mask2, x, b)
        b = torch.where(mask3, c, b)
        b = torch.where(mask4, c, b)
        b = torch.where(mask5, x, b)
        
        r = r + m
        g = g + m
        b = b + m
        
        return torch.stack([r, g, b], dim=3)


class MultiTextureBlender:
    """
    Blend up to 4 textures with individual blend modes, opacity, and masks
    Professional layer-based compositing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        blend_modes = [
            "normal", "multiply", "screen", "overlay", "soft_light", "hard_light",
            "color_dodge", "color_burn", "linear_dodge", "linear_burn",
            "difference", "subtract", "lighter", "darker"
        ]
        
        return {
            "required": {
                "base": ("IMAGE",),
            },
            "optional": {
                "layer1": ("IMAGE",),
                "layer1_mode": (blend_modes, {"default": "normal"}),
                "layer1_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "layer1_mask": ("IMAGE",),
                
                "layer2": ("IMAGE",),
                "layer2_mode": (blend_modes, {"default": "multiply"}),
                "layer2_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "layer2_mask": ("IMAGE",),
                
                "layer3": ("IMAGE",),
                "layer3_mode": (blend_modes, {"default": "overlay"}),
                "layer3_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "layer3_mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "blend_multi"
    CATEGORY = "Texture Alchemist/Blending"
    
    def blend_multi(self, base, **kwargs):
        """Blend multiple layers"""
        
        print(f"\n{'='*60}")
        print(f"Multi-Texture Blender")
        print(f"{'='*60}")
        
        result = base.clone()
        blender = BlendModeUtility()
        
        layers_applied = 0
        
        for i in range(1, 4):
            layer_key = f"layer{i}"
            mode_key = f"layer{i}_mode"
            opacity_key = f"layer{i}_opacity"
            mask_key = f"layer{i}_mask"
            
            if layer_key in kwargs and kwargs[layer_key] is not None:
                layer = kwargs[layer_key]
                mode = kwargs.get(mode_key, "normal")
                opacity = kwargs.get(opacity_key, 1.0)
                mask = kwargs.get(mask_key, None)
                
                print(f"\n  Layer {i}:")
                print(f"    Mode: {mode}")
                print(f"    Opacity: {opacity:.2f}")
                print(f"    Masked: {'Yes' if mask is not None else 'No'}")
                
                result = blender.apply_blend(result, layer, mode, opacity, mask)[0]
                layers_applied += 1
        
        print(f"\n✓ Blended {layers_applied} layers")
        print(f"{'='*60}\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "BlendModeUtility": BlendModeUtility,
    "MultiTextureBlender": MultiTextureBlender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlendModeUtility": "Blend Mode Utility",
    "MultiTextureBlender": "Multi-Texture Blender",
}

