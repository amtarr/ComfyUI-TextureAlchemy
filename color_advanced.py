"""
Advanced Color Tools for TextureAlchemy
Levels, color matching, auto contrast, temperature controls
"""

import torch
import torch.nn.functional as F


class LevelsAdjustment:
    """
    Adjust input/output levels per channel (like Photoshop Levels)
    Essential color grading tool
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "input_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "input_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01, "display": "number"}),
                "output_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "output_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),
                "mode": (["rgb", "red", "green", "blue", "luminosity"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist/Color"
    
    def adjust(self, image, input_black, input_white, input_gamma, output_black, output_white, mode):
        """Apply levels adjustment"""
        
        print(f"\n{'='*60}")
        print(f"Levels Adjustment - {mode.upper()}")
        print(f"{'='*60}")
        
        if mode == "rgb":
            result = self._apply_levels(image, input_black, input_white, input_gamma, output_black, output_white)
        
        elif mode in ["red", "green", "blue"]:
            result = image.clone()
            channel_idx = {"red": 0, "green": 1, "blue": 2}[mode]
            channel = result[:, :, :, channel_idx:channel_idx+1]
            adjusted = self._apply_levels(channel, input_black, input_white, input_gamma, output_black, output_white)
            result[:, :, :, channel_idx:channel_idx+1] = adjusted
        
        else:  # luminosity
            lum = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
            lum_adjusted = self._apply_levels(lum, input_black, input_white, input_gamma, output_black, output_white)
            scale = lum_adjusted / (lum + 1e-7)
            result = image * scale
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Levels adjusted")
        print(f"  Input: [{input_black:.2f}, {input_white:.2f}] γ={input_gamma:.2f}")
        print(f"  Output: [{output_black:.2f}, {output_white:.2f}]")
        print(f"{'='*60}\n")
        
        return (result,)
    
    def _apply_levels(self, image, in_black, in_white, gamma, out_black, out_white):
        """Apply levels transformation"""
        # Input levels
        result = (image - in_black) / (in_white - in_black + 1e-7)
        result = torch.clamp(result, 0.0, 1.0)
        
        # Gamma
        result = torch.pow(result, 1.0 / gamma)
        
        # Output levels
        result = result * (out_white - out_black) + out_black
        
        return result


class AutoContrastLevels:
    """
    Automatic histogram normalization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "clip_percent": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Percent of histogram to clip at extremes"
                }),
                "mode": (["luminosity", "per_channel"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "auto_contrast"
    CATEGORY = "Texture Alchemist/Color"
    
    def auto_contrast(self, image, clip_percent, mode):
        """Auto contrast"""
        
        print(f"\n{'='*60}")
        print(f"Auto Contrast - {mode.upper()}")
        print(f"{'='*60}")
        
        if mode == "per_channel":
            result = image.clone()
            for c in range(min(3, image.shape[3])):
                channel = image[:, :, :, c]
                
                # Calculate percentiles
                sorted_vals = torch.sort(channel.flatten())[0]
                n = sorted_vals.shape[0]
                
                clip_low_idx = int(n * clip_percent / 100.0)
                clip_high_idx = int(n * (1.0 - clip_percent / 100.0))
                
                vmin = sorted_vals[clip_low_idx]
                vmax = sorted_vals[clip_high_idx]
                
                # Normalize
                channel_norm = (channel - vmin) / (vmax - vmin + 1e-7)
                result[:, :, :, c] = torch.clamp(channel_norm, 0.0, 1.0)
        
        else:  # luminosity
            lum = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
            
            sorted_vals = torch.sort(lum.flatten())[0]
            n = sorted_vals.shape[0]
            
            clip_low_idx = int(n * clip_percent / 100.0)
            clip_high_idx = int(n * (1.0 - clip_percent / 100.0))
            
            vmin = sorted_vals[clip_low_idx]
            vmax = sorted_vals[clip_high_idx]
            
            # Scale image
            result = (image - vmin) / (vmax - vmin + 1e-7)
            result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Auto contrast applied")
        print(f"  Clip: {clip_percent:.1f}%")
        print(f"{'='*60}\n")
        
        return (result,)


class TemperatureTint:
    """
    Color temperature and tint adjustment
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Cool (-) to Warm (+)"
                }),
                "tint": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Green (-) to Magenta (+)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist/Color"
    
    def adjust(self, image, temperature, tint, strength):
        """Apply temperature and tint"""
        
        print(f"\n{'='*60}")
        print(f"Temperature/Tint Adjustment")
        print(f"{'='*60}")
        
        result = image.clone()
        
        # Temperature (blue-yellow axis)
        if temperature != 0.0:
            temp_shift = temperature * strength
            result[:, :, :, 0] += temp_shift * 0.5  # Red
            result[:, :, :, 2] -= temp_shift * 0.5  # Blue
        
        # Tint (green-magenta axis)
        if tint != 0.0:
            tint_shift = tint * strength
            result[:, :, :, 1] -= tint_shift * 0.5  # Green
            result[:, :, :, 0] += tint_shift * 0.25  # Red
            result[:, :, :, 2] += tint_shift * 0.25  # Blue
        
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Color adjusted")
        print(f"  Temperature: {temperature:+.2f}")
        print(f"  Tint: {tint:+.2f}")
        print(f"{'='*60}\n")
        
        return (result,)


class ColorMatchTransfer:
    """
    Match colors from reference image
    Histogram matching technique
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "mode": (["mean_std", "histogram"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("result",)
    FUNCTION = "match_colors"
    CATEGORY = "Texture Alchemist/Color"
    
    def match_colors(self, image, reference, strength, mode):
        """Match colors to reference"""
        
        print(f"\n{'='*60}")
        print(f"Color Match Transfer - {mode.upper()}")
        print(f"{'='*60}")
        
        if mode == "mean_std":
            result = self._match_mean_std(image, reference)
        else:
            result = self._match_histogram(image, reference)
        
        # Blend with original
        result = image * (1.0 - strength) + result * strength
        result = torch.clamp(result, 0.0, 1.0)
        
        print(f"✓ Colors matched")
        print(f"  Strength: {strength:.2f}")
        print(f"{'='*60}\n")
        
        return (result,)
    
    def _match_mean_std(self, image, reference):
        """Match mean and standard deviation"""
        result = image.clone()
        
        for c in range(min(3, image.shape[3])):
            img_channel = image[:, :, :, c]
            ref_channel = reference[:, :, :, c]
            
            img_mean = img_channel.mean()
            img_std = img_channel.std()
            ref_mean = ref_channel.mean()
            ref_std = ref_channel.std()
            
            # Normalize and rescale
            normalized = (img_channel - img_mean) / (img_std + 1e-7)
            matched = normalized * ref_std + ref_mean
            
            result[:, :, :, c] = matched
        
        return result
    
    def _match_histogram(self, image, reference):
        """Histogram matching (simplified)"""
        result = image.clone()
        
        for c in range(min(3, image.shape[3])):
            img_channel = image[:, :, :, c].flatten()
            ref_channel = reference[:, :, :, c].flatten()
            
            # Sort both
            img_sorted, img_indices = torch.sort(img_channel)
            ref_sorted = torch.sort(ref_channel)[0]
            
            # Match histograms
            matched = torch.zeros_like(img_sorted)
            n_img = img_sorted.shape[0]
            n_ref = ref_sorted.shape[0]
            
            for i in range(n_img):
                ref_idx = int((i / n_img) * n_ref)
                matched[i] = ref_sorted[min(ref_idx, n_ref-1)]
            
            # Unsort
            result[:, :, :, c] = matched[torch.argsort(img_indices)].view(image.shape[0], image.shape[1], image.shape[2])
        
        return result


# Node registration
NODE_CLASS_MAPPINGS = {
    "LevelsAdjustment": LevelsAdjustment,
    "AutoContrastLevels": AutoContrastLevels,
    "TemperatureTint": TemperatureTint,
    "ColorMatchTransfer": ColorMatchTransfer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LevelsAdjustment": "Levels Adjustment",
    "AutoContrastLevels": "Auto Contrast/Levels",
    "TemperatureTint": "Temperature & Tint",
    "ColorMatchTransfer": "Color Match/Transfer",
}

