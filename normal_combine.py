"""
Normal Map Combiner Node
Properly combines multiple normal maps using mathematically correct blending
"""

import torch


class NormalMapCombiner:
    """
    Combine two normal maps using proper blending techniques
    
    Supports multiple blending modes:
    - Reoriented Normal Mapping (RNM): Most accurate, preserves detail
    - Whiteout/Overlay: Fast and good quality, industry standard
    - Linear: Simple blend (not recommended, but included for comparison)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_normal": ("IMAGE",),
                "detail_normal": ("IMAGE",),
                "blend_mode": (["reoriented", "whiteout", "linear"], {
                    "default": "reoriented",
                    "tooltip": "Blending algorithm: Reoriented (best), Whiteout (fast), Linear (simple)"
                }),
                "detail_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Strength of the detail normal map"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_normal",)
    FUNCTION = "combine"
    CATEGORY = "PBR"
    
    def normalize_normal(self, normal):
        """Normalize normal map vectors to unit length"""
        # Normal maps store XYZ in RGB channels
        # Make sure they're unit vectors
        length = torch.sqrt(torch.sum(normal * normal, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-6)  # Avoid division by zero
        return normal / length
    
    def blend_linear(self, base, detail, strength):
        """Simple linear blend (not mathematically correct but simple)"""
        # Interpolate between base and detail
        blended = base * (1.0 - strength) + detail * strength
        return self.normalize_normal(blended)
    
    def blend_whiteout(self, base, detail, strength):
        """
        Whiteout/Overlay blending (industry standard)
        Fast and produces good results
        Also called "UDN blending" or "Unreal blending"
        """
        # Adjust detail strength
        # Blend detail toward neutral (0.5, 0.5, 1.0) based on strength
        neutral = torch.tensor([0.5, 0.5, 1.0], device=detail.device, dtype=detail.dtype)
        neutral = neutral.view(1, 1, 1, 3)
        detail_adjusted = detail * strength + neutral * (1.0 - strength)
        
        # Convert from [0,1] to [-1,1] range
        base_unpacked = base * 2.0 - 1.0
        detail_unpacked = detail_adjusted * 2.0 - 1.0
        
        # Whiteout blend formula
        # n1.xy + n2.xy, n1.z * n2.z
        combined = torch.zeros_like(base_unpacked)
        combined[..., 0:2] = base_unpacked[..., 0:2] + detail_unpacked[..., 0:2]
        combined[..., 2:3] = base_unpacked[..., 2:3] * detail_unpacked[..., 2:3]
        
        # Normalize
        combined = self.normalize_normal(combined)
        
        # Convert back to [0,1] range
        return combined * 0.5 + 0.5
    
    def blend_reoriented(self, base, detail, strength):
        """
        Reoriented Normal Mapping (RNM)
        Most accurate method, preserves detail and handles steep angles well
        Reference: http://blog.selfshadow.com/publications/blending-in-detail/
        """
        # Adjust detail strength
        neutral = torch.tensor([0.5, 0.5, 1.0], device=detail.device, dtype=detail.dtype)
        neutral = neutral.view(1, 1, 1, 3)
        detail_adjusted = detail * strength + neutral * (1.0 - strength)
        
        # Convert from [0,1] to [-1,1] range
        base_unpacked = base * 2.0 - 1.0
        detail_unpacked = detail_adjusted * 2.0 - 1.0
        
        # Reoriented Normal Mapping formula
        # t = n1.xy * n2.z + n2.xy
        # result = normalize(t.x, t.y, n1.z)
        t = base_unpacked[..., 0:2] * detail_unpacked[..., 2:3] + detail_unpacked[..., 0:2]
        
        combined = torch.zeros_like(base_unpacked)
        combined[..., 0:2] = t
        combined[..., 2:3] = base_unpacked[..., 2:3]
        
        # Normalize
        combined = self.normalize_normal(combined)
        
        # Convert back to [0,1] range
        return combined * 0.5 + 0.5
    
    def combine(self, base_normal, detail_normal, blend_mode, detail_strength):
        """Combine two normal maps"""
        
        print("\n" + "="*60)
        print("Normal Map Combiner")
        print("="*60)
        print(f"Base shape: {base_normal.shape}")
        print(f"Detail shape: {detail_normal.shape}")
        print(f"Blend mode: {blend_mode}")
        print(f"Detail strength: {detail_strength}")
        
        # Ensure both normals have 3 channels
        if base_normal.shape[-1] != 3:
            print(f"\n✗ ERROR: Base normal must have 3 channels (RGB), got {base_normal.shape[-1]}")
            return (base_normal,)
        
        if detail_normal.shape[-1] != 3:
            print(f"\n✗ ERROR: Detail normal must have 3 channels (RGB), got {detail_normal.shape[-1]}")
            return (base_normal,)
        
        # Match batch sizes if needed
        if base_normal.shape[0] != detail_normal.shape[0]:
            # Use first batch from whichever has multiple
            if base_normal.shape[0] > 1:
                base_normal = base_normal[0:1]
            if detail_normal.shape[0] > 1:
                detail_normal = detail_normal[0:1]
        
        # Match spatial dimensions if needed (simple nearest neighbor resize)
        if base_normal.shape[1:3] != detail_normal.shape[1:3]:
            print(f"\n⚠ Warning: Resizing detail normal to match base")
            # Use PyTorch interpolate
            detail_normal = torch.nn.functional.interpolate(
                detail_normal.permute(0, 3, 1, 2),  # BHWC -> BCHW
                size=base_normal.shape[1:3],
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        # Apply blending
        if blend_mode == "linear":
            combined = self.blend_linear(base_normal, detail_normal, detail_strength)
            print("\n✓ Combined using Linear blend")
        elif blend_mode == "whiteout":
            combined = self.blend_whiteout(base_normal, detail_normal, detail_strength)
            print("\n✓ Combined using Whiteout/Overlay blend")
        elif blend_mode == "reoriented":
            combined = self.blend_reoriented(base_normal, detail_normal, detail_strength)
            print("\n✓ Combined using Reoriented Normal Mapping (RNM)")
        else:
            combined = base_normal
            print(f"\n✗ Unknown blend mode: {blend_mode}")
        
        print(f"  Output range: [{combined.min():.3f}, {combined.max():.3f}]")
        print("\n" + "="*60)
        print("✓ Normal Combination Complete")
        print("="*60 + "\n")
        
        return (combined,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NormalMapCombiner": NormalMapCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapCombiner": "Normal Map Combiner",
}

