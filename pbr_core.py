"""
PBR Core Nodes
Extraction and basic adjustment of PBR maps from Marigold outputs
"""

import torch


class PBRExtractor:
    """
    Extracts PBR maps from Marigold IID Appearance and Lighting outputs
    
    Inputs:
    - Marigold Appearance output (3 batches: 0=albedo, 1=rough/metal, 2=unused)
    - Marigold Lighting output (3 batches: 0=albedo, 1=AO/Lighting, 2=unused)
    
    Outputs: PBR_PIPE, Albedo, Roughness, Metallic, Lighting
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Marigold outputs
                "marigold_appearance": ("IMAGE",),
                "marigold_lighting": ("IMAGE",),
                
                # Albedo choice
                "albedo_source": (["appearance", "lighting"], {
                    "default": "lighting",
                    "tooltip": "Which Marigold output to use for albedo"
                }),
                
                # Gamma correction - separate controls
                "gamma_albedo": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for albedo (applied to both appearance and lighting sources)"
                }),
                "gamma_metal_rough": ("FLOAT", {
                    "default": 2.2,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for metallic and roughness maps"
                }),
                "gamma_lighting_ao": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for lighting and AO maps"
                }),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "extract"
    CATEGORY = "Texture Alchemist"
    
    def apply_gamma(self, image, gamma):
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def channel_to_rgb(self, channel):
        """Convert single channel to RGB"""
        if channel.shape[-1] == 1:
            return channel.repeat(1, 1, 1, 3)
        return channel
    
    def grayscale(self, image):
        """Convert to grayscale"""
        if image.shape[-1] == 1:
            return image
        weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                               device=image.device, dtype=image.dtype)
        gray = torch.sum(image * weights, dim=-1, keepdim=True)
        return self.channel_to_rgb(gray)
    
    def extract(self, marigold_appearance, marigold_lighting, albedo_source, gamma_albedo, gamma_metal_rough, gamma_lighting_ao):
        """Extract PBR maps"""
        
        print("\n" + "="*60)
        print("PBR Extractor")
        print("="*60)
        print(f"Appearance shape: {marigold_appearance.shape}")
        print(f"Lighting shape: {marigold_lighting.shape}")
        print(f"Albedo source: {albedo_source}")
        print(f"Gamma - Albedo: {gamma_albedo}, Metal/Rough: {gamma_metal_rough}, Lighting/AO: {gamma_lighting_ao}")
        
        # ===== ALBEDO =====
        if albedo_source == "appearance":
            # Get index 0 from appearance
            albedo = marigold_appearance[0:1] if marigold_appearance.shape[0] > 0 else marigold_appearance
            print("\n✓ Albedo from Appearance (index 0)")
        else:
            # Get index 0 from lighting
            albedo = marigold_lighting[0:1] if marigold_lighting.shape[0] > 0 else marigold_lighting
            print("\n✓ Albedo from Lighting (index 0)")
        
        albedo = self.apply_gamma(albedo, gamma_albedo)
        print(f"  Gamma: {gamma_albedo}")
        print(f"  Range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        
        # ===== ROUGHNESS (Red channel from Appearance index 1) =====
        if marigold_appearance.shape[0] >= 2:
            material_batch = marigold_appearance[1:2]
            roughness = material_batch[:, :, :, 0:1]  # Red channel
            roughness = self.channel_to_rgb(roughness)
            roughness = self.apply_gamma(roughness, gamma_metal_rough)
            print("\n✓ Roughness from Appearance index 1, RED channel")
            print(f"  Gamma: {gamma_metal_rough}")
            print(f"  Range: [{roughness.min():.3f}, {roughness.max():.3f}]")
        else:
            print("\n✗ WARNING: Appearance doesn't have index 1, using zeros")
            roughness = torch.zeros_like(albedo)
        
        # ===== METALLIC (Green channel from Appearance index 1) =====
        if marigold_appearance.shape[0] >= 2:
            material_batch = marigold_appearance[1:2]
            metallic = material_batch[:, :, :, 1:2]  # Green channel
            metallic = self.channel_to_rgb(metallic)
            metallic = self.apply_gamma(metallic, gamma_metal_rough)
            print("\n✓ Metallic from Appearance index 1, GREEN channel")
            print(f"  Gamma: {gamma_metal_rough}")
            print(f"  Range: [{metallic.min():.3f}, {metallic.max():.3f}]")
        else:
            print("\n✗ WARNING: Appearance doesn't have index 1, using zeros")
            metallic = torch.zeros_like(albedo)
        
        # ===== LIGHTING / AO (Lighting index 1) =====
        if marigold_lighting.shape[0] >= 2:
            lighting = marigold_lighting[1:2]
            lighting = self.grayscale(lighting)
            lighting = self.apply_gamma(lighting, gamma_lighting_ao)
            print("\n✓ Lighting/AO from Lighting index 1")
            print(f"  Gamma: {gamma_lighting_ao}")
            print(f"  Range: [{lighting.min():.3f}, {lighting.max():.3f}]")
        else:
            print("\n✗ WARNING: Lighting doesn't have index 1, using albedo luminance")
            lighting = self.grayscale(albedo)
        
        # ===== CREATE PBR PIPE =====
        pbr_pipe = {
            "albedo": albedo,
            "normal": None,
            "ao": lighting,  # Store as 'ao' in pipe for compatibility
            "height": None,
            "roughness": roughness,
            "metallic": metallic,
            "transparency": None,
        }
        
        print("\n" + "="*60)
        print("✓ Extraction Complete")
        print("✓ PBR Pipeline created")
        print("="*60 + "\n")
        
        return (pbr_pipe,)


class PBRAdjuster:
    """
    Adjust brightness, contrast, and invert for each PBR map
    
    Takes: Albedo, AO, Roughness, Metallic
    Outputs: Adjusted versions
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Inputs
                "albedo": ("IMAGE",),
                "ao": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                
                # Albedo adjustments
                "albedo_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "albedo_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "albedo_invert": ("BOOLEAN", {"default": False}),
                
                # AO adjustments
                "ao_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ao_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "ao_invert": ("BOOLEAN", {"default": False}),
                
                # Roughness adjustments
                "roughness_brightness": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "roughness_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "roughness_invert": ("BOOLEAN", {"default": False}),
                
                # Metallic adjustments
                "metallic_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "metallic_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "metallic_invert": ("BOOLEAN", {"default": False}),
                
                # Embed transparency option
                "embed_transparency": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Embed transparency map into albedo's alpha channel"
                }),
            },
            "optional": {
                "transparency": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "ao", "roughness", "metallic")
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist"
    
    def adjust_map(self, image, brightness, contrast, invert):
        """Apply brightness, contrast, and invert to a map"""
        # Apply brightness
        adjusted = image * brightness
        
        # Apply contrast (around midpoint 0.5)
        adjusted = (adjusted - 0.5) * contrast + 0.5
        
        # Clamp
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
        
        # Invert if requested
        if invert:
            adjusted = 1.0 - adjusted
        
        return adjusted
    
    def adjust(self, albedo, ao, roughness, metallic,
               albedo_brightness, albedo_contrast, albedo_invert,
               ao_brightness, ao_contrast, ao_invert,
               roughness_brightness, roughness_contrast, roughness_invert,
               metallic_brightness, metallic_contrast, metallic_invert,
               embed_transparency, transparency=None):
        """Adjust all PBR maps"""
        
        print("\n" + "="*60)
        print("PBR Adjuster")
        print("="*60)
        
        # Adjust Albedo
        albedo_adj = self.adjust_map(albedo, albedo_brightness, albedo_contrast, albedo_invert)
        print(f"Albedo: brightness={albedo_brightness}, contrast={albedo_contrast}, invert={albedo_invert}")
        print(f"  Range: [{albedo_adj.min():.3f}, {albedo_adj.max():.3f}]")
        
        # Adjust AO
        ao_adj = self.adjust_map(ao, ao_brightness, ao_contrast, ao_invert)
        print(f"\nAO: brightness={ao_brightness}, contrast={ao_contrast}, invert={ao_invert}")
        print(f"  Range: [{ao_adj.min():.3f}, {ao_adj.max():.3f}]")
        
        # Adjust Roughness
        roughness_adj = self.adjust_map(roughness, roughness_brightness, roughness_contrast, roughness_invert)
        print(f"\nRoughness: brightness={roughness_brightness}, contrast={roughness_contrast}, invert={roughness_invert}")
        print(f"  Range: [{roughness_adj.min():.3f}, {roughness_adj.max():.3f}]")
        
        # Adjust Metallic
        metallic_adj = self.adjust_map(metallic, metallic_brightness, metallic_contrast, metallic_invert)
        print(f"\nMetallic: brightness={metallic_brightness}, contrast={metallic_contrast}, invert={metallic_invert}")
        print(f"  Range: [{metallic_adj.min():.3f}, {metallic_adj.max():.3f}]")
        
        # ===== EMBED TRANSPARENCY INTO ALBEDO ALPHA =====
        if transparency is not None and embed_transparency:
            # Convert transparency to grayscale if needed
            trans_gray = transparency
            if trans_gray.shape[-1] == 3:
                weights = torch.tensor([0.299, 0.587, 0.114], 
                                      device=trans_gray.device, dtype=trans_gray.dtype)
                trans_gray = torch.sum(trans_gray * weights, dim=-1, keepdim=True)
            
            # Resize transparency to match albedo if needed
            if trans_gray.shape[1:3] != albedo_adj.shape[1:3]:
                trans_gray = torch.nn.functional.interpolate(
                    trans_gray.permute(0, 3, 1, 2),
                    size=albedo_adj.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Ensure albedo is RGB (3 channels)
            if albedo_adj.shape[-1] == 4:
                albedo_adj = albedo_adj[:, :, :, :3]
            elif albedo_adj.shape[-1] == 1:
                albedo_adj = albedo_adj.repeat(1, 1, 1, 3)
            
            # Concatenate transparency as alpha channel
            albedo_adj = torch.cat([albedo_adj, trans_gray], dim=-1)
            print("\n✓ Transparency embedded into albedo alpha channel")
            print(f"  Albedo shape: {albedo_adj.shape}")
        
        print("\n" + "="*60)
        print("✓ Adjustments Complete")
        print("="*60 + "\n")
        
        return (albedo_adj, ao_adj, roughness_adj, metallic_adj)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PBRExtractor": PBRExtractor,
    "PBRAdjuster": PBRAdjuster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PBRExtractor": "PBR Extractor (Marigold)",
    "PBRAdjuster": "PBR Adjuster",
}

