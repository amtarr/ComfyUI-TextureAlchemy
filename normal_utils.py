"""
Custom ComfyUI Node: PBR Material Processor
Simplified version that takes Marigold/Lotus outputs and processes them
with unified brightness/gamma controls and seed management
"""

import torch


class PBRMaterialProcessor:
    """
    Takes raw outputs from Marigold IID and processes them into PBR maps
    With unified brightness controls for roughness and metallic
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Marigold outputs (these are 3-batch images typically)
                "marigold_appearance": ("IMAGE",),
                "marigold_lighting": ("IMAGE",),
                
                # Brightness controls
                "roughness_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Adjust roughness map brightness"
                }),
                "metallic_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Adjust metallic map brightness"
                }),
                "albedo_gamma": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Gamma correction for albedo"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "roughness", "metallic", "ao")
    FUNCTION = "process_materials"
    CATEGORY = "Texture Alchemist/Materials"
    
    def apply_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def adjust_brightness(self, image: torch.Tensor, brightness: float) -> torch.Tensor:
        """Adjust image brightness"""
        return torch.clamp(image * brightness, 0.0, 1.0)
    
    def grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale"""
        if image.shape[-1] == 1:
            return image
        # Luminance weights
        weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype)
        return torch.sum(image * weights, dim=-1, keepdim=True)
    
    def channel_to_rgb(self, channel: torch.Tensor) -> torch.Tensor:
        """Convert single channel to RGB"""
        if channel.shape[-1] == 1:
            return channel.repeat(1, 1, 1, 3)
        return channel
    
    def process_materials(self, marigold_appearance, marigold_lighting,
                         roughness_brightness, metallic_brightness, albedo_gamma):
        """
        Process Marigold outputs into clean PBR maps
        
        Marigold IID Appearance typically outputs 3 images:
        - [0]: Base appearance
        - [1]: Material channels (R=roughness, G=metallic)
        - [2]: Additional data
        
        Marigold IID Lighting typically outputs 3 images:
        - [0]: Albedo/diffuse color
        - [1]: Ambient occlusion / lighting
        - [2]: Additional lighting data
        """
        
        # Process Appearance output
        # Apply gamma first
        appearance = self.apply_gamma(marigold_appearance, albedo_gamma)
        
        # Extract material channels (typically batch index 1)
        if appearance.shape[0] >= 2:
            material_channels = appearance[1:2]
        else:
            material_channels = appearance[0:1]
        
        # Extract roughness (red channel)
        roughness = material_channels[:, :, :, 0:1]
        roughness = self.channel_to_rgb(roughness)
        roughness = self.adjust_brightness(roughness, roughness_brightness)
        
        # Extract metallic (green channel)
        metallic = material_channels[:, :, :, 1:2]
        metallic = self.channel_to_rgb(metallic)
        # Metallic uses inverse gamma effect
        metallic_gamma = max(0.1, 2.0 - metallic_brightness)
        metallic = self.apply_gamma(metallic, metallic_gamma)
        
        # Process Lighting output
        lighting = self.apply_gamma(marigold_lighting, 0.45)
        
        # Extract albedo (batch index 0)
        if lighting.shape[0] >= 1:
            albedo = lighting[0:1]
        else:
            albedo = lighting
        
        # Extract or generate AO (batch index 1 if available)
        if lighting.shape[0] >= 2:
            ao = lighting[1:2]
        else:
            ao = albedo
        
        # Convert AO to grayscale
        ao = self.grayscale(ao)
        ao = self.channel_to_rgb(ao)
        
        return (albedo, roughness, metallic, ao)


class PBRNormalProcessor:
    """
    Process Lotus normal map outputs with flip control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
                "flip_green": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flip green channel (DirectX vs OpenGL)"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize to -1 to 1 range"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "process_normal"
    CATEGORY = "Texture Alchemist/Materials"
    
    def process_normal(self, normal_map, flip_green, normalize):
        """Process normal map with optional green channel flip"""
        
        normal = normal_map.clone()
        
        # Flip green channel if requested
        if flip_green and normal.shape[-1] >= 3:
            normal[:, :, :, 1] = 1.0 - normal[:, :, :, 1]
        
        # Normalize if requested
        if normalize:
            # Convert from 0-1 to -1 to 1
            normal = (normal * 2.0) - 1.0
            # Normalize vectors
            norm = torch.sqrt(torch.sum(normal ** 2, dim=-1, keepdim=True))
            norm = torch.clamp(norm, min=1e-8)
            normal = normal / norm
            # Convert back to 0-1
            normal = (normal + 1.0) / 2.0
        
        return (normal,)


class PBRHeightProcessor:
    """
    Process Lotus depth/height map with remap control
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("IMAGE",),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert height values"
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remap to 0-1 range"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("height",)
    FUNCTION = "process_height"
    CATEGORY = "Texture Alchemist/Materials"
    
    def process_height(self, depth_map, invert, normalize):
        """Process height/depth map"""
        
        height = depth_map.clone()
        
        # Convert to grayscale if needed
        if height.shape[-1] > 1:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=height.device, dtype=height.dtype)
            height = torch.sum(height * weights, dim=-1, keepdim=True)
            height = height.repeat(1, 1, 1, 3)
        
        # Normalize to 0-1
        if normalize:
            h_min = height.min()
            h_max = height.max()
            if h_max - h_min > 1e-6:
                height = (height - h_min) / (h_max - h_min)
        
        # Invert if requested
        if invert:
            height = 1.0 - height
        
        return (height,)


class FrankenMapExtractor:
    """
    Extract PBR maps from FrankenMap format
    FrankenMap channel layout:
    - Red Channel: Grayscale/Albedo
    - Green Channel: Height
    - Blue Channel: Roughness
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frankenmap": ("IMAGE",),
                
                # Gamma controls for each channel
                "albedo_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Gamma correction for albedo (from red channel)"
                }),
                "height_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Gamma correction for height (from green channel)"
                }),
                "roughness_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Gamma correction for roughness (from blue channel)"
                }),
                
                # Brightness/contrast controls
                "albedo_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Albedo brightness multiplier"
                }),
                "height_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Height contrast multiplier"
                }),
                "roughness_brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Roughness brightness multiplier"
                }),
                
                # Inversion controls
                "invert_height": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert height values (bumps become indents)"
                }),
                "invert_roughness": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert roughness values (rough becomes smooth)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "PBR_PIPE")
    RETURN_NAMES = ("albedo", "height", "roughness", "pbr_pipe")
    FUNCTION = "extract"
    CATEGORY = "Texture Alchemist/Extractors"
    
    def apply_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def adjust_brightness(self, image: torch.Tensor, brightness: float) -> torch.Tensor:
        """Adjust image brightness"""
        return torch.clamp(image * brightness, 0.0, 1.0)
    
    def adjust_contrast(self, image: torch.Tensor, contrast: float) -> torch.Tensor:
        """Adjust image contrast around midpoint"""
        midpoint = 0.5
        return torch.clamp((image - midpoint) * contrast + midpoint, 0.0, 1.0)
    
    def channel_to_rgb(self, channel: torch.Tensor) -> torch.Tensor:
        """Convert single channel to RGB"""
        if channel.shape[-1] == 1:
            return channel.repeat(1, 1, 1, 3)
        return channel
    
    def extract(self, frankenmap, albedo_gamma, height_gamma, roughness_gamma,
               albedo_brightness, height_contrast, roughness_brightness,
               invert_height, invert_roughness):
        """Extract PBR maps from FrankenMap"""
        
        print("\n" + "="*60)
        print("FrankenMap PBR Extractor")
        print("="*60)
        print(f"Input shape: {frankenmap.shape}")
        
        # Validate input
        if frankenmap.shape[-1] < 3:
            print("⚠ Warning: FrankenMap should have 3 channels (RGB)")
            print("  Padding with zeros for missing channels")
            # Pad with zeros if needed
            missing = 3 - frankenmap.shape[-1]
            padding = torch.zeros(
                (frankenmap.shape[0], frankenmap.shape[1], frankenmap.shape[2], missing),
                device=frankenmap.device, dtype=frankenmap.dtype
            )
            frankenmap = torch.cat([frankenmap, padding], dim=-1)
        
        # Extract channels
        # Red = Albedo/Grayscale
        red_channel = frankenmap[:, :, :, 0:1]
        # Green = Height
        green_channel = frankenmap[:, :, :, 1:2]
        # Blue = Roughness
        blue_channel = frankenmap[:, :, :, 2:3]
        
        print(f"\n📊 CHANNEL EXTRACTION:")
        print(f"  Red (Albedo): range [{red_channel.min():.3f}, {red_channel.max():.3f}]")
        print(f"  Green (Height): range [{green_channel.min():.3f}, {green_channel.max():.3f}]")
        print(f"  Blue (Roughness): range [{blue_channel.min():.3f}, {blue_channel.max():.3f}]")
        
        # Process Albedo (from red channel)
        print(f"\n🎨 PROCESSING ALBEDO:")
        print(f"  Gamma: {albedo_gamma}")
        print(f"  Brightness: {albedo_brightness}")
        
        albedo = red_channel
        albedo = self.apply_gamma(albedo, albedo_gamma)
        albedo = self.adjust_brightness(albedo, albedo_brightness)
        albedo = self.channel_to_rgb(albedo)
        
        print(f"  Output range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        
        # Process Height (from green channel)
        print(f"\n⛰️  PROCESSING HEIGHT:")
        print(f"  Gamma: {height_gamma}")
        print(f"  Contrast: {height_contrast}")
        print(f"  Invert: {invert_height}")
        
        height = green_channel
        height = self.apply_gamma(height, height_gamma)
        height = self.adjust_contrast(height, height_contrast)
        
        if invert_height:
            height = 1.0 - height
            print(f"  ✓ Height inverted")
        
        height = self.channel_to_rgb(height)
        
        print(f"  Output range: [{height.min():.3f}, {height.max():.3f}]")
        
        # Process Roughness (from blue channel)
        print(f"\n🔨 PROCESSING ROUGHNESS:")
        print(f"  Gamma: {roughness_gamma}")
        print(f"  Brightness: {roughness_brightness}")
        print(f"  Invert: {invert_roughness}")
        
        roughness = blue_channel
        roughness = self.apply_gamma(roughness, roughness_gamma)
        roughness = self.adjust_brightness(roughness, roughness_brightness)
        
        if invert_roughness:
            roughness = 1.0 - roughness
            print(f"  ✓ Roughness inverted")
        
        roughness = self.channel_to_rgb(roughness)
        
        print(f"  Output range: [{roughness.min():.3f}, {roughness.max():.3f}]")
        
        # Create PBR pipe
        pbr_pipe = {
            "albedo": albedo,
            "normal": None,
            "ao": None,
            "height": height,
            "roughness": roughness,
            "metallic": None,
            "transparency": None,
            "emission": None,
        }
        
        print(f"\n✓ FrankenMap extraction complete")
        print(f"  Extracted: Albedo, Height, Roughness")
        print(f"  PBR Pipe created")
        print("="*60 + "\n")
        
        return (albedo, height, roughness, pbr_pipe)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PBRMaterialProcessor": PBRMaterialProcessor,
    "PBRNormalProcessor": PBRNormalProcessor,
    "PBRHeightProcessor": PBRHeightProcessor,
    "FrankenMapExtractor": FrankenMapExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PBRMaterialProcessor": "PBR Material Processor (Marigold)",
    "PBRNormalProcessor": "PBR Normal Processor (Lotus)",
    "PBRHeightProcessor": "PBR Height Processor (Lotus)",
    "FrankenMapExtractor": "PBR Extractor (FrankenMap)",
}
