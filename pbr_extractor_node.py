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


class LightingExtractor:
    """
    Extract lighting components from Marigold Lighting output
    Marigold Lighting outputs 3 images in a batch:
    - Image 0: Albedo (0-1, linear space)
    - Image 1: Diffuse shading (0-1, linear space)
    - Image 2: Non-diffuse residual (0-1, linear space)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "marigold_lighting": ("IMAGE",),
                
                # Gamma controls for each output
                "albedo_gamma": ("FLOAT", {
                    "default": 0.4545,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for albedo (linear space)"
                }),
                "diffuse_shading_gamma": ("FLOAT", {
                    "default": 0.4545,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for diffuse shading (linear space)"
                }),
                "non_diffuse_gamma": ("FLOAT", {
                    "default": 0.4545,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for non-diffuse residual (linear space)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "diffuse_shading", "non_diffuse_residual")
    FUNCTION = "extract"
    CATEGORY = "Texture Alchemist/Extractors"
    
    def apply_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def extract(self, marigold_lighting, albedo_gamma, diffuse_shading_gamma, non_diffuse_gamma):
        """Extract lighting components from Marigold batch output"""
        
        print("\n" + "="*60)
        print("Lighting Extractor (Marigold)")
        print("="*60)
        print(f"Input shape: {marigold_lighting.shape}")
        print(f"Batch size: {marigold_lighting.shape[0]}")
        
        # Extract each image from the batch
        if marigold_lighting.shape[0] >= 1:
            albedo = marigold_lighting[0:1]
            print(f"\n🎨 Albedo (Image 0):")
            print(f"  Range: [{albedo.min():.3f}, {albedo.max():.3f}]")
            print(f"  Gamma: {albedo_gamma}")
            albedo = self.apply_gamma(albedo, albedo_gamma)
            print(f"  Output range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        else:
            print("⚠ Warning: No albedo image found in batch")
            albedo = marigold_lighting[0:1]
        
        if marigold_lighting.shape[0] >= 2:
            diffuse_shading = marigold_lighting[1:2]
            print(f"\n💡 Diffuse Shading (Image 1):")
            print(f"  Range: [{diffuse_shading.min():.3f}, {diffuse_shading.max():.3f}]")
            print(f"  Gamma: {diffuse_shading_gamma}")
            diffuse_shading = self.apply_gamma(diffuse_shading, diffuse_shading_gamma)
            print(f"  Output range: [{diffuse_shading.min():.3f}, {diffuse_shading.max():.3f}]")
        else:
            print("⚠ Warning: No diffuse shading image found in batch, using albedo")
            diffuse_shading = albedo
        
        if marigold_lighting.shape[0] >= 3:
            non_diffuse = marigold_lighting[2:3]
            print(f"\n✨ Non-Diffuse Residual (Image 2):")
            print(f"  Range: [{non_diffuse.min():.3f}, {non_diffuse.max():.3f}]")
            print(f"  Gamma: {non_diffuse_gamma}")
            non_diffuse = self.apply_gamma(non_diffuse, non_diffuse_gamma)
            print(f"  Output range: [{non_diffuse.min():.3f}, {non_diffuse.max():.3f}]")
        else:
            print("⚠ Warning: No non-diffuse residual found in batch, using albedo")
            non_diffuse = albedo
        
        print(f"\n✓ Lighting extraction complete")
        print("="*60 + "\n")
        
        return (albedo, diffuse_shading, non_diffuse)


class AppearanceExtractor:
    """
    Extract appearance components from Marigold Appearance output
    Marigold Appearance outputs 3 images in a batch:
    - Image 0: Albedo (0-1, sRGB space)
    - Image 1: Roughness and Metallicity (0-1, linear space)
      - Red channel: Roughness
      - Green channel: Metallicity
    - Image 2: Uncertainty maps (optional, for ensemble size > 2)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "marigold_appearance": ("IMAGE",),
                
                # Gamma controls for each output
                "albedo_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for albedo (sRGB space)"
                }),
                "roughness_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for roughness (linear space)"
                }),
                "metallic_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for metallic (linear space)"
                }),
                "uncertainty_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction for uncertainty maps"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "roughness", "metallic", "uncertainty")
    FUNCTION = "extract"
    CATEGORY = "Texture Alchemist/Extractors"
    
    def apply_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def channel_to_rgb(self, channel: torch.Tensor) -> torch.Tensor:
        """Convert single channel to RGB"""
        if channel.shape[-1] == 1:
            return channel.repeat(1, 1, 1, 3)
        return channel
    
    def extract(self, marigold_appearance, albedo_gamma, roughness_gamma, metallic_gamma, uncertainty_gamma):
        """Extract appearance components from Marigold batch output"""
        
        print("\n" + "="*60)
        print("Appearance Extractor (Marigold)")
        print("="*60)
        print(f"Input shape: {marigold_appearance.shape}")
        print(f"Batch size: {marigold_appearance.shape[0]}")
        
        # Extract albedo (Image 0 - sRGB space)
        if marigold_appearance.shape[0] >= 1:
            albedo = marigold_appearance[0:1]
            print(f"\n🎨 Albedo (Image 0 - sRGB):")
            print(f"  Range: [{albedo.min():.3f}, {albedo.max():.3f}]")
            print(f"  Gamma: {albedo_gamma}")
            albedo = self.apply_gamma(albedo, albedo_gamma)
            print(f"  Output range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        else:
            print("⚠ Warning: No albedo image found in batch")
            albedo = marigold_appearance[0:1]
        
        # Extract material channels (Image 1 - linear space)
        if marigold_appearance.shape[0] >= 2:
            material_channels = marigold_appearance[1:2]
            print(f"\n🔨 Material Channels (Image 1 - linear):")
            print(f"  Shape: {material_channels.shape}")
            print(f"  Range: [{material_channels.min():.3f}, {material_channels.max():.3f}]")
            
            # Extract roughness (red channel)
            roughness = material_channels[:, :, :, 0:1]
            print(f"\n  🟥 Roughness (Red Channel):")
            print(f"    Range: [{roughness.min():.3f}, {roughness.max():.3f}]")
            print(f"    Gamma: {roughness_gamma}")
            roughness = self.apply_gamma(roughness, roughness_gamma)
            roughness = self.channel_to_rgb(roughness)
            print(f"    Output range: [{roughness.min():.3f}, {roughness.max():.3f}]")
            
            # Extract metallic (green channel)
            metallic = material_channels[:, :, :, 1:2]
            print(f"\n  🟩 Metallicity (Green Channel):")
            print(f"    Range: [{metallic.min():.3f}, {metallic.max():.3f}]")
            print(f"    Gamma: {metallic_gamma}")
            metallic = self.apply_gamma(metallic, metallic_gamma)
            metallic = self.channel_to_rgb(metallic)
            print(f"    Output range: [{metallic.min():.3f}, {metallic.max():.3f}]")
        else:
            print("⚠ Warning: No material channels found in batch, using albedo")
            roughness = albedo
            metallic = albedo
        
        # Extract uncertainty maps (Image 2 - optional)
        if marigold_appearance.shape[0] >= 3:
            uncertainty = marigold_appearance[2:3]
            print(f"\n📊 Uncertainty Maps (Image 2):")
            print(f"  Range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
            print(f"  Gamma: {uncertainty_gamma}")
            uncertainty = self.apply_gamma(uncertainty, uncertainty_gamma)
            print(f"  Output range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
        else:
            print("⚠ Note: No uncertainty maps in batch (requires ensemble size > 2)")
            # Create a zero/neutral uncertainty map
            uncertainty = torch.zeros_like(albedo)
        
        print(f"\n✓ Appearance extraction complete")
        print("="*60 + "\n")
        
        return (albedo, roughness, metallic, uncertainty)


class FrankenMapExtractor:
    """
    Extract PBR maps from FrankenMap format
    FrankenMap channel layout:
    - Red Channel: Grayscale
    - Green Channel: Height
    - Blue Channel: Roughness
    
    Outputs grayscale, height, roughness, original image, and PBR pipe
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
                    "tooltip": "Gamma correction for grayscale (from red channel)"
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
                    "tooltip": "Grayscale brightness multiplier"
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
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "PBR_PIPE")
    RETURN_NAMES = ("grayscale", "height", "roughness", "original_image", "pbr_pipe")
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
        print(f"  Red (Grayscale): range [{red_channel.min():.3f}, {red_channel.max():.3f}]")
        print(f"  Green (Height): range [{green_channel.min():.3f}, {green_channel.max():.3f}]")
        print(f"  Blue (Roughness): range [{blue_channel.min():.3f}, {blue_channel.max():.3f}]")
        
        # Process Grayscale (from red channel)
        print(f"\n🎨 PROCESSING GRAYSCALE:")
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
            "image": frankenmap,  # Store original input
        }
        
        print(f"\n✓ FrankenMap extraction complete")
        print(f"  Extracted: Grayscale, Height, Roughness")
        print(f"  Original image stored in pipe")
        print(f"  PBR Pipe created")
        print("="*60 + "\n")
        
        return (albedo, height, roughness, frankenmap, pbr_pipe)


class BasicLightingBuilder:
    """
    Combine lighting components into final composite image
    Formula: Composite = Albedo * Shading + Residual
    
    Perfect for recombining outputs from Lighting Extractor (Marigold)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE",),
                "shading": ("IMAGE",),
                "residual": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composite",)
    FUNCTION = "build"
    CATEGORY = "Texture Alchemist/Materials"
    
    def build(self, albedo, shading, residual):
        """Combine lighting components using multiplicative and additive blending"""
        
        print("\n" + "="*60)
        print("Basic Lighting Builder")
        print("="*60)
        print(f"Albedo shape: {albedo.shape}")
        print(f"Shading shape: {shading.shape}")
        print(f"Residual shape: {residual.shape}")
        
        # Step 1: Multiply shading over albedo
        print(f"\n📐 Step 1: Multiply Shading over Albedo")
        print(f"  Albedo range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        print(f"  Shading range: [{shading.min():.3f}, {shading.max():.3f}]")
        
        base = albedo * shading
        print(f"  Result range: [{base.min():.3f}, {base.max():.3f}]")
        
        # Step 2: Add residual
        print(f"\n➕ Step 2: Add Residual")
        print(f"  Base range: [{base.min():.3f}, {base.max():.3f}]")
        print(f"  Residual range: [{residual.min():.3f}, {residual.max():.3f}]")
        
        composite = base + residual
        print(f"  Pre-clamp range: [{composite.min():.3f}, {composite.max():.3f}]")
        
        # Clamp to valid range
        composite = torch.clamp(composite, 0.0, 1.0)
        
        print(f"\n✓ Lighting composite complete")
        print(f"  Final range: [{composite.min():.3f}, {composite.max():.3f}]")
        print(f"  Formula: Composite = Albedo * Shading + Residual")
        print("="*60 + "\n")
        
        return (composite,)


class CompositeMaskAdjuster:
    """
    Apply mask-based adjustments to all three lighting passes simultaneously
    Allows targeted artistic changes that affect Albedo, Shading, and Residual consistently
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
                
                # Adjustment controls
                "brightness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Brightness multiplier in masked areas"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Contrast adjustment in masked areas"
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma correction in masked areas"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Saturation adjustment in masked areas (affects albedo only)"
                }),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "shading": ("IMAGE",),
                "residual": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "shading", "residual")
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist/Materials"
    
    def apply_brightness(self, image: torch.Tensor, brightness: float) -> torch.Tensor:
        """Apply brightness adjustment"""
        return image * brightness
    
    def apply_contrast(self, image: torch.Tensor, contrast: float) -> torch.Tensor:
        """Apply contrast adjustment around midpoint"""
        midpoint = 0.5
        return (image - midpoint) * contrast + midpoint
    
    def apply_gamma(self, image: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply gamma correction"""
        return torch.pow(torch.clamp(image, 0.0, 1.0), gamma)
    
    def apply_saturation(self, image: torch.Tensor, saturation: float) -> torch.Tensor:
        """Apply saturation adjustment (luminance-based)"""
        # Calculate luminance
        if image.shape[-1] >= 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype)
            luminance = torch.sum(image[..., :3] * weights, dim=-1, keepdim=True)
            
            # Blend between luminance (desaturated) and original (saturated)
            result = luminance + (image[..., :3] - luminance) * saturation
            
            # If there are extra channels (alpha, etc.), preserve them
            if image.shape[-1] > 3:
                result = torch.cat([result, image[..., 3:]], dim=-1)
            
            return result
        else:
            # Grayscale image, saturation has no effect
            return image
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Convert mask to grayscale if needed
        if mask.shape[-1] > 1:
            weights = torch.tensor([0.299, 0.587, 0.114], device=mask.device, dtype=mask.dtype)
            mask = torch.sum(mask[..., :3] * weights, dim=-1, keepdim=True)
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def adjust(self, mask, brightness, contrast, gamma, saturation, albedo=None, shading=None, residual=None):
        """Apply masked adjustments to all three passes"""
        
        print("\n" + "="*60)
        print("Composite Mask Adjuster")
        print("="*60)
        print(f"Mask shape: {mask.shape}")
        if albedo is not None:
            print(f"Albedo shape: {albedo.shape}")
        if shading is not None:
            print(f"Shading shape: {shading.shape}")
        if residual is not None:
            print(f"Residual shape: {residual.shape}")
        print(f"\nAdjustments:")
        print(f"  Brightness: {brightness}")
        print(f"  Contrast: {contrast}")
        print(f"  Gamma: {gamma}")
        print(f"  Saturation: {saturation}")
        
        print(f"\nMask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Process Albedo (if provided)
        if albedo is not None:
            mask_albedo = self.prepare_mask(mask, albedo.shape)
            
            print(f"\n🎨 Processing Albedo:")
            print(f"  Input range: [{albedo.min():.3f}, {albedo.max():.3f}]")
            
            albedo_adjusted = albedo.clone()
            albedo_adjusted = self.apply_brightness(albedo_adjusted, brightness)
            albedo_adjusted = self.apply_contrast(albedo_adjusted, contrast)
            albedo_adjusted = self.apply_gamma(albedo_adjusted, gamma)
            albedo_adjusted = self.apply_saturation(albedo_adjusted, saturation)
            albedo_adjusted = torch.clamp(albedo_adjusted, 0.0, 1.0)
            
            # Blend based on mask
            albedo_final = albedo * (1.0 - mask_albedo) + albedo_adjusted * mask_albedo
            print(f"  Output range: [{albedo_final.min():.3f}, {albedo_final.max():.3f}]")
        else:
            albedo_final = None
            print(f"\n🎨 Albedo: Not provided (skipped)")
        
        # Process Shading (if provided)
        if shading is not None:
            mask_shading = self.prepare_mask(mask, shading.shape)
            
            print(f"\n💡 Processing Shading:")
            print(f"  Input range: [{shading.min():.3f}, {shading.max():.3f}]")
            
            shading_adjusted = shading.clone()
            shading_adjusted = self.apply_brightness(shading_adjusted, brightness)
            shading_adjusted = self.apply_contrast(shading_adjusted, contrast)
            shading_adjusted = self.apply_gamma(shading_adjusted, gamma)
            shading_adjusted = torch.clamp(shading_adjusted, 0.0, 1.0)
            
            # Blend based on mask
            shading_final = shading * (1.0 - mask_shading) + shading_adjusted * mask_shading
            print(f"  Output range: [{shading_final.min():.3f}, {shading_final.max():.3f}]")
        else:
            shading_final = None
            print(f"\n💡 Shading: Not provided (skipped)")
        
        # Process Residual (if provided)
        if residual is not None:
            mask_residual = self.prepare_mask(mask, residual.shape)
            
            print(f"\n✨ Processing Residual:")
            print(f"  Input range: [{residual.min():.3f}, {residual.max():.3f}]")
            
            residual_adjusted = residual.clone()
            residual_adjusted = self.apply_brightness(residual_adjusted, brightness)
            residual_adjusted = self.apply_contrast(residual_adjusted, contrast)
            residual_adjusted = self.apply_gamma(residual_adjusted, gamma)
            residual_adjusted = torch.clamp(residual_adjusted, 0.0, 1.0)
            
            # Blend based on mask
            residual_final = residual * (1.0 - mask_residual) + residual_adjusted * mask_residual
            print(f"  Output range: [{residual_final.min():.3f}, {residual_final.max():.3f}]")
        else:
            residual_final = None
            print(f"\n✨ Residual: Not provided (skipped)")
        
        print(f"\n✓ Composite mask adjustment complete")
        print("="*60 + "\n")
        
        return (albedo_final, shading_final, residual_final)


class CompositeMaskExtractor:
    """
    Extract masks from lighting passes based on various criteria
    Useful for creating targeted adjustments based on lighting properties
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Extraction method
                "extract_from": (["Albedo", "Shading", "Residual", "Combined"], {
                    "default": "Shading",
                    "tooltip": "Which pass to extract mask from"
                }),
                
                "method": (["Luminance", "Threshold", "Range", "Inverse Luminance"], {
                    "default": "Luminance",
                    "tooltip": "Method for mask extraction"
                }),
                
                # Threshold controls
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Threshold value for Threshold method"
                }),
                
                "range_min": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum value for Range method"
                }),
                
                "range_max": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Maximum value for Range method"
                }),
                
                # Post-processing
                "feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Soften mask edges (0 = hard edges)"
                }),
                
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "Gamma adjustment for mask"
                }),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "shading": ("IMAGE",),
                "residual": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "extract"
    CATEGORY = "Texture Alchemist/Materials"
    
    def get_luminance(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate luminance from RGB image"""
        if image.shape[-1] >= 3:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device, dtype=image.dtype)
            return torch.sum(image[..., :3] * weights, dim=-1, keepdim=True)
        else:
            return image[..., :1]
    
    def apply_feather(self, mask: torch.Tensor, feather: float) -> torch.Tensor:
        """Soften mask edges"""
        if feather <= 0.0:
            return mask
        
        # Simple feathering using contrast reduction around threshold
        feather_strength = 1.0 - feather * 2.0  # 0.0-0.5 maps to 1.0-0.0
        midpoint = 0.5
        return torch.clamp((mask - midpoint) * feather_strength + midpoint, 0.0, 1.0)
    
    def extract(self, extract_from, method, threshold, range_min, range_max, feather, gamma,
                albedo=None, shading=None, residual=None):
        """Extract mask from lighting passes"""
        
        print("\n" + "="*60)
        print("Composite Mask Extractor")
        print("="*60)
        print(f"Extract from: {extract_from}")
        print(f"Method: {method}")
        
        # Select source image
        if extract_from == "Albedo":
            if albedo is None:
                raise ValueError("Albedo input required when extract_from is 'Albedo'")
            source = albedo
            print(f"Source: Albedo, range [{source.min():.3f}, {source.max():.3f}]")
        elif extract_from == "Shading":
            if shading is None:
                raise ValueError("Shading input required when extract_from is 'Shading'")
            source = shading
            print(f"Source: Shading, range [{source.min():.3f}, {source.max():.3f}]")
        elif extract_from == "Residual":
            if residual is None:
                raise ValueError("Residual input required when extract_from is 'Residual'")
            source = residual
            print(f"Source: Residual, range [{source.min():.3f}, {source.max():.3f}]")
        else:  # Combined
            # Average all available passes
            available_passes = []
            if albedo is not None:
                available_passes.append(albedo)
            if shading is not None:
                available_passes.append(shading)
            if residual is not None:
                available_passes.append(residual)
            
            if not available_passes:
                raise ValueError("At least one pass required when extract_from is 'Combined'")
            
            source = sum(available_passes) / len(available_passes)
            print(f"Source: Combined ({len(available_passes)} passes), range [{source.min():.3f}, {source.max():.3f}]")
        
        # Extract mask based on method
        if method == "Luminance":
            print(f"\n🔍 Extracting by Luminance")
            mask = self.get_luminance(source)
            
        elif method == "Inverse Luminance":
            print(f"\n🔍 Extracting by Inverse Luminance")
            mask = 1.0 - self.get_luminance(source)
            
        elif method == "Threshold":
            print(f"\n🔍 Extracting by Threshold")
            print(f"  Threshold: {threshold}")
            luminance = self.get_luminance(source)
            mask = (luminance >= threshold).float()
            
        elif method == "Range":
            print(f"\n🔍 Extracting by Range")
            print(f"  Range: [{range_min}, {range_max}]")
            luminance = self.get_luminance(source)
            mask = ((luminance >= range_min) & (luminance <= range_max)).float()
        
        print(f"  Mask range (pre-processing): [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Apply feathering
        if feather > 0.0:
            print(f"\n🌫️  Applying feather: {feather}")
            mask = self.apply_feather(mask, feather)
            print(f"  Mask range (post-feather): [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Apply gamma
        if gamma != 1.0:
            print(f"\n🎚️  Applying gamma: {gamma}")
            mask = torch.pow(torch.clamp(mask, 0.0, 1.0), gamma)
            print(f"  Mask range (post-gamma): [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Convert to RGB for compatibility
        if mask.shape[-1] == 1:
            mask = mask.repeat(1, 1, 1, 3)
        
        print(f"\n✓ Mask extraction complete")
        print(f"  Final mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        print("="*60 + "\n")
        
        return (mask,)


class LightingPassMaskApplicator:
    """
    Apply a mask to lighting passes (albedo, shading, residual)
    Outputs the masked-out version of each input (mask defines what to keep)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
            },
            "optional": {
                "albedo": ("IMAGE",),
                "shading": ("IMAGE",),
                "residual": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "shading", "residual")
    FUNCTION = "apply_mask"
    CATEGORY = "Texture Alchemist/Materials"
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Convert mask to grayscale if needed
        if mask.shape[-1] > 1:
            weights = torch.tensor([0.299, 0.587, 0.114], device=mask.device, dtype=mask.dtype)
            mask = torch.sum(mask[..., :3] * weights, dim=-1, keepdim=True)
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def apply_mask(self, mask, albedo=None, shading=None, residual=None):
        """Apply mask to all provided passes"""
        
        print("\n" + "="*60)
        print("Lighting Pass Mask Applicator")
        print("="*60)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        # Apply mask to albedo if provided
        if albedo is not None:
            mask_albedo = self.prepare_mask(mask, albedo.shape)
            albedo_masked = albedo * mask_albedo
            print(f"\n🎨 Albedo:")
            print(f"  Input range: [{albedo.min():.3f}, {albedo.max():.3f}]")
            print(f"  Output range: [{albedo_masked.min():.3f}, {albedo_masked.max():.3f}]")
        else:
            albedo_masked = None
            print(f"\n🎨 Albedo: Not provided")
        
        # Apply mask to shading if provided
        if shading is not None:
            mask_shading = self.prepare_mask(mask, shading.shape)
            shading_masked = shading * mask_shading
            print(f"\n💡 Shading:")
            print(f"  Input range: [{shading.min():.3f}, {shading.max():.3f}]")
            print(f"  Output range: [{shading_masked.min():.3f}, {shading_masked.max():.3f}]")
        else:
            shading_masked = None
            print(f"\n💡 Shading: Not provided")
        
        # Apply mask to residual if provided
        if residual is not None:
            mask_residual = self.prepare_mask(mask, residual.shape)
            residual_masked = residual * mask_residual
            print(f"\n✨ Residual:")
            print(f"  Input range: [{residual.min():.3f}, {residual.max():.3f}]")
            print(f"  Output range: [{residual_masked.min():.3f}, {residual_masked.max():.3f}]")
        else:
            residual_masked = None
            print(f"\n✨ Residual: Not provided")
        
        print(f"\n✓ Mask applied to all passes")
        print("="*60 + "\n")
        
        return (albedo_masked, shading_masked, residual_masked)


class PassMaskApplicator:
    """
    Apply a mask to up to 6 images
    Outputs the masked-out version of each input (mask defines what to keep)
    General purpose mask applicator for any images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("IMAGE",),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6")
    FUNCTION = "apply_mask"
    CATEGORY = "Texture Alchemist/Adjustment"
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Convert mask to grayscale if needed
        if mask.shape[-1] > 1:
            weights = torch.tensor([0.299, 0.587, 0.114], device=mask.device, dtype=mask.dtype)
            mask = torch.sum(mask[..., :3] * weights, dim=-1, keepdim=True)
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def apply_mask(self, mask, image_1=None, image_2=None, image_3=None, 
                   image_4=None, image_5=None, image_6=None):
        """Apply mask to all provided images"""
        
        print("\n" + "="*60)
        print("Pass Mask Applicator")
        print("="*60)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        results = []
        images = [image_1, image_2, image_3, image_4, image_5, image_6]
        
        for i, image in enumerate(images, 1):
            if image is not None:
                mask_prepared = self.prepare_mask(mask, image.shape)
                masked_image = image * mask_prepared
                results.append(masked_image)
                print(f"\n📷 Image {i}:")
                print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")
                print(f"  Output range: [{masked_image.min():.3f}, {masked_image.max():.3f}]")
            else:
                results.append(None)
                print(f"\n📷 Image {i}: Not provided")
        
        print(f"\n✓ Mask applied to all images")
        print("="*60 + "\n")
        
        return tuple(results)


class MultimattePassApplicator:
    """
    Apply a specific channel from a multimatte mask to up to 6 images
    Extracts Red, Green, or Blue channel and uses it as a mask
    Perfect for VFX workflows with channel-packed masks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "multimatte_mask": ("IMAGE",),
                "channel": (["Red", "Green", "Blue", "Luminance"], {
                    "default": "Red",
                    "tooltip": "Which channel to use as the mask"
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6")
    FUNCTION = "apply_mask"
    CATEGORY = "Texture Alchemist/Adjustment"
    
    def extract_channel(self, multimatte: torch.Tensor, channel: str) -> torch.Tensor:
        """Extract specific channel from multimatte"""
        if channel == "Luminance":
            # Calculate luminance from RGB
            if multimatte.shape[-1] >= 3:
                weights = torch.tensor([0.2989, 0.5870, 0.1140], device=multimatte.device, dtype=multimatte.dtype)
                return torch.sum(multimatte[..., :3] * weights, dim=-1, keepdim=True)
            else:
                return multimatte[..., 0:1]
        
        if multimatte.shape[-1] < 3 and channel != "Luminance":
            print(f"⚠ Warning: Multimatte has fewer than 3 channels, using first channel")
            return multimatte[..., 0:1]
        
        if channel == "Red":
            return multimatte[..., 0:1]
        elif channel == "Green":
            return multimatte[..., 1:2]
        elif channel == "Blue":
            return multimatte[..., 2:3]
        else:
            return multimatte[..., 0:1]
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Mask is already single channel from extract_channel
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def apply_mask(self, multimatte_mask, channel, image_1=None, image_2=None, image_3=None, 
                   image_4=None, image_5=None, image_6=None):
        """Apply selected channel mask to all provided images"""
        
        print("\n" + "="*60)
        print("Multimatte Pass Applicator")
        print("="*60)
        print(f"Multimatte shape: {multimatte_mask.shape}")
        print(f"Selected channel: {channel}")
        
        # Extract the selected channel
        mask = self.extract_channel(multimatte_mask, channel)
        print(f"Extracted mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        results = []
        images = [image_1, image_2, image_3, image_4, image_5, image_6]
        
        for i, image in enumerate(images, 1):
            if image is not None:
                mask_prepared = self.prepare_mask(mask, image.shape)
                masked_image = image * mask_prepared
                results.append(masked_image)
                print(f"\n📷 Image {i}:")
                print(f"  Input range: [{image.min():.3f}, {image.max():.3f}]")
                print(f"  Output range: [{masked_image.min():.3f}, {masked_image.max():.3f}]")
            else:
                results.append(None)
                print(f"\n📷 Image {i}: Not provided")
        
        print(f"\n✓ {channel} channel mask applied to all images")
        print("="*60 + "\n")
        
        return tuple(results)


class SinglePassMaskApplicator:
    """
    Apply a mask to a single image
    Simple mask applicator for one image at a time
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_mask"
    CATEGORY = "Texture Alchemist/Adjustment"
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Convert mask to grayscale if needed
        if mask.shape[-1] > 1:
            weights = torch.tensor([0.299, 0.587, 0.114], device=mask.device, dtype=mask.dtype)
            mask = torch.sum(mask[..., :3] * weights, dim=-1, keepdim=True)
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def apply_mask(self, image, mask):
        """Apply mask to image"""
        
        print("\n" + "="*60)
        print("Single Pass Mask Applicator")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        mask_prepared = self.prepare_mask(mask, image.shape)
        masked_image = image * mask_prepared
        
        print(f"\nInput range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Output range: [{masked_image.min():.3f}, {masked_image.max():.3f}]")
        print(f"\n✓ Mask applied")
        print("="*60 + "\n")
        
        return (masked_image,)


class SingleMultimattePassApplicator:
    """
    Apply a specific channel from a multimatte mask to a single image
    Extracts Red, Green, Blue, or Luminance channel and uses it as a mask
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "multimatte_mask": ("IMAGE",),
                "channel": (["Red", "Green", "Blue", "Luminance"], {
                    "default": "Red",
                    "tooltip": "Which channel to use as the mask"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_mask"
    CATEGORY = "Texture Alchemist/Adjustment"
    
    def extract_channel(self, multimatte: torch.Tensor, channel: str) -> torch.Tensor:
        """Extract specific channel from multimatte"""
        if channel == "Luminance":
            # Calculate luminance from RGB
            if multimatte.shape[-1] >= 3:
                weights = torch.tensor([0.2989, 0.5870, 0.1140], device=multimatte.device, dtype=multimatte.dtype)
                return torch.sum(multimatte[..., :3] * weights, dim=-1, keepdim=True)
            else:
                return multimatte[..., 0:1]
        
        if multimatte.shape[-1] < 3 and channel != "Luminance":
            print(f"⚠ Warning: Multimatte has fewer than 3 channels, using first channel")
            return multimatte[..., 0:1]
        
        if channel == "Red":
            return multimatte[..., 0:1]
        elif channel == "Green":
            return multimatte[..., 1:2]
        elif channel == "Blue":
            return multimatte[..., 2:3]
        else:
            return multimatte[..., 0:1]
    
    def prepare_mask(self, mask: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Prepare mask to match target image shape"""
        # Mask is already single channel from extract_channel
        
        # Expand mask to match target channels
        if mask.shape[-1] != target_shape[-1]:
            mask = mask.repeat(1, 1, 1, target_shape[-1])
        
        # Clamp mask to [0, 1]
        mask = torch.clamp(mask, 0.0, 1.0)
        
        return mask
    
    def apply_mask(self, image, multimatte_mask, channel):
        """Apply selected channel mask to image"""
        
        print("\n" + "="*60)
        print("Single Multimatte Pass Applicator")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Multimatte shape: {multimatte_mask.shape}")
        print(f"Selected channel: {channel}")
        
        # Extract the selected channel
        mask = self.extract_channel(multimatte_mask, channel)
        print(f"Extracted mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        
        mask_prepared = self.prepare_mask(mask, image.shape)
        masked_image = image * mask_prepared
        
        print(f"\nInput range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Output range: [{masked_image.min():.3f}, {masked_image.max():.3f}]")
        print(f"\n✓ {channel} channel mask applied")
        print("="*60 + "\n")
        
        return (masked_image,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PBRMaterialProcessor": PBRMaterialProcessor,
    "PBRNormalProcessor": PBRNormalProcessor,
    "PBRHeightProcessor": PBRHeightProcessor,
    "LightingExtractor": LightingExtractor,
    "AppearanceExtractor": AppearanceExtractor,
    "BasicLightingBuilder": BasicLightingBuilder,
    "CompositeMaskAdjuster": CompositeMaskAdjuster,
    "CompositeMaskExtractor": CompositeMaskExtractor,
    "LightingPassMaskApplicator": LightingPassMaskApplicator,
    "PassMaskApplicator": PassMaskApplicator,
    "MultimattePassApplicator": MultimattePassApplicator,
    "SinglePassMaskApplicator": SinglePassMaskApplicator,
    "SingleMultimattePassApplicator": SingleMultimattePassApplicator,
    "FrankenMapExtractor": FrankenMapExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PBRMaterialProcessor": "PBR Material Processor (Marigold)",
    "PBRNormalProcessor": "PBR Normal Processor (Lotus)",
    "PBRHeightProcessor": "PBR Height Processor (Lotus)",
    "LightingExtractor": "Lighting Extractor (Marigold)",
    "AppearanceExtractor": "Appearance Extractor (Marigold)",
    "BasicLightingBuilder": "Basic Lighting Builder",
    "CompositeMaskAdjuster": "Composite Mask Adjuster",
    "CompositeMaskExtractor": "Composite Mask Extractor",
    "LightingPassMaskApplicator": "Lighting Pass Mask Applicator",
    "PassMaskApplicator": "Pass Mask Applicator",
    "MultimattePassApplicator": "Multimatte Pass Applicator",
    "SinglePassMaskApplicator": "Single Pass Mask Applicator",
    "SingleMultimattePassApplicator": "Single Multimatte Pass Applicator",
    "FrankenMapExtractor": "PBR Extractor (FrankenMap)",
}
