"""
PBR Material Processor Nodes
Simple, clear extraction and adjustment of PBR maps from Marigold outputs
"""

import torch


class PBRExtractor:
    """
    Extracts PBR maps from Marigold IID Appearance and Lighting outputs
    
    Inputs:
    - Marigold Appearance output (3 batches: 0=albedo, 1=rough/metal, 2=unused)
    - Marigold Lighting output (3 batches: 0=albedo, 1=AO, 2=unused)
    
    Outputs: Albedo, Roughness, Metallic, AO
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
                    "default": "appearance",
                    "tooltip": "Which Marigold output to use for albedo"
                }),
                
                # Gamma correction - separate controls
                "gamma_albedo": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for albedo"
                }),
                "gamma_appearance": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for appearance maps (roughness, metallic)"
                }),
                "gamma_lighting": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gamma correction for lighting maps (AO)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "roughness", "metallic", "ao")
    FUNCTION = "extract"
    CATEGORY = "PBR"
    
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
    
    def extract(self, marigold_appearance, marigold_lighting, albedo_source, gamma_albedo, gamma_appearance, gamma_lighting):
        """Extract PBR maps"""
        
        print("\n" + "="*60)
        print("PBR Extractor")
        print("="*60)
        print(f"Appearance shape: {marigold_appearance.shape}")
        print(f"Lighting shape: {marigold_lighting.shape}")
        print(f"Albedo source: {albedo_source}")
        print(f"Gamma - Albedo: {gamma_albedo}, Appearance: {gamma_appearance}, Lighting: {gamma_lighting}")
        
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
            roughness = self.apply_gamma(roughness, gamma_appearance)
            print("\n✓ Roughness from Appearance index 1, RED channel")
            print(f"  Gamma: {gamma_appearance}")
            print(f"  Range: [{roughness.min():.3f}, {roughness.max():.3f}]")
        else:
            print("\n✗ WARNING: Appearance doesn't have index 1, using zeros")
            roughness = torch.zeros_like(albedo)
        
        # ===== METALLIC (Green channel from Appearance index 1) =====
        if marigold_appearance.shape[0] >= 2:
            material_batch = marigold_appearance[1:2]
            metallic = material_batch[:, :, :, 1:2]  # Green channel
            metallic = self.channel_to_rgb(metallic)
            metallic = self.apply_gamma(metallic, gamma_appearance)
            print("\n✓ Metallic from Appearance index 1, GREEN channel")
            print(f"  Gamma: {gamma_appearance}")
            print(f"  Range: [{metallic.min():.3f}, {metallic.max():.3f}]")
        else:
            print("\n✗ WARNING: Appearance doesn't have index 1, using zeros")
            metallic = torch.zeros_like(albedo)
        
        # ===== AO (Lighting index 1) =====
        if marigold_lighting.shape[0] >= 2:
            ao = marigold_lighting[1:2]
            ao = self.grayscale(ao)
            ao = self.apply_gamma(ao, gamma_lighting)
            print("\n✓ AO from Lighting index 1")
            print(f"  Gamma: {gamma_lighting}")
            print(f"  Range: [{ao.min():.3f}, {ao.max():.3f}]")
        else:
            print("\n✗ WARNING: Lighting doesn't have index 1, using albedo luminance")
            ao = self.grayscale(albedo)
        
        print("\n" + "="*60)
        print("✓ Extraction Complete")
        print("="*60 + "\n")
        
        return (albedo, roughness, metallic, ao)


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
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "ao", "roughness", "metallic")
    FUNCTION = "adjust"
    CATEGORY = "PBR"
    
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
               metallic_brightness, metallic_contrast, metallic_invert):
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
        
        print("\n" + "="*60)
        print("✓ Adjustments Complete")
        print("="*60 + "\n")
        
        return (albedo_adj, ao_adj, roughness_adj, metallic_adj)


class LotusHeightProcessor:
    """
    Process Lotus depth/height map output
    Connect: LotusSampler → VAEDecode → This node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lotus_depth": ("IMAGE",),
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert height values"
                }),
                "bit_depth": (["8-bit", "16-bit", "32-bit"], {
                    "default": "16-bit",
                    "tooltip": "Output bit depth (save as EXR for 16/32-bit)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("height",)
    FUNCTION = "process"
    CATEGORY = "PBR"
    
    def process(self, lotus_depth, invert, bit_depth):
        """Process height map"""
        
        height = lotus_depth.clone()
        
        # Convert to grayscale
        if height.shape[-1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                   device=height.device, dtype=height.dtype)
            height = torch.sum(height * weights, dim=-1, keepdim=True)
            height = height.repeat(1, 1, 1, 3)
        
        # Normalize to 0-1
        h_min = height.min()
        h_max = height.max()
        if h_max - h_min > 1e-6:
            height = (height - h_min) / (h_max - h_min)
        
        # Invert if requested
        if invert:
            height = 1.0 - height
        
        print(f"✓ Height processed ({bit_depth})")
        if bit_depth in ["16-bit", "32-bit"]:
            print(f"  Note: Save as OpenEXR to preserve {bit_depth} precision")
        
        return (height,)


class AOApproximator:
    """
    Generate Ambient Occlusion map from height and normal maps
    Uses height sampling to detect concave areas that would be occluded
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Sampling radius in pixels (larger = broader occlusion)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "AO intensity multiplier"
                }),
                "samples": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Number of sampling directions (more = better quality, slower)"
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Contrast adjustment for AO"
                }),
                "use_normal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use normal map to improve AO quality (if provided)"
                }),
            },
            "optional": {
                "normal": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ao",)
    FUNCTION = "generate_ao"
    CATEGORY = "PBR"
    
    def generate_ao(self, height, radius, strength, samples, contrast, use_normal, normal=None):
        """Generate AO from height map"""
        
        print("\n" + "="*60)
        print("AO Approximator")
        print("="*60)
        print(f"Height shape: {height.shape}")
        if normal is not None:
            print(f"Normal shape: {normal.shape}")
        print(f"Radius: {radius}, Samples: {samples}, Strength: {strength}, Contrast: {contrast}")
        print(f"Use normal: {use_normal}")
        
        # Convert height to grayscale if needed
        height_map = height.clone()
        if height_map.shape[-1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                   device=height_map.device, dtype=height_map.dtype)
            height_map = torch.sum(height_map * weights, dim=-1, keepdim=True)
        elif height_map.shape[-1] != 1:
            height_map = height_map[..., 0:1]
        
        batch, h, w, channels = height.shape
        
        # Initialize AO map (start fully lit)
        ao = torch.ones((batch, h, w, 1), device=height.device, dtype=height.dtype)
        
        # Generate sampling angles
        import math
        angles = [2.0 * math.pi * i / samples for i in range(samples)]
        
        print(f"\n⚙ Computing AO with {samples} samples at radius {radius}...")
        
        # For each sampling direction
        occlusion_sum = torch.zeros_like(ao)
        
        for angle in angles:
            # Calculate offset
            dx = int(round(math.cos(angle) * radius))
            dy = int(round(math.sin(angle) * radius))
            
            # Skip if no offset
            if dx == 0 and dy == 0:
                continue
            
            # Sample height at offset position (with boundary handling)
            # Pad the height map to handle boundaries
            pad_h = abs(dy) if dy != 0 else 0
            pad_w = abs(dx) if dx != 0 else 0
            
            # Use reflection padding
            height_padded = torch.nn.functional.pad(
                height_map.permute(0, 3, 1, 2),  # BHWC -> BCHW
                (pad_w, pad_w, pad_h, pad_h),
                mode='replicate'
            ).permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Calculate the sampling positions
            y_start = max(0, dy) + pad_h
            y_end = y_start + h
            x_start = max(0, dx) + pad_w
            x_end = x_start + w
            
            # Center position
            center_y = pad_h
            center_x = pad_w
            
            # Sample neighbor and center
            neighbor_height = height_padded[:, y_start:y_end, x_start:x_end, :]
            center_height = height_padded[:, center_y:center_y+h, center_x:center_x+w, :]
            
            # Calculate height difference (positive if neighbor is higher = occlusion)
            height_diff = neighbor_height - center_height
            
            # Convert to occlusion (only positive differences contribute)
            occlusion = torch.clamp(height_diff, 0.0, 1.0)
            
            occlusion_sum += occlusion
        
        # Average the occlusion
        if samples > 0:
            occlusion_avg = occlusion_sum / samples
        else:
            occlusion_avg = occlusion_sum
        
        # Apply strength
        occlusion_avg = occlusion_avg * strength
        
        # Convert to AO (1.0 = no occlusion, 0.0 = full occlusion)
        ao = 1.0 - torch.clamp(occlusion_avg, 0.0, 1.0)
        
        # Apply contrast
        if contrast != 1.0:
            ao = (ao - 0.5) * contrast + 0.5
            ao = torch.clamp(ao, 0.0, 1.0)
        
        # Optional: Use normal map to bias AO
        if use_normal and normal is not None:
            print("✓ Applying normal-based bias")
            
            # Convert normal to grayscale or use blue channel (up direction)
            if normal.shape[-1] >= 3:
                # Use blue channel (Z/up direction)
                # Areas facing up get less AO, areas facing down/sideways get more
                normal_z = normal[..., 2:3]
                
                # Resize to match AO if needed
                if normal_z.shape[1:3] != ao.shape[1:3]:
                    normal_z = torch.nn.functional.interpolate(
                        normal_z.permute(0, 3, 1, 2),
                        size=ao.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Blend AO with normal bias
                # Faces pointing up (normal_z close to 1.0) = less AO
                # Faces pointing sideways/down (normal_z close to 0.5 or less) = more AO
                normal_bias = torch.clamp((1.0 - normal_z) * 0.5, 0.0, 0.5)
                ao = ao * (1.0 - normal_bias)
        
        # Convert to RGB
        ao_rgb = ao.repeat(1, 1, 1, 3)
        
        print(f"\n✓ AO generated")
        print(f"  Range: [{ao_rgb.min():.3f}, {ao_rgb.max():.3f}]")
        print("\n" + "="*60)
        print("✓ AO Approximation Complete")
        print("="*60 + "\n")
        
        return (ao_rgb,)


class PBRCombiner:
    """
    Combine all PBR maps into a single PBR_PIPE for easy workflow connections
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE",),
            },
            "optional": {
                "normal": ("IMAGE",),
                "ao": ("IMAGE",),
                "height": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "transparency": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "combine"
    CATEGORY = "PBR/Pipeline"
    
    def combine(self, albedo, normal=None, ao=None, height=None, roughness=None, metallic=None, transparency=None):
        """Combine PBR maps into a pipeline"""
        
        print("\n" + "="*60)
        print("PBR Combiner")
        print("="*60)
        print(f"Albedo: {albedo.shape}")
        if normal is not None:
            print(f"Normal: {normal.shape}")
        if ao is not None:
            print(f"AO: {ao.shape}")
        if height is not None:
            print(f"Height: {height.shape}")
        if roughness is not None:
            print(f"Roughness: {roughness.shape}")
        if metallic is not None:
            print(f"Metallic: {metallic.shape}")
        if transparency is not None:
            print(f"Transparency: {transparency.shape}")
        
        # Create PBR pipeline tuple
        pbr_pipe = {
            "albedo": albedo,
            "normal": normal,
            "ao": ao,
            "height": height,
            "roughness": roughness,
            "metallic": metallic,
            "transparency": transparency,
        }
        
        print("\n✓ PBR Pipeline created")
        print("="*60 + "\n")
        
        return (pbr_pipe,)


class PBRPipelineAdjuster:
    """
    Advanced PBR adjustment node using the PBR_PIPE
    Applies AO to albedo/roughness, adjusts all maps with various controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
                
                # AO controls
                "ao_strength_albedo": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Multiply AO over albedo (darkens occluded areas)"
                }),
                "ao_strength_roughness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Multiply AO over roughness (occluded areas become rougher)"
                }),
                
                # Roughness controls
                "roughness_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Brightness multiplier for roughness"
                }),
                
                # Metallic controls
                "metallic_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Brightness multiplier for metallic"
                }),
                
                # Normal controls
                "normal_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Normal map intensity (1.0=original, 0.0=flat, 2.0=exaggerated)"
                }),
                "invert_normal_green": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Flip green channel (OpenGL vs DirectX)"
                }),
                
                # Transparency controls
                "invert_transparency": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert transparency map"
                }),
                
                # Albedo controls
                "albedo_dimmer": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Darken albedo (0.0=no change, 1.0=black)"
                }),
                "albedo_saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Albedo color saturation (1.0=original, 0.0=grayscale, >1.0=vibrant)"
                }),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("pbr_pipe", "albedo", "normal", "ao", "height", "roughness", "metallic", "transparency")
    FUNCTION = "adjust"
    CATEGORY = "PBR/Pipeline"
    
    def apply_saturation(self, image, saturation):
        """Apply saturation adjustment to an image"""
        if saturation == 1.0:
            return image
        
        # Convert to grayscale
        weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                               device=image.device, dtype=image.dtype)
        gray = torch.sum(image * weights, dim=-1, keepdim=True)
        
        # Blend between grayscale and original based on saturation
        if saturation < 1.0:
            # Desaturate
            result = image * saturation + gray.repeat(1, 1, 1, 3) * (1.0 - saturation)
        else:
            # Oversaturate
            result = image + (image - gray.repeat(1, 1, 1, 3)) * (saturation - 1.0)
        
        return torch.clamp(result, 0.0, 1.0)
    
    def adjust_normal_strength(self, normal, strength):
        """Adjust normal map strength"""
        if strength == 1.0:
            return normal
        
        # Convert to [-1, 1] range
        normal_unpacked = normal * 2.0 - 1.0
        
        # Blend between flat normal (0,0,1) and the actual normal
        flat = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
        flat = flat.view(1, 1, 1, 3)
        
        # Interpolate
        normal_unpacked = normal_unpacked * strength + flat * (1.0 - strength)
        
        # Normalize
        length = torch.sqrt(torch.sum(normal_unpacked * normal_unpacked, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-6)
        normal_unpacked = normal_unpacked / length
        
        # Convert back to [0, 1]
        return normal_unpacked * 0.5 + 0.5
    
    def adjust(self, pbr_pipe, ao_strength_albedo, ao_strength_roughness, roughness_strength, 
               metallic_strength, normal_strength, invert_normal_green, invert_transparency,
               albedo_dimmer, albedo_saturation):
        """Apply adjustments to PBR pipeline"""
        
        print("\n" + "="*60)
        print("PBR Pipeline Adjuster")
        print("="*60)
        
        # Extract maps from pipe
        albedo = pbr_pipe["albedo"].clone() if pbr_pipe["albedo"] is not None else None
        normal = pbr_pipe["normal"].clone() if pbr_pipe["normal"] is not None else None
        ao = pbr_pipe["ao"].clone() if pbr_pipe["ao"] is not None else None
        height = pbr_pipe["height"].clone() if pbr_pipe["height"] is not None else None
        roughness = pbr_pipe["roughness"].clone() if pbr_pipe["roughness"] is not None else None
        metallic = pbr_pipe["metallic"].clone() if pbr_pipe["metallic"] is not None else None
        transparency = pbr_pipe["transparency"].clone() if pbr_pipe["transparency"] is not None else None
        
        # ===== ALBEDO ADJUSTMENTS =====
        if albedo is not None:
            # Apply saturation
            if albedo_saturation != 1.0:
                albedo = self.apply_saturation(albedo, albedo_saturation)
                print(f"✓ Albedo saturation: {albedo_saturation}")
            
            # Apply dimmer (darken)
            if albedo_dimmer > 0.0:
                albedo = albedo * (1.0 - albedo_dimmer)
                print(f"✓ Albedo dimmer: {albedo_dimmer}")
            
            # Apply AO to albedo
            if ao is not None and ao_strength_albedo > 0.0:
                # Convert AO to grayscale if needed
                ao_gray = ao
                if ao_gray.shape[-1] == 3:
                    weights = torch.tensor([0.299, 0.587, 0.114], 
                                          device=ao_gray.device, dtype=ao_gray.dtype)
                    ao_gray = torch.sum(ao_gray * weights, dim=-1, keepdim=True)
                    ao_gray = ao_gray.repeat(1, 1, 1, 3)
                
                # Resize AO to match albedo if needed
                if ao_gray.shape[1:3] != albedo.shape[1:3]:
                    ao_gray = torch.nn.functional.interpolate(
                        ao_gray.permute(0, 3, 1, 2),
                        size=albedo.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Apply AO: blend between original albedo and AO-multiplied albedo
                albedo_ao = albedo * ao_gray
                albedo = albedo * (1.0 - ao_strength_albedo) + albedo_ao * ao_strength_albedo
                albedo = torch.clamp(albedo, 0.0, 1.0)
                print(f"✓ AO applied to albedo: strength {ao_strength_albedo}")
            
            print(f"  Albedo range: [{albedo.min():.3f}, {albedo.max():.3f}]")
        
        # ===== ROUGHNESS ADJUSTMENTS =====
        if roughness is not None:
            # Apply strength (brightness)
            if roughness_strength != 1.0:
                roughness = roughness * roughness_strength
                roughness = torch.clamp(roughness, 0.0, 1.0)
                print(f"✓ Roughness strength: {roughness_strength}")
            
            # Apply AO to roughness
            if ao is not None and ao_strength_roughness > 0.0:
                # Convert AO to grayscale if needed
                ao_gray = ao
                if ao_gray.shape[-1] == 3:
                    weights = torch.tensor([0.299, 0.587, 0.114], 
                                          device=ao_gray.device, dtype=ao_gray.dtype)
                    ao_gray = torch.sum(ao_gray * weights, dim=-1, keepdim=True)
                    ao_gray = ao_gray.repeat(1, 1, 1, 3)
                
                # Resize AO to match roughness if needed
                if ao_gray.shape[1:3] != roughness.shape[1:3]:
                    ao_gray = torch.nn.functional.interpolate(
                        ao_gray.permute(0, 3, 1, 2),
                        size=roughness.shape[1:3],
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Invert AO for roughness (dark AO = more rough)
                ao_inverted = 1.0 - ao_gray
                roughness_ao = roughness + ao_inverted * ao_strength_roughness
                roughness = torch.clamp(roughness_ao, 0.0, 1.0)
                print(f"✓ AO applied to roughness: strength {ao_strength_roughness}")
            
            print(f"  Roughness range: [{roughness.min():.3f}, {roughness.max():.3f}]")
        
        # ===== METALLIC ADJUSTMENTS =====
        if metallic is not None:
            if metallic_strength != 1.0:
                metallic = metallic * metallic_strength
                metallic = torch.clamp(metallic, 0.0, 1.0)
                print(f"✓ Metallic strength: {metallic_strength}")
                print(f"  Metallic range: [{metallic.min():.3f}, {metallic.max():.3f}]")
        
        # ===== NORMAL ADJUSTMENTS =====
        if normal is not None:
            # Adjust strength
            if normal_strength != 1.0:
                normal = self.adjust_normal_strength(normal, normal_strength)
                print(f"✓ Normal strength: {normal_strength}")
            
            # Invert green channel
            if invert_normal_green and normal.shape[-1] >= 3:
                normal = normal.clone()
                normal[..., 1] = 1.0 - normal[..., 1]
                print("✓ Normal green channel inverted")
            
            print(f"  Normal range: [{normal.min():.3f}, {normal.max():.3f}]")
        
        # ===== TRANSPARENCY ADJUSTMENTS =====
        if transparency is not None:
            if invert_transparency:
                transparency = 1.0 - transparency
                print("✓ Transparency inverted")
                print(f"  Transparency range: [{transparency.min():.3f}, {transparency.max():.3f}]")
        
        # Create output pipe
        output_pipe = {
            "albedo": albedo,
            "normal": normal,
            "ao": ao,
            "height": height,
            "roughness": roughness,
            "metallic": metallic,
            "transparency": transparency,
        }
        
        print("\n" + "="*60)
        print("✓ Pipeline Adjustments Complete")
        print("="*60 + "\n")
        
        return (output_pipe, albedo, normal, ao, height, roughness, metallic, transparency)


class PBRSplitter:
    """
    Extract individual maps from a PBR_PIPE for saving or further processing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "normal", "ao", "height", "roughness", "metallic", "transparency")
    FUNCTION = "split"
    CATEGORY = "PBR/Pipeline"
    
    def split(self, pbr_pipe):
        """Extract maps from PBR pipeline"""
        
        print("\n" + "="*60)
        print("PBR Splitter")
        print("="*60)
        
        albedo = pbr_pipe.get("albedo", None)
        normal = pbr_pipe.get("normal", None)
        ao = pbr_pipe.get("ao", None)
        height = pbr_pipe.get("height", None)
        roughness = pbr_pipe.get("roughness", None)
        metallic = pbr_pipe.get("metallic", None)
        transparency = pbr_pipe.get("transparency", None)
        
        # Create placeholder for None values (ComfyUI needs something)
        def create_placeholder():
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        print("Extracted maps:")
        if albedo is not None:
            print(f"  ✓ Albedo: {albedo.shape}")
        else:
            albedo = create_placeholder()
            print(f"  ✗ Albedo: None (using placeholder)")
        
        if normal is not None:
            print(f"  ✓ Normal: {normal.shape}")
        else:
            normal = create_placeholder()
            print(f"  ✗ Normal: None (using placeholder)")
        
        if ao is not None:
            print(f"  ✓ AO: {ao.shape}")
        else:
            ao = create_placeholder()
            print(f"  ✗ AO: None (using placeholder)")
        
        if height is not None:
            print(f"  ✓ Height: {height.shape}")
        else:
            height = create_placeholder()
            print(f"  ✗ Height: None (using placeholder)")
        
        if roughness is not None:
            print(f"  ✓ Roughness: {roughness.shape}")
        else:
            roughness = create_placeholder()
            print(f"  ✗ Roughness: None (using placeholder)")
        
        if metallic is not None:
            print(f"  ✓ Metallic: {metallic.shape}")
        else:
            metallic = create_placeholder()
            print(f"  ✗ Metallic: None (using placeholder)")
        
        if transparency is not None:
            print(f"  ✓ Transparency: {transparency.shape}")
        else:
            transparency = create_placeholder()
            print(f"  ✗ Transparency: None (using placeholder)")
        
        print("\n✓ Maps extracted from pipeline")
        print("="*60 + "\n")
        
        return (albedo, normal, ao, height, roughness, metallic, transparency)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PBRExtractor": PBRExtractor,
    "PBRAdjuster": PBRAdjuster,
    "LotusHeightProcessor": LotusHeightProcessor,
    "AOApproximator": AOApproximator,
    "PBRCombiner": PBRCombiner,
    "PBRPipelineAdjuster": PBRPipelineAdjuster,
    "PBRSplitter": PBRSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PBRExtractor": "PBR Extractor (Marigold)",
    "PBRAdjuster": "PBR Adjuster",
    "LotusHeightProcessor": "Height Processor (Lotus)",
    "AOApproximator": "AO Approximator",
    "PBRCombiner": "PBR Combiner",
    "PBRPipelineAdjuster": "PBR Pipeline Adjuster",
    "PBRSplitter": "PBR Splitter",
}
