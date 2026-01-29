"""
Relight Utilities
Image relighting based on normal map dot product - inspired by Blender's Normal node
"""

import torch
import numpy as np


class NormalMapRelighter:
    """
    Relight an image using a normal map and light direction
    Uses dot product between normal vectors and light direction to calculate lighting
    Similar to Blender's Normal node shader
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE",),
                
                # Light direction (normalized vector)
                "light_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction X component"
                }),
                "light_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction Y component"
                }),
                "light_z": ("FLOAT", {
                    "default": 1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction Z component (pointing toward camera)"
                }),
                
                # Light properties
                "light_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Light intensity multiplier"
                }),
                
                "ambient_light": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ambient light level (minimum brightness)"
                }),
                
                # Normal map format
                "normal_format": (["OpenGL", "DirectX"], {
                    "default": "OpenGL",
                    "tooltip": "Normal map format (affects Y channel)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("relit_image", "lighting_mask")
    FUNCTION = "relight"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def decode_normal_map(self, normal_map: torch.Tensor, format: str) -> torch.Tensor:
        """
        Decode normal map from [0,1] color space to [-1,1] vector space
        """
        # Convert from 0-1 to -1 to 1
        normals = normal_map * 2.0 - 1.0
        
        # Flip Y for DirectX format
        if format == "DirectX":
            normals[:, :, :, 1] = -normals[:, :, :, 1]
        
        # Normalize the vectors
        length = torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        normals = normals / length
        
        return normals
    
    def calculate_lighting(self, normals: torch.Tensor, light_dir: torch.Tensor, 
                          intensity: float, ambient: float) -> torch.Tensor:
        """
        Calculate lighting using dot product between normals and light direction
        """
        # Normalize light direction
        light_length = torch.sqrt(torch.sum(light_dir ** 2))
        if light_length > 1e-8:
            light_dir = light_dir / light_length
        
        # Calculate dot product (N · L)
        # normals shape: [B, H, W, 3]
        # light_dir shape: [3]
        dot_product = torch.sum(normals * light_dir, dim=-1, keepdim=True)
        
        # Clamp dot product to [0, 1] (only front-facing surfaces are lit)
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        
        # Apply intensity and ambient light
        lighting = dot_product * intensity + ambient
        
        # Clamp final lighting
        lighting = torch.clamp(lighting, 0.0, 1.0)
        
        return lighting
    
    def relight(self, image, normal_map, light_x, light_y, light_z, 
               light_intensity, ambient_light, normal_format):
        """
        Relight image using normal map and light direction
        """
        
        print("\n" + "="*60)
        print("Normal Map Relighter")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Light direction: ({light_x:.3f}, {light_y:.3f}, {light_z:.3f})")
        print(f"Light intensity: {light_intensity}")
        print(f"Ambient light: {ambient_light}")
        print(f"Normal format: {normal_format}")
        
        # Decode normal map
        print(f"\n🗺️  Decoding normal map...")
        normals = self.decode_normal_map(normal_map, normal_format)
        print(f"  Normal range: [{normals.min():.3f}, {normals.max():.3f}]")
        
        # Create light direction vector
        light_dir = torch.tensor([light_x, light_y, light_z], 
                                device=image.device, dtype=image.dtype)
        
        # Calculate lighting
        print(f"\n💡 Calculating lighting...")
        lighting_mask = self.calculate_lighting(normals, light_dir, 
                                                light_intensity, ambient_light)
        print(f"  Lighting mask range: [{lighting_mask.min():.3f}, {lighting_mask.max():.3f}]")
        
        # Apply lighting to image
        # Expand lighting mask to match image channels
        if image.shape[-1] > 1:
            lighting_expanded = lighting_mask.repeat(1, 1, 1, image.shape[-1])
        else:
            lighting_expanded = lighting_mask
        
        relit_image = image * lighting_expanded
        
        # Convert lighting mask to RGB for output
        lighting_mask_rgb = lighting_mask.repeat(1, 1, 1, 3)
        
        print(f"\n✓ Relighting complete")
        print(f"  Relit image range: [{relit_image.min():.3f}, {relit_image.max():.3f}]")
        print("="*60 + "\n")
        
        return (relit_image, lighting_mask_rgb)


class NormalMapRelighterAdvanced:
    """
    Advanced relighting with multiple light types and properties
    Supports directional, point, and ambient lighting
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE",),
                
                # Light direction (normalized vector)
                "light_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction X component"
                }),
                "light_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction Y component"
                }),
                "light_z": ("FLOAT", {
                    "default": 1.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light direction Z component"
                }),
                
                # Light properties
                "diffuse_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Diffuse light intensity"
                }),
                
                "specular_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Specular highlight intensity"
                }),
                
                "specular_power": ("FLOAT", {
                    "default": 32.0,
                    "min": 1.0,
                    "max": 128.0,
                    "step": 1.0,
                    "tooltip": "Specular power (shininess)"
                }),
                
                "ambient_light": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ambient light level"
                }),
                
                # Light color
                "light_color_r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light color red channel"
                }),
                "light_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light color green channel"
                }),
                "light_color_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Light color blue channel"
                }),
                
                # Normal map format
                "normal_format": (["OpenGL", "DirectX"], {
                    "default": "OpenGL",
                    "tooltip": "Normal map format"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("relit_image", "diffuse_lighting", "specular_lighting")
    FUNCTION = "relight"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def decode_normal_map(self, normal_map: torch.Tensor, format: str) -> torch.Tensor:
        """Decode normal map from color space to vector space"""
        normals = normal_map * 2.0 - 1.0
        
        if format == "DirectX":
            normals[:, :, :, 1] = -normals[:, :, :, 1]
        
        length = torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        normals = normals / length
        
        return normals
    
    def calculate_diffuse(self, normals: torch.Tensor, light_dir: torch.Tensor, 
                         intensity: float) -> torch.Tensor:
        """Calculate diffuse lighting (Lambertian)"""
        dot_product = torch.sum(normals * light_dir, dim=-1, keepdim=True)
        dot_product = torch.clamp(dot_product, 0.0, 1.0)
        return dot_product * intensity
    
    def calculate_specular(self, normals: torch.Tensor, light_dir: torch.Tensor, 
                          view_dir: torch.Tensor, intensity: float, power: float) -> torch.Tensor:
        """Calculate specular lighting (Blinn-Phong)"""
        # Calculate halfway vector
        half_vector = light_dir + view_dir
        half_length = torch.sqrt(torch.sum(half_vector ** 2))
        if half_length > 1e-8:
            half_vector = half_vector / half_length
        
        # Calculate specular term
        spec_dot = torch.sum(normals * half_vector, dim=-1, keepdim=True)
        spec_dot = torch.clamp(spec_dot, 0.0, 1.0)
        specular = torch.pow(spec_dot, power) * intensity
        
        return specular
    
    def relight(self, image, normal_map, light_x, light_y, light_z,
               diffuse_intensity, specular_intensity, specular_power, ambient_light,
               light_color_r, light_color_g, light_color_b, normal_format):
        """Advanced relighting with diffuse and specular components"""
        
        print("\n" + "="*60)
        print("Advanced Normal Map Relighter")
        print("="*60)
        print(f"Light direction: ({light_x:.3f}, {light_y:.3f}, {light_z:.3f})")
        print(f"Light color: ({light_color_r:.3f}, {light_color_g:.3f}, {light_color_b:.3f})")
        
        # Decode normal map
        normals = self.decode_normal_map(normal_map, normal_format)
        
        # Normalize light and view directions
        light_dir = torch.tensor([light_x, light_y, light_z], 
                                device=image.device, dtype=image.dtype)
        light_length = torch.sqrt(torch.sum(light_dir ** 2))
        if light_length > 1e-8:
            light_dir = light_dir / light_length
        
        # View direction (camera looking at surface)
        view_dir = torch.tensor([0.0, 0.0, 1.0], 
                               device=image.device, dtype=image.dtype)
        
        # Calculate lighting components
        diffuse = self.calculate_diffuse(normals, light_dir, diffuse_intensity)
        specular = self.calculate_specular(normals, light_dir, view_dir, 
                                          specular_intensity, specular_power)
        
        # Create light color tensor
        light_color = torch.tensor([light_color_r, light_color_g, light_color_b], 
                                  device=image.device, dtype=image.dtype)
        
        # Combine lighting components
        total_lighting = diffuse + ambient_light
        total_lighting = torch.clamp(total_lighting, 0.0, 1.0)
        
        # Expand to RGB
        total_lighting_rgb = total_lighting.repeat(1, 1, 1, 3) * light_color
        specular_rgb = specular.repeat(1, 1, 1, 3) * light_color
        
        # Apply to image
        relit_image = image * total_lighting_rgb + specular_rgb
        relit_image = torch.clamp(relit_image, 0.0, 1.0)
        
        # Output masks
        diffuse_rgb = total_lighting.repeat(1, 1, 1, 3)
        specular_output = specular.repeat(1, 1, 1, 3)
        
        print(f"\n✓ Advanced relighting complete")
        print(f"  Diffuse range: [{diffuse.min():.3f}, {diffuse.max():.3f}]")
        print(f"  Specular range: [{specular.min():.3f}, {specular.max():.3f}]")
        print("="*60 + "\n")
        
        return (relit_image, diffuse_rgb, specular_output)


class NormalMapRelighterIBL:
    """
    Image-Based Lighting (IBL) using equirectangular environment maps
    Samples environment map colors based on normal directions and reflection vectors
    Perfect for realistic HDRI lighting
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "environment_map": ("IMAGE",),
                
                # Environment rotation
                "environment_rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Rotate environment map (degrees)"
                }),
                
                # Lighting properties
                "diffuse_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Diffuse lighting intensity"
                }),
                
                "specular_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Specular reflection intensity"
                }),
                
                "specular_power": ("FLOAT", {
                    "default": 32.0,
                    "min": 1.0,
                    "max": 128.0,
                    "step": 1.0,
                    "tooltip": "Specular power (blur amount - lower = blurrier)"
                }),
                
                "ambient_light": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ambient light level"
                }),
                
                # Normal map format
                "normal_format": (["OpenGL", "DirectX"], {
                    "default": "OpenGL",
                    "tooltip": "Normal map format"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("relit_image", "diffuse_lighting", "specular_lighting")
    FUNCTION = "relight_ibl"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def decode_normal_map(self, normal_map: torch.Tensor, format: str) -> torch.Tensor:
        """Decode normal map from color space to vector space"""
        normals = normal_map * 2.0 - 1.0
        
        if format == "DirectX":
            normals[:, :, :, 1] = -normals[:, :, :, 1]
        
        length = torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        normals = normals / length
        
        return normals
    
    def direction_to_equirect_uv(self, direction: torch.Tensor, rotation_radians: float) -> torch.Tensor:
        """
        Convert 3D direction vector to equirectangular UV coordinates
        direction: [B, H, W, 3] normalized vectors
        returns: [B, H, W, 2] UV coordinates in range [0, 1]
        """
        x = direction[:, :, :, 0]
        y = direction[:, :, :, 1]
        z = direction[:, :, :, 2]
        
        # Calculate latitude (theta) and longitude (phi)
        # theta = arcsin(y) - vertical angle
        # phi = atan2(x, z) - horizontal angle
        
        theta = torch.asin(torch.clamp(y, -1.0, 1.0))
        phi = torch.atan2(x, z)
        
        # Apply rotation to phi
        phi = phi + rotation_radians
        
        # Convert to UV coordinates [0, 1]
        u = (phi / (2 * np.pi)) + 0.5
        v = (theta / np.pi) + 0.5
        
        # Wrap U coordinate
        u = u % 1.0
        
        # Clamp V coordinate
        v = torch.clamp(v, 0.0, 1.0)
        
        # Stack to create UV tensor
        uv = torch.stack([u, v], dim=-1)
        
        return uv
    
    def sample_equirect(self, env_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """
        Sample equirectangular map using UV coordinates with bilinear interpolation
        env_map: [B, H, W, C]
        uv: [B, H, W, 2]
        returns: [B, H, W, C]
        """
        batch_env, height_env, width_env, channels = env_map.shape
        batch_uv, height_uv, width_uv, _ = uv.shape
        
        # Convert UV to pixel coordinates
        # U maps to width, V maps to height
        px = uv[:, :, :, 0] * (width_env - 1)
        py = uv[:, :, :, 1] * (height_env - 1)
        
        # Bilinear interpolation
        px0 = torch.floor(px).long()
        py0 = torch.floor(py).long()
        px1 = px0 + 1
        py1 = py0 + 1
        
        # Clamp coordinates
        px0 = torch.clamp(px0, 0, width_env - 1)
        px1 = torch.clamp(px1, 0, width_env - 1)
        py0 = torch.clamp(py0, 0, height_env - 1)
        py1 = torch.clamp(py1, 0, height_env - 1)
        
        # Get fractional parts
        fx = px - px0.float()
        fy = py - py0.float()
        
        # Expand dimensions for broadcasting with channels
        fx = fx.unsqueeze(-1)  # [B, H, W, 1]
        fy = fy.unsqueeze(-1)  # [B, H, W, 1]
        
        # Sample four corners using advanced indexing
        # We need to create batch indices that match the spatial dimensions
        batch_idx = torch.arange(batch_env, device=env_map.device).view(batch_env, 1, 1).expand(batch_uv, height_uv, width_uv)
        
        # Sample all four corners
        c00 = env_map[batch_idx, py0, px0]  # [B, H, W, C]
        c01 = env_map[batch_idx, py0, px1]  # [B, H, W, C]
        c10 = env_map[batch_idx, py1, px0]  # [B, H, W, C]
        c11 = env_map[batch_idx, py1, px1]  # [B, H, W, C]
        
        # Bilinear interpolation
        c0 = c00 * (1 - fx) + c01 * fx
        c1 = c10 * (1 - fx) + c11 * fx
        result = c0 * (1 - fy) + c1 * fy
        
        return result
    
    def calculate_reflection(self, normals: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        """
        Calculate reflection vector: R = 2(N·V)N - V
        normals: [B, H, W, 3]
        view_dir: [3] viewing direction
        returns: [B, H, W, 3] reflection vectors
        """
        # Calculate N·V
        dot_nv = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        
        # R = 2(N·V)N - V
        reflection = 2.0 * dot_nv * normals - view_dir
        
        # Normalize
        length = torch.sqrt(torch.sum(reflection ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        reflection = reflection / length
        
        return reflection
    
    def blur_environment_sample(self, env_map: torch.Tensor, uv: torch.Tensor, 
                                blur_amount: float) -> torch.Tensor:
        """
        Sample environment map with simple blur for roughness simulation
        Lower specular_power = more blur
        """
        # For now, just sample directly
        # In a more advanced version, could implement mipmap-based sampling
        result = self.sample_equirect(env_map, uv)
        
        # Simple blur approximation by averaging nearby samples
        if blur_amount > 1.0:
            # This is a simplified version - proper IBL would use prefiltered mipmaps
            pass
        
        return result
    
    def relight_ibl(self, image, normal_map, environment_map, environment_rotation,
                   diffuse_intensity, specular_intensity, specular_power, ambient_light,
                   normal_format):
        """
        Relight image using Image-Based Lighting with equirectangular environment map
        """
        
        print("\n" + "="*60)
        print("Image-Based Lighting (IBL) Relighter")
        print("="*60)
        print(f"Image shape: {image.shape}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Environment map shape: {environment_map.shape}")
        print(f"Environment rotation: {environment_rotation}°")
        
        # Convert rotation to radians
        rotation_rad = environment_rotation * (np.pi / 180.0)
        
        # Decode normal map
        print(f"\n🗺️  Decoding normal map...")
        normals = self.decode_normal_map(normal_map, normal_format)
        
        # View direction (camera looking at surface)
        view_dir = torch.tensor([0.0, 0.0, 1.0], 
                               device=image.device, dtype=image.dtype)
        
        # Calculate diffuse lighting from environment
        print(f"\n💡 Calculating diffuse lighting from environment...")
        diffuse_uv = self.direction_to_equirect_uv(normals, rotation_rad)
        diffuse_colors = self.sample_equirect(environment_map, diffuse_uv)
        
        # Apply diffuse intensity
        diffuse_lighting = diffuse_colors * diffuse_intensity + ambient_light
        diffuse_lighting = torch.clamp(diffuse_lighting, 0.0, 1.0)
        
        print(f"  Diffuse range: [{diffuse_lighting.min():.3f}, {diffuse_lighting.max():.3f}]")
        
        # Calculate specular lighting from reflections
        print(f"\n✨ Calculating specular reflections...")
        reflection_vectors = self.calculate_reflection(normals, view_dir)
        specular_uv = self.direction_to_equirect_uv(reflection_vectors, rotation_rad)
        
        # Sample with blur based on specular power (inverse relationship)
        blur_factor = 128.0 / max(specular_power, 1.0)
        specular_colors = self.blur_environment_sample(environment_map, specular_uv, blur_factor)
        
        # Calculate Fresnel-like falloff for specular
        view_dot_normal = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        view_dot_normal = torch.clamp(view_dot_normal, 0.0, 1.0)
        
        # Simple Fresnel approximation: more reflection at grazing angles
        fresnel = torch.pow(1.0 - view_dot_normal, 5.0)
        fresnel = fresnel * 0.5 + 0.5  # Soften the effect
        
        specular_lighting = specular_colors * specular_intensity * fresnel
        specular_lighting = torch.clamp(specular_lighting, 0.0, 1.0)
        
        print(f"  Specular range: [{specular_lighting.min():.3f}, {specular_lighting.max():.3f}]")
        
        # Combine diffuse and specular
        relit_image = image * diffuse_lighting + specular_lighting
        relit_image = torch.clamp(relit_image, 0.0, 1.0)
        
        print(f"\n✓ IBL relighting complete")
        print(f"  Final image range: [{relit_image.min():.3f}, {relit_image.max():.3f}]")
        print("="*60 + "\n")
        
        return (relit_image, diffuse_lighting, specular_lighting)


class NormalMapRelighterIBL_PBR:
    """
    PBR-aware Image-Based Lighting with roughness and metallic control
    Uses PBR material properties for physically accurate relighting
    Non-metallic surfaces won't look metallic!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffuse_image": ("IMAGE", {"tooltip": "Albedo/diffuse color map"}),
                "normal_map": ("IMAGE",),
                "environment_map": ("IMAGE",),
                
                # Environment rotation
                "environment_rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Rotate environment map (degrees)"
                }),
                
                # Lighting properties
                "diffuse_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Diffuse lighting intensity"
                }),
                
                "specular_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Specular reflection intensity"
                }),
                
                "ambient_light": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Ambient light level"
                }),
                
                # Roughness control
                "roughness_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Global roughness multiplier (affects both diffuse and specular blur)"
                }),
                
                "roughness_affects_diffuse": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much roughness affects diffuse detail (0=uniform, 1=fully controlled by roughness)"
                }),
                
                "diffuse_contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Diffuse softness control (0=soft/compressed highlights, 1=normal, 2=enhanced contrast). Lower values reduce metallic look"
                }),
                
                # Diffuse color control
                "use_colored_diffuse": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use environment colors for diffuse (False = grayscale shading only)"
                }),
                
                "diffuse_quality": (["Flat Ambient (No Direction)",
                                    "Pure Matte (No Environment)", 
                                    "Soft (Heavy Blur)", 
                                    "Detailed (Light Blur)",
                                    "Hybrid (Lambert + Env Colors)",
                                    "True Hemisphere Sampling"], {
                    "default": "Flat Ambient (No Direction)",
                    "tooltip": "Diffuse quality - Flat Ambient = pure even lighting (no reflection look)"
                }),
                
                # Normal map format
                "normal_format": (["OpenGL", "DirectX"], {
                    "default": "OpenGL",
                    "tooltip": "Normal map format"
                }),
            },
            "optional": {
                "roughness_map": ("IMAGE",),
                "metallic_map": ("IMAGE",),
                "irradiance_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("relit_image", "diffuse_lighting", "specular_lighting", "diffuse_color", "key_light_direction")
    FUNCTION = "relight_ibl_pbr"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def decode_normal_map(self, normal_map: torch.Tensor, format: str) -> torch.Tensor:
        """Decode normal map from color space to vector space"""
        normals = normal_map * 2.0 - 1.0
        
        if format == "DirectX":
            normals[:, :, :, 1] = -normals[:, :, :, 1]
        
        length = torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        normals = normals / length
        
        return normals
    
    def prepare_pbr_map(self, pbr_map: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Convert PBR map to grayscale and match target shape"""
        if pbr_map.shape[-1] > 1:
            # Convert to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], device=pbr_map.device, dtype=pbr_map.dtype)
            pbr_map = torch.sum(pbr_map[..., :3] * weights, dim=-1, keepdim=True)
        
        # Expand to match target channels if needed
        if pbr_map.shape[-1] != target_shape[-1]:
            pbr_map = pbr_map.repeat(1, 1, 1, target_shape[-1])
        
        return pbr_map
    
    def direction_to_equirect_uv(self, direction: torch.Tensor, rotation_radians: float) -> torch.Tensor:
        """Convert 3D direction vector to equirectangular UV coordinates"""
        x = direction[:, :, :, 0]
        y = direction[:, :, :, 1]
        z = direction[:, :, :, 2]
        
        theta = torch.asin(torch.clamp(y, -1.0, 1.0))
        phi = torch.atan2(x, z)
        phi = phi + rotation_radians
        
        u = (phi / (2 * np.pi)) + 0.5
        v = (theta / np.pi) + 0.5
        
        u = u % 1.0
        v = torch.clamp(v, 0.0, 1.0)
        
        uv = torch.stack([u, v], dim=-1)
        return uv
    
    def sample_equirect(self, env_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Sample equirectangular map using UV coordinates with bilinear interpolation"""
        batch_env, height_env, width_env, channels = env_map.shape
        batch_uv, height_uv, width_uv, _ = uv.shape
        
        px = uv[:, :, :, 0] * (width_env - 1)
        py = uv[:, :, :, 1] * (height_env - 1)
        
        px0 = torch.floor(px).long()
        py0 = torch.floor(py).long()
        px1 = px0 + 1
        py1 = py0 + 1
        
        px0 = torch.clamp(px0, 0, width_env - 1)
        px1 = torch.clamp(px1, 0, width_env - 1)
        py0 = torch.clamp(py0, 0, height_env - 1)
        py1 = torch.clamp(py1, 0, height_env - 1)
        
        fx = px - px0.float()
        fy = py - py0.float()
        fx = fx.unsqueeze(-1)
        fy = fy.unsqueeze(-1)
        
        batch_idx = torch.arange(batch_env, device=env_map.device).view(batch_env, 1, 1).expand(batch_uv, height_uv, width_uv)
        
        c00 = env_map[batch_idx, py0, px0]
        c01 = env_map[batch_idx, py0, px1]
        c10 = env_map[batch_idx, py1, px0]
        c11 = env_map[batch_idx, py1, px1]
        
        c0 = c00 * (1 - fx) + c01 * fx
        c1 = c10 * (1 - fx) + c11 * fx
        result = c0 * (1 - fy) + c1 * fy
        
        return result
    
    def calculate_reflection(self, normals: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        """Calculate reflection vector: R = 2(N·V)N - V"""
        dot_nv = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        reflection = 2.0 * dot_nv * normals - view_dir
        
        length = torch.sqrt(torch.sum(reflection ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        reflection = reflection / length
        
        return reflection
    
    def calculate_fresnel_schlick(self, view_dot_normal: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Fresnel-Schlick approximation
        f0: base reflectivity (0.04 for dielectrics, albedo for metals)
        """
        return f0 + (1.0 - f0) * torch.pow(1.0 - view_dot_normal, 5.0)
    
    def sample_equirect_blurred(self, env_map: torch.Tensor, uv: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """
        Sample environment map with blur based on roughness
        roughness: [B, H, W, C] where 0=smooth, 1=rough
        """
        # Convert roughness to average grayscale
        roughness_gray = torch.mean(roughness, dim=-1, keepdim=True)  # [B, H, W, 1]
        
        # For each pixel, sample multiple nearby UV coordinates based on roughness
        # This simulates blurred reflections for rough surfaces
        
        batch_uv, height_uv, width_uv, _ = uv.shape
        
        # Sample center point
        center_sample = self.sample_equirect(env_map, uv)
        
        # For rough surfaces, sample additional points in a pattern
        # Roughness determines blur radius
        blur_radius = roughness_gray * 0.05  # Max 5% of image size blur
        
        # Create offset pattern (8 samples around center)
        offsets = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)
        ]
        
        # Accumulate samples
        result = center_sample
        weight = 1.0
        
        for ox, oy in offsets:
            # Create offset UV
            offset_uv = uv.clone()
            offset_uv[:, :, :, 0] += ox * blur_radius.squeeze(-1)
            offset_uv[:, :, :, 1] += oy * blur_radius.squeeze(-1)
            
            # Wrap/clamp UV
            offset_uv[:, :, :, 0] = offset_uv[:, :, :, 0] % 1.0
            offset_uv[:, :, :, 1] = torch.clamp(offset_uv[:, :, :, 1], 0.0, 1.0)
            
            # Sample and accumulate
            sample = self.sample_equirect(env_map, offset_uv)
            sample_weight = roughness_gray  # Weight by roughness
            result += sample * sample_weight
            weight += sample_weight
        
        # Average all samples
        result = result / weight
        
        return result
    
    def relight_ibl_pbr(self, diffuse_image, normal_map, environment_map, environment_rotation,
                       diffuse_intensity, specular_intensity, ambient_light,
                       roughness_multiplier, roughness_affects_diffuse, diffuse_contrast, use_colored_diffuse, diffuse_quality, 
                       normal_format, roughness_map=None, metallic_map=None, irradiance_map=None):
        """
        PBR-aware IBL relighting with roughness and metallic control
        """
        
        print("\n" + "="*60)
        print("PBR Image-Based Lighting Relighter")
        print("="*60)
        print(f"Diffuse image shape: {diffuse_image.shape}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Environment map shape: {environment_map.shape}")
        
        # Convert rotation to radians
        rotation_rad = environment_rotation * (np.pi / 180.0)
        
        # Decode normal map
        normals = self.decode_normal_map(normal_map, normal_format)
        
        # Process PBR maps
        if roughness_map is not None:
            roughness = self.prepare_pbr_map(roughness_map, diffuse_image.shape)
            print(f"Roughness map: [{roughness.min():.3f}, {roughness.max():.3f}]")
        else:
            # Default roughness (semi-glossy)
            roughness = torch.ones_like(diffuse_image) * 0.5
            print(f"Roughness map: Using default (0.5)")
        
        # Apply roughness multiplier
        roughness = roughness * roughness_multiplier
        roughness = torch.clamp(roughness, 0.0, 1.0)
        print(f"Roughness (after multiplier {roughness_multiplier}): [{roughness.min():.3f}, {roughness.max():.3f}]")
        
        if metallic_map is not None:
            metallic = self.prepare_pbr_map(metallic_map, diffuse_image.shape)
            print(f"Metallic map: [{metallic.min():.3f}, {metallic.max():.3f}]")
        else:
            # Default non-metallic
            metallic = torch.zeros_like(diffuse_image)
            print(f"Metallic map: Using default (0.0 - non-metallic)")
        
        # View direction
        view_dir = torch.tensor([0.0, 0.0, 1.0], 
                               device=diffuse_image.device, dtype=diffuse_image.dtype)
        
        # Calculate diffuse lighting from environment
        print(f"\n💡 Calculating PBR diffuse lighting...")
        print(f"  Diffuse quality mode: {diffuse_quality}")
        print(f"  Roughness affects diffuse: {roughness_affects_diffuse:.2f}")
        
        # Check if we have a pre-computed irradiance map
        if irradiance_map is not None:
            print(f"  Using provided irradiance map (pre-computed)")
            diffuse_uv = self.direction_to_equirect_uv(normals, rotation_rad)
            
            # Even with irradiance, we can apply additional roughness-based blur
            if roughness_affects_diffuse > 0.0:
                # Base irradiance + roughness-modulated extra blur
                base_blur = roughness * roughness_affects_diffuse * 0.3  # 0 to 0.3 extra blur
                diffuse_colors = self.sample_equirect_blurred(irradiance_map, diffuse_uv, base_blur)
                print(f"  Applied roughness-based blur to irradiance (range: [{base_blur.min():.3f}, {base_blur.max():.3f}])")
            else:
                diffuse_colors = self.sample_equirect(irradiance_map, diffuse_uv)
        
        elif diffuse_quality == "Flat Ambient (No Direction)":
            # PURE AMBIENT: No directional lighting at all, just average environment color
            # This ensures ZERO reflection/refraction look - pure flat lighting
            print(f"  🎨 Using Flat Ambient mode (pure non-directional)")
            
            # Get average environment color
            avg_env_color = torch.mean(environment_map, dim=(1, 2), keepdim=True)
            
            # Broadcast to image dimensions (same color everywhere)
            batch, height, width, _ = normals.shape
            diffuse_colors = avg_env_color.expand(batch, height, width, 3)
            
            print(f"    Average environment color: RGB({avg_env_color[0,0,0,0]:.3f}, {avg_env_color[0,0,0,1]:.3f}, {avg_env_color[0,0,0,2]:.3f})")
            print(f"    Result: Completely flat/even lighting (no directionality)")
            
        elif diffuse_quality == "Pure Matte (No Environment)":
            # Pure matte diffuse - simple directional lighting with no environment sampling
            light_dir = torch.tensor([0.2, 0.3, 0.9], device=diffuse_image.device, dtype=diffuse_image.dtype)
            light_dir = light_dir / torch.sqrt(torch.sum(light_dir ** 2))
            
            # Calculate N·L (Lambert)
            dot_nl = torch.sum(normals * light_dir, dim=-1, keepdim=True)
            dot_nl = torch.clamp(dot_nl, 0.0, 1.0)
            
            # Pure matte shading (grayscale)
            diffuse_colors = dot_nl.repeat(1, 1, 1, 3)
            print(f"  Using pure matte shading (no environment, clay-like)")
            
        elif diffuse_quality == "Hybrid (Lambert + Env Colors)":
            # SWITCHLIGHT-STYLE: Lambert shading with environment colors
            # This creates pure diffuse shading without looking metallic
            print(f"  🎬 Using Hybrid mode (Switchlight-style)")
            
            # Step 1: Calculate pure Lambert shading (black & white)
            # Use top hemisphere average as light direction
            light_dir = torch.tensor([0.0, 0.0, 1.0], device=diffuse_image.device, dtype=diffuse_image.dtype)
            dot_nl = torch.sum(normals * light_dir, dim=-1, keepdim=True)
            dot_nl = torch.clamp(dot_nl, 0.0, 1.0)
            
            print(f"    Lambert shading range: [{dot_nl.min():.3f}, {dot_nl.max():.3f}]")
            
            # Step 2: If colored diffuse is enabled, map to environment colors
            if use_colored_diffuse:
                # Get environment colors (using mean-based approach, not extreme quantiles)
                # Convert environment to luminance
                env_lum_weights = torch.tensor([0.299, 0.587, 0.114], device=environment_map.device, dtype=environment_map.dtype)
                env_luminance = torch.sum(environment_map * env_lum_weights, dim=-1, keepdim=True)
                
                # Find dark and light colors (less extreme than 10th/90th percentile)
                dark_color = torch.quantile(environment_map.flatten(0, 2), 0.2, dim=0, keepdim=True)  # 20th percentile (less extreme)
                light_color = torch.quantile(environment_map.flatten(0, 2), 0.8, dim=0, keepdim=True)  # 80th percentile (less extreme)
                
                # Reshape for broadcasting
                dark_color = dark_color.reshape(1, 1, 1, 3)
                light_color = light_color.reshape(1, 1, 1, 3)
                
                # Reduce color intensity to avoid looking specular
                # Mix with grayscale to reduce saturation
                avg_color = (dark_color + light_color) / 2.0
                dark_color = dark_color * 0.5 + avg_color * 0.5  # 50% desaturate
                light_color = light_color * 0.5 + avg_color * 0.5
                
                # Map Lambert shading to color range
                diffuse_colors = dark_color + (light_color - dark_color) * dot_nl
                
                print(f"    Dark color: [{dark_color[0,0,0,0]:.3f}, {dark_color[0,0,0,1]:.3f}, {dark_color[0,0,0,2]:.3f}]")
                print(f"    Light color: [{light_color[0,0,0,0]:.3f}, {light_color[0,0,0,1]:.3f}, {light_color[0,0,0,2]:.3f}]")
            else:
                # Pure grayscale Lambert (no color from environment)
                diffuse_colors = dot_nl.repeat(1, 1, 1, 3)
                print(f"    Using pure grayscale Lambert (no environment color)")
            
        elif diffuse_quality == "True Hemisphere Sampling":
            # TRUE PHYSICALLY-BASED: Integrate over hemisphere
            # This is the most accurate but also slowest
            print(f"  🌐 Using True Hemisphere Sampling (slow but accurate)")
            
            batch, height, width, _ = normals.shape
            num_samples = 64  # Can be adjusted for quality vs speed tradeoff
            
            # Initialize accumulator
            diffuse_colors = torch.zeros_like(diffuse_image)
            total_weight = torch.zeros_like(diffuse_image[:, :, :, 0:1])  # Track total weights
            
            print(f"    Sampling {num_samples} directions per pixel...")
            
            # Generate sample directions using Fibonacci sphere (hemisphere only)
            indices = torch.arange(0, num_samples, dtype=torch.float32, device=diffuse_image.device)
            phi = torch.acos(1 - (indices + 0.5) / num_samples)  # Only upper hemisphere (0 to π/2)
            theta = np.pi * (1 + 5**0.5) * indices
            
            # Convert to Cartesian coordinates
            sample_dirs = torch.stack([
                torch.sin(phi) * torch.cos(theta),
                torch.sin(phi) * torch.sin(theta),
                torch.cos(phi)  # z always positive (hemisphere)
            ], dim=-1)  # [num_samples, 3]
            
            # For each sample direction
            for i in range(num_samples):
                sample_dir = sample_dirs[i:i+1]  # [1, 3]
                
                # Calculate N·L for this sample
                dot_nl = torch.sum(normals * sample_dir, dim=-1, keepdim=True)
                dot_nl = torch.clamp(dot_nl, 0.0, 1.0)
                
                # Convert direction to UV coordinates
                # Expand sample_dir to match batch dimensions
                sample_dir_expanded = sample_dir.expand(batch, height, width, 3)
                sample_uv = self.direction_to_equirect_uv(sample_dir_expanded, rotation_rad)
                
                # Sample environment
                env_sample = self.sample_equirect(environment_map, sample_uv)
                
                # Accumulate: weight by N·L (Lambert's cosine law)
                diffuse_colors += env_sample * dot_nl
                total_weight += dot_nl
                
                if (i + 1) % 16 == 0:
                    progress = (i + 1) / num_samples * 100
                    print(f"    Progress: {progress:.0f}%")
            
            # Average by total weight (proper Monte Carlo integration)
            # Avoid division by zero
            total_weight = torch.clamp(total_weight, min=1e-8)
            diffuse_colors = diffuse_colors / total_weight
            
            print(f"    Hemisphere sampling complete!")
            print(f"    Diffuse range: [{diffuse_colors.min():.3f}, {diffuse_colors.max():.3f}]")
            
        elif diffuse_quality == "Soft (Heavy Blur)":
            # Heavy environment blur for soft diffuse, modulated by roughness
            diffuse_uv = self.direction_to_equirect_uv(normals, rotation_rad)
            # Base: 0.95, add roughness influence
            base_blur = 0.95
            roughness_influence = roughness * roughness_affects_diffuse * 0.05
            diffuse_blur = base_blur + roughness_influence
            diffuse_colors = self.sample_equirect_blurred(environment_map, diffuse_uv, diffuse_blur)
            print(f"  Using heavy blur with roughness modulation")
            print(f"  Blur range: [{diffuse_blur.min():.3f}, {diffuse_blur.max():.3f}]")
            
        else:  # "Detailed (Light Blur)"
            # Lighter blur - more environment detail, strongly affected by roughness
            diffuse_uv = self.direction_to_equirect_uv(normals, rotation_rad)
            # Base: 0.5-0.7, roughness adds 0 to 0.3
            base_blur = 0.5 + (1.0 - roughness_affects_diffuse) * 0.2
            roughness_influence = roughness * roughness_affects_diffuse * 0.3
            diffuse_blur = base_blur + roughness_influence
            diffuse_colors = self.sample_equirect_blurred(environment_map, diffuse_uv, diffuse_blur)
            print(f"  Using detailed blur with roughness control")
            print(f"  Blur range: [{diffuse_blur.min():.3f}, {diffuse_blur.max():.3f}]")
        
        # Convert diffuse to grayscale (luminance) for non-colored shading
        # Colored lighting looks metallic, grayscale looks like proper shading
        # NOTE: Hybrid mode handles this internally, so skip for Hybrid
        if not use_colored_diffuse and diffuse_quality != "Hybrid (Lambert + Env Colors)":
            # Convert to luminance for natural light/shadow appearance
            weights = torch.tensor([0.299, 0.587, 0.114], device=diffuse_colors.device, dtype=diffuse_colors.dtype)
            diffuse_luminance = torch.sum(diffuse_colors * weights, dim=-1, keepdim=True)
            diffuse_colors = diffuse_luminance.repeat(1, 1, 1, 3)
            print(f"  Converting to grayscale (natural light/shadow appearance)")
        elif diffuse_quality != "Hybrid (Lambert + Env Colors)":
            print(f"  Using colored diffuse (environment colors)")
        
        # Apply diffuse contrast control to reduce metallic look
        # Use a soft compression curve instead of linear blend to grey
        if diffuse_contrast != 1.0:
            if diffuse_contrast < 1.0:
                # Compress highlights using a soft curve (reduces metallic look)
                # This applies a power curve that compresses bright values while preserving gradients
                # contrast = 0 → strong compression (very matte)
                # contrast = 1 → no change
                power = 1.0 + (1.0 - diffuse_contrast) * 2.0  # Range: 1.0 to 3.0
                diffuse_colors = torch.pow(diffuse_colors, 1.0 / power)
                
                print(f"  Applied highlight compression: {diffuse_contrast:.2f} (power: {power:.2f})")
            else:
                # Enhance contrast (values > 1.0)
                # Apply inverse power curve to increase contrast
                power = 1.0 / diffuse_contrast
                diffuse_colors = torch.pow(diffuse_colors, power)
                
                print(f"  Enhanced contrast: {diffuse_contrast:.2f} (power: {power:.2f})")
            
            diffuse_colors = torch.clamp(diffuse_colors, 0.0, 1.0)
            print(f"  Diffuse range after adjustment: [{diffuse_colors.min():.3f}, {diffuse_colors.max():.3f}]")
        
        # Calculate ambient light as average environment color (GI-style)
        # Use irradiance map if available, otherwise average the environment
        if irradiance_map is not None and ambient_light > 0.0:
            # Average the irradiance map for ambient
            ambient_color = torch.mean(irradiance_map, dim=(1, 2), keepdim=True)
            print(f"  Ambient from irradiance: RGB({ambient_color[0,0,0,0]:.3f}, {ambient_color[0,0,0,1]:.3f}, {ambient_color[0,0,0,2]:.3f})")
        elif ambient_light > 0.0:
            # Average the environment map for ambient (GI-style)
            ambient_color = torch.mean(environment_map, dim=(1, 2), keepdim=True)
            print(f"  Ambient from environment avg: RGB({ambient_color[0,0,0,0]:.3f}, {ambient_color[0,0,0,1]:.3f}, {ambient_color[0,0,0,2]:.3f})")
        else:
            # No ambient
            ambient_color = torch.zeros_like(diffuse_image[:, 0:1, 0:1, :])
            print(f"  No ambient light")
        
        # Apply ambient light intensity
        ambient_contribution = ambient_color * ambient_light
        
        # Create pure diffuse shading (without metallic masking) for output preview
        diffuse_shading_pure = diffuse_colors * diffuse_intensity + ambient_contribution
        diffuse_shading_pure = torch.clamp(diffuse_shading_pure, 0.0, 1.0)
        
        # Create diffuse for actual lighting calculation (with metallic masking)
        # In PBR, metals don't have diffuse reflection
        diffuse_for_lighting = diffuse_colors * diffuse_intensity * (1.0 - metallic) + ambient_contribution
        diffuse_for_lighting = torch.clamp(diffuse_for_lighting, 0.0, 1.0)
        
        print(f"  Final diffuse shading range: [{diffuse_shading_pure.min():.3f}, {diffuse_shading_pure.max():.3f}]")
        
        # Calculate specular reflections
        print(f"\n✨ Calculating PBR specular reflections...")
        reflection_vectors = self.calculate_reflection(normals, view_dir)
        specular_uv = self.direction_to_equirect_uv(reflection_vectors, rotation_rad)
        
        # Sample environment with roughness-based blur
        # Black roughness (0) = sharp reflections, White roughness (1) = blurred reflections
        specular_colors = self.sample_equirect_blurred(environment_map, specular_uv, roughness)
        print(f"  Applied roughness-based blur to specular reflections")
        
        # Calculate Fresnel
        view_dot_normal = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        view_dot_normal = torch.clamp(view_dot_normal, 0.0, 1.0)
        
        # F0: Base reflectivity
        # Dielectrics: ~0.04 (4% reflection)
        # Metals: Use albedo color
        f0_dielectric = torch.ones_like(diffuse_image) * 0.04
        f0_metal = diffuse_image  # Metals reflect colored light
        f0 = f0_dielectric * (1.0 - metallic) + f0_metal * metallic
        
        # Apply Fresnel
        fresnel = self.calculate_fresnel_schlick(view_dot_normal, f0)
        
        # Specular intensity: Metals get more specular, non-metals less
        # White metallic (1) = full specular, Black metallic (0) = minimal specular
        metal_boost = 1.0 + metallic * 2.0  # Metals get 3x boost
        
        specular_lighting = specular_colors * specular_intensity * fresnel * metal_boost
        specular_lighting = torch.clamp(specular_lighting, 0.0, 1.0)
        
        print(f"  Specular range: [{specular_lighting.min():.3f}, {specular_lighting.max():.3f}]")
        print(f"  Fresnel applied: grazing angles get more reflection")
        print(f"  Roughness applied: white (rough) = blurred, black (smooth) = sharp")
        print(f"  Metallic applied: white (metal) = strong colored reflection, black (non-metal) = weak")
        
        # Combine lighting
        # For metals: specular is the main component (colored by albedo)
        # For dielectrics: diffuse is main, specular is additive
        relit_image = diffuse_image * diffuse_for_lighting + specular_lighting
        relit_image = torch.clamp(relit_image, 0.0, 1.0)
        
        # Create additional outputs
        print(f"\n📤 Creating additional outputs...")
        
        # Diffuse Color: Input albedo with ONLY ambient/GI applied (no directional diffuse, no specular)
        # This is: Albedo × Ambient GI only - for Switchlight-style compositing
        diffuse_color = diffuse_image * ambient_contribution
        diffuse_color = torch.clamp(diffuse_color, 0.0, 1.0)
        print(f"  Diffuse Color: Albedo × Ambient GI only (no directional lighting)")
        print(f"    Range: [{diffuse_color.min():.3f}, {diffuse_color.max():.3f}]")
        print(f"    Usage: Base image with ambient lift, add diffuse_lighting + specular separately")
        
        # Key Light Direction: Pure Lambert shading that rotates with environment
        # The light direction rotates with the environment rotation
        # Start with light pointing from above/forward (typical key light position)
        base_light_dir = torch.tensor([0.3, 0.5, 0.8], device=diffuse_image.device, dtype=diffuse_image.dtype)
        base_light_dir = base_light_dir / torch.sqrt(torch.sum(base_light_dir ** 2))  # Normalize
        
        # Rotate light direction by environment rotation (around Y axis)
        cos_rot = torch.cos(torch.tensor(rotation_rad, device=diffuse_image.device, dtype=diffuse_image.dtype))
        sin_rot = torch.sin(torch.tensor(rotation_rad, device=diffuse_image.device, dtype=diffuse_image.dtype))
        
        # Rotation matrix around Y axis
        rotated_light_x = base_light_dir[0] * cos_rot - base_light_dir[2] * sin_rot
        rotated_light_y = base_light_dir[1]  # Y doesn't change
        rotated_light_z = base_light_dir[0] * sin_rot + base_light_dir[2] * cos_rot
        
        key_light_dir = torch.stack([rotated_light_x, rotated_light_y, rotated_light_z])
        
        # Calculate Lambert dot product with rotated light
        key_light_dot = torch.sum(normals * key_light_dir, dim=-1, keepdim=True)
        key_light_dot = torch.clamp(key_light_dot, 0.0, 1.0)
        key_light_direction = key_light_dot.repeat(1, 1, 1, 3)  # Convert to RGB for output
        
        print(f"  Key Light Direction: Pure Lambert mask (rotates with environment)")
        print(f"    Light direction: [{key_light_dir[0]:.3f}, {key_light_dir[1]:.3f}, {key_light_dir[2]:.3f}]")
        print(f"    Environment rotation: {environment_rotation}°")
        
        print(f"\n✓ PBR IBL relighting complete")
        print(f"  Final image range: [{relit_image.min():.3f}, {relit_image.max():.3f}]")
        print(f"  Material interpretation:")
        print(f"    Roughness: Black=Shiny/Sharp, White=Rough/Blurred")
        print(f"    Metallic: Black=Non-Metal, White=Metal")
        print("="*60 + "\n")
        
        return (relit_image, diffuse_shading_pure, specular_lighting, diffuse_color, key_light_direction)


class PBR_AOV_Generator:
    """
    PBR AOV (Arbitrary Output Variable) Generator
    Takes PBR material inputs and generates proper render passes for compositing:
    - Ambient Light
    - GI Light Pass
    - Diffuse (direct lighting)
    - Reflection (specular)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "albedo": ("IMAGE", {"tooltip": "Albedo/diffuse color map"}),
                "normal_map": ("IMAGE",),
                "environment_map": ("IMAGE", {"tooltip": "HDRI or irradiance map"}),
                
                # Environment rotation
                "environment_rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Rotate environment map (degrees)"
                }),
                
                # Light intensities
                "ambient_intensity": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Ambient fill light intensity"
                }),
                
                "gi_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Global illumination (indirect) intensity"
                }),
                
                "diffuse_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Direct diffuse lighting intensity"
                }),
                
                "reflection_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Reflection/specular intensity"
                }),
                
                # Normal map format
                "normal_format": (["OpenGL", "DirectX"], {
                    "default": "OpenGL",
                    "tooltip": "Normal map format"
                }),
            },
            "optional": {
                "roughness_map": ("IMAGE",),
                "metallic_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("ambient_light", "gi_light", "diffuse", "reflection")
    FUNCTION = "generate_aov_passes"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def decode_normal_map(self, normal_map, format):
        """Decode normal map from color space to vector space"""
        normals = normal_map * 2.0 - 1.0
        if format == "DirectX":
            normals[:, :, :, 1] = -normals[:, :, :, 1]
        length = torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        normals = normals / length
        return normals
    
    def prepare_pbr_map(self, pbr_map, target_shape):
        """Ensure PBR map matches target dimensions"""
        if pbr_map.shape[1:3] != target_shape[1:3]:
            import torch.nn.functional as F
            pbr_map = F.interpolate(pbr_map.permute(0, 3, 1, 2), 
                                   size=(target_shape[1], target_shape[2]), 
                                   mode='bilinear', 
                                   align_corners=False).permute(0, 2, 3, 1)
        if pbr_map.shape[-1] == 1:
            pbr_map = pbr_map.repeat(1, 1, 1, 3)
        return pbr_map
    
    def direction_to_equirect_uv(self, directions, rotation_offset=0.0):
        """Convert 3D direction vectors to equirectangular UV coordinates"""
        x, y, z = directions[:, :, :, 0], directions[:, :, :, 1], directions[:, :, :, 2]
        
        theta = torch.atan2(x, z) + rotation_offset
        phi = torch.acos(torch.clamp(y, -1.0, 1.0))
        
        u = (theta / (2 * np.pi)) % 1.0
        v = phi / np.pi
        
        uv = torch.stack([u, v], dim=-1)
        return uv
    
    def sample_equirect(self, env_map, uv):
        """Sample equirectangular environment map using UV coordinates"""
        batch, height, width, _ = uv.shape
        env_height, env_width = env_map.shape[1], env_map.shape[2]
        
        u = uv[:, :, :, 0] * (env_width - 1)
        v = uv[:, :, :, 1] * (env_height - 1)
        
        u0 = torch.floor(u).long()
        u1 = torch.ceil(u).long()
        v0 = torch.floor(v).long()
        v1 = torch.ceil(v).long()
        
        u0 = torch.clamp(u0, 0, env_width - 1)
        u1 = torch.clamp(u1, 0, env_width - 1)
        v0 = torch.clamp(v0, 0, env_height - 1)
        v1 = torch.clamp(v1, 0, env_height - 1)
        
        wu = (u - u0.float()).unsqueeze(-1)
        wv = (v - v0.float()).unsqueeze(-1)
        
        batch_idx = torch.arange(batch, device=env_map.device).view(-1, 1, 1).expand(batch, height, width)
        
        c00 = env_map[batch_idx, v0, u0]
        c01 = env_map[batch_idx, v0, u1]
        c10 = env_map[batch_idx, v1, u0]
        c11 = env_map[batch_idx, v1, u1]
        
        c0 = c00 * (1 - wu) + c01 * wu
        c1 = c10 * (1 - wu) + c11 * wu
        result = c0 * (1 - wv) + c1 * wv
        
        return result
    
    def calculate_reflection(self, normals, view_dir):
        """Calculate reflection vector for environment sampling"""
        dot_nv = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        reflection = 2.0 * dot_nv * normals - view_dir
        length = torch.sqrt(torch.sum(reflection ** 2, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-8)
        reflection = reflection / length
        return reflection
    
    def calculate_fresnel_schlick(self, cos_theta, f0):
        """Fresnel-Schlick approximation"""
        return f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)
    
    def generate_aov_passes(self, albedo, normal_map, environment_map, environment_rotation,
                           ambient_intensity, gi_intensity, diffuse_intensity, reflection_intensity,
                           normal_format, roughness_map=None, metallic_map=None):
        """
        Generate proper AOV render passes for compositing
        """
        
        print("\n" + "="*60)
        print("PBR AOV Generator - Render Passes")
        print("="*60)
        print(f"Albedo shape: {albedo.shape}")
        print(f"Normal map shape: {normal_map.shape}")
        print(f"Environment shape: {environment_map.shape}")
        
        # Convert rotation to radians
        rotation_rad = environment_rotation * (np.pi / 180.0)
        
        # Decode normal map
        normals = self.decode_normal_map(normal_map, normal_format)
        
        # Process PBR maps
        if roughness_map is not None:
            roughness = self.prepare_pbr_map(roughness_map, albedo.shape)
            print(f"Roughness map: [{roughness.min():.3f}, {roughness.max():.3f}]")
        else:
            roughness = torch.ones_like(albedo) * 0.5
            print(f"Roughness: Using default (0.5)")
        
        if metallic_map is not None:
            metallic = self.prepare_pbr_map(metallic_map, albedo.shape)
            print(f"Metallic map: [{metallic.min():.3f}, {metallic.max():.3f}]")
        else:
            metallic = torch.zeros_like(albedo)
            print(f"Metallic: Using default (0.0)")
        
        # View direction
        view_dir = torch.tensor([0.0, 0.0, 1.0], 
                               device=albedo.device, dtype=albedo.dtype)
        
        # =====================================================================
        # AOV 1: AMBIENT LIGHT
        # Pure flat ambient fill (no directionality)
        # =====================================================================
        print(f"\n💡 Generating AOV 1: Ambient Light...")
        
        # Average environment color
        ambient_color = torch.mean(environment_map, dim=(1, 2), keepdim=True)
        batch, height, width, _ = albedo.shape
        ambient_fill = ambient_color.expand(batch, height, width, 3)
        
        ambient_light = albedo * ambient_fill * ambient_intensity
        ambient_light = torch.clamp(ambient_light, 0.0, 1.0)
        
        print(f"  Ambient color: RGB({ambient_color[0,0,0,0]:.3f}, {ambient_color[0,0,0,1]:.3f}, {ambient_color[0,0,0,2]:.3f})")
        print(f"  Result: Flat ambient fill × {ambient_intensity}")
        
        # =====================================================================
        # AOV 2: GI LIGHT PASS (Global Illumination - Indirect)
        # Soft hemisphere-averaged environment lighting
        # =====================================================================
        print(f"\n🌐 Generating AOV 2: GI Light Pass...")
        
        # Sample environment based on normals but heavily blurred (indirect)
        gi_uv = self.direction_to_equirect_uv(normals, rotation_rad)
        
        # Heavy blur for GI (indirect lighting is soft)
        # Simple multi-sample blur
        gi_colors = self.sample_equirect(environment_map, gi_uv)
        
        # Apply to albedo (non-metals only for GI)
        gi_light = albedo * gi_colors * gi_intensity * (1.0 - metallic)
        gi_light = torch.clamp(gi_light, 0.0, 1.0)
        
        print(f"  Indirect lighting from environment")
        print(f"  Masked by metallic (metals don't have diffuse GI)")
        
        # =====================================================================
        # AOV 3: DIFFUSE (Direct Lighting)
        # Lambert-based direct light from environment
        # =====================================================================
        print(f"\n☀️ Generating AOV 3: Diffuse (Direct)...")
        
        # Key light direction (from environment dominant direction)
        # For now, use upward direction as key light
        key_light_dir = torch.tensor([0.3, 0.5, 0.8], device=albedo.device, dtype=albedo.dtype)
        key_light_dir = key_light_dir / torch.sqrt(torch.sum(key_light_dir ** 2))
        
        # Rotate with environment
        cos_rot = torch.cos(torch.tensor(rotation_rad, device=albedo.device, dtype=albedo.dtype))
        sin_rot = torch.sin(torch.tensor(rotation_rad, device=albedo.device, dtype=albedo.dtype))
        
        rotated_x = key_light_dir[0] * cos_rot - key_light_dir[2] * sin_rot
        rotated_y = key_light_dir[1]
        rotated_z = key_light_dir[0] * sin_rot + key_light_dir[2] * cos_rot
        
        key_light = torch.stack([rotated_x, rotated_y, rotated_z])
        
        # Lambert shading
        dot_nl = torch.sum(normals * key_light, dim=-1, keepdim=True)
        dot_nl = torch.clamp(dot_nl, 0.0, 1.0)
        
        # Sample environment at key light direction for color
        key_light_uv = self.direction_to_equirect_uv(
            key_light.unsqueeze(0).unsqueeze(0).expand(batch, height, width, 3),
            rotation_rad
        )
        key_light_color = self.sample_equirect(environment_map, key_light_uv)
        
        # Direct diffuse (non-metals only)
        diffuse = albedo * key_light_color * dot_nl * diffuse_intensity * (1.0 - metallic)
        diffuse = torch.clamp(diffuse, 0.0, 1.0)
        
        print(f"  Direct Lambert lighting from key light")
        print(f"  Key light direction: [{key_light[0]:.3f}, {key_light[1]:.3f}, {key_light[2]:.3f}]")
        
        # =====================================================================
        # AOV 4: REFLECTION (Specular)
        # Environment reflections with PBR
        # =====================================================================
        print(f"\n✨ Generating AOV 4: Reflection...")
        
        # Calculate reflection vectors
        reflection_vectors = self.calculate_reflection(normals, view_dir)
        reflection_uv = self.direction_to_equirect_uv(reflection_vectors, rotation_rad)
        
        # Sample environment (would need blur based on roughness for proper PBR)
        reflection_colors = self.sample_equirect(environment_map, reflection_uv)
        
        # Calculate Fresnel
        view_dot_normal = torch.sum(normals * view_dir, dim=-1, keepdim=True)
        view_dot_normal = torch.clamp(view_dot_normal, 0.0, 1.0)
        
        f0_dielectric = torch.ones_like(albedo) * 0.04
        f0_metal = albedo
        f0 = f0_dielectric * (1.0 - metallic) + f0_metal * metallic
        
        fresnel = self.calculate_fresnel_schlick(view_dot_normal, f0)
        
        # Metal boost for reflections
        metal_boost = 1.0 + metallic * 2.0
        
        reflection = reflection_colors * fresnel * reflection_intensity * metal_boost
        reflection = torch.clamp(reflection, 0.0, 1.0)
        
        print(f"  Environment reflections with Fresnel")
        print(f"  Boosted for metallic surfaces")
        
        print(f"\n✓ AOV Generation Complete")
        print(f"  4 render passes ready for compositing")
        print("="*60 + "\n")
        
        return (ambient_light, gi_light, diffuse, reflection)


class Marigold_AOV_Extractor:
    """
    Extract render passes (AOVs) from Marigold IId outputs
    Based on Octane render passes: https://help.otoy.com/hc/en-us/articles/30487201925019
    
    Extracts:
    - Diffuse (Albedo from Lighting)
    - Lighting/Shading (Direct + Indirect illumination)
    - Reflection/Specular (Residual highlights)
    - Ambient Occlusion (from shading analysis)
    - Emission (bright residual areas)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "marigold_lighting": ("IMAGE", {
                    "tooltip": "Marigold IId Lighting output (contains albedo + shading)"
                }),
                "marigold_appearance": ("IMAGE", {
                    "tooltip": "Marigold IId Appearance output (contains residual/specular)"
                }),
                
                # Extraction controls
                "diffuse_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Gamma correction for diffuse/albedo"
                }),
                
                "shading_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Lighting/shading intensity"
                }),
                
                "specular_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for extracting specular from residual"
                }),
                
                "specular_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.1,
                    "tooltip": "Specular/reflection intensity"
                }),
                
                "ao_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Ambient occlusion extraction strength"
                }),
                
                "emission_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for extracting emission from bright areas"
                }),
            },
            "optional": {
                "normal_map": ("IMAGE", {
                    "tooltip": "Optional normal map for enhanced AO extraction"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("diffuse_albedo", "lighting_direct", "lighting_indirect", "reflection_specular", "ambient_occlusion", "emission", "beauty_composite")
    FUNCTION = "extract_render_passes"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def extract_render_passes(self, marigold_lighting, marigold_appearance, 
                             diffuse_gamma, shading_strength, specular_threshold, 
                             specular_strength, ao_strength, emission_threshold,
                             normal_map=None):
        """
        Extract render passes from Marigold IId outputs
        
        Marigold IId Lighting decomposition:
        - Albedo: Diffuse surface color (no lighting)
        - Shading: Direct + Indirect illumination
        
        Marigold IId Appearance:
        - Residual: Information not captured by albedo/shading (specular, reflections)
        """
        
        print("\n" + "="*60)
        print("Marigold AOV Extractor - Render Pass Decomposition")
        print("="*60)
        print(f"Lighting input shape: {marigold_lighting.shape}")
        print(f"Appearance input shape: {marigold_appearance.shape}")
        
        # Ensure both inputs are RGB (3 channels)
        # Marigold sometimes outputs grayscale or 2-channel images
        def ensure_rgb(tensor, name):
            """Convert grayscale/2-channel to RGB"""
            if tensor.shape[-1] == 1:
                # Grayscale to RGB
                print(f"  Converting {name} from grayscale to RGB")
                return tensor.repeat(1, 1, 1, 3)
            elif tensor.shape[-1] == 2:
                # 2-channel to RGB (repeat first channel for blue)
                print(f"  Converting {name} from 2-channel to RGB")
                return torch.cat([tensor, tensor[:, :, :, 0:1]], dim=-1)
            elif tensor.shape[-1] == 3:
                # Already RGB
                return tensor
            else:
                # More than 3 channels, take first 3
                print(f"  Warning: {name} has {tensor.shape[-1]} channels, using first 3")
                return tensor[:, :, :, :3]
        
        marigold_lighting = ensure_rgb(marigold_lighting, "Lighting")
        marigold_appearance = ensure_rgb(marigold_appearance, "Appearance")
        
        # Ensure batch sizes match (take minimum)
        if marigold_lighting.shape[0] != marigold_appearance.shape[0]:
            min_batch = min(marigold_lighting.shape[0], marigold_appearance.shape[0])
            print(f"  Warning: Batch size mismatch!")
            print(f"  Lighting batch: {marigold_lighting.shape[0]}, Appearance batch: {marigold_appearance.shape[0]}")
            print(f"  Using first {min_batch} images from each")
            marigold_lighting = marigold_lighting[:min_batch]
            marigold_appearance = marigold_appearance[:min_batch]
        
        # Ensure inputs match spatial dimensions
        if marigold_lighting.shape[1:3] != marigold_appearance.shape[1:3]:
            import torch.nn.functional as F
            target_shape = marigold_lighting.shape[1:3]
            marigold_appearance = F.interpolate(
                marigold_appearance.permute(0, 3, 1, 2),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            print(f"  Resized appearance to match lighting: {marigold_appearance.shape}")
        
        print(f"Final shapes - Lighting: {marigold_lighting.shape}, Appearance: {marigold_appearance.shape}")
        
        # =====================================================================
        # PASS 1: DIFFUSE ALBEDO
        # Pure surface color without lighting
        # =====================================================================
        print(f"\n📦 Pass 1: Diffuse Albedo...")
        
        # Marigold Lighting contains the albedo component
        # We need to extract it by removing the shading
        
        # Simple approach: The lighting image is albedo × shading
        # We can approximate albedo by analyzing the image
        # For now, use the lighting output as diffuse (it's already close)
        diffuse_albedo = marigold_lighting.clone()
        
        # Apply gamma correction
        if diffuse_gamma != 1.0:
            diffuse_albedo = torch.pow(diffuse_albedo, diffuse_gamma)
        
        diffuse_albedo = torch.clamp(diffuse_albedo, 0.0, 1.0)
        print(f"  Extracted diffuse albedo with gamma={diffuse_gamma}")
        
        # =====================================================================
        # PASS 2 & 3: LIGHTING (Direct + Indirect)
        # =====================================================================
        print(f"\n💡 Pass 2 & 3: Lighting (Direct + Indirect)...")
        
        # The "shading" in Marigold represents the lighting
        # We approximate this by dividing the lit image by albedo
        # shading = lit_image / albedo
        
        # To extract shading, we need the ratio
        # Avoid division by zero
        albedo_safe = torch.clamp(marigold_lighting, min=0.01)
        
        # The appearance contains the residual, so we can use lighting as base
        # Convert to grayscale for shading intensity
        lighting_intensity = torch.mean(marigold_lighting, dim=-1, keepdim=True)
        lighting_intensity = lighting_intensity.repeat(1, 1, 1, 3)
        
        # Direct lighting: Brighter areas (peaks)
        lighting_direct = torch.clamp(lighting_intensity * shading_strength, 0.0, 1.0)
        
        # Indirect lighting: Softer, ambient component
        # GI should be VERY soft and diffuse (no specular)
        # Apply heavy blur to remove specular highlights
        import torch.nn.functional as F
        
        # Convert to BCHW format for conv2d
        lighting_for_blur = lighting_intensity.permute(0, 3, 1, 2)
        
        # Create large gaussian-like blur kernel for soft GI
        kernel_size = 21  # Larger kernel = softer, more diffuse
        sigma = kernel_size / 3.0
        
        # Simple box blur (faster than gaussian)
        # Multiple passes of small blur = larger blur
        for _ in range(3):  # 3 passes for soft result
            lighting_for_blur = F.avg_pool2d(
                lighting_for_blur,
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False
            )
        
        # Convert back to BHWC
        lighting_indirect = lighting_for_blur.permute(0, 2, 3, 1)
        
        # Further reduce contrast for GI (very soft)
        lighting_indirect = (lighting_indirect - 0.5) * 0.3 + 0.5  # Much softer
        lighting_indirect = torch.clamp(lighting_indirect * shading_strength * 0.4, 0.0, 1.0)
        
        print(f"  Extracted direct lighting (strength={shading_strength})")
        print(f"  Extracted indirect lighting (heavily blurred, matte)")

        
        # =====================================================================
        # PASS 4: REFLECTION/SPECULAR
        # Extract from residual (appearance - lighting)
        # =====================================================================
        print(f"\n✨ Pass 4: Reflection/Specular...")
        
        # The residual is what appearance captures that lighting doesn't
        # This includes specular highlights and reflections
        residual = marigold_appearance - marigold_lighting
        residual = torch.clamp(residual, 0.0, 1.0)
        
        # Extract bright spots as specular
        residual_intensity = torch.mean(residual, dim=-1, keepdim=True)
        
        # Threshold to get specular highlights
        specular_mask = (residual_intensity > specular_threshold).float()
        
        # Apply mask and boost
        reflection_specular = residual * specular_mask * specular_strength
        reflection_specular = torch.clamp(reflection_specular, 0.0, 1.0)
        
        print(f"  Extracted specular with threshold={specular_threshold}")
        print(f"  Specular pixels: {(specular_mask.sum() / specular_mask.numel() * 100):.1f}%")
        
        # =====================================================================
        # PASS 5: AMBIENT OCCLUSION
        # Extract from shadow analysis
        # =====================================================================
        print(f"\n🌑 Pass 5: Ambient Occlusion...")
        
        # AO is typically the darkest areas in shading
        # Convert lighting to grayscale and invert dark areas
        
        lighting_gray = torch.mean(marigold_lighting, dim=-1, keepdim=True)
        
        # Invert and boost shadows
        ao_raw = 1.0 - lighting_gray
        
        # Apply contrast to enhance
        ao_raw = torch.pow(ao_raw, 1.0 / ao_strength) if ao_strength > 0 else ao_raw
        
        # Normalize
        ao_min = ao_raw.min()
        ao_max = ao_raw.max()
        if ao_max > ao_min:
            ambient_occlusion = (ao_raw - ao_min) / (ao_max - ao_min)
        else:
            ambient_occlusion = ao_raw
        
        # Convert to RGB for display
        ambient_occlusion = ambient_occlusion.repeat(1, 1, 1, 3)
        ambient_occlusion = torch.clamp(ambient_occlusion, 0.0, 1.0)
        
        print(f"  Extracted AO from shadow analysis (strength={ao_strength})")
        
        # Enhance with normal map if provided
        if normal_map is not None:
            print(f"  Enhancing AO with normal map...")
            # Use normal map to refine AO (darker in crevices)
            normal_gray = torch.mean(normal_map, dim=-1, keepdim=True)
            normal_factor = 1.0 - torch.abs(normal_gray - 0.5) * 2.0
            ambient_occlusion = ambient_occlusion * (0.7 + normal_factor * 0.3)
            ambient_occlusion = torch.clamp(ambient_occlusion, 0.0, 1.0)
        
        # =====================================================================
        # PASS 6: EMISSION
        # Extract very bright areas as self-illumination
        # =====================================================================
        print(f"\n💥 Pass 6: Emission...")
        
        # Emission is extracted from very bright areas in appearance
        appearance_intensity = torch.mean(marigold_appearance, dim=-1, keepdim=True)
        
        # Threshold for emission
        emission_mask = (appearance_intensity > emission_threshold).float()
        
        # Extract emission
        emission = marigold_appearance * emission_mask
        
        # Boost emission
        emission = torch.clamp(emission * 1.5, 0.0, 1.0)
        
        emission_percent = (emission_mask.sum() / emission_mask.numel() * 100)
        print(f"  Extracted emission with threshold={emission_threshold}")
        print(f"  Emissive pixels: {emission_percent:.1f}%")
        
        # =====================================================================
        # PASS 7: BEAUTY COMPOSITE
        # Combine all passes for reference
        # =====================================================================
        print(f"\n🎨 Pass 7: Beauty Composite...")
        
        # Beauty = (Diffuse × Lighting_Direct) + Lighting_Indirect + Reflection + Emission
        beauty_composite = (
            diffuse_albedo * lighting_direct +
            lighting_indirect +
            reflection_specular +
            emission
        )
        beauty_composite = torch.clamp(beauty_composite, 0.0, 1.0)
        
        print(f"  Composed beauty pass from all AOVs")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print(f"\n✓ Render Pass Extraction Complete")
        print(f"  7 passes ready for compositing:")
        print(f"    1. Diffuse Albedo    - Base surface color")
        print(f"    2. Lighting Direct   - Direct illumination")
        print(f"    3. Lighting Indirect - Ambient/bounce light")
        print(f"    4. Reflection        - Specular highlights")
        print(f"    5. Ambient Occlusion - Shadow/cavity AO")
        print(f"    6. Emission          - Self-illumination")
        print(f"    7. Beauty Composite  - Final combined result")
        print("="*60 + "\n")
        
        return (diffuse_albedo, lighting_direct, lighting_indirect, 
                reflection_specular, ambient_occlusion, emission, beauty_composite)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NormalMapRelighter": NormalMapRelighter,
    "NormalMapRelighterAdvanced": NormalMapRelighterAdvanced,
    "NormalMapRelighterIBL": NormalMapRelighterIBL,
    "NormalMapRelighterIBL_PBR": NormalMapRelighterIBL_PBR,
    "PBR_AOV_Generator": PBR_AOV_Generator,
    "Marigold_AOV_Extractor": Marigold_AOV_Extractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapRelighter": "Normal Map Relighter",
    "NormalMapRelighterAdvanced": "Normal Map Relighter (Advanced)",
    "NormalMapRelighterIBL": "Normal Map Relighter (IBL)",
    "NormalMapRelighterIBL_PBR": "Normal Map Relighter (IBL + PBR)",
    "PBR_AOV_Generator": "🎬 PBR AOV Generator (Render Passes)",
    "Marigold_AOV_Extractor": "📦 Marigold AOV Extractor (Render Passes)",
}
