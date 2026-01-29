"""
Environment Map Processing Utilities
Pre-processing tools for Image-Based Lighting
"""

import torch
import numpy as np


class EnvironmentMapToIrradiance:
    """
    Convert environment map (HDRI) to irradiance map for diffuse lighting
    Performs hemisphere convolution for physically accurate diffuse IBL
    
    Three quality modes:
    - Fast: Heavy blur approximation
    - Balanced: Monte Carlo sampling with moderate samples
    - Accurate: Monte Carlo sampling with many samples
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "environment_map": ("IMAGE",),
                
                "quality": (["Fast (Blur)", "Balanced (128 samples)", "Accurate (512 samples)"], {
                    "default": "Balanced (128 samples)",
                    "tooltip": "Quality vs speed tradeoff"
                }),
                
                "compute_device": (["GPU (CUDA)", "CPU"], {
                    "default": "GPU (CUDA)",
                    "tooltip": "GPU is 10-100x faster for hemisphere sampling"
                }),
                
                "output_resolution": (["Same as Input", "512x256", "1024x512", "2048x1024"], {
                    "default": "512x256",
                    "tooltip": "Output resolution - lower is faster"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("irradiance_map",)
    FUNCTION = "generate_irradiance"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def sample_environment(self, env_map: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """
        Sample environment map given 3D directions
        directions: [N, 3] normalized vectors
        returns: [N, C] colors
        """
        x = directions[:, 0]
        y = directions[:, 1]
        z = directions[:, 2]
        
        # Convert to spherical coordinates
        theta = torch.asin(torch.clamp(y, -1.0, 1.0))
        phi = torch.atan2(x, z)
        
        # Convert to UV
        u = (phi / (2 * np.pi)) + 0.5
        v = (theta / np.pi) + 0.5
        
        # Wrap/clamp
        u = u % 1.0
        v = torch.clamp(v, 0.0, 1.0)
        
        # Sample with bilinear interpolation
        batch, height, width, channels = env_map.shape
        
        px = u * (width - 1)
        py = v * (height - 1)
        
        px0 = torch.floor(px).long()
        py0 = torch.floor(py).long()
        px1 = torch.clamp(px0 + 1, 0, width - 1)
        py1 = torch.clamp(py0 + 1, 0, height - 1)
        px0 = torch.clamp(px0, 0, width - 1)
        py0 = torch.clamp(py0, 0, height - 1)
        
        fx = (px - px0.float()).unsqueeze(-1)
        fy = (py - py0.float()).unsqueeze(-1)
        
        # Sample corners
        c00 = env_map[0, py0, px0]
        c01 = env_map[0, py0, px1]
        c10 = env_map[0, py1, px0]
        c11 = env_map[0, py1, px1]
        
        # Bilinear interpolation
        c0 = c00 * (1 - fx) + c01 * fx
        c1 = c10 * (1 - fx) + c11 * fx
        result = c0 * (1 - fy) + c1 * fy
        
        return result
    
    def generate_irradiance_hemisphere_sampling_gpu(self, env_map: torch.Tensor, 
                                                     width: int, height: int, 
                                                     num_samples: int) -> torch.Tensor:
        """
        GPU-accelerated hemisphere sampling using batched operations
        Much faster than CPU version (10-100x speedup)
        """
        print(f"  🚀 GPU-accelerated generation with {num_samples} samples per pixel...")
        print(f"  Output resolution: {width}x{height}")
        
        device = env_map.device
        dtype = env_map.dtype
        channels = env_map.shape[-1]
        
        # Create all output pixel coordinates at once
        v_coords = torch.linspace(0, 1, height, device=device, dtype=dtype)
        u_coords = torch.linspace(0, 1, width, device=device, dtype=dtype)
        u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='xy')
        
        # Convert to directions (normals) for all pixels
        phi = (u_grid - 0.5) * 2 * np.pi
        theta = (v_grid - 0.5) * np.pi
        
        # Normal directions [H, W, 3]
        normals = torch.stack([
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta),
            torch.cos(theta) * torch.cos(phi)
        ], dim=-1)
        
        # Normalize
        normals = normals / torch.sqrt(torch.sum(normals ** 2, dim=-1, keepdim=True))
        
        # Build tangent space for all pixels [H, W, 3]
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 1, 3)
        up = up.expand(height, width, 3)
        
        # Handle near-vertical normals
        vertical_mask = torch.abs(normals[..., 1]) > 0.999
        up_alt = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        up = torch.where(vertical_mask.unsqueeze(-1), up_alt, up)
        
        # Tangent vectors
        tangents = torch.linalg.cross(up, normals)
        tangents = tangents / torch.sqrt(torch.sum(tangents ** 2, dim=-1, keepdim=True))
        bitangents = torch.linalg.cross(normals, tangents)
        
        # Generate random samples for hemisphere [num_samples, 2]
        r1 = torch.rand(num_samples, device=device, dtype=dtype)
        r2 = torch.rand(num_samples, device=device, dtype=dtype)
        
        # Cosine-weighted hemisphere sampling
        phi_sample = 2 * np.pi * r1
        cos_theta = torch.sqrt(r2)
        sin_theta = torch.sqrt(1 - r2)
        
        # Local directions [num_samples, 3]
        local_dirs = torch.stack([
            torch.cos(phi_sample) * sin_theta,
            torch.sin(phi_sample) * sin_theta,
            cos_theta
        ], dim=-1)
        
        # Initialize accumulator
        irradiance = torch.zeros((height, width, channels), device=device, dtype=dtype)
        
        # Process in batches to avoid OOM
        batch_size = 32  # samples per batch
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_local = local_dirs[start_idx:end_idx]  # [B, 3]
            
            # Transform to world space: [H, W, 1, 3] * [1, 1, B, 3] -> [H, W, B, 3]
            world_dirs = (tangents.unsqueeze(2) * batch_local[:, 0].view(1, 1, -1, 1) +
                         bitangents.unsqueeze(2) * batch_local[:, 1].view(1, 1, -1, 1) +
                         normals.unsqueeze(2) * batch_local[:, 2].view(1, 1, -1, 1))
            
            # Flatten for sampling [H*W*B, 3]
            world_dirs_flat = world_dirs.reshape(-1, 3)
            
            # Sample environment
            colors_flat = self.sample_environment(env_map, world_dirs_flat)  # [H*W*B, C]
            
            # Reshape and accumulate [H, W, B, C]
            colors = colors_flat.reshape(height, width, -1, channels)
            irradiance += colors.sum(dim=2)
            
            # Progress
            if batch_idx % max(1, num_batches // 10) == 0:
                progress = (batch_idx / num_batches) * 100
                print(f"    Progress: {progress:.1f}%")
        
        # Average
        irradiance = irradiance / num_samples
        
        print(f"  ✓ GPU generation complete!")
        return irradiance.unsqueeze(0)
    
    def generate_irradiance_hemisphere_sampling(self, env_map: torch.Tensor, 
                                                width: int, height: int, 
                                                num_samples: int) -> torch.Tensor:
        """
        Generate irradiance map using Monte Carlo hemisphere sampling
        For each texel, sample hemisphere and integrate with Lambert cosine
        """
        print(f"  Generating irradiance map with {num_samples} samples per pixel...")
        print(f"  Output resolution: {width}x{height}")
        
        device = env_map.device
        dtype = env_map.dtype
        channels = env_map.shape[-1]
        
        # Create output irradiance map
        irradiance = torch.zeros((1, height, width, channels), device=device, dtype=dtype)
        
        # For each pixel in output
        for y in range(height):
            # Print progress more frequently
            progress_pct = (y / height) * 100
            if y % 8 == 0:  # Every 8 rows instead of 32
                print(f"    Progress: {progress_pct:.1f}% ({y}/{height} rows)")
            
            for x in range(width):
                # Convert pixel to direction (normal)
                u = x / (width - 1)
                v = y / (height - 1)
                
                phi = (u - 0.5) * 2 * np.pi
                theta = (v - 0.5) * np.pi
                
                # Normal direction for this pixel
                nx = np.cos(theta) * np.sin(phi)
                ny = np.sin(theta)
                nz = np.cos(theta) * np.cos(phi)
                
                normal = torch.tensor([nx, ny, nz], device=device, dtype=dtype)
                normal = normal / torch.sqrt(torch.sum(normal ** 2))
                
                # Build tangent space
                up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
                if torch.abs(normal[1]) > 0.999:
                    up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
                
                tangent = torch.linalg.cross(up, normal)
                tangent = tangent / torch.sqrt(torch.sum(tangent ** 2))
                bitangent = torch.linalg.cross(normal, tangent)
                
                # Sample hemisphere
                accumulated_color = torch.zeros(channels, device=device, dtype=dtype)
                total_weight = 0.0
                
                for _ in range(num_samples):
                    # Random hemisphere sample (cosine weighted)
                    r1 = torch.rand(1, device=device, dtype=dtype).item()
                    r2 = torch.rand(1, device=device, dtype=dtype).item()
                    
                    # Cosine-weighted hemisphere sampling
                    phi_sample = 2 * np.pi * r1
                    cos_theta = np.sqrt(r2)
                    sin_theta = np.sqrt(1 - r2)
                    
                    # Local direction
                    local_x = np.cos(phi_sample) * sin_theta
                    local_y = np.sin(phi_sample) * sin_theta
                    local_z = cos_theta
                    
                    # Transform to world space
                    sample_dir = (tangent * local_x + 
                                 bitangent * local_y + 
                                 normal * local_z)
                    
                    # Sample environment
                    color = self.sample_environment(env_map, sample_dir.unsqueeze(0))[0]
                    
                    # Lambert cosine term (already in cosine-weighted sampling)
                    accumulated_color += color
                    total_weight += 1.0
                
                # Average
                irradiance[0, y, x] = accumulated_color / total_weight
        
        print(f"  Irradiance map generation complete!")
        return irradiance
    
    def generate_irradiance_fast(self, env_map: torch.Tensor, 
                                 width: int, height: int) -> torch.Tensor:
        """
        Fast irradiance approximation using heavy Gaussian-like blur
        """
        print(f"  Using fast approximation (heavy blur)...")
        
        # If resolution needs to change, we'd need to implement downsampling
        # For now, just return heavily blurred version
        
        # Simple box blur approximation (multiple passes)
        result = env_map.clone()
        
        # Multiple blur passes to approximate integration
        for _ in range(8):
            # Simple averaging with neighbors
            padded = torch.nn.functional.pad(result, (0, 0, 1, 1, 1, 1), mode='replicate')
            
            blurred = (
                padded[:, :-2, :-2, :] + padded[:, :-2, 1:-1, :] + padded[:, :-2, 2:, :] +
                padded[:, 1:-1, :-2, :] + padded[:, 1:-1, 1:-1, :] + padded[:, 1:-1, 2:, :] +
                padded[:, 2:, :-2, :] + padded[:, 2:, 1:-1, :] + padded[:, 2:, 2:, :]
            ) / 9.0
            
            result = blurred
        
        # Downsample if needed
        if result.shape[2] != width or result.shape[1] != height:
            print(f"  Resampling from {result.shape[2]}x{result.shape[1]} to {width}x{height}")
            # Simple nearest neighbor downsampling
            result = torch.nn.functional.interpolate(
                result.permute(0, 3, 1, 2), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        return result
    
    def generate_irradiance(self, environment_map, quality, compute_device, output_resolution):
        """Generate irradiance map from environment map"""
        
        print("\n" + "="*60)
        print("Environment Map to Irradiance")
        print("="*60)
        print(f"Input shape: {environment_map.shape}")
        print(f"Quality: {quality}")
        print(f"Compute device: {compute_device}")
        print(f"Output resolution: {output_resolution}")
        
        # Determine output size
        if output_resolution == "Same as Input":
            out_width = environment_map.shape[2]
            out_height = environment_map.shape[1]
        else:
            res_map = {
                "512x256": (512, 256),
                "1024x512": (1024, 512),
                "2048x1024": (2048, 1024)
            }
            out_width, out_height = res_map[output_resolution]
        
        # Check if GPU is available
        use_gpu = compute_device == "GPU (CUDA)" and torch.cuda.is_available()
        if compute_device == "GPU (CUDA)" and not torch.cuda.is_available():
            print("  ⚠ GPU requested but CUDA not available, falling back to CPU")
            use_gpu = False
        
        # Generate based on quality
        if quality == "Fast (Blur)":
            irradiance = self.generate_irradiance_fast(environment_map, out_width, out_height)
        elif use_gpu and quality != "Fast (Blur)":
            # GPU accelerated hemisphere sampling
            num_samples = 128 if quality == "Balanced (128 samples)" else 512
            irradiance = self.generate_irradiance_hemisphere_sampling_gpu(
                environment_map, out_width, out_height, num_samples
            )
        else:
            # CPU hemisphere sampling
            num_samples = 128 if quality == "Balanced (128 samples)" else 512
            irradiance = self.generate_irradiance_hemisphere_sampling(
                environment_map, out_width, out_height, num_samples
            )
        
        print(f"\nOutput shape: {irradiance.shape}")
        print(f"Irradiance range: [{irradiance.min():.3f}, {irradiance.max():.3f}]")
        print("="*60 + "\n")
        
        return (irradiance,)


class HDRIToIrradianceFast:
    """
    Super fast HDRI to Irradiance conversion
    Simple downscale + blur - instant results for testing
    Perfect for iteration and quick previews
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdri": ("IMAGE",),
                
                "blur_strength": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of blur passes (more = softer)"
                }),
                
                "output_size": (["256x128", "512x256", "1024x512"], {
                    "default": "512x256",
                    "tooltip": "Output resolution"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("irradiance_map",)
    FUNCTION = "convert_fast"
    CATEGORY = "Texture Alchemist/Lighting"
    
    def convert_fast(self, hdri, blur_strength, output_size):
        """Ultra-fast irradiance conversion using downscale + blur"""
        
        print("\n" + "="*60)
        print("⚡ HDRI to Irradiance (FAST)")
        print("="*60)
        print(f"Input shape: {hdri.shape}")
        print(f"Blur passes: {blur_strength}")
        
        # Parse output size
        size_map = {
            "256x128": (256, 128),
            "512x256": (512, 256),
            "1024x512": (1024, 512)
        }
        out_width, out_height = size_map[output_size]
        
        # Downscale first (if needed)
        if hdri.shape[2] != out_width or hdri.shape[1] != out_height:
            print(f"  Downscaling to {out_width}x{out_height}...")
            result = torch.nn.functional.interpolate(
                hdri.permute(0, 3, 1, 2), 
                size=(out_height, out_width), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            result = hdri.clone()
        
        # Apply multiple blur passes
        print(f"  Applying {blur_strength} blur passes...")
        for pass_idx in range(blur_strength):
            # Simple 3x3 box blur
            padded = torch.nn.functional.pad(result, (0, 0, 1, 1, 1, 1), mode='replicate')
            
            result = (
                padded[:, :-2, :-2, :] + padded[:, :-2, 1:-1, :] + padded[:, :-2, 2:, :] +
                padded[:, 1:-1, :-2, :] + padded[:, 1:-1, 1:-1, :] + padded[:, 1:-1, 2:, :] +
                padded[:, 2:, :-2, :] + padded[:, 2:, 1:-1, :] + padded[:, 2:, 2:, :]
            ) / 9.0
            
            if pass_idx % 2 == 0 and pass_idx > 0:
                print(f"    Pass {pass_idx}/{blur_strength}")
        
        print(f"\n✓ Fast conversion complete!")
        print(f"  Output shape: {result.shape}")
        print(f"  Processing time: ~instant ⚡")
        print("="*60 + "\n")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "EnvironmentMapToIrradiance": EnvironmentMapToIrradiance,
    "HDRIToIrradianceFast": HDRIToIrradianceFast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnvironmentMapToIrradiance": "Environment Map to Irradiance",
    "HDRIToIrradianceFast": "HDRI to Irradiance (FAST)",
}
