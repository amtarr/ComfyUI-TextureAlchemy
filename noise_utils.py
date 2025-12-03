"""
Noise and Pattern Generation for TextureAlchemy
Procedural noise, patterns, and imperfections
"""

import torch
import torch.nn.functional as F
import math


class ProceduralNoiseGenerator:
    """
    Generate various types of procedural noise
    Perlin, Simplex-like, Voronoi, Cellular patterns
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "noise_type": (["perlin", "fbm", "turbulence", "voronoi", "cellular", "white"],),
                "scale": ("FLOAT", {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 500.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Noise scale/frequency"
                }),
                "octaves": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Number of detail levels"
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Amplitude falloff per octave"
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Frequency multiplier per octave"
                }),
                "tileable": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("noise",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Procedural"
    
    def generate(self, width, height, noise_type, scale, octaves, persistence, lacunarity, tileable, seed):
        """Generate procedural noise"""
        
        print(f"\n{'='*60}")
        print(f"Procedural Noise Generator - {noise_type.upper()}")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        torch.manual_seed(seed)
        
        if noise_type == "white":
            # Pure white noise
            noise = torch.rand((1, height, width, 1), device=device, dtype=dtype)
        
        elif noise_type == "voronoi":
            noise = self._voronoi_noise(width, height, scale, seed, device, dtype)
        
        elif noise_type == "cellular":
            noise = self._cellular_noise(width, height, scale, seed, device, dtype)
        
        else:
            # Perlin-like, FBM, or Turbulence
            noise = self._fbm_noise(width, height, scale, octaves, persistence, lacunarity, 
                                   noise_type, tileable, seed, device, dtype)
        
        # Convert to 3-channel RGB
        noise = noise.repeat(1, 1, 1, 3)
        noise = torch.clamp(noise, 0.0, 1.0)
        
        print(f"✓ Noise generated")
        print(f"  Type: {noise_type}")
        print(f"  Resolution: {width}×{height}")
        print(f"  Scale: {scale:.1f}")
        print(f"  Octaves: {octaves}")
        print(f"  Tileable: {tileable}")
        print(f"{'='*60}\n")
        
        return (noise,)
    
    def _fbm_noise(self, width, height, scale, octaves, persistence, lacunarity, mode, tileable, seed, device, dtype):
        """Fractional Brownian Motion noise"""
        noise = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for octave in range(octaves):
            octave_noise = self._perlin_noise(
                width, height, scale / frequency, 
                seed + octave, tileable, device, dtype
            )
            
            if mode == "turbulence":
                octave_noise = torch.abs(octave_noise)
            
            noise += octave_noise * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
        
        # Normalize
        noise = noise / (max_value + 1e-7)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
        
        return noise
    
    def _perlin_noise(self, width, height, scale, seed, tileable, device, dtype):
        """Simplified Perlin-like noise"""
        torch.manual_seed(seed)
        
        grid_h = max(int(height / scale), 2) + 2
        grid_w = max(int(width / scale), 2) + 2
        
        if tileable:
            # Create tileable by wrapping edges
            grid = torch.rand((grid_h, grid_w), device=device, dtype=dtype)
        else:
            grid = torch.rand((grid_h, grid_w), device=device, dtype=dtype)
        
        # Smooth interpolation
        upsampled = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bicubic',
            align_corners=False
        )
        
        noise = upsampled.permute(0, 2, 3, 1)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-7)
        
        return noise
    
    def _voronoi_noise(self, width, height, scale, seed, device, dtype):
        """Voronoi/Worley noise"""
        torch.manual_seed(seed)
        
        num_points = max(int((width * height) / (scale * scale)), 10)
        
        # Random point positions
        points_x = torch.rand(num_points, device=device, dtype=dtype) * width
        points_y = torch.rand(num_points, device=device, dtype=dtype) * height
        
        # Create coordinate grid
        y_coords = torch.arange(height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.arange(width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        # Find distance to nearest point
        min_dist = torch.ones((height, width), device=device, dtype=dtype) * float('inf')
        
        for i in range(num_points):
            dx = x_coords - points_x[i]
            dy = y_coords - points_y[i]
            dist = torch.sqrt(dx * dx + dy * dy)
            min_dist = torch.min(min_dist, dist)
        
        # Normalize
        noise = min_dist / (min_dist.max() + 1e-7)
        noise = noise.unsqueeze(0).unsqueeze(3)
        
        return noise
    
    def _cellular_noise(self, width, height, scale, seed, device, dtype):
        """Cellular noise (second nearest distance)"""
        torch.manual_seed(seed)
        
        num_points = max(int((width * height) / (scale * scale)), 10)
        
        points_x = torch.rand(num_points, device=device, dtype=dtype) * width
        points_y = torch.rand(num_points, device=device, dtype=dtype) * height
        
        y_coords = torch.arange(height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.arange(width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        # Find two nearest distances
        distances = []
        for i in range(num_points):
            dx = x_coords - points_x[i]
            dy = y_coords - points_y[i]
            dist = torch.sqrt(dx * dx + dy * dy)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=0)
        sorted_dist, _ = torch.sort(distances, dim=0)
        
        # Use second nearest - first nearest for cell pattern
        noise = sorted_dist[1] - sorted_dist[0]
        noise = noise / (noise.max() + 1e-7)
        noise = noise.unsqueeze(0).unsqueeze(3)
        
        return noise


class PatternGenerator:
    """
    Generate geometric patterns (brick, tile, hexagon, scales, weave)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "pattern_type": (["brick", "tile", "hexagon", "scales", "weave", "checker", "grid"],),
                "size": ("FLOAT", {
                    "default": 50.0,
                    "min": 5.0,
                    "max": 500.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Pattern element size"
                }),
                "gap": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Gap between elements"
                }),
                "randomness": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Pattern variation"
                }),
                "mortar_color": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Gap/mortar brightness"
                }),
                "tile_color": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Tile brightness"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pattern", "normal")
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Procedural"
    
    def generate(self, width, height, pattern_type, size, gap, randomness, mortar_color, tile_color, seed):
        """Generate geometric pattern"""
        
        print(f"\n{'='*60}")
        print(f"Pattern Generator - {pattern_type.upper()}")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        torch.manual_seed(seed)
        
        if pattern_type == "brick":
            pattern = self._brick_pattern(width, height, size, gap, randomness, mortar_color, tile_color, device, dtype)
        elif pattern_type == "hexagon":
            pattern = self._hexagon_pattern(width, height, size, gap, mortar_color, tile_color, device, dtype)
        elif pattern_type == "scales":
            pattern = self._scales_pattern(width, height, size, gap, mortar_color, tile_color, device, dtype)
        elif pattern_type == "checker":
            pattern = self._checker_pattern(width, height, size, mortar_color, tile_color, device, dtype)
        else:  # tile, weave, grid
            pattern = self._tile_pattern(width, height, size, gap, randomness, mortar_color, tile_color, device, dtype)
        
        # Generate simple normal map from pattern (height-based)
        normal = self._pattern_to_normal(pattern)
        
        # Convert to 3-channel
        pattern = pattern.repeat(1, 1, 1, 3)
        
        print(f"✓ Pattern generated")
        print(f"  Type: {pattern_type}")
        print(f"  Resolution: {width}×{height}")
        print(f"  Size: {size:.1f}")
        print(f"{'='*60}\n")
        
        return (torch.clamp(pattern, 0.0, 1.0), torch.clamp(normal, 0.0, 1.0))
    
    def _brick_pattern(self, width, height, size, gap, randomness, mortar_color, tile_color, device, dtype):
        """Classic brick pattern"""
        pattern = torch.ones((1, height, width, 1), device=device, dtype=dtype) * tile_color
        
        brick_height = int(size)
        brick_width = int(size * 2)
        gap_size = int(gap)
        
        for y in range(0, height, brick_height):
            # Offset every other row
            row_num = y // brick_height
            offset = (brick_width // 2) if row_num % 2 else 0
            
            # Horizontal mortar
            if y + gap_size < height:
                pattern[:, y:y+gap_size, :, :] = mortar_color
            
            # Vertical mortar
            for x in range(-offset, width, brick_width):
                x_start = max(0, x)
                x_end = min(width, x + gap_size)
                if x_start < width:
                    pattern[:, max(0, y):min(height, y+brick_height), x_start:x_end, :] = mortar_color
        
        return pattern
    
    def _tile_pattern(self, width, height, size, gap, randomness, mortar_color, tile_color, device, dtype):
        """Square tile pattern"""
        pattern = torch.ones((1, height, width, 1), device=device, dtype=dtype) * tile_color
        
        tile_size = int(size)
        gap_size = int(gap)
        
        for y in range(0, height, tile_size):
            pattern[:, y:min(height, y+gap_size), :, :] = mortar_color
        
        for x in range(0, width, tile_size):
            pattern[:, :, x:min(width, x+gap_size), :] = mortar_color
        
        return pattern
    
    def _checker_pattern(self, width, height, size, mortar_color, tile_color, device, dtype):
        """Checkerboard pattern"""
        pattern = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        
        tile_size = int(size)
        
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                row = y // tile_size
                col = x // tile_size
                
                color = tile_color if (row + col) % 2 == 0 else mortar_color
                
                pattern[:, 
                       y:min(height, y+tile_size), 
                       x:min(width, x+tile_size), 
                       :] = color
        
        return pattern
    
    def _hexagon_pattern(self, width, height, size, gap, mortar_color, tile_color, device, dtype):
        """Hexagonal pattern (simplified)"""
        # Simplified hex pattern using circles
        pattern = torch.ones((1, height, width, 1), device=device, dtype=dtype) * mortar_color
        
        hex_size = size
        hex_spacing = hex_size * 0.866  # sqrt(3)/2
        
        y_coords = torch.arange(height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.arange(width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        for row in range(int(-height / hex_spacing), int(2 * height / hex_spacing)):
            for col in range(int(-width / hex_size), int(2 * width / hex_size)):
                center_x = col * hex_size + (hex_size / 2 if row % 2 else 0)
                center_y = row * hex_spacing
                
                dx = x_coords - center_x
                dy = y_coords - center_y
                dist = torch.sqrt(dx * dx + dy * dy)
                
                mask = dist < (hex_size / 2 - gap)
                pattern[:, :, :, 0] = torch.where(mask, 
                                                  torch.ones_like(pattern[:, :, :, 0]) * tile_color,
                                                  pattern[:, :, :, 0])
        
        return pattern
    
    def _scales_pattern(self, width, height, size, gap, mortar_color, tile_color, device, dtype):
        """Fish scale / roof tile pattern"""
        pattern = torch.ones((1, height, width, 1), device=device, dtype=dtype) * mortar_color
        
        scale_size = size
        scale_spacing = scale_size * 0.5
        
        y_coords = torch.arange(height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.arange(width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        for row in range(int(-height / scale_spacing), int(2 * height / scale_spacing)):
            for col in range(int(-width / scale_size), int(2 * width / scale_size)):
                center_x = col * scale_size + (scale_size / 2 if row % 2 else 0)
                center_y = row * scale_spacing
                
                dx = x_coords - center_x
                dy = (y_coords - center_y) * 1.5  # Elongate vertically
                dist = torch.sqrt(dx * dx + dy * dy)
                
                # Only draw bottom half of circle
                mask = (dist < (scale_size / 2 - gap)) & (y_coords > center_y)
                pattern[:, :, :, 0] = torch.where(mask,
                                                  torch.ones_like(pattern[:, :, :, 0]) * tile_color,
                                                  pattern[:, :, :, 0])
        
        return pattern
    
    def _pattern_to_normal(self, pattern):
        """Generate simple normal map from pattern"""
        # Sobel operator for normal generation
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=pattern.dtype, device=pattern.device)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=pattern.dtype, device=pattern.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        pattern_t = pattern.permute(0, 3, 1, 2)
        
        dx = F.conv2d(F.pad(pattern_t, (1, 1, 1, 1), mode='replicate'), sobel_x)
        dy = F.conv2d(F.pad(pattern_t, (1, 1, 1, 1), mode='replicate'), sobel_y)
        
        # Create normal map
        dz = torch.ones_like(dx)
        length = torch.sqrt(dx*dx + dy*dy + dz*dz)
        
        nx = -dx / (length + 1e-7)
        ny = -dy / (length + 1e-7)
        nz = dz / (length + 1e-7)
        
        # Convert to 0-1 range
        nx = (nx + 1.0) * 0.5
        ny = (ny + 1.0) * 0.5
        nz = (nz + 1.0) * 0.5
        
        normal = torch.cat([nx, ny, nz], dim=1)
        normal = normal.permute(0, 2, 3, 1)
        
        return normal


class ScratchesGenerator:
    """
    Generate procedural scratches and surface imperfections
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "density": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Number of scratches"
                }),
                "length_min": ("FLOAT", {
                    "default": 50.0,
                    "min": 5.0,
                    "max": 500.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "length_max": ("FLOAT", {
                    "default": 200.0,
                    "min": 5.0,
                    "max": 1000.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "width_scratch": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Scratch width in pixels"
                }),
                "angle_variation": ("FLOAT", {
                    "default": 180.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "Angular spread (0=vertical, 180=all directions)"
                }),
                "intensity": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Scratch brightness"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("scratches", "height")
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Procedural"
    
    def generate(self, width, height, density, length_min, length_max, width_scratch, 
                 angle_variation, intensity, seed):
        """Generate procedural scratches"""
        
        print(f"\n{'='*60}")
        print(f"Scratches Generator")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        torch.manual_seed(seed)
        
        # Start with black background
        scratches = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        
        # Calculate number of scratches based on density
        num_scratches = int((width * height * density) / 10000)
        
        print(f"  Generating {num_scratches} scratches...")
        
        for i in range(num_scratches):
            # Random start position
            start_x = torch.rand(1).item() * width
            start_y = torch.rand(1).item() * height
            
            # Random length
            length = length_min + torch.rand(1).item() * (length_max - length_min)
            
            # Random angle
            angle = (torch.rand(1).item() - 0.5) * angle_variation
            angle_rad = math.radians(angle)
            
            # End position
            end_x = start_x + length * math.cos(angle_rad)
            end_y = start_y + length * math.sin(angle_rad)
            
            # Draw scratch as a line
            self._draw_line(scratches, start_x, start_y, end_x, end_y, width_scratch, intensity)
        
        # Create height map (inverted scratches for depth)
        height_map = 1.0 - scratches * 0.3
        
        # Convert to 3-channel
        scratches_rgb = scratches.repeat(1, 1, 1, 3)
        height_rgb = height_map.repeat(1, 1, 1, 3)
        
        print(f"✓ Scratches generated")
        print(f"  Count: {num_scratches}")
        print(f"  Resolution: {width}×{height}")
        print(f"{'='*60}\n")
        
        return (torch.clamp(scratches_rgb, 0.0, 1.0), torch.clamp(height_rgb, 0.0, 1.0))
    
    def _draw_line(self, image, x0, y0, x1, y1, width, intensity):
        """Draw a line on the image (Bresenham's algorithm with width)"""
        batch, h, w, c = image.shape
        
        # Clip to image bounds
        x0, y0 = max(0, min(w-1, x0)), max(0, min(h-1, y0))
        x1, y1 = max(0, min(w-1, x1)), max(0, min(h-1, y1))
        
        # Simple line drawing
        steps = int(max(abs(x1 - x0), abs(y1 - y0)))
        if steps == 0:
            return
        
        for i in range(steps):
            t = i / steps
            x = int(x0 + t * (x1 - x0))
            y = int(y0 + t * (y1 - y0))
            
            # Draw with width
            half_width = int(width / 2)
            for dy in range(-half_width, half_width + 1):
                for dx in range(-half_width, half_width + 1):
                    px = x + dx
                    py = y + dy
                    
                    if 0 <= px < w and 0 <= py < h:
                        # Soft falloff at edges
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist <= width / 2:
                            fade = 1.0 - (dist / (width / 2))
                            current = image[0, py, px, 0].item()
                            image[0, py, px, 0] = max(current, intensity * fade)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ProceduralNoiseGenerator": ProceduralNoiseGenerator,
    "PatternGenerator": PatternGenerator,
    "ScratchesGenerator": ScratchesGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProceduralNoiseGenerator": "Procedural Noise Generator",
    "PatternGenerator": "Pattern Generator",
    "ScratchesGenerator": "Scratches Generator",
}

