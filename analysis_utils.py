"""
Analysis and Utility Tools for TextureAlchemy
Texture analyzer, UV tools, atlas packer
"""

import torch
import torch.nn.functional as F


class TextureAnalyzer:
    """
    Analyze texture properties and statistics
    Resolution, color space, seamless detection, stats
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "check_seamless": ("BOOLEAN", {"default": True}),
                "edge_tolerance": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Tolerance for seamless detection"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("analysis", "visualization")
    FUNCTION = "analyze"
    CATEGORY = "Texture Alchemist/Analysis"
    OUTPUT_NODE = True
    
    def analyze(self, image, check_seamless, edge_tolerance):
        """Analyze texture"""
        
        print(f"\n{'='*60}")
        print(f"TEXTURE ANALYZER")
        print(f"{'='*60}")
        
        batch, height, width, channels = image.shape
        
        # Basic info
        resolution = f"{width}×{height}"
        aspect_ratio = width / height
        megapixels = (width * height) / 1000000
        
        # Color analysis
        mean_rgb = [image[:, :, :, i].mean().item() for i in range(min(3, channels))]
        std_rgb = [image[:, :, :, i].std().item() for i in range(min(3, channels))]
        min_vals = [image[:, :, :, i].min().item() for i in range(min(3, channels))]
        max_vals = [image[:, :, :, i].max().item() for i in range(min(3, channels))]
        
        # Luminosity
        if channels >= 3:
            lum = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
        else:
            lum = image[:, :, :, 0]
        
        brightness = lum.mean().item()
        contrast = lum.std().item()
        
        # Seamless check
        is_seamless_h = False
        is_seamless_v = False
        
        if check_seamless:
            # Check horizontal seam
            left = image[:, :, 0:5, :]
            right = image[:, :, -5:, :]
            h_diff = torch.abs(left - right).mean().item()
            is_seamless_h = h_diff < edge_tolerance
            
            # Check vertical seam
            top = image[:, 0:5, :, :]
            bottom = image[:, -5:, :, :]
            v_diff = torch.abs(top - bottom).mean().item()
            is_seamless_v = v_diff < edge_tolerance
        
        # Build analysis report
        report = []
        report.append(f"📊 TEXTURE ANALYSIS REPORT")
        report.append(f"")
        report.append(f"🖼️ RESOLUTION:")
        report.append(f"  Size: {resolution}")
        report.append(f"  Aspect Ratio: {aspect_ratio:.2f}:1")
        report.append(f"  Megapixels: {megapixels:.2f}MP")
        report.append(f"  Channels: {channels}")
        report.append(f"")
        report.append(f"🎨 COLOR STATISTICS:")
        if channels >= 3:
            report.append(f"  Mean RGB: ({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f})")
            report.append(f"  Std Dev: ({std_rgb[0]:.3f}, {std_rgb[1]:.3f}, {std_rgb[2]:.3f})")
            report.append(f"  Range R: [{min_vals[0]:.3f}, {max_vals[0]:.3f}]")
            report.append(f"  Range G: [{min_vals[1]:.3f}, {max_vals[1]:.3f}]")
            report.append(f"  Range B: [{min_vals[2]:.3f}, {max_vals[2]:.3f}]")
        report.append(f"  Brightness: {brightness:.3f}")
        report.append(f"  Contrast: {contrast:.3f}")
        report.append(f"")
        
        if check_seamless:
            report.append(f"🔄 SEAMLESS CHECK:")
            report.append(f"  Horizontal: {'✓ SEAMLESS' if is_seamless_h else '✗ NOT SEAMLESS'}")
            report.append(f"  Vertical: {'✓ SEAMLESS' if is_seamless_v else '✗ NOT SEAMLESS'}")
            report.append(f"  Edge difference: H={h_diff:.4f}, V={v_diff:.4f}")
            report.append(f"")
        
        # Create visualization
        viz = self._create_visualization(image, mean_rgb, is_seamless_h, is_seamless_v)
        
        report_str = "\n".join(report)
        
        # Print to console
        for line in report:
            print(f"  {line}")
        
        print(f"{'='*60}\n")
        
        return (report_str, viz)
    
    def _create_visualization(self, image, mean_rgb, seamless_h, seamless_v):
        """Create visual analysis"""
        batch, h, w, c = image.shape
        
        # Create avg color swatch
        avg_color = torch.tensor(mean_rgb + [1.0] * (3 - len(mean_rgb)), 
                                device=image.device, dtype=image.dtype)
        swatch = avg_color.view(1, 1, 1, 3).repeat(1, h//4, w//4, 1)
        
        # Create edge visualization if checking seamless
        if seamless_h or seamless_v:
            edges = torch.zeros_like(image)
            # Mark edges
            edges[:, :, 0:3, :] = 1.0 if seamless_v else torch.tensor([1.0, 0.0, 0.0])  # Top
            edges[:, :, -3:, :] = 1.0 if seamless_v else torch.tensor([1.0, 0.0, 0.0])  # Bottom
            edges[:, 0:3, :, :] = 1.0 if seamless_h else torch.tensor([1.0, 0.0, 0.0])  # Left
            edges[:, -3:, :, :] = 1.0 if seamless_h else torch.tensor([1.0, 0.0, 0.0])  # Right
            
            # Blend with image
            viz = image * 0.7 + edges * 0.3
        else:
            viz = image
        
        # Add swatch in corner
        viz[:, :h//4, :w//4, :] = swatch
        
        return torch.clamp(viz, 0.0, 1.0)


class UVCheckerGenerator:
    """
    Generate UV test patterns for checking UV mapping
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 256}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 256}),
                "pattern_type": (["grid", "checker", "numbered", "gradient"],),
                "grid_size": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Number of grid divisions"
                }),
                "color_mode": (["rainbow", "grayscale", "red_green"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("uv_checker",)
    FUNCTION = "generate"
    CATEGORY = "Texture Alchemist/Analysis"
    
    def generate(self, width, height, pattern_type, grid_size, color_mode):
        """Generate UV checker pattern"""
        
        print(f"\n{'='*60}")
        print(f"UV Checker Generator - {pattern_type.upper()}")
        print(f"{'='*60}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        
        # Create coordinate grid
        y_coords = torch.linspace(0, 1, height, device=device, dtype=dtype).view(-1, 1).repeat(1, width)
        x_coords = torch.linspace(0, 1, width, device=device, dtype=dtype).view(1, -1).repeat(height, 1)
        
        # Create pattern based on type
        if pattern_type == "grid":
            pattern = self._grid_pattern(x_coords, y_coords, grid_size, color_mode)
        elif pattern_type == "checker":
            pattern = self._checker_pattern(x_coords, y_coords, grid_size)
        elif pattern_type == "gradient":
            pattern = torch.stack([x_coords, y_coords, torch.ones_like(x_coords) * 0.5], dim=2)
        else:  # numbered
            pattern = self._grid_pattern(x_coords, y_coords, grid_size, color_mode)
        
        pattern = pattern.unsqueeze(0)  # Add batch dim
        pattern = torch.clamp(pattern, 0.0, 1.0)
        
        print(f"✓ UV checker generated")
        print(f"  Resolution: {width}×{height}")
        print(f"  Grid: {grid_size}×{grid_size}")
        print(f"{'='*60}\n")
        
        return (pattern,)
    
    def _grid_pattern(self, x, y, grid_size, color_mode):
        """Create grid pattern"""
        grid_x = (x * grid_size) % 1.0
        grid_y = (y * grid_size) % 1.0
        
        # Draw grid lines
        line_width = 0.05
        lines = ((grid_x < line_width) | (grid_y < line_width)).float()
        
        # Create colored cells
        cell_x = torch.floor(x * grid_size).long() % 3
        cell_y = torch.floor(y * grid_size).long() % 3
        
        if color_mode == "rainbow":
            r = ((cell_x + cell_y) % 3) / 2.0
            g = ((cell_x * 2) % 3) / 2.0
            b = ((cell_y * 2) % 3) / 2.0
        elif color_mode == "red_green":
            r = cell_x.float() / 2.0
            g = cell_y.float() / 2.0
            b = torch.zeros_like(x)
        else:  # grayscale
            val = ((cell_x + cell_y) % 2).float() * 0.5 + 0.25
            r = g = b = val
        
        # Combine
        r = r * (1.0 - lines) + lines
        g = g * (1.0 - lines) + lines
        b = b * (1.0 - lines) + lines
        
        return torch.stack([r, g, b], dim=2)
    
    def _checker_pattern(self, x, y, grid_size):
        """Create checker pattern"""
        cell_x = torch.floor(x * grid_size).long()
        cell_y = torch.floor(y * grid_size).long()
        
        checker = ((cell_x + cell_y) % 2).float()
        
        return checker.unsqueeze(2).repeat(1, 1, 3)


class TextureAtlasBuilder:
    """
    Combine multiple textures into a texture atlas
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (["2x2", "3x3", "4x4", "1x2", "1x3", "1x4", "2x1", "3x1", "4x1"],),
                "spacing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Pixels between textures"
                }),
                "background_color": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "tex1": ("IMAGE",),
                "tex2": ("IMAGE",),
                "tex3": ("IMAGE",),
                "tex4": ("IMAGE",),
                "tex5": ("IMAGE",),
                "tex6": ("IMAGE",),
                "tex7": ("IMAGE",),
                "tex8": ("IMAGE",),
                "tex9": ("IMAGE",),
                "tex10": ("IMAGE",),
                "tex11": ("IMAGE",),
                "tex12": ("IMAGE",),
                "tex13": ("IMAGE",),
                "tex14": ("IMAGE",),
                "tex15": ("IMAGE",),
                "tex16": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("atlas",)
    FUNCTION = "build_atlas"
    CATEGORY = "Texture Alchemist/Analysis"
    
    def build_atlas(self, layout, spacing, background_color, **textures):
        """Build texture atlas"""
        
        print(f"\n{'='*60}")
        print(f"Texture Atlas Builder - {layout}")
        print(f"{'='*60}")
        
        # Parse layout
        rows, cols = map(int, layout.split('x'))
        max_textures = rows * cols
        
        # Collect textures
        tex_list = []
        for i in range(1, max_textures + 1):
            key = f"tex{i}"
            if key in textures and textures[key] is not None:
                tex_list.append(textures[key])
        
        if not tex_list:
            print("⚠ No textures provided!")
            return (torch.zeros((1, 512, 512, 3)),)
        
        # Find max dimensions
        max_h = max([t.shape[1] for t in tex_list])
        max_w = max([t.shape[2] for t in tex_list])
        
        # Calculate atlas size
        atlas_w = cols * max_w + (cols - 1) * spacing
        atlas_h = rows * max_h + (rows - 1) * spacing
        
        device = tex_list[0].device
        dtype = tex_list[0].dtype
        
        # Create atlas
        atlas = torch.ones((1, atlas_h, atlas_w, 3), device=device, dtype=dtype) * background_color
        
        # Place textures
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= len(tex_list):
                    break
                
                tex = tex_list[idx]
                
                # Resize to max dimensions
                if tex.shape[1:3] != (max_h, max_w):
                    tex = F.interpolate(
                        tex.permute(0, 3, 1, 2),
                        size=(max_h, max_w),
                        mode='bilinear',
                        align_corners=False
                    ).permute(0, 2, 3, 1)
                
                # Calculate position
                y = row * (max_h + spacing)
                x = col * (max_w + spacing)
                
                # Place
                atlas[:, y:y+max_h, x:x+max_w, :] = tex
                
                idx += 1
        
        print(f"✓ Atlas created")
        print(f"  Layout: {rows}×{cols}")
        print(f"  Textures: {len(tex_list)}")
        print(f"  Cell size: {max_w}×{max_h}")
        print(f"  Atlas size: {atlas_w}×{atlas_h}")
        print(f"{'='*60}\n")
        
        return (atlas,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "TextureAnalyzer": TextureAnalyzer,
    "UVCheckerGenerator": UVCheckerGenerator,
    "TextureAtlasBuilder": TextureAtlasBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextureAnalyzer": "Texture Analyzer",
    "UVCheckerGenerator": "UV Checker Generator",
    "TextureAtlasBuilder": "Texture Atlas Builder",
}

