"""
Normal Map Utilities
Tools for combining, processing, and adjusting normal maps
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
    CATEGORY = "PBR/Normals"
    
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


class LotusNormalProcessor:
    """
    Process Lotus normal map output with channel controls
    Connect: LotusSampler → VAEDecode → This node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "invert_red": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert red channel (X axis)"
                }),
                "invert_green": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Invert green channel (Y axis) - OpenGL vs DirectX"
                }),
                "invert_blue": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert blue channel (Z axis)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Normal map strength (1.0 = original)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "process"
    CATEGORY = "PBR/Normals"
    
    def process(self, normal, invert_red, invert_green, invert_blue, strength):
        """Process normal map"""
        
        print("\n" + "="*60)
        print("Lotus Normal Processor")
        print("="*60)
        print(f"Input shape: {normal.shape}")
        print(f"Invert - R:{invert_red}, G:{invert_green}, B:{invert_blue}")
        print(f"Strength: {strength}")
        
        result = normal.clone()
        
        # Ensure 3 channels
        if result.shape[-1] < 3:
            print("\n✗ ERROR: Normal map must have at least 3 channels (RGB)")
            return (normal,)
        
        # Convert to [-1, 1] range for processing
        result = result * 2.0 - 1.0
        
        # Invert channels as requested
        if invert_red:
            result[..., 0] = -result[..., 0]
            print("✓ Red channel inverted")
        
        if invert_green:
            result[..., 1] = -result[..., 1]
            print("✓ Green channel inverted")
        
        if invert_blue:
            result[..., 2] = -result[..., 2]
            print("✓ Blue channel inverted")
        
        # Apply strength
        # Blend between flat normal (0,0,1) and the processed normal
        if strength != 1.0:
            flat_normal = torch.tensor([0.0, 0.0, 1.0], 
                                      device=result.device, dtype=result.dtype)
            flat_normal = flat_normal.view(1, 1, 1, 3)
            result[..., :3] = result[..., :3] * strength + flat_normal * (1.0 - strength)
            
            # Normalize
            length = torch.sqrt(torch.sum(result[..., :3] * result[..., :3], dim=-1, keepdim=True))
            length = torch.clamp(length, min=1e-6)
            result[..., :3] = result[..., :3] / length
            
            print(f"✓ Strength applied: {strength}")
        
        # Convert back to [0, 1] range
        result = result * 0.5 + 0.5
        
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print("\n" + "="*60)
        print("✓ Normal Processing Complete")
        print("="*60 + "\n")
        
        return (result,)


# Node registration
class NormalToDepth:
    """
    Convert normal maps to depth/height maps
    Uses integration and approximation methods
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "method": (["integration", "blue_channel", "hybrid"], {
                    "default": "hybrid",
                    "tooltip": "Conversion method: integration (accurate), blue_channel (fast), hybrid (balanced)"
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Depth strength multiplier"
                }),
                "iterations": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Integration iterations (higher = more accurate but slower)"
                }),
                "blur_radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Smoothing blur radius"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Normal"
    
    def convert(self, normal, method, strength, iterations, blur_radius):
        """Convert normal map to depth map"""
        
        print("\n" + "="*60)
        print("Normal to Depth Converter")
        print("="*60)
        print(f"Input shape: {normal.shape}, dtype: {normal.dtype}")
        print(f"Method: {method}")
        print(f"Strength: {strength}")
        
        # Ensure input is float32
        normal = normal.float()
        
        batch, height, width, channels = normal.shape
        device = normal.device
        dtype = torch.float32  # Always use float32 for processing
        
        if method == "blue_channel":
            # Simple method: use blue channel as depth approximation
            print("Using blue channel method (fast)")
            if channels >= 3:
                depth = normal[:, :, :, 2:3]  # Blue channel
            else:
                depth = normal[:, :, :, 0:1]
            
            depth = depth * strength
            
        elif method == "integration":
            # Integration method: integrate gradients from normal map
            print(f"Using integration method ({iterations} iterations)")
            depth = self._integrate_gradients(normal, iterations, strength)
            
        else:  # hybrid
            # Combine both methods
            print(f"Using hybrid method ({iterations} iterations)")
            
            # Get blue channel approximation
            if channels >= 3:
                blue_depth = normal[:, :, :, 2:3]
            else:
                blue_depth = normal[:, :, :, 0:1]
            
            # Get integrated depth
            integrated_depth = self._integrate_gradients(normal, iterations, 1.0)
            
            # Blend them
            depth = (blue_depth * 0.3 + integrated_depth * 0.7) * strength
        
        # Apply smoothing if requested
        if blur_radius > 0:
            depth = self._apply_blur(depth, blur_radius)
            print(f"Applied blur: radius={blur_radius}")
        
        # Normalize to 0-1 range and ensure float32
        depth = depth.float()  # Ensure float32
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        # Ensure output is in correct format
        depth = torch.clamp(depth, 0.0, 1.0)
        
        # Convert to 3-channel RGB for display compatibility
        if depth.shape[-1] == 1:
            depth = depth.repeat(1, 1, 1, 3)
        
        print(f"✓ Depth map generated")
        print(f"  Output shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"  Output range: [{depth.min():.3f}, {depth.max():.3f}]")
        print("="*60 + "\n")
        
        return (depth,)
    
    def _integrate_gradients(self, normal, iterations, strength):
        """Integrate normal map gradients to reconstruct depth"""
        batch, height, width, channels = normal.shape
        device = normal.device
        dtype = normal.dtype
        
        # Convert normal map from [0,1] to [-1,1]
        normal_scaled = normal * 2.0 - 1.0
        
        # Extract gradients from red and green channels
        if channels >= 2:
            grad_x = normal_scaled[:, :, :, 0:1]  # Red = X gradient
            grad_y = normal_scaled[:, :, :, 1:2]  # Green = Y gradient
        else:
            grad_x = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
            grad_y = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Initialize height map
        depth = torch.zeros((batch, height, width, 1), device=device, dtype=dtype)
        
        # Iterative integration (Jacobi method)
        for iteration in range(iterations):
            depth_old = depth.clone()
            
            # Compute neighbors
            depth_left = torch.cat([depth[:, :, :1, :], depth[:, :, :-1, :]], dim=2)
            depth_right = torch.cat([depth[:, :, 1:, :], depth[:, :, -1:, :]], dim=2)
            depth_up = torch.cat([depth[:, :1, :, :], depth[:, :-1, :, :]], dim=1)
            depth_down = torch.cat([depth[:, 1:, :, :], depth[:, -1:, :, :]], dim=1)
            
            # Integrate gradients
            depth = 0.25 * (
                depth_left + grad_x * strength +
                depth_right - grad_x * strength +
                depth_up + grad_y * strength +
                depth_down - grad_y * strength
            )
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0 and iteration > 0:
                diff = torch.abs(depth - depth_old).mean()
                if diff < 1e-5:
                    print(f"  Converged at iteration {iteration}")
                    break
        
        return depth
    
    def _apply_blur(self, image, radius):
        """Apply Gaussian blur"""
        if radius <= 0:
            return image
        
        batch, height, width, channels = image.shape
        
        # Create Gaussian kernel
        kernel_size = int(radius * 4) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        sigma = radius
        kernel_range = torch.arange(kernel_size, device=image.device, dtype=image.dtype)
        kernel_center = kernel_size // 2
        kernel = torch.exp(-0.5 * ((kernel_range - kernel_center) / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Reshape for convolution
        image_flat = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        # Apply horizontal blur
        kernel_h = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
        padding_h = kernel_size // 2
        image_flat = torch.nn.functional.pad(image_flat, (padding_h, padding_h, 0, 0), mode='replicate')
        image_flat = torch.nn.functional.conv2d(image_flat, kernel_h, groups=channels)
        
        # Apply vertical blur
        kernel_v = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
        padding_v = kernel_size // 2
        image_flat = torch.nn.functional.pad(image_flat, (0, 0, padding_v, padding_v), mode='replicate')
        image_flat = torch.nn.functional.conv2d(image_flat, kernel_v, groups=channels)
        
        # Back to BHWC
        result = image_flat.permute(0, 2, 3, 1)
        
        return result


class HeightToNormal:
    """
    Convert height/depth maps to normal maps
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "height": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Normal map strength/intensity"
                }),
                "method": (["sobel", "scharr", "prewitt"], {
                    "default": "sobel",
                    "tooltip": "Edge detection method"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Normal"
    
    def convert(self, height, strength, method):
        """Convert height map to normal map"""
        
        print("\n" + "="*60)
        print("Height to Normal Converter")
        print("="*60)
        print(f"Input shape: {height.shape}")
        print(f"Strength: {strength}")
        print(f"Method: {method}")
        
        # Convert to grayscale if needed
        if height.shape[-1] > 1:
            weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                                   device=height.device, dtype=height.dtype)
            height_gray = torch.sum(height * weights, dim=-1, keepdim=True)
        else:
            height_gray = height
        
        batch, h, w, channels = height_gray.shape
        
        # Get gradient kernels
        if method == "sobel":
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    device=height.device, dtype=height.dtype) / 8.0
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    device=height.device, dtype=height.dtype) / 8.0
        elif method == "scharr":
            kernel_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], 
                                    device=height.device, dtype=height.dtype) / 32.0
            kernel_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], 
                                    device=height.device, dtype=height.dtype) / 32.0
        else:  # prewitt
            kernel_x = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], 
                                    device=height.device, dtype=height.dtype) / 6.0
            kernel_y = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], 
                                    device=height.device, dtype=height.dtype) / 6.0
        
        # Reshape for convolution
        kernel_x = kernel_x.view(1, 1, 3, 3)
        kernel_y = kernel_y.view(1, 1, 3, 3)
        
        height_bchw = height_gray.permute(0, 3, 1, 2)
        
        # Compute gradients
        grad_x = torch.nn.functional.conv2d(height_bchw, kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(height_bchw, kernel_y, padding=1)
        
        # Convert gradients to normal
        # Normal = (-dz/dx, -dz/dy, 1)
        normal_x = -grad_x * strength
        normal_y = -grad_y * strength
        normal_z = torch.ones_like(normal_x)
        
        # Normalize
        normal = torch.cat([normal_x, normal_y, normal_z], dim=1)
        length = torch.sqrt(torch.sum(normal ** 2, dim=1, keepdim=True)) + 1e-8
        normal = normal / length
        
        # Convert from [-1,1] to [0,1] for display
        normal = (normal + 1.0) / 2.0
        
        # Back to BHWC
        normal = normal.permute(0, 2, 3, 1)
        
        print(f"✓ Normal map generated")
        print(f"  Output shape: {normal.shape}")
        print("="*60 + "\n")
        
        return (normal,)


class NormalConverter:
    """
    Convert between DirectX and OpenGL normal map formats
    (Flips green channel)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "conversion": (["DirectX_to_OpenGL", "OpenGL_to_DirectX", "auto_detect"], {
                    "default": "auto_detect",
                    "tooltip": "Conversion direction (both flip green channel)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "convert"
    CATEGORY = "Texture Alchemist/Normal"
    
    def convert(self, normal, conversion):
        """Convert normal map between DirectX and OpenGL formats"""
        
        print("\n" + "="*60)
        print("Normal Map Format Converter")
        print("="*60)
        print(f"Input shape: {normal.shape}")
        print(f"Conversion: {conversion}")
        
        result = normal.clone()
        
        # Flip green channel (Y axis)
        if result.shape[-1] >= 2:
            result[:, :, :, 1] = 1.0 - result[:, :, :, 1]
            print(f"✓ Green channel flipped")
        else:
            print("⚠ Not enough channels to convert")
        
        print("="*60 + "\n")
        
        return (result,)


class NormalFormatValidator:
    """
    Detect and validate whether a normal map is OpenGL or DirectX format
    Analyzes green channel to determine Y-axis direction
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_map": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("visualization", "detected_format")
    FUNCTION = "validate"
    CATEGORY = "Texture Alchemist/Normal"
    OUTPUT_NODE = True
    
    def validate(self, normal_map):
        """Validate normal map format (OpenGL vs DirectX)"""
        
        print("\n" + "="*60)
        print("Normal Format Validator")
        print("="*60)
        print(f"Input shape: {normal_map.shape}")
        
        batch, height, width, channels = normal_map.shape
        device = normal_map.device
        dtype = normal_map.dtype
        
        # Extract green channel (Y-axis direction differs between formats)
        if channels >= 3:
            green = normal_map[:, :, :, 1]
        else:
            print("⚠ Warning: Normal map has less than 3 channels")
            return (normal_map, "UNKNOWN - Not RGB")
        
        # Analyze green channel distribution
        green_mean = green.mean().item()
        green_median = green.median().item()
        
        # Count pixels above and below 0.5 (neutral)
        above_half = (green > 0.5).float().mean().item()
        below_half = (green < 0.5).float().mean().item()
        
        # Calculate histogram-based detection
        # OpenGL: Tends to have more pixels > 0.5 (Y points up)
        # DirectX: Tends to have more pixels < 0.5 (Y points down, inverted)
        
        # Detect format based on bias
        bias = above_half - below_half
        
        if abs(bias) < 0.05:
            # Very close to 0.5 distribution - ambiguous or flat
            detected = "AMBIGUOUS"
            confidence = "Low"
            reason = "Flat or equal distribution"
        elif bias > 0:
            # More pixels above 0.5 = OpenGL
            detected = "OpenGL"
            confidence_value = abs(bias) * 100
            if confidence_value > 20:
                confidence = "High"
            elif confidence_value > 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            reason = f"Green bias: +{bias:.2%}"
        else:
            # More pixels below 0.5 = DirectX
            detected = "DirectX"
            confidence_value = abs(bias) * 100
            if confidence_value > 20:
                confidence = "High"
            elif confidence_value > 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            reason = f"Green bias: {bias:.2%}"
        
        # Create visualization
        visualization = self._create_visualization(
            normal_map, green, detected, confidence, 
            green_mean, above_half, below_half, bias, device, dtype
        )
        
        # Print detailed analysis
        print(f"\n📊 GREEN CHANNEL ANALYSIS:")
        print(f"  Mean: {green_mean:.3f}")
        print(f"  Median: {green_median:.3f}")
        print(f"  Pixels > 0.5: {above_half*100:.1f}% (up-facing)")
        print(f"  Pixels < 0.5: {below_half*100:.1f}% (down-facing)")
        print(f"  Bias: {bias:+.3f}")
        
        print(f"\n🎯 DETECTION RESULT:")
        print(f"  Format: {detected}")
        print(f"  Confidence: {confidence}")
        print(f"  Reason: {reason}")
        
        if detected == "OpenGL":
            print(f"\n✓ OpenGL Format")
            print(f"  • Green channel points UP (+Y)")
            print(f"  • Compatible with: Blender, Unity, Three.js")
            print(f"  • Standard in most modern engines")
        elif detected == "DirectX":
            print(f"\n✓ DirectX Format")
            print(f"  • Green channel points DOWN (-Y)")
            print(f"  • Compatible with: Unreal Engine, 3ds Max, Maya")
            print(f"  • 💡 Use 'Normal Format Converter' to flip to OpenGL")
        else:
            print(f"\n⚠ Ambiguous Result")
            print(f"  Possible reasons:")
            print(f"  • Normal map is mostly flat (no vertical detail)")
            print(f"  • Equal distribution of up/down facing normals")
            print(f"  • Non-standard or corrupted normal map")
        
        print("="*60 + "\n")
        
        result_string = f"{detected} | Confidence: {confidence} | Bias: {bias:+.2%}"
        
        return (visualization, result_string)
    
    def _create_visualization(self, normal_map, green, detected, confidence, 
                             green_mean, above_half, below_half, bias, device, dtype):
        """Create visual analysis panels"""
        batch, height, width, channels = normal_map.shape
        
        # Panel 1: Original normal map
        panel1 = normal_map.clone()
        
        # Panel 2: Green channel only (as RGB)
        green_rgb = green.unsqueeze(-1).repeat(1, 1, 1, 3)
        panel2 = green_rgb
        
        # Panel 3: Above/below 0.5 visualization
        # White = above 0.5 (up), Black = below 0.5 (down)
        threshold_vis = torch.where(
            green.unsqueeze(-1) > 0.5,
            torch.ones((batch, height, width, 3), device=device, dtype=dtype),
            torch.zeros((batch, height, width, 3), device=device, dtype=dtype)
        )
        panel3 = threshold_vis
        
        # Panel 4: Color-coded format indicator
        if "OpenGL" in detected:
            # Green = OpenGL (up-facing dominant)
            color = torch.tensor([0.2, 1.0, 0.2], device=device, dtype=dtype)
        elif "DirectX" in detected:
            # Red = DirectX (down-facing dominant)
            color = torch.tensor([1.0, 0.2, 0.2], device=device, dtype=dtype)
        else:
            # Yellow = Ambiguous
            color = torch.tensor([1.0, 1.0, 0.2], device=device, dtype=dtype)
        
        # Create gradient background based on bias
        gradient = torch.linspace(0.3, 0.7, height, device=device, dtype=dtype)
        gradient = gradient.view(1, -1, 1, 1).expand(batch, -1, width, 1)
        
        panel4 = gradient.repeat(1, 1, 1, 3) * color.view(1, 1, 1, 3)
        
        # Overlay statistics text would require PIL, so just use solid indicator
        # Blend format color with threshold visualization
        panel4 = panel4 * 0.5 + threshold_vis * 0.5
        
        # Stack panels horizontally
        visualization = torch.cat([panel1, panel2, panel3, panel4], dim=2)
        
        return visualization


class NormalIntensity:
    """
    Adjust normal map intensity/strength
    Blends between flat normal (0,0,1) and the actual normal
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Normal intensity (0.0=flat, 1.0=original, >1.0=enhanced)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "adjust_intensity"
    CATEGORY = "Texture Alchemist/Normal"
    
    def adjust_intensity(self, normal, intensity):
        """Adjust normal map intensity"""
        
        print("\n" + "="*60)
        print("Normal Intensity Adjuster")
        print("="*60)
        print(f"Input shape: {normal.shape}")
        print(f"Intensity: {intensity}")
        
        if intensity == 1.0:
            print("✓ Intensity is 1.0, no change needed")
            print("="*60 + "\n")
            return (normal,)
        
        # Convert from [0,1] to [-1,1] range
        normal_unpacked = normal * 2.0 - 1.0
        
        # Flat normal is (0, 0, 1) in tangent space
        flat = torch.tensor([0.0, 0.0, 1.0], device=normal.device, dtype=normal.dtype)
        flat = flat.view(1, 1, 1, 3)
        
        # Interpolate: result = flat + (normal - flat) * intensity
        # When intensity = 0: result = flat
        # When intensity = 1: result = normal
        # When intensity > 1: exaggerated normal
        normal_adjusted = flat + (normal_unpacked - flat) * intensity
        
        # Normalize the vectors
        length = torch.sqrt(torch.sum(normal_adjusted * normal_adjusted, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-6)
        normal_adjusted = normal_adjusted / length
        
        # Convert back to [0,1] range
        result = normal_adjusted * 0.5 + 0.5
        
        print(f"✓ Normal intensity adjusted")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print("="*60 + "\n")
        
        return (result,)


class SharpenNormal:
    """
    Sharpen normal maps to enhance detail
    Uses unsharp mask technique in tangent space
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Blur radius for unsharp mask"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("normal",)
    FUNCTION = "sharpen"
    CATEGORY = "Texture Alchemist/Normal"
    
    def sharpen(self, normal, strength, radius):
        """Sharpen normal map"""
        
        print("\n" + "="*60)
        print("Normal Sharpener")
        print("="*60)
        print(f"Input shape: {normal.shape}")
        print(f"Strength: {strength}")
        print(f"Radius: {radius}")
        
        if strength == 0.0:
            print("✓ Strength is 0.0, no sharpening applied")
            print("="*60 + "\n")
            return (normal,)
        
        # Convert to tangent space [-1,1]
        normal_unpacked = normal * 2.0 - 1.0
        
        # Create Gaussian blur kernel
        kernel_size = int(radius * 4) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian kernel
        sigma = radius
        x = torch.arange(kernel_size, dtype=normal.dtype, device=normal.device) - kernel_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # Convert to 2D kernel
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Apply blur to each channel separately
        normal_bchw = normal_unpacked.permute(0, 3, 1, 2)
        blurred_channels = []
        
        for i in range(3):
            channel = normal_bchw[:, i:i+1, :, :]
            kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
            blurred = torch.nn.functional.conv2d(channel, kernel, padding=kernel_size//2)
            blurred_channels.append(blurred)
        
        blurred = torch.cat(blurred_channels, dim=1)
        blurred = blurred.permute(0, 2, 3, 1)
        
        # Unsharp mask: sharpened = original + (original - blurred) * strength
        detail = normal_unpacked - blurred
        sharpened = normal_unpacked + detail * strength
        
        # Normalize the vectors (very important for normals!)
        length = torch.sqrt(torch.sum(sharpened * sharpened, dim=-1, keepdim=True))
        length = torch.clamp(length, min=1e-6)
        sharpened = sharpened / length
        
        # Convert back to [0,1] range
        result = sharpened * 0.5 + 0.5
        
        print(f"✓ Normal map sharpened")
        print(f"  Kernel size: {kernel_size}x{kernel_size}")
        print(f"  Output range: [{result.min():.3f}, {result.max():.3f}]")
        print("="*60 + "\n")
        
        return (result,)


class SharpenDepth:
    """
    Sharpen depth/height maps to enhance detail
    Uses unsharp mask technique
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Sharpening strength"
                }),
                "radius": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "Blur radius for unsharp mask"
                }),
                "clamp_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clamp output to 0-1 range"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "sharpen"
    CATEGORY = "Texture Alchemist/Height"
    
    def sharpen(self, depth, strength, radius, clamp_output):
        """Sharpen depth/height map"""
        
        print("\n" + "="*60)
        print("Depth Sharpener")
        print("="*60)
        print(f"Input shape: {depth.shape}")
        print(f"Strength: {strength}")
        print(f"Radius: {radius}")
        print(f"Clamp output: {clamp_output}")
        
        if strength == 0.0:
            print("✓ Strength is 0.0, no sharpening applied")
            print("="*60 + "\n")
            return (depth,)
        
        # Create Gaussian blur kernel
        kernel_size = int(radius * 4) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create 1D Gaussian kernel
        sigma = radius
        x = torch.arange(kernel_size, dtype=depth.dtype, device=depth.device) - kernel_size // 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # Convert to 2D kernel
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Apply blur to each channel separately
        depth_bchw = depth.permute(0, 3, 1, 2)
        blurred_channels = []
        
        for i in range(depth.shape[-1]):
            channel = depth_bchw[:, i:i+1, :, :]
            kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
            blurred = torch.nn.functional.conv2d(channel, kernel, padding=kernel_size//2)
            blurred_channels.append(blurred)
        
        blurred = torch.cat(blurred_channels, dim=1)
        blurred = blurred.permute(0, 2, 3, 1)
        
        # Unsharp mask: sharpened = original + (original - blurred) * strength
        detail = depth - blurred
        sharpened = depth + detail * strength
        
        # Clamp if requested
        if clamp_output:
            sharpened = torch.clamp(sharpened, 0.0, 1.0)
        
        print(f"✓ Depth map sharpened")
        print(f"  Kernel size: {kernel_size}x{kernel_size}")
        print(f"  Output range: [{sharpened.min():.3f}, {sharpened.max():.3f}]")
        print("="*60 + "\n")
        
        return (sharpened,)


NODE_CLASS_MAPPINGS = {
    "NormalMapCombiner": NormalMapCombiner,
    "LotusNormalProcessor": LotusNormalProcessor,
    "NormalToDepth": NormalToDepth,
    "HeightToNormal": HeightToNormal,
    "NormalConverter": NormalConverter,
    "NormalFormatValidator": NormalFormatValidator,
    "NormalIntensity": NormalIntensity,
    "SharpenNormal": SharpenNormal,
    "SharpenDepth": SharpenDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NormalMapCombiner": "Normal Map Combiner",
    "LotusNormalProcessor": "Normal Processor (Lotus)",
    "NormalToDepth": "Normal to Depth Converter",
    "HeightToNormal": "Height to Normal Converter",
    "NormalConverter": "Normal Format Converter (DX↔GL)",
    "NormalFormatValidator": "Normal Format Validator (OGL vs DX)",
    "NormalIntensity": "Normal Intensity Adjuster",
    "SharpenNormal": "Sharpen Normal",
    "SharpenDepth": "Sharpen Depth",
}

