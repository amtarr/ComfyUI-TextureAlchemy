"""
PBR Pipeline System
Combiner, advanced adjuster, and splitter for clean workflows
"""

import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path
import traceback


class PBRCombiner:
    """
    Combine all PBR maps into a single PBR_PIPE for easy workflow connections
    Can also add/override maps in an existing PBR_PIPE
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "pbr_pipe": ("PBR_PIPE",),
                "albedo": ("IMAGE",),
                "normal": ("IMAGE",),
                "ao": ("IMAGE",),
                "height": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "transparency": ("IMAGE",),
                "emission": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "combine"
    CATEGORY = "Texture Alchemist/Pipeline"
    
    def combine(self, pbr_pipe=None, albedo=None, normal=None, ao=None, height=None, roughness=None, metallic=None, transparency=None, emission=None):
        """Combine PBR maps into a pipeline"""
        
        print("\n" + "="*60)
        print("PBR Combiner")
        print("="*60)
        
        # Start with existing pipe or create new
        if pbr_pipe is not None:
            print("✓ Merging with existing PBR_PIPE")
            result_pipe = pbr_pipe.copy()
        else:
            print("✓ Creating new PBR_PIPE")
            result_pipe = {
                "albedo": None,
                "normal": None,
                "ao": None,
                "height": None,
                "roughness": None,
                "metallic": None,
                "transparency": None,
                "emission": None,
            }
        
        # Override with provided inputs
        if albedo is not None:
            result_pipe["albedo"] = albedo
            print(f"  Albedo: {albedo.shape}")
        if normal is not None:
            result_pipe["normal"] = normal
            print(f"  Normal: {normal.shape}")
        if ao is not None:
            result_pipe["ao"] = ao
            print(f"  AO: {ao.shape}")
        if height is not None:
            result_pipe["height"] = height
            print(f"  Height: {height.shape}")
        if roughness is not None:
            result_pipe["roughness"] = roughness
            print(f"  Roughness: {roughness.shape}")
        if metallic is not None:
            result_pipe["metallic"] = metallic
            print(f"  Metallic: {metallic.shape}")
        if transparency is not None:
            result_pipe["transparency"] = transparency
            print(f"  Transparency: {transparency.shape}")
        if emission is not None:
            result_pipe["emission"] = emission
            print(f"  Emission: {emission.shape}")
        
        # Show what's in the final pipe
        available = [k for k, v in result_pipe.items() if v is not None]
        print(f"\nFinal pipe contains: {', '.join(available)}")
        
        print("\n✓ PBR Pipeline created")
        print("="*60 + "\n")
        
        return (result_pipe,)


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
                "embed_transparency": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Embed transparency map into albedo's alpha channel"
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
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "adjust"
    CATEGORY = "Texture Alchemist/Pipeline"
    
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
               embed_transparency, albedo_dimmer, albedo_saturation):
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
        emission = pbr_pipe.get("emission", None)
        if emission is not None:
            emission = emission.clone()
        
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
        
        # ===== EMBED TRANSPARENCY INTO ALBEDO ALPHA =====
        if albedo is not None and transparency is not None and embed_transparency:
            # Convert transparency to grayscale if needed
            trans_gray = transparency
            if trans_gray.shape[-1] == 3:
                weights = torch.tensor([0.299, 0.587, 0.114], 
                                      device=trans_gray.device, dtype=trans_gray.dtype)
                trans_gray = torch.sum(trans_gray * weights, dim=-1, keepdim=True)
            
            # Resize transparency to match albedo if needed
            if trans_gray.shape[1:3] != albedo.shape[1:3]:
                trans_gray = torch.nn.functional.interpolate(
                    trans_gray.permute(0, 3, 1, 2),
                    size=albedo.shape[1:3],
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            # Ensure albedo is RGB (3 channels)
            if albedo.shape[-1] == 4:
                albedo = albedo[:, :, :, :3]
            elif albedo.shape[-1] == 1:
                albedo = albedo.repeat(1, 1, 1, 3)
            
            # Concatenate transparency as alpha channel
            albedo = torch.cat([albedo, trans_gray], dim=-1)
            print("✓ Transparency embedded into albedo alpha channel")
            print(f"  Albedo shape: {albedo.shape}")
        
        # Create output pipe
        output_pipe = {
            "albedo": albedo,
            "normal": normal,
            "ao": ao,
            "height": height,
            "roughness": roughness,
            "metallic": metallic,
            "transparency": transparency,
            "emission": emission,
        }
        
        print("\n" + "="*60)
        print("✓ Pipeline Adjustments Complete")
        print("="*60 + "\n")
        
        return (output_pipe,)


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
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo", "normal", "ao", "height", "roughness", "metallic", "transparency", "emission")
    FUNCTION = "split"
    CATEGORY = "Texture Alchemist/Pipeline"
    
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
        emission = pbr_pipe.get("emission", None)
        
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
        
        if emission is not None:
            print(f"  ✓ Emission: {emission.shape}")
        else:
            emission = create_placeholder()
            print(f"  ✗ Emission: None (using placeholder)")
        
        print("\n✓ Maps extracted from pipeline")
        print("="*60 + "\n")
        
        return (albedo, normal, ao, height, roughness, metallic, transparency, emission)


class PBRSaver:
    """
    Save all maps from PBR_PIPE with custom naming and enumeration
    Example: bricks_albedo_001.png, bricks_metallic_001.png
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
                "base_name": ("STRING", {
                    "default": "material",
                    "tooltip": "Base name for saved files (e.g., 'bricks')"
                }),
                "output_path": ("STRING", {
                    "default": "pbr_materials",
                    "tooltip": "Folder name inside ComfyUI/output (or full path if absolute)"
                }),
                "file_format": (["png", "png16", "jpg", "exr", "tiff"], {
                    "default": "png",
                    "tooltip": "Image format (png16/EXR for high bit depth)"
                }),
                "enumeration_mode": (["enumerate", "overwrite"], {
                    "default": "enumerate",
                    "tooltip": "Enumerate (001, 002...) or overwrite existing files"
                }),
                "starting_number": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Starting number for enumeration"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save"
    CATEGORY = "Texture Alchemist/Pipeline"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute (don't cache)
        return float("nan")
    
    def tensor_to_pil(self, tensor, bit_depth=8):
        """Convert tensor to PIL Image with specified bit depth"""
        # Tensor is in [B, H, W, C] format
        # Take first batch
        if tensor.shape[0] > 1:
            tensor = tensor[0:1]
        
        # Convert to numpy
        img_np = tensor[0].cpu().numpy()
        
        # Clamp to 0-1
        img_np = np.clip(img_np, 0.0, 1.0)
        
        # Convert based on bit depth
        if bit_depth == 16:
            # 16-bit (0-65535)
            img_np = (img_np * 65535).astype(np.uint16)
        else:
            # 8-bit (0-255)
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL with appropriate mode
        if img_np.shape[-1] == 1:
            # Grayscale
            img_np = img_np[:, :, 0]
            mode = 'I;16' if bit_depth == 16 else 'L'
            pil_image = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[-1] == 3:
            # RGB
            mode = 'RGB'
            pil_image = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[-1] == 4:
            # RGBA
            mode = 'RGBA'
            pil_image = Image.fromarray(img_np, mode=mode)
        else:
            # Default to RGB, take first 3 channels
            mode = 'RGB'
            pil_image = Image.fromarray(img_np[:, :, :3], mode=mode)
        
        return pil_image
    
    def ensure_format_compatible(self, pil_image, file_format, map_type):
        """Ensure PIL image is compatible with the target file format"""
        # JPG doesn't support alpha channel
        if file_format == "jpg" and pil_image.mode == "RGBA":
            print(f"  ⚠ Warning: {map_type} has alpha channel, but JPG doesn't support it")
            print(f"    Converting RGBA to RGB (compositing over white background)")
            # Create white background
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            # Composite the image over white background
            background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
            return background
        
        # Grayscale with JPG
        if file_format == "jpg" and pil_image.mode == "L":
            # Convert grayscale to RGB for JPG
            return pil_image.convert('RGB')
        
        return pil_image
    
    def find_next_number(self, output_dir, base_name, map_type, file_format, starting_number):
        """Find the next available number for enumeration"""
        # png16 saves as .png extension
        ext = "png" if file_format == "png16" else file_format
        
        number = starting_number
        while True:
            filename = f"{base_name}_{map_type}_{number:03d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                return number
            number += 1
            # Safety limit
            if number > 9999:
                return starting_number
    
    def save(self, pbr_pipe, base_name, output_path, file_format, enumeration_mode, starting_number):
        """Save all maps from PBR pipeline"""
        
        print("\n" + "="*60)
        print("PBR Saver - EXECUTING")
        print("="*60)
        print(f"Base name: {base_name}")
        print(f"Output path: {output_path}")
        print(f"Format: {file_format}")
        print(f"Enumeration: {enumeration_mode}")
        print(f"PBR Pipe type: {type(pbr_pipe)}")
        print(f"PBR Pipe keys: {pbr_pipe.keys() if isinstance(pbr_pipe, dict) else 'Not a dict'}")
        
        # Create output directory
        try:
            output_dir = Path(output_path)
            
            if not output_dir.is_absolute():
                # Detect ComfyUI structure (portable vs standard)
                cwd = Path.cwd()
                
                # Check if we're in ComfyUI portable (has ComfyUI subfolder)
                portable_comfy = cwd / "ComfyUI"
                if portable_comfy.exists() and portable_comfy.is_dir():
                    # Portable: ComfyUI/ComfyUI/output
                    comfy_output = portable_comfy / "output"
                    print(f"Detected ComfyUI Portable structure")
                else:
                    # Standard: ComfyUI/output
                    comfy_output = cwd / "output"
                    print(f"Using standard ComfyUI structure")
                
                # Create output folder if it doesn't exist
                if not comfy_output.exists():
                    comfy_output.mkdir(parents=True, exist_ok=True)
                
                output_dir = comfy_output / output_path
            
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_str = str(output_dir.absolute())
            print(f"Output directory: {output_dir_str}")
        except Exception as e:
            print(f"✗ Error creating directory: {str(e)}")
            traceback.print_exc()
            # Return empty placeholder batch on error
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder,)
        
        # Get maps from pipe
        maps = {
            "albedo": pbr_pipe.get("albedo", None),
            "normal": pbr_pipe.get("normal", None),
            "ao": pbr_pipe.get("ao", None),
            "height": pbr_pipe.get("height", None),
            "roughness": pbr_pipe.get("roughness", None),
            "metallic": pbr_pipe.get("metallic", None),
            "transparency": pbr_pipe.get("transparency", None),
            "emission": pbr_pipe.get("emission", None),
        }
        
        # Debug: show what maps we have
        available_maps = [k for k, v in maps.items() if v is not None]
        print(f"Available maps: {available_maps}")
        
        # Determine enumeration number
        if enumeration_mode == "enumerate":
            # Find the next available number across all map types
            next_number = starting_number
            for map_type, tensor in maps.items():
                if tensor is not None:
                    num = self.find_next_number(output_dir_str, base_name, map_type, file_format, starting_number)
                    next_number = max(next_number, num)
            file_number = next_number
        else:
            # Overwrite mode - use starting number
            file_number = starting_number
        
        saved_files = []
        
        # Save each map
        for map_type, tensor in maps.items():
            if tensor is not None:
                try:
                    print(f"\nProcessing {map_type}...")
                    print(f"  Tensor shape: {tensor.shape}")
                    print(f"  Tensor type: {tensor.dtype}")
                    
                    # Determine bit depth based on format
                    bit_depth = 16 if file_format == "png16" else 8
                    
                    # Convert tensor to PIL Image
                    pil_image = self.tensor_to_pil(tensor, bit_depth=bit_depth)
                    print(f"  PIL image size: {pil_image.size}, mode: {pil_image.mode}")
                    
                    # Generate filename (png16 saves as .png)
                    actual_ext = "png" if file_format == "png16" else file_format
                    if enumeration_mode == "enumerate":
                        filename = f"{base_name}_{map_type}_{file_number:03d}.{actual_ext}"
                    else:
                        filename = f"{base_name}_{map_type}.{actual_ext}"
                    
                    filepath = os.path.join(output_dir_str, filename)
                    
                    # Ensure image is compatible with the target format
                    pil_image = self.ensure_format_compatible(pil_image, file_format, map_type)
                    
                    # Save with appropriate settings
                    if file_format == "png" or file_format == "png16":
                        # PNG supports alpha channel and 16-bit
                        if file_format == "png16":
                            pil_image.save(filepath, format='PNG', compress_level=4, bits=16)
                            print(f"  ✓ Saved as 16-bit PNG")
                        else:
                            pil_image.save(filepath, format='PNG', compress_level=4)
                        if pil_image.mode == "RGBA":
                            print(f"  ✓ Saved with alpha channel")
                    elif file_format == "jpg":
                        pil_image.save(filepath, format='JPEG', quality=95)
                    elif file_format == "exr":
                        # For EXR, save as 32-bit float (supports alpha)
                        # Convert back to float numpy array
                        img_np = tensor[0].cpu().numpy()
                        img_np = np.clip(img_np, 0.0, 1.0)
                        
                        # Try to save as EXR (requires OpenEXR or imageio)
                        try:
                            import imageio
                            imageio.imwrite(filepath, img_np.astype(np.float32))
                            print(f"  ✓ Saved as 32-bit float EXR")
                            if img_np.shape[-1] == 4:
                                print(f"  ✓ Saved with alpha channel")
                        except ImportError:
                            print(f"  ⚠ Warning: imageio not available, saving {map_type} as PNG instead")
                            filename = f"{base_name}_{map_type}_{file_number:03d}.png"
                            filepath = os.path.join(output_dir_str, filename)
                            pil_image.save(filepath, format='PNG', compress_level=4)
                    elif file_format == "tiff":
                        # TIFF supports alpha channel
                        pil_image.save(filepath, format='TIFF')
                        if pil_image.mode == "RGBA":
                            print(f"  ✓ Saved with alpha channel")
                    
                    saved_files.append(filename)
                    print(f"  ✓ Saved: {filename}")
                    print(f"  Full path: {filepath}")
                    
                except Exception as e:
                    print(f"  ✗ Error saving {map_type}: {str(e)}")
                    traceback.print_exc()
        
        if saved_files:
            print(f"\n✓ Saved {len(saved_files)} maps to: {output_dir_str}")
            if enumeration_mode == "enumerate":
                print(f"  Number: {file_number:03d}")
        else:
            print("\n✗ No maps to save (check if PBR_PIPE has data)")
        
        print("="*60 + "\n")
        
        # Collect all available maps into a batch for preview
        image_batch = []
        map_order = ["albedo", "normal", "ao", "height", "roughness", "metallic", "transparency", "emission"]
        
        for map_name in map_order:
            if maps[map_name] is not None:
                img = maps[map_name]
                
                # Handle different channel counts
                if img.shape[-1] == 1:
                    # Grayscale - convert to RGB
                    img_rgb = img.repeat(1, 1, 1, 3)
                    image_batch.append(img_rgb)
                    
                elif img.shape[-1] == 3:
                    # RGB - use as is
                    image_batch.append(img)
                    
                elif img.shape[-1] == 4:
                    # RGBA - composite over checkerboard to show transparency
                    img_rgb = img[:, :, :, :3]
                    img_alpha = img[:, :, :, 3:4]
                    
                    # Create checkerboard pattern
                    h, w = img.shape[1], img.shape[2]
                    checker_size = max(16, min(h, w) // 32)
                    
                    y_coords = torch.arange(h, device=img.device).unsqueeze(1).expand(h, w)
                    x_coords = torch.arange(w, device=img.device).unsqueeze(0).expand(h, w)
                    checkerboard = ((y_coords // checker_size + x_coords // checker_size) % 2).float()
                    checkerboard = checkerboard * 0.25 + 0.5  # Light/medium gray
                    checkerboard = checkerboard.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                    
                    # Composite: result = img_rgb * alpha + checkerboard * (1 - alpha)
                    img_composited = img_rgb * img_alpha + checkerboard * (1.0 - img_alpha)
                    
                    image_batch.append(img_composited)
                    print(f"  Preview: Added {map_name} (RGBA composited over checkerboard)")
                    
                elif img.shape[-1] > 4:
                    # More than 4 channels - take first 3
                    img_rgb = img[:, :, :, :3]
                    image_batch.append(img_rgb)
        
        # If we have images, concatenate them into a batch
        if image_batch:
            # Concatenate along batch dimension
            output_batch = torch.cat(image_batch, dim=0)
            print(f"Output batch: {len(image_batch)} images, shape: {output_batch.shape}")
        else:
            # Return placeholder if nothing was saved
            output_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            print("No images to display")
        
        return (output_batch,)


class PBRPipePreview:
    """
    Preview node for PBR Pipe
    Passes through the pipe and outputs all images as a batch for easy preview
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE", "IMAGE")
    RETURN_NAMES = ("pbr_pipe", "preview_batch")
    FUNCTION = "preview"
    CATEGORY = "Texture Alchemist/Pipeline"
    
    def preview(self, pbr_pipe):
        """Create preview batch from PBR pipe and pass through"""
        
        print("\n" + "="*60)
        print("PBR Pipe Preview")
        print("="*60)
        
        if not pbr_pipe:
            print("⚠ Empty PBR pipe")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (pbr_pipe, placeholder)
        
        # Collect all available images
        image_batch = []
        map_order = ["albedo", "normal", "ao", "height", "roughness", "metallic", "lighting", "transparency", "emission"]
        
        for map_name in map_order:
            if map_name in pbr_pipe and pbr_pipe[map_name] is not None:
                img = pbr_pipe[map_name]
                
                # Ensure correct format
                if not isinstance(img, torch.Tensor):
                    continue
                
                # Ensure 4D tensor (batch, height, width, channels)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                
                # Handle different channel counts
                original_channels = img.shape[-1]
                
                if img.shape[-1] == 1:
                    # Grayscale - convert to RGB
                    img_rgb = img.repeat(1, 1, 1, 3)
                    image_batch.append(img_rgb)
                    print(f"✓ Added {map_name}: {img_rgb.shape}")
                    
                elif img.shape[-1] == 3:
                    # RGB - use as is
                    image_batch.append(img)
                    print(f"✓ Added {map_name}: {img.shape}")
                    
                elif img.shape[-1] == 4:
                    # RGBA - composite over checkerboard to show transparency
                    img_rgb = img[:, :, :, :3]
                    img_alpha = img[:, :, :, 3:4]
                    
                    # Create checkerboard pattern (alternating dark/light squares)
                    h, w = img.shape[1], img.shape[2]
                    checker_size = max(16, min(h, w) // 32)  # Adaptive checker size
                    
                    # Create checkerboard using modulo
                    y_coords = torch.arange(h, device=img.device).unsqueeze(1).expand(h, w)
                    x_coords = torch.arange(w, device=img.device).unsqueeze(0).expand(h, w)
                    checkerboard = ((y_coords // checker_size + x_coords // checker_size) % 2).float()
                    checkerboard = checkerboard * 0.25 + 0.5  # Light gray (0.75) and medium gray (0.5)
                    checkerboard = checkerboard.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                    
                    # Composite: result = img_rgb * alpha + checkerboard * (1 - alpha)
                    img_composited = img_rgb * img_alpha + checkerboard * (1.0 - img_alpha)
                    
                    image_batch.append(img_composited)
                    print(f"✓ Added {map_name} (RGBA composited over checkerboard): {img_composited.shape}")
                    
                elif img.shape[-1] > 4:
                    # More than 4 channels - take first 3
                    img_rgb = img[:, :, :, :3]
                    image_batch.append(img_rgb)
                    print(f"✓ Added {map_name}: {img_rgb.shape} (trimmed from {original_channels} channels)")
        
        # Concatenate all images into a batch
        if image_batch:
            preview_batch = torch.cat(image_batch, dim=0)
            print(f"\n✓ Preview batch created")
            print(f"  Total images: {len(image_batch)}")
            print(f"  Batch shape: {preview_batch.shape}")
        else:
            print("⚠ No images found in pipe")
            preview_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        print("="*60 + "\n")
        
        return (pbr_pipe, preview_batch)


class EmbedAlpha:
    """
    Embed an alpha image into another image's alpha channel
    Takes a base image and an alpha image, outputs base with embedded alpha
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "alpha_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_with_alpha",)
    FUNCTION = "embed_alpha"
    CATEGORY = "Texture Alchemist/Utilities"
    
    def embed_alpha(self, base_image, alpha_image):
        """Embed alpha_image into base_image's alpha channel"""
        
        print("\n" + "="*60)
        print("Embed Alpha")
        print("="*60)
        print(f"Base image shape: {base_image.shape}")
        print(f"Alpha image shape: {alpha_image.shape}")
        
        # Convert alpha to grayscale if needed
        alpha_gray = alpha_image
        if alpha_gray.shape[-1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], 
                                  device=alpha_gray.device, dtype=alpha_gray.dtype)
            alpha_gray = torch.sum(alpha_gray * weights, dim=-1, keepdim=True)
            print("✓ Alpha image converted to grayscale")
        elif alpha_gray.shape[-1] == 4:
            # If alpha_image has alpha, use it
            alpha_gray = alpha_gray[:, :, :, 3:4]
            print("✓ Using existing alpha channel from alpha_image")
        
        # Resize alpha to match base if needed
        if alpha_gray.shape[1:3] != base_image.shape[1:3]:
            alpha_gray = torch.nn.functional.interpolate(
                alpha_gray.permute(0, 3, 1, 2),
                size=base_image.shape[1:3],
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            print(f"✓ Alpha resized to match base: {alpha_gray.shape[1:3]}")
        
        # Ensure base is RGB (3 channels)
        base_rgb = base_image
        if base_rgb.shape[-1] == 4:
            base_rgb = base_rgb[:, :, :, :3]
            print("✓ Removed existing alpha from base image")
        elif base_rgb.shape[-1] == 1:
            base_rgb = base_rgb.repeat(1, 1, 1, 3)
            print("✓ Converted grayscale base to RGB")
        elif base_rgb.shape[-1] > 4:
            base_rgb = base_rgb[:, :, :, :3]
            print("✓ Trimmed base image to RGB")
        
        # Concatenate alpha channel
        result = torch.cat([base_rgb, alpha_gray], dim=-1)
        
        print(f"\n✓ Alpha embedded successfully")
        print(f"  Output shape: {result.shape}")
        print(f"  Alpha range: [{alpha_gray.min():.3f}, {alpha_gray.max():.3f}]")
        print("="*60 + "\n")
        
        return (result,)


class PBRSaverWithMetadata:
    """
    Save all maps from PBR_PIPE with workflow metadata embedded
    Allows drag-and-drop back into ComfyUI to recover the workflow
    Example: bricks_albedo_001.png, bricks_metallic_001.png (with embedded workflow)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
                "base_name": ("STRING", {
                    "default": "material",
                    "tooltip": "Base name for saved files (e.g., 'bricks')"
                }),
                "output_path": ("STRING", {
                    "default": "pbr_materials",
                    "tooltip": "Folder name inside ComfyUI/output (or full path if absolute)"
                }),
                "file_format": (["png", "png16"], {
                    "default": "png",
                    "tooltip": "Image format (metadata only supported for PNG)"
                }),
                "enumeration_mode": (["enumerate", "overwrite"], {
                    "default": "enumerate",
                    "tooltip": "Enumerate (001, 002...) or overwrite existing files"
                }),
                "starting_number": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 9999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Starting number for enumeration"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_with_metadata"
    CATEGORY = "Texture Alchemist/Pipeline"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute (don't cache)
        return float("nan")
    
    def tensor_to_pil(self, tensor, bit_depth=8):
        """Convert tensor to PIL Image with specified bit depth"""
        # Tensor is in [B, H, W, C] format
        # Take first batch
        if tensor.shape[0] > 1:
            tensor = tensor[0:1]
        
        # Convert to numpy
        img_np = tensor[0].cpu().numpy()
        
        # Clamp to 0-1
        img_np = np.clip(img_np, 0.0, 1.0)
        
        # Convert based on bit depth
        if bit_depth == 16:
            # 16-bit (0-65535)
            img_np = (img_np * 65535).astype(np.uint16)
        else:
            # 8-bit (0-255)
            img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to PIL with appropriate mode
        if img_np.shape[-1] == 1:
            # Grayscale
            img_np = img_np[:, :, 0]
            mode = 'I;16' if bit_depth == 16 else 'L'
            pil_image = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[-1] == 3:
            # RGB
            mode = 'RGB'
            pil_image = Image.fromarray(img_np, mode=mode)
        elif img_np.shape[-1] == 4:
            # RGBA
            mode = 'RGBA'
            pil_image = Image.fromarray(img_np, mode=mode)
        else:
            # Default to RGB, take first 3 channels
            mode = 'RGB'
            pil_image = Image.fromarray(img_np[:, :, :3], mode=mode)
        
        return pil_image
    
    def find_next_number(self, output_dir, base_name, map_type, file_format, starting_number):
        """Find the next available number for enumeration"""
        # png16 saves as .png extension
        ext = "png"
        
        number = starting_number
        while True:
            filename = f"{base_name}_{map_type}_{number:03d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                return number
            number += 1
            # Safety limit
            if number > 9999:
                return starting_number
    
    def save_with_metadata(self, pbr_pipe, base_name, output_path, file_format, 
                           enumeration_mode, starting_number, prompt=None, extra_pnginfo=None):
        """Save all maps from PBR pipeline with workflow metadata"""
        from PIL.PngImagePlugin import PngInfo
        import json
        
        print("\n" + "="*60)
        print("PBR Saver with Metadata - EXECUTING")
        print("="*60)
        print(f"Base name: {base_name}")
        print(f"Output path: {output_path}")
        print(f"Format: {file_format}")
        print(f"Enumeration: {enumeration_mode}")
        print(f"Metadata: {'✓ Available' if prompt else '✗ Not available'}")
        
        # Create output directory
        try:
            output_dir = Path(output_path)
            
            if not output_dir.is_absolute():
                # Detect ComfyUI structure (portable vs standard)
                cwd = Path.cwd()
                
                # Check if we're in ComfyUI portable (has ComfyUI subfolder)
                portable_comfy = cwd / "ComfyUI"
                if portable_comfy.exists() and portable_comfy.is_dir():
                    # Portable: ComfyUI/ComfyUI/output
                    comfy_output = portable_comfy / "output"
                    print(f"Detected ComfyUI Portable structure")
                else:
                    # Standard: ComfyUI/output
                    comfy_output = cwd / "output"
                    print(f"Using standard ComfyUI structure")
                
                # Create output folder if it doesn't exist
                if not comfy_output.exists():
                    comfy_output.mkdir(parents=True, exist_ok=True)
                
                output_dir = comfy_output / output_path
            
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir_str = str(output_dir.absolute())
            print(f"Output directory: {output_dir_str}")
        except Exception as e:
            print(f"✗ Error creating directory: {str(e)}")
            traceback.print_exc()
            # Return empty placeholder batch on error
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder,)
        
        # Get maps from pipe
        maps = {
            "albedo": pbr_pipe.get("albedo", None),
            "normal": pbr_pipe.get("normal", None),
            "ao": pbr_pipe.get("ao", None),
            "height": pbr_pipe.get("height", None),
            "roughness": pbr_pipe.get("roughness", None),
            "metallic": pbr_pipe.get("metallic", None),
            "transparency": pbr_pipe.get("transparency", None),
            "emission": pbr_pipe.get("emission", None),
        }
        
        # Debug: show what maps we have
        available_maps = [k for k, v in maps.items() if v is not None]
        print(f"Available maps: {available_maps}")
        
        # Determine enumeration number
        if enumeration_mode == "enumerate":
            # Find the next available number across all map types
            next_number = starting_number
            for map_type, tensor in maps.items():
                if tensor is not None:
                    num = self.find_next_number(output_dir_str, base_name, map_type, file_format, starting_number)
                    next_number = max(next_number, num)
            file_number = next_number
        else:
            # Overwrite mode - use starting number
            file_number = starting_number
        
        # Prepare PNG metadata
        metadata = None
        if prompt is not None:
            metadata = PngInfo()
            
            # Add prompt (workflow data)
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
                print("✓ Embedding prompt metadata")
            
            # Add extra pnginfo (workflow JSON)
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value) if not isinstance(value, str) else value)
                print(f"✓ Embedding {len(extra_pnginfo)} extra metadata fields")
        
        saved_files = []
        
        # Save each map
        for map_type, tensor in maps.items():
            if tensor is not None:
                try:
                    print(f"\nProcessing {map_type}...")
                    print(f"  Tensor shape: {tensor.shape}")
                    
                    # Determine bit depth based on format
                    bit_depth = 16 if file_format == "png16" else 8
                    
                    # Convert tensor to PIL Image
                    pil_image = self.tensor_to_pil(tensor, bit_depth=bit_depth)
                    print(f"  PIL image size: {pil_image.size}, mode: {pil_image.mode}")
                    
                    # Generate filename
                    if enumeration_mode == "enumerate":
                        filename = f"{base_name}_{map_type}_{file_number:03d}.png"
                    else:
                        filename = f"{base_name}_{map_type}.png"
                    
                    filepath = os.path.join(output_dir_str, filename)
                    
                    # Save with metadata
                    if file_format == "png16":
                        pil_image.save(filepath, format='PNG', pnginfo=metadata, 
                                     compress_level=4, bits=16)
                        print(f"  ✓ Saved as 16-bit PNG with metadata")
                    else:
                        pil_image.save(filepath, format='PNG', pnginfo=metadata, 
                                     compress_level=4)
                        print(f"  ✓ Saved as 8-bit PNG with metadata")
                    
                    if pil_image.mode == "RGBA":
                        print(f"  ✓ Saved with alpha channel")
                    
                    saved_files.append(filename)
                    print(f"  ✓ Saved: {filename}")
                    
                except Exception as e:
                    print(f"  ✗ Error saving {map_type}: {str(e)}")
                    traceback.print_exc()
        
        if saved_files:
            print(f"\n✓ Saved {len(saved_files)} maps with workflow metadata to: {output_dir_str}")
            if enumeration_mode == "enumerate":
                print(f"  Number: {file_number:03d}")
            print(f"  🎯 Drag any saved image into ComfyUI to recover the workflow!")
        else:
            print("\n✗ No maps to save (check if PBR_PIPE has data)")
        
        print("="*60 + "\n")
        
        # Collect all available maps into a batch for preview
        image_batch = []
        map_order = ["albedo", "normal", "ao", "height", "roughness", "metallic", "transparency", "emission"]
        
        for map_name in map_order:
            if maps[map_name] is not None:
                img = maps[map_name]
                
                # Handle different channel counts
                if img.shape[-1] == 1:
                    # Grayscale - convert to RGB
                    img_rgb = img.repeat(1, 1, 1, 3)
                    image_batch.append(img_rgb)
                    
                elif img.shape[-1] == 3:
                    # RGB - use as is
                    image_batch.append(img)
                    
                elif img.shape[-1] == 4:
                    # RGBA - composite over checkerboard to show transparency
                    img_rgb = img[:, :, :, :3]
                    img_alpha = img[:, :, :, 3:4]
                    
                    # Create checkerboard pattern
                    h, w = img.shape[1], img.shape[2]
                    checker_size = max(16, min(h, w) // 32)
                    
                    y_coords = torch.arange(h, device=img.device).unsqueeze(1).expand(h, w)
                    x_coords = torch.arange(w, device=img.device).unsqueeze(0).expand(h, w)
                    checkerboard = ((y_coords // checker_size + x_coords // checker_size) % 2).float()
                    checkerboard = checkerboard * 0.25 + 0.5  # Light/medium gray
                    checkerboard = checkerboard.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
                    
                    # Composite: result = img_rgb * alpha + checkerboard * (1 - alpha)
                    img_composited = img_rgb * img_alpha + checkerboard * (1.0 - img_alpha)
                    
                    image_batch.append(img_composited)
                    
                elif img.shape[-1] > 4:
                    # More than 4 channels - take first 3
                    img_rgb = img[:, :, :, :3]
                    image_batch.append(img_rgb)
        
        # If we have images, concatenate them into a batch
        if image_batch:
            # Concatenate along batch dimension
            output_batch = torch.cat(image_batch, dim=0)
            print(f"Output batch: {len(image_batch)} images, shape: {output_batch.shape}")
        else:
            # Return placeholder if nothing was saved
            output_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        return (output_batch,)


class PBRPipeToBatch:
    """
    Convert PBR Pipe to Image Batch
    Extracts all available maps and outputs them as a single image batch
    Perfect for batch processing with upscalers or other nodes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pbr_pipe": ("PBR_PIPE",),
            },
            "optional": {
                "include_albedo": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include albedo map in batch"
                }),
                "include_normal": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include normal map in batch"
                }),
                "include_ao": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include AO map in batch"
                }),
                "include_height": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include height map in batch"
                }),
                "include_roughness": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include roughness map in batch"
                }),
                "include_metallic": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include metallic map in batch"
                }),
                "include_transparency": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include transparency map in batch"
                }),
                "include_emission": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include emission map in batch"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "pipe_to_batch"
    CATEGORY = "Texture Alchemist/Pipeline"
    
    def pipe_to_batch(self, pbr_pipe, include_albedo=True, include_normal=True, 
                     include_ao=True, include_height=True, include_roughness=True,
                     include_metallic=True, include_transparency=True, include_emission=True):
        """Convert PBR pipe to image batch"""
        
        print("\n" + "="*60)
        print("PBR Pipe to Image Batch")
        print("="*60)
        
        if not pbr_pipe:
            print("⚠ Empty PBR pipe")
            placeholder = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (placeholder,)
        
        # Map order and inclusion flags
        map_config = [
            ("albedo", include_albedo),
            ("normal", include_normal),
            ("ao", include_ao),
            ("height", include_height),
            ("roughness", include_roughness),
            ("metallic", include_metallic),
            ("transparency", include_transparency),
            ("emission", include_emission),
        ]
        
        # Collect images
        image_batch = []
        
        for map_name, include in map_config:
            if not include:
                continue
                
            if map_name not in pbr_pipe or pbr_pipe[map_name] is None:
                continue
                
            img = pbr_pipe[map_name]
            
            # Ensure correct format
            if not isinstance(img, torch.Tensor):
                continue
            
            # Ensure 4D tensor (batch, height, width, channels)
            if img.dim() == 3:
                img = img.unsqueeze(0)
            
            # Convert to RGB format (3 channels)
            if img.shape[-1] == 1:
                # Grayscale - convert to RGB
                img_rgb = img.repeat(1, 1, 1, 3)
                image_batch.append(img_rgb)
                print(f"✓ Added {map_name}: {img_rgb.shape} (grayscale -> RGB)")
                
            elif img.shape[-1] == 3:
                # RGB - use as is
                image_batch.append(img)
                print(f"✓ Added {map_name}: {img.shape} (RGB)")
                
            elif img.shape[-1] == 4:
                # RGBA - composite over black background (preserve color/alpha info)
                img_rgb = img[:, :, :, :3]
                img_alpha = img[:, :, :, 3:4]
                
                # Composite over black: result = img_rgb * alpha
                img_composited = img_rgb * img_alpha
                
                image_batch.append(img_composited)
                print(f"✓ Added {map_name}: {img_composited.shape} (RGBA -> RGB, composited)")
                
            elif img.shape[-1] > 4:
                # More than 4 channels - take first 3
                img_rgb = img[:, :, :, :3]
                image_batch.append(img_rgb)
                print(f"✓ Added {map_name}: {img_rgb.shape} (multi-channel -> RGB)")
        
        # Concatenate all images into a batch
        if image_batch:
            output_batch = torch.cat(image_batch, dim=0)
            print(f"\n✓ Image batch created")
            print(f"  Total images: {len(image_batch)}")
            print(f"  Batch shape: {output_batch.shape}")
        else:
            print("⚠ No images found in pipe")
            output_batch = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        print("="*60 + "\n")
        
        return (output_batch,)


class PBRBatchToPipe:
    """
    Convert Image Batch back to PBR Pipe
    Takes a batch of images (in order) and reconstructs a PBR pipe
    Perfect for converting processed images back to a PBR workflow
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "map_order": ("STRING", {
                    "default": "albedo,normal,ao,height,roughness,metallic,transparency,emission",
                    "tooltip": "Comma-separated list of map names in batch order"
                }),
            },
            "optional": {
                "existing_pipe": ("PBR_PIPE",),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "batch_to_pipe"
    CATEGORY = "Texture Alchemist/Pipeline"
    
    def batch_to_pipe(self, image_batch, map_order, existing_pipe=None):
        """Convert image batch back to PBR pipe"""
        
        print("\n" + "="*60)
        print("PBR Batch to Pipe")
        print("="*60)
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Map order: {map_order}")
        
        # Parse map order
        map_names = [name.strip() for name in map_order.split(",")]
        batch_size = image_batch.shape[0]
        
        # Start with existing pipe or create new
        if existing_pipe is not None:
            print("✓ Updating existing PBR_PIPE")
            result_pipe = existing_pipe.copy()
        else:
            print("✓ Creating new PBR_PIPE")
            result_pipe = {
                "albedo": None,
                "normal": None,
                "ao": None,
                "height": None,
                "roughness": None,
                "metallic": None,
                "transparency": None,
                "emission": None,
            }
        
        # Assign images to maps
        for i, map_name in enumerate(map_names):
            if i >= batch_size:
                print(f"⚠ Warning: {map_name} specified but no image at index {i}")
                break
            
            # Extract single image from batch
            img = image_batch[i:i+1]
            
            # Store in pipe
            if map_name in result_pipe:
                result_pipe[map_name] = img
                print(f"✓ {map_name}: {img.shape}")
            else:
                print(f"⚠ Unknown map name: {map_name}")
        
        # Show what's in the final pipe
        available = [k for k, v in result_pipe.items() if v is not None]
        print(f"\nFinal pipe contains: {', '.join(available)}")
        
        print("\n✓ PBR Pipe created from batch")
        print("="*60 + "\n")
        
        return (result_pipe,)


class PBRMaterialMixer:
    """
    Mix/blend two complete PBR material pipes
    Supports various blend modes and masking
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pipe": ("PBR_PIPE",),
                "overlay_pipe": ("PBR_PIPE",),
                "blend_mode": (["mix", "multiply", "overlay", "add", "screen"], {
                    "default": "mix",
                    "tooltip": "Blend mode for combining materials"
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Blend amount (0=base only, 1=overlay only)"
                }),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("PBR_PIPE",)
    RETURN_NAMES = ("pbr_pipe",)
    FUNCTION = "mix_materials"
    CATEGORY = "Texture Alchemist/Pipeline"
    
    def mix_materials(self, base_pipe, overlay_pipe, blend_mode, blend_strength, mask=None):
        """Mix two PBR material pipes"""
        
        print("\n" + "="*60)
        print("PBR Material Mixer")
        print("="*60)
        print(f"Blend mode: {blend_mode}")
        print(f"Blend strength: {blend_strength}")
        
        # Create output pipe
        mixed_pipe = {}
        
        # Get list of all available maps
        all_maps = set(list(base_pipe.keys()) + list(overlay_pipe.keys()))
        
        for map_name in all_maps:
            base_map = base_pipe.get(map_name)
            overlay_map = overlay_pipe.get(map_name)
            
            # Skip if neither has this map
            if base_map is None and overlay_map is None:
                continue
            
            # If only one has the map, use it
            if base_map is None:
                mixed_pipe[map_name] = overlay_map
                print(f"  {map_name}: overlay only")
                continue
            if overlay_map is None:
                mixed_pipe[map_name] = base_map
                print(f"  {map_name}: base only")
                continue
            
            # Both have the map - blend them
            # Resize if needed
            if overlay_map.shape[1:3] != base_map.shape[1:3]:
                overlay_map = self._resize_to_match(overlay_map, base_map)
            
            # Apply blend mode
            if blend_mode == "mix":
                blended = base_map * (1.0 - blend_strength) + overlay_map * blend_strength
            elif blend_mode == "multiply":
                blended = base_map * (1.0 + (overlay_map - 0.5) * blend_strength)
            elif blend_mode == "overlay":
                # Photoshop-style overlay
                low = 2.0 * base_map * overlay_map
                high = 1.0 - 2.0 * (1.0 - base_map) * (1.0 - overlay_map)
                blended = torch.where(base_map < 0.5, low, high)
                blended = base_map * (1.0 - blend_strength) + blended * blend_strength
            elif blend_mode == "add":
                blended = base_map + overlay_map * blend_strength
            elif blend_mode == "screen":
                blended = 1.0 - (1.0 - base_map) * (1.0 - overlay_map * blend_strength)
            else:
                blended = base_map
            
            # Apply mask if provided
            if mask is not None:
                mask_gray = self._to_grayscale(mask)
                if mask_gray.shape[1:3] != base_map.shape[1:3]:
                    mask_gray = self._resize_to_match(mask_gray, base_map)
                blended = base_map * (1.0 - mask_gray) + blended * mask_gray
            
            # Clamp and store
            mixed_pipe[map_name] = torch.clamp(blended, 0.0, 1.0)
            print(f"  {map_name}: blended")
        
        print(f"✓ Materials mixed")
        print(f"  Total maps: {len(mixed_pipe)}")
        print("="*60 + "\n")
        
        return (mixed_pipe,)
    
    def _resize_to_match(self, source, target):
        """Resize source to match target"""
        target_h, target_w = target.shape[1:3]
        source_bchw = source.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            source_bchw, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        return resized.permute(0, 2, 3, 1)
    
    def _to_grayscale(self, image):
        """Convert to grayscale"""
        if image.shape[-1] == 1:
            return image
        weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                               device=image.device, dtype=image.dtype)
        return torch.sum(image * weights, dim=-1, keepdim=True)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PBRCombiner": PBRCombiner,
    "PBRPipelineAdjuster": PBRPipelineAdjuster,
    "PBRSplitter": PBRSplitter,
    "PBRSaver": PBRSaver,
    "PBRSaverWithMetadata": PBRSaverWithMetadata,
    "PBRPipePreview": PBRPipePreview,
    "PBRPipeToBatch": PBRPipeToBatch,
    "PBRBatchToPipe": PBRBatchToPipe,
    "PBRMaterialMixer": PBRMaterialMixer,
    "EmbedAlpha": EmbedAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PBRCombiner": "PBR Combiner",
    "PBRPipelineAdjuster": "PBR Pipeline Adjuster",
    "PBRSplitter": "PBR Splitter",
    "PBRSaver": "PBR Saver",
    "PBRSaverWithMetadata": "PBR Saver (with Metadata)",
    "PBRPipePreview": "PBR Pipe Preview",
    "PBRPipeToBatch": "PBR Pipe to Batch",
    "PBRBatchToPipe": "PBR Batch to Pipe",
    "PBRMaterialMixer": "PBR Material Mixer",
    "EmbedAlpha": "Embed Alpha",
}

