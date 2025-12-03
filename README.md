# TextureAlchemy

Transform Images into complete material sets ready for game engines, 3D software, and rendering pipelines.*

Complete workflow suite for ComfyUI. Extract, process, and enhance physically-based rendering textures with AI-powered alchemy.

---

## 🎯 What It Does

Creates complete PBR material sets from images using AI (Marigold/Lotus) with professional post-processing tools:

**Material Maps:**
- **Albedo** - Base color/diffuse
- **Normal** - Surface normals (OpenGL/DirectX compatible)
- **Roughness** - Surface roughness
- **Metallic** - Metalness/reflectivity
- **AO** - Ambient occlusion
- **Height** - Displacement/parallax
- **Curvature** - Edge detection for wear/weathering

**Professional Tools:**
- Seamless tiling, texture scaling, triplanar projection
- Channel packing (save VRAM), normal/height conversion
- Material mixing, wear generation, detail blending
- HSV color adjustment, gradient mapping
- Interactive color ramp with visual UI

## 📦 Installation

1. Copy `ComfyUI_PBR_MaterialProcessor` folder to `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Find all nodes in **Texture Alchemist** category in ComfyUI's node library

**No additional requirements needed** - uses ComfyUI's built-in libraries (PyTorch, PIL, NumPy).

---

## 📚 Node Categories

**All nodes are in the "Texture Alchemist" category in ComfyUI!**

To find nodes: Right-click in ComfyUI → Add Node → **Texture Alchemist** → Choose subcategory

### ⚗️ Texture Alchemist/Core (2 nodes)
Extract and adjust PBR materials from AI outputs

### ⚗️ Texture Alchemist/Pipeline (6 nodes)
Bundle maps, adjust in batch, mix materials, save sets

### ⚗️ Texture Alchemist/Normal (4 nodes)
Process, combine, convert normal maps

### ⚗️ Texture Alchemist/Color (4 nodes)
Recolor, adjust HSV, color ramps

### ⚗️ Texture Alchemist/Effects (4 nodes)
Wear generation, curvature, detail blending

### ⚗️ Texture Alchemist/Texture (8 nodes)
Tiling, scaling, projection, optimization

### ⚗️ Texture Alchemist/Channel (8 nodes)
Pack/unpack RGB channels, grayscale conversion

### ⚗️ Texture Alchemist/Maps (3 nodes)
Height processing, AO generation

### ⚗️ Texture Alchemist/Materials (User nodes)
Your custom PBR extractors

---

## 🔌 The Nodes (Detailed Reference)

### Core Processing

#### 1. **PBR Extractor (Marigold)**
Extracts PBR maps from Marigold AI outputs into a PBR_PIPE.
Needs the Marigold model loader from [ComfyUI-Marigold ]([url](https://github.com/kijai/ComfyUI-Marigold))

**Inputs:**
- `marigold_appearance` - Marigold output (appearance model)
- `marigold_lighting` - Marigold output (lighting model)
- `albedo_source` - "appearance" or "lighting"
- `gamma_albedo` - Gamma for albedo (0.45)
- `gamma_metal_rough` - Gamma for metallic and roughness (2.2)
- `gamma_lighting_ao` - Gamma for lighting and AO (0.45)

**Outputs:**
- `pbr_pipe` - Complete PBR pipeline with albedo, roughness, metallic, lighting

**Tips:**
- Default `gamma_metal_rough=2.2` provides strong contrast; adjust 1.8-2.6 if needed
- Default `albedo_source="lighting"` provides cleaner albedo; try "appearance" for alternative look
- `gamma_albedo` affects albedo regardless of source (appearance or lighting)

#### 2. **PBR Adjuster**
Simple brightness/contrast/invert adjustments for individual maps.

**Inputs:**
- Individual maps (albedo, ao, roughness, metallic)
- Per-map: brightness (0-3x), contrast (0-3x), invert (bool)

**Outputs:**
- Adjusted versions of input maps

**Use Cases:**
- Quick map adjustments before combining
- Fine-tune individual channels

#### 3. **AO Approximator**
Generate ambient occlusion from height + normal maps.

**Inputs:**
- `height` - Height/displacement map (required)
- `normal` - Normal map (optional, improves quality)
- `radius` - Sample radius in pixels (8-16 typical)
- `strength` - AO intensity (1.0)
- `samples` - Quality vs speed (16-32)
- `contrast` - Contrast boost (1.0)

**Outputs:**
- `ao` - Generated ambient occlusion map

**Tips:**
- More samples = better quality but slower
- Larger radius = broader, softer AO
- Use both height + normal for best results

---

### Pipeline System

#### 4. **PBR Combiner**
Bundle individual maps into a single PBR_PIPE connection.

**Inputs (all optional):**
- `pbr_pipe` - Existing pipe to merge with
- `albedo`, `normal`, `ao`, `height`, `roughness`, `metallic`, `transparency`, `emission` ⭐

**Outputs:**
- `pbr_pipe` - All maps in one bundle

**Use Cases:**
- Create pipe from individual maps
- Add normal/height to extractor output
- Override specific maps in existing pipe
- **Add emission maps** for glowing/emissive materials (lights, lava, screens, etc.)

#### 5. **PBR Pipeline Adjuster**
Advanced batch adjustments with AO integration.

**Inputs:**
- `pbr_pipe` - PBR pipeline
- `ao_strength_albedo` (1.0) - Darken albedo with AO
- `ao_strength_roughness` (0.0) - Add AO to roughness (weathering)
- `roughness_strength` (1.0) - Brightness multiplier
- `metallic_strength` (1.0) - Brightness multiplier
- `normal_strength` (1.0) - Flatten (0.0) or exaggerate (2.0)
- `invert_normal_green` - OpenGL ↔ DirectX
- `invert_transparency` - Flip alpha
- `albedo_dimmer` (0.0-1.0) - Darken (1.0 = black)
- `albedo_saturation` (1.0) - Color intensity

**Outputs:**
- `pbr_pipe` - Adjusted pipeline

**Tips:**
- `ao_strength_albedo=1.0-1.2` for realistic darkening
- `ao_strength_roughness=0.2-0.5` for weathered look
- `albedo_saturation=1.2-1.5` for vibrant colors

#### 6. **PBR Splitter**
Extract individual maps from PBR_PIPE.

**Outputs:**
- All map types as separate outputs: albedo, normal, ao, height, roughness, metallic, transparency, emission ⭐

#### 7. **PBR Saver**  💾
Save complete material sets with auto-naming and enumeration.

**Inputs:**
- `pbr_pipe` - Pipeline to save
- `base_name` - File prefix (e.g., "bricks")
- `output_path` - Folder in ComfyUI/output (default: "pbr_materials")
- `file_format` - png, jpg, exr, tiff
- `enumeration_mode` - "enumerate" or "overwrite"
- `starting_number` - Starting index (1)

**Outputs:**
- `images` - Batch of all saved maps (for preview)

**Example Output:**
```
ComfyUI/output/pbr_materials/
  bricks_albedo_001.png
  bricks_normal_001.png
  bricks_ao_001.png
  bricks_roughness_001.png
  bricks_metallic_001.png
```

**Tips:**
- Auto-detects portable vs standard ComfyUI install
- Connect output to Preview Image to see all maps
- Use EXR for 32-bit precision

#### 8. **PBR Pipe Preview**
Passthrough node that outputs all maps as a batch for preview.

**Inputs:**
- `pbr_pipe` - Pipeline to preview

**Outputs:**
- `pbr_pipe` - Unchanged passthrough
- `preview_batch` - All maps in one batch

**Use Cases:**
- Debug pipeline at any stage
- Preview without saving
- Quick visual check

#### 9. **PBR Material Mixer**
Blend two complete PBR materials together.

**Inputs:**
- `base_pipe` - Base material
- `overlay_pipe` - Overlay material
- `blend_mode` - mix, multiply, overlay, add, screen
- `blend_strength` (0.0-1.0) - Blend amount
- `mask` (optional) - Blend mask

**Outputs:**
- `pbr_pipe` - Mixed material

**Use Cases:**
- Layer materials (brick + moss)
- Create material variations
- Mask-based blending

---

### Normal & Height Processing

#### 10. **Normal Processor (Lotus)**
Process Lotus normal maps with channel control.

**Inputs:**
- `normal` - Lotus output
- `invert_red`, `invert_green`, `invert_blue` - Per-channel inversion
- `strength` (0.0-2.0) - Normal intensity

**Outputs:**
- `normal` - Processed normal map

**Tips:**
- `invert_green=True` for OpenGL format
- Adjust `strength` to flatten or exaggerate bumps

#### 11. **Normal Map Combiner**
Combine two normal maps using proper math.

**Inputs:**
- `base_normal` - Primary normal
- `detail_normal` - Detail to add
- `blend_mode` - reoriented (best), whiteout, linear
- `detail_strength` (0.0-2.0) - Detail intensity

**Outputs:**
- `combined_normal` - Blended normal

**Use Cases:**
- Add micro-detail to base normals
- Layer multiple detail passes
- Combine procedural + photo normals

#### 12. **Normal to Depth Converter**
Convert normal maps to height/depth maps.

**Inputs:**
- `normal` - Normal map
- `method` - integration (accurate), blue_channel (fast), hybrid (balanced)
- `strength` (1.0) - Depth intensity
- `iterations` (50) - Integration accuracy
- `blur_radius` (1.0) - Smoothing

**Outputs:**
- `depth` - Generated depth/height map

**Tips:**
- Use "hybrid" for best speed/quality balance
- More iterations = more accurate but slower

#### 13. **Height to Normal Converter** ⭐ NEW
Generate normal maps from height/depth maps.

**Inputs:**
- `height` - Height/displacement map
- `strength` (1.0) - Normal map intensity
- `method` - sobel (balanced), scharr (detailed), prewitt (smooth)

**Outputs:**
- `normal` - Generated normal map

**Use Cases:**
- Create normals from Lotus depth output
- Generate normals from procedural height
- Convert displacement to normal

#### 14. **Normal Format Converter (DX↔GL)** ⭐ NEW
Convert between DirectX and OpenGL normal map formats.

**Inputs:**
- `normal` - Normal map
- `conversion` - DirectX_to_OpenGL, OpenGL_to_DirectX, auto_detect

**Outputs:**
- `normal` - Converted normal map

**Tips:**
- Flips green (Y) channel
- Use when normals look inverted
- Most game engines use OpenGL

#### 15. **Normal Format Validator (OGL vs DX)** ⭐ NEW
Automatically detects and validates normal map format (OpenGL vs DirectX).

**Inputs:**
- `normal_map` - Normal map to analyze

**Outputs:**
- `visualization` - 4-panel visual analysis
  - Panel 1: Original normal map
  - Panel 2: Green channel isolated
  - Panel 3: Threshold map (white=up, black=down)
  - Panel 4: Format indicator (green=OGL, red=DX)
- `detected_format` - String result with confidence

**Detection Method:**
- Analyzes green channel distribution
- **OpenGL:** More pixels > 0.5 (Y points UP)
- **DirectX:** More pixels < 0.5 (Y points DOWN)
- Confidence based on bias strength

**Console Output:**
```
📊 GREEN CHANNEL ANALYSIS:
  Pixels > 0.5: 73.2% (up-facing)
  Pixels < 0.5: 26.8% (down-facing)
  Bias: +0.464

🎯 DETECTION RESULT:
  Format: OpenGL
  Confidence: High
```

**Confidence Levels:**
- **High** (>20% bias) - Very clear format
- **Medium** (10-20%) - Likely correct
- **Low** (5-10%) - Uncertain
- **Ambiguous** (<5%) - Flat map or balanced normals

**Use Cases:**
- Verify normals from unknown sources
- Debug "why does my normal map look wrong?"
- QA for asset pipelines
- Educational tool

**Tips:**
- Check visualization if result is ambiguous
- Flat normal maps may return AMBIGUOUS
- Use with "Normal Format Converter" for fixing

---

### Color & Effects

#### 16. **Color Ramp** 🎨✨
Map grayscale to colors with **visual gradient preview** (like Blender!).

**Features:**
- 🌈 Real-time gradient preview in node
- 🎯 Interactive color stop markers (click to add, double-click to remove)
- 🎨 Click color swatches to open color picker
- ↔️ Drag markers to reposition
- 📋 8 built-in presets

**Inputs:**
- `image` - Input (converts to grayscale)
- `preset` - grayscale, heat, rainbow, gold_metal, rust, copper, blue_metal, custom
- `interpolation` - linear, ease_in, ease_out, constant
- `color_stops` - JSON data (managed by visual widget)

**Outputs:**
- `image` - Recolored output

**Presets:**
- `heat` - Black → Red → Orange → Yellow (data visualization)
- `rainbow` - Full spectrum
- `gold_metal` - Dark bronze → Gold → Bright
- `rust`, `copper`, `blue_metal` - Metallic variations

**Use Cases:**
- Stylized albedo from AO/height
- Heat map visualizations
- Quick color variations
- Artistic effects

#### 17. **Simple Recolor**
Quick two-color gradient.

**Inputs:**
- `dark_color`, `light_color` - RGB values
- `blend_mode` - linear or smooth

#### 17. **HSV Adjuster** ⭐ NEW
Adjust hue, saturation, and value (better than RGB adjustments).

**Inputs:**
- `image` - Input image
- `hue_shift` (-0.5 to 0.5) - Rotate color wheel
- `saturation` (0.0-3.0) - Color intensity (0=grayscale, >1=vivid)
- `value` (0.0-3.0) - Brightness

**Outputs:**
- `image` - Adjusted image

**Use Cases:**
- Color grade albedo
- Create material variations
- Adjust saturation without affecting brightness
- Preserve material properties better than RGB

#### 18. **Curvature Map Generator** ⭐ NEW
Detect edges and crevices for wear/weathering masks.

**Inputs:**
- `input_map` - Normal or height map
- `input_type` - normal or height
- `strength` (1.0) - Detection sensitivity
- `blur_radius` (1.0) - Smoothing

**Outputs:**
- `curvature` - Edge detection mask

**Use Cases:**
- Create wear masks (edges = more wear)
- Procedural damage
- Detail masking
- AO enhancement

#### 19. **Detail Map Blender** ⭐ NEW
Add micro-detail without washing out base maps.

**Inputs:**
- `base` - Base map
- `detail` - Detail map
- `map_type` - normal (RNM blend), roughness (multiply), generic (overlay)
- `strength` (1.0) - Detail intensity
- `mask` (optional) - Blend mask

**Outputs:**
- `blended` - Combined map

**Tips:**
- Use `map_type=normal` for proper normal blending
- Use `map_type=roughness` for surface detail
- Mask controls where detail appears

#### 20. **Wear & Edge Damage Generator** ⭐ NEW
Procedural wear and weathering effects.

**Inputs:**
- `albedo` - Base albedo
- `wear_strength` (0.5) - Overall wear
- `edge_wear` (0.7) - Damage on edges (needs curvature)
- `dirt_strength` (0.3) - Dirt in crevices (needs AO)
- `curvature` (optional) - For edge detection
- `ao` (optional) - For cavity dirt

**Outputs:**
- `worn_albedo` - Weathered albedo
- `wear_mask` - Mask showing wear areas

**Use Cases:**
- Realistic weathering
- Procedural damage
- Aged materials
- Dirt accumulation

#### 21. **Gradient Map (Mask)** ⭐ NEW
Create selection masks from value ranges.

**Inputs:**
- `image` - Input map
- `input_range_min`, `input_range_max` - Value range to select
- `invert` - Flip mask
- `smoothness` (0.0-0.5) - Edge softness

**Outputs:**
- `mask` - Selection mask

**Use Cases:**
- Select height ranges
- Isolate value ranges
- Create soft selections
- Layer masking

---

### Texture Utilities

#### 22. **Texture Offset** ⭐ NEW
Offset, rotate, and wrap textures with seamless tiling support and edge mask.

**Inputs:**
- `image` - Input texture
- `offset_x` (-1.0 to 1.0) - Horizontal shift (fraction of width)
- `offset_y` (-1.0 to 1.0) - Vertical shift (fraction of height)
- `rotation` (-360° to 360°) - Rotation angle
- `wrap_mode` - repeat (tile), clamp (extend), mirror (reflect)
- `edge_mask_width` (0.1) - Edge mask width for affected areas

**Outputs:**
- `image` - Offset/rotated texture
- `edge_mask` - White mask showing transformed/affected edges

**Wrap Modes:**
- **repeat** - True circular/seamless wrapping (perfect for tileable textures)
- **clamp** - Extends edge pixels (useful for bordered images)
- **mirror** - Reflects at edges (creates symmetrical patterns)

**Technical Note:**
- `repeat` mode uses `torch.roll()` for offset (perfect circular wrapping)
- `repeat` mode with rotation tiles 3x3, rotates, then crops center (maintains seamless edges!)
- Other modes use PyTorch's `grid_sample()` with border/reflection padding

**Use Cases:**
- Adjust texture alignment for UV mapping
- Test tiling at different offsets
- Rotate textures without re-rendering
- Fix texture orientation issues

**Tips:**
- Use `repeat` mode for seamless textures (maintains tiling even with rotation!)
- `offset_x/y = 0.5` shifts by half (useful for seam testing)
- Combine with Seamless Tiling for perfect results

---

#### 23. **Texture Tiler** ⭐ NEW
Create grid of repeated textures (2x2, 3x3, etc.) with two modes.

**Inputs:**
- `image` - Input texture
- `tile_x` (1-8) - Horizontal tile count
- `tile_y` (1-8) - Vertical tile count
- `scale_to_input` (False) - Scale output back to input size

**Outputs:**
- `image` - Tiled grid texture

**Two Modes:**

**Mode 1: Scale to Input OFF (default)**
- Output size = Input × (tile_x, tile_y)
- Example: 512×512 input, 3×3 tiles → 1536×1536 output
- Use for: Creating large texture sheets, high-res previews

**Mode 2: Scale to Input ON**
- Output size = Input size (each tile is smaller)
- Example: 512×512 input, 3×3 tiles → 512×512 output (tiles are 170×170)
- Use for: Previewing tiling density, testing scale variations

**Use Cases:**
- Preview how textures tile at different scales
- Create larger texture sheets (scale OFF)
- Test seamless tiling quality
- Visualize density/frequency (scale ON)

**Tips:**
- **Scale OFF:** Best for checking seams at full quality
- **Scale ON:** Best for seeing overall tiling pattern/density
- Use after Seamless Tiling Maker to preview results
- 2x2 is ideal for checking seams
- Larger grids (4x4+) useful with scale ON to see pattern repetition

**Example Workflows:**
```
MODE 1 (Large Preview):
Load Texture (512×512)
   ↓
Seamless Tiling Maker
   ↓
Texture Tiler (3x3, scale_to_input: OFF)
   ↓
Preview (1536×1536 - see all tiles at full size!)

MODE 2 (Density Preview):
Load Texture (512×512)
   ↓
Seamless Tiling Maker
   ↓
Texture Tiler (4x4, scale_to_input: ON)
   ↓
Preview (512×512 - see how dense the pattern looks!)
```

---

#### 24. **Texture Equalizer** ⭐ NEW
Remove uneven lighting and shadows from textures using High Pass technique.

**Inputs:**
- `image` - Texture with uneven lighting/shadows
- `radius` (100) - High pass radius for detail preservation (50-150 typical)
- `strength` (1.0) - Effect intensity (0.0=off, 1.0=full, 2.0=exaggerated)
- `preserve_color` (True) - Keep original hue/saturation, only fix brightness
- `method` (overlay) - Blend method: overlay, soft_light, or linear_light

**Outputs:**
- `image` - Equalized texture with normalized lighting
- `average_color` - Extracted average color (for debugging/visualization)

**How It Works:**
Based on the professional [Photoshop technique](https://tolas.wordpress.com/2009/05/26/tutorial-how-to-equalize-textures-in-photoshop/):
1. Extracts average color from texture
2. Applies High Pass filter to isolate details
3. Inverts high pass and blends using selected method
4. Optionally preserves original color hue

**Parameters Explained:**

**Radius (50-150):**
- **50** - Gentle correction, preserves gradients
- **100** - Standard correction ⭐ (recommended)
- **150** - Aggressive flattening, removes all gradients

**Strength (0.0-2.0):**
- **0.0** - No effect (original)
- **0.5** - Subtle correction
- **1.0** - Full correction ⭐
- **2.0** - Exaggerated/stylized

**Preserve Color:**
- **ON** - Keeps original hue/saturation, fixes only brightness (recommended for albedo)
- **OFF** - Full equalization including color shifts (useful for height maps)

**Method (Blend Mode):**
- **overlay** - Photoshop standard method ⭐ (recommended)
  - Balanced correction, works well for most textures
  - Matches the classic Photoshop equalization tutorial
- **soft_light** - Gentle, subtle correction
  - Less aggressive than overlay
  - Good for textures with mild lighting issues
- **linear_light** - Strong, aggressive correction
  - Maximum correction power
  - Use for heavily shadowed textures

**Use Cases:**
- Remove shadows from photos taken in uneven lighting
- Clean up scanned textures
- Prepare textures for seamless tiling (remove lighting variations)
- Normalize albedo maps before PBR extraction
- Fix height maps with lighting baked in
- Remove vignetting from camera photos

**Perfect For:**
- Photos with directional lighting
- Textures with cast shadows
- Camera vignetting (darker edges)
- Uneven illumination from lighting setup
- Baked lighting in scans

**Before/After Examples:**
```
BEFORE: Photo with shadow on left side, bright on right
AFTER: Evenly lit, shadow removed, details preserved!

BEFORE: Scanned texture with dark edges (vignetting)
AFTER: Uniform brightness edge-to-edge

BEFORE: Height map with baked shadows
AFTER: Clean height data, no lighting artifacts
```

**Pro Tips:**
- Start with `radius: 100, strength: 1.0, preserve_color: ON`
- For subtle effects, reduce strength to 0.5-0.7
- For aggressive flattening, increase radius to 150+
- Always enable `preserve_color` for albedo/diffuse maps
- Disable `preserve_color` for grayscale maps (height, roughness, etc.)
- Apply BEFORE making texture seamless for best results

**Troubleshooting:**
- **Output too dark/black:** 
  - **Try method="overlay"** instead of linear_light ⭐
  - Check console output for value ranges
  - View `average_color` output to verify extraction
  - Try lower radius (30-70)
  - Try lower strength (0.5-0.7)
- **Output washed out:** 
  - Increase radius or strength
  - Try method="linear_light" for stronger correction
- **Colors look wrong:** Enable `preserve_color`
- **Not enough effect:** 
  - Increase radius to 150-200
  - Try method="linear_light"
  - Increase strength to 1.5-2.0

**Workflow:**
```
Load Photo (uneven lighting)
   ↓
Texture Equalizer
├─ radius: 100
├─ strength: 1.0
└─ preserve_color: ON
   ↓
Seamless Tiling Maker
   ↓
PBR Extractor
   ↓
Perfect material with no shadow artifacts!
```

**Reference:** Based on [Tolas' Photoshop technique](https://tolas.wordpress.com/2009/05/26/tutorial-how-to-equalize-textures-in-photoshop/)

#### 26. **Seamless Tiling Maker**
Make textures tile seamlessly with edge mask for inpainting.

**Inputs:**
- `image` - Input texture
- `method` - mirror (fast), blend_edges (best), offset (simple)
- `blend_width` (0.1) - Edge blend width and mask feathering

**Outputs:**
- `image` - Seamless texture
- `edge_mask` - White mask showing edge/seam regions (for inpainting)

**Methods:**
- `mirror` - Flip and average (fast, good for organic)
- `blend_edges` - Smooth edge transitions (best quality)
- `offset` - Shift by half (creates center cross seam)

**Edge Mask:**
- White (1.0) at edges/seams where inpainting would help
- Black (0.0) in center where texture is unchanged
- Gradient based on `blend_width` parameter
- Perfect for: Inpainting cleanup, selective processing, edge detection

**Use Cases:**
- Game textures (seamless + inpaint mask)
- Repeating materials
- Inpainting cleanup (use mask to target seams)
- Tile-able patterns

#### 27. **Square Maker** ⭐ NEW
Convert any image to a perfect square by cropping or scaling.

**Inputs:**
- `image` - Input texture (any dimensions)
- `method` - crop (maintain aspect) or scale (stretch)
- `square_size` - shortest_edge, longest_edge, or custom
- `custom_size` (1024) - Custom square size (64-8192, multiples of 64)
- `crop_position` - 9 positions for crop alignment
- `scaling_method` - bicubic, bilinear, lanczos, nearest

**Outputs:**
- `image` - Perfect square image

**Methods:**

**Crop (Recommended):**
- Maintains aspect ratio
- Removes excess from edges
- No distortion
- Choose from 9 crop positions:
  ```
  Top Left    | Top Center    | Top Right
  Middle Left | Center        | Middle Right
  Bottom Left | Bottom Center | Bottom Right
  ```

**Scale:**
- Stretches to square
- No content loss
- May distort image
- Useful when all content must be preserved

**Square Size Modes:**

**shortest_edge** ⭐ Recommended for crop:
- Square size = shortest dimension
- Example: 1920×1080 → 1080×1080 (crops width)
- No quality loss, only crops excess

**longest_edge:**
- Square size = longest dimension
- Example: 1920×1080 → 1920×1920 (upscales height if crop, or uses 1920 if custom)
- May upscale one dimension

**custom:**
- Specify exact square size
- Scales to match, then crops if needed
- Perfect for specific requirements (512, 1024, 2048, etc.)

**Use Cases:**
- AI models requiring square inputs (Stable Diffusion, etc.)
- Game engines with square texture requirements
- Instagram/social media (square format)
- Cube maps and skyboxes
- Preparing textures for tiling

**Examples:**

```
Example 1: Landscape to Square (Crop Center)
Input: 1920×1080 (landscape photo)
Settings:
  - method: crop
  - square_size: shortest_edge
  - crop_position: center
Output: 1080×1080 (center portion, no distortion)

Example 2: Portrait to Square (Crop Top Center)
Input: 1080×1920 (portrait photo, face at top)
Settings:
  - method: crop
  - square_size: shortest_edge
  - crop_position: top_center
Output: 1080×1080 (top portion with face)

Example 3: Exact Size for AI Model
Input: 3000×2000 (any size)
Settings:
  - method: crop
  - square_size: custom
  - custom_size: 1024
  - crop_position: center
Output: 1024×1024 (perfect for SD)

Example 4: Stretch to Square
Input: 1920×1080
Settings:
  - method: scale
  - square_size: longest_edge
Output: 1920×1920 (stretched, all content preserved)
```

**Pro Tips:**
- Use **crop + shortest_edge** for no quality loss
- Use **center** position for most images
- Use **top_center** for portraits (keeps faces)
- Use **bottom_center** for landscapes (keeps ground)
- Combine with **Smart Texture Resizer** for GPU-optimized squares
- Use **custom** size for specific AI model requirements

**Workflow Example:**
```
Load Image (any size)
   ↓
Square Maker
├─ method: crop
├─ square_size: custom
├─ custom_size: 1024
└─ crop_position: center
   ↓
Smart Texture Resizer (optional)
├─ target_megapixels: 1.0
└─ multiple_of: 64
   ↓
Perfect 1024×1024 texture, GPU-optimized!
```

#### 28. **Smart Texture Resizer** ⭐ NEW
Intelligently resize textures to optimal GPU-friendly resolutions.

**Inputs:**
- `image` - Input texture
- `target_megapixels` (0.25-16.0 MP) - Target size in megapixels
- `multiple_of` - 4, 8, 16, 32, 64, 128, 256 (ensures dimensions are multiples)
- `resize_mode` - fit_within, fit_exact, no_upscale
- `scaling_method` - bicubic, bilinear, lanczos, nearest

**Outputs:**
- `image` - Optimized texture

**Resize Modes:**
- **fit_within** - Stay under target MP, never exceed (safe for VRAM limits)
- **fit_exact** - Get as close to target MP as possible
- **no_upscale** - Only downscale, never increase size

**Multiple Benefits:**
- **32** - Standard GPU optimization (recommended for most textures)
- **64** - Extra GPU friendly, some game engines prefer this
- **16** - Good for smaller textures
- **8** - AI model compatibility (Stable Diffusion, etc.)
- **4** - Minimum GPU alignment

**Why This Matters:**
- GPUs process textures most efficiently at multiples of 32
- Game engines often require power-of-2 or specific multiples
- Reduces VRAM waste and improves performance
- AI models (SD, ControlNet) often need multiples of 8/64

**Examples:**
```
Input: 1920×1080 (2.07 MP, not GPU optimized)
Target: 2.0 MP, multiple_of: 32
Output: 1536×864 (1.33 MP, both divisible by 32!)

Input: 4096×4096 (16.7 MP, too large)
Target: 4.0 MP, multiple_of: 64
Output: 2048×2048 (4.2 MP, perfect square, divisible by 64!)

Input: 3000×2000 (6.0 MP, odd dimensions)
Target: 4.0 MP, multiple_of: 32, mode: fit_within
Output: 1920×1280 (2.46 MP, GPU optimized)
```

**Use Cases:**
- Optimize textures for game engines
- Prepare images for AI models (SD needs multiples of 8/64)
- Reduce VRAM usage while maintaining quality
- Ensure GPU-friendly dimensions automatically

#### 29. **Texture Scaler**
Manual texture resolution scaling.

**Inputs:**
- `image` - Input texture
- `scale_factor` (0.125-8.0x) - Size multiplier
- `method` - nearest (pixel art), bilinear (fast), bicubic (quality), lanczos (best)

**Outputs:**
- `image` - Scaled texture

**Tips:**
- Use `nearest` for pixel art preservation
- Use `bicubic` or `lanczos` for photographic textures
- Scale up to 8x or down to 0.125x
- For GPU optimization, use **Smart Texture Resizer** instead

#### 30. **Triplanar Projection** ⭐ NEW
Remove UV seams with XYZ projection blending.

**Inputs:**
- `image` - Input texture
- `blend_sharpness` (1.0) - Transition sharpness
- `scale` (1.0) - Texture tiling

**Outputs:**
- `image` - Projected texture

**Use Cases:**
- Remove UV seams
- Organic surfaces
- Procedural textures
- Seamless projections

---

### Channel Tools

#### 31. **Grayscale to Color** ⭐ NEW
Convert grayscale images to RGB by repeating the channel.

**Inputs:**
- `grayscale` - Single-channel grayscale image

**Outputs:**
- `color` - RGB image (grayscale repeated across R, G, B)

**Use Cases:**
- Convert height/AO maps to RGB for display
- Prepare grayscale for nodes requiring RGB input
- Visualization purposes

#### 32. **Color to Grayscale** ⭐ NEW
Convert RGB images to grayscale using various methods.

**Inputs:**
- `color` - RGB image
- `method` - luminance (perceptual), average, lightness, red_only, green_only, blue_only

**Outputs:**
- `grayscale` - Grayscale image (as RGB for display)

**Methods:**
- **luminance** - Perceptual (Rec. 709: 0.2126R + 0.7152G + 0.0722B) - Best for photos
- **average** - Simple average (R+G+B)/3 - Fast
- **lightness** - (max+min)/2 - Preserves brightness range
- **red/green/blue_only** - Extract single channel

**Use Cases:**
- Convert albedo to grayscale for masks
- Create height maps from photos
- Channel extraction

---

#### 33. **Channel Packer (RGB)**
Pack 3 grayscale maps into RGB channels (saves VRAM).

**Inputs:**
- `red_channel` - Grayscale map for R
- `green_channel` (optional) - For G
- `blue_channel` (optional) - For B
- `preset` - custom, orm_unity, orm_unreal, rma

**Outputs:**
- `packed` - RGB packed image

**Common Packing:**
- **ORM** (Unity): AO(R) + Roughness(G) + Metallic(B)
- **RMA**: Roughness(R) + Metallic(G) + AO(B)

**Use Cases:**
- Reduce texture count for games
- Save VRAM
- Standard game engine formats

#### 34. **Channel Unpacker (RGB)**
Extract RGB channels back to grayscale maps.

**Inputs:**
- `packed` - RGB packed image
- `output_channels` - all, r_only, g_only, b_only, rg, rb, gb

**Outputs:**
- `red`, `green`, `blue` - Extracted channels (as RGB for display)

**Use Cases:**
- Extract from packed textures
- Separate channels for editing
- Convert ORM back to individual maps

#### 35. **Channel Packer (ORMA)** ⭐ NEW
Advanced ORM packing with alpha channel support.

**Inputs:**
- `occlusion` - AO map (R channel)
- `roughness` - Roughness map (G channel)
- `metallic` - Metallic map (B channel)
- `alpha` (optional) - Transparency/mask (A channel, defaults to 1.0 if not provided)

**Outputs:**
- `orma_packed` - RGBA packed image

**Format:**
- **R** - Occlusion (AO)
- **G** - Roughness
- **B** - Metallic
- **A** - Alpha (transparency/mask)

**Use Cases:**
- Game engines with transparency support
- Vegetation/foliage materials (alpha for cutout)
- Decals with transparency
- Advanced material packing with masks

**Advantage over ORM:**
- Supports transparency/opacity maps
- Perfect for masked materials
- 4 maps in 1 texture!

#### 36. **Channel Packer (RMA)** ⭐ NEW
Pack RMA format (alternative to ORM).

**Inputs:**
- `roughness` - Roughness map (R channel)
- `metallic` - Metallic map (G channel)
- `ao` - Ambient Occlusion map (B channel)

**Outputs:**
- `rma_packed` - RGB packed image

**Format:**
- **R** - Roughness
- **G** - Metallic
- **B** - AO (Ambient Occlusion)

**Use Cases:**
- Some game engines prefer RMA over ORM
- Alternative channel layout
- Consistency with certain workflows

**RMA vs ORM:**
- **RMA**: Roughness, Metallic, AO
- **ORM**: AO, Roughness, Metallic
- Same maps, different channel order

#### 37. **Channel Packer (RMAA)** ⭐ NEW
Advanced RMA packing with alpha channel support.

**Inputs:**
- `roughness` - Roughness map (R channel)
- `metallic` - Metallic map (G channel)
- `ao` - AO map (B channel)
- `alpha` (optional) - Transparency/mask (A channel, defaults to 1.0 if not provided)

**Outputs:**
- `rmaa_packed` - RGBA packed image

**Format:**
- **R** - Roughness
- **G** - Metallic
- **B** - AO (Ambient Occlusion)
- **A** - Alpha (transparency/mask)

**Use Cases:**
- RMA format with transparency support
- Vegetation/foliage materials (alpha for cutout)
- Decals with transparency
- Alternative to ORMA with different channel order

**RMAA vs ORMA:**
- **RMAA**: Roughness, Metallic, AO, Alpha
- **ORMA**: AO, Roughness, Metallic, Alpha
- Both support 4 channels, different channel order preference

#### 38. **Channel Unpacker (RMA)** ⭐ NEW
Unpack RMA format back to individual maps.

**Inputs:**
- `rma_packed` - RGB packed RMA image

**Outputs:**
- `roughness` - Extracted roughness (as RGB for display)
- `metallic` - Extracted metallic (as RGB for display)
- `ao` - Extracted AO (as RGB for display)

**Use Cases:**
- Extract from RMA packed textures
- Edit individual RMA channels
- Convert RMA to separate maps for processing

---

### Additional Processing

#### 39. **Height Processor (Lotus)**
Process Lotus depth output.

**Inputs:**
- `lotus_depth` - Lotus output
- `invert` - Flip values
- `bit_depth` - 8, 16, or 32-bit

**Outputs:**
- `height` - Processed height map

---

## 💡 Emission Maps

**NEW in v2.1!** Full emission map support throughout the PBR pipeline.

**What are Emission Maps?**
- Self-illuminated/glowing parts of materials
- Used for lights, screens, lava, neon signs, glowing eyes, etc.
- Not affected by scene lighting

**Usage:**
```
Load Emission Texture
   ↓
PBR Combiner
├─ albedo: (your albedo)
├─ normal: (your normal)
├─ roughness: (your roughness)
├─ metallic: (your metallic)
└─ emission: (glowing parts) ⭐
   ↓
PBR Pipe → Game Engine
```

**Common Uses:**
- **Screens/Monitors** - Emissive UI elements
- **Lights** - Light fixtures, bulbs, LEDs
- **Lava/Fire** - Glowing hot materials
- **Neon Signs** - Bright glowing text
- **Sci-Fi** - Glowing panels, energy effects
- **Magic Effects** - Glowing runes, spells

**Workflow Example:**
```
Albedo → Brightness/Contrast → Mask bright areas → Emission map
                                                      ↓
                                            PBR Combiner (emission input)
```

---

## 🎮 Example Workflows

### ⚡ Quick Start (Fastest)
```
Load Image
   ↓
   ├─> Marigold (appearance) ─┐
   └─> Marigold (lighting) ────┼─> PBR Extractor ──> PBR Saver ──> Preview Image
                               │      (pbr_pipe)        (images)     (view all)
                               └─ Done in 3 nodes!
```

### 🎯 Complete Workflow (Recommended)
```
Load Image
   ↓
   ├─> Marigold (appearance) ─┐
   ├─> Marigold (lighting) ────┼─> PBR Extractor ──> pbr_pipe ─┐
   ├─> Lotus (normal) ─────────┤                                │
   └─> Lotus (height) ─────────┘                                ↓
                                                    PBR Combiner (merge)
                                                    ├─ pbr_pipe
                                                    ├─ normal (Lotus)
                                                    └─ height (Lotus)
                                                                ↓
                                                    Height to Normal ──┐
                                                    (generate backup)  │
                                                                ↓      ↓
                                                    PBR Pipeline Adjuster
                                                    ├─ ao_strength_albedo: 1.0
                                                    ├─ roughness_strength: 1.2
                                                    ├─ albedo_saturation: 1.1
                                                    └─ normal_strength: 1.0
                                                                ↓
                                                    PBR Pipe Preview ──> Preview
                                                    (check before save)
                                                                ↓
                                                    PBR Saver
                                                    ├─ base_name: "material_name"
                                                    └─ format: png
                                                                ↓
                                                    Preview Image
                                                    (final review)
```

### 🎨 Advanced: Material Mixing
```
Material A (pbr_pipe) ─┐
                       ├─> PBR Material Mixer ──> mixed_pipe
Material B (pbr_pipe) ─┤     ├─ blend_mode: overlay
Mask (optional) ───────┘     ├─ strength: 0.5
                             └─ mask: (optional)
```

### 🛠️ Advanced: Weathering & Wear
```
PBR Pipe ──> Curvature Generator ──┐
              (from normal/height)  │
                                   ├─> Wear Generator ──> worn_pipe
AO Map ───────────────────────────┘    ├─ edge_wear: 0.7
                                        ├─ dirt_strength: 0.3
                                        └─ wear_strength: 0.5
```

### 📦 Advanced: Channel Packing
```
Roughness ──┐
            ├─> Channel Packer ──> ORM_packed.png
Metallic ───┤    (preset: orm_unity)
AO ─────────┘    R=AO, G=Roughness, B=Metallic
```

### 🎨 Advanced: Stylized Materials
```
Height Map ──> Gradient Map ──> Color Ramp ──> Stylized Albedo
               ├─ min: 0.2     (heat preset)
               └─ max: 0.8
```

---

## 💡 Pro Tips

### Extraction & Quality
1. **Washed out maps?** Adjust `gamma_metal_rough` (try 1.2-1.6)
2. **Better AO?** Use AO Approximator with height + normal, 24-32 samples
3. **Sharper normals?** Use `method=scharr` in Height to Normal converter
4. **Cross-platform normals?** Use Normal Format Converter

### Material Enhancement
5. **Realistic wear:** Curvature Generator → Wear Generator (edge_wear=0.7)
6. **Weathering:** Set `ao_strength_roughness=0.3` in Pipeline Adjuster
7. **Add detail:** Detail Map Blender with RNM mode for normals
8. **Seamless textures:** Seamless Tiling Maker with blend_edges method

### Color & Appearance
9. **Color variations:** HSV Adjuster (hue_shift for different colors)
10. **Vibrant materials:** Increase `albedo_saturation` (1.2-1.5)
11. **Stylized look:** Color Ramp with custom gradients or presets
12. **Heat maps:** Color Ramp heat preset on AO or height

### Workflow Optimization
13. **Clean workflow:** Use Pipeline system (fewer connections)
14. **Debug easily:** PBR Pipe Preview at any stage
15. **Batch materials:** PBR Saver enumerate mode
16. **Save VRAM:** Channel Packer for ORM textures

### Advanced Techniques
17. **Layer materials:** PBR Material Mixer with masks
18. **Procedural damage:** Gradient Map → Color Ramp → Wear Generator
19. **Detail layers:** Base → Detail Map Blender (x3) → fine detail
20. **Smart masks:** Curvature + Gradient Map for precise selections

---

## 🎯 Common Use Cases

### Game Development
- **Optimize textures:** Channel Packer (ORM format)
- **Tile-able materials:** Seamless Tiling Maker
- **LOD textures:** Texture Scaler for multiple resolutions
- **Batch export:** PBR Saver with enumerate mode

### Architectural Visualization
- **Weathered materials:** Wear Generator + Curvature
- **Detail enhancement:** Detail Map Blender
- **Material variations:** HSV Adjuster + Material Mixer
- **High precision:** Save as EXR (32-bit)

### 3D Printing
- **Height maps:** Normal to Depth Converter
- **Clean normals:** Normal Format Converter
- **Seamless patterns:** Seamless Tiling Maker + Triplanar Projection

### VFX & Film
- **Material layering:** Material Mixer with masks
- **Procedural aging:** Wear Generator + Gradient Maps
- **Color grading:** HSV Adjuster + Color Ramp
- **High-res export:** Texture Scaler + EXR format

---

## 🔧 Technical Details

### PBR_PIPE Format
Internal dictionary structure:
```python
{
    'albedo': IMAGE tensor,
    'normal': IMAGE tensor,
    'ao': IMAGE tensor,
    'height': IMAGE tensor,
    'roughness': IMAGE tensor,
    'metallic': IMAGE tensor,
    'lighting': IMAGE tensor,
    'transparency': IMAGE tensor
}
```

### Image Formats
- **PNG**: 8-bit, standard
- **JPG**: 8-bit, compressed
- **EXR**: 16/32-bit, lossless
- **TIFF**: 16-bit, compatible

### Performance Notes
- **Color Ramp:** Fully GPU-accelerated (100x faster than v1.0)
- **Channel Packing:** Instant, no quality loss
- **Integration methods:** CPU-intensive (Normal to Depth hybrid mode recommended)
- **Wear Generator:** GPU-accelerated, scales with resolution

---

## ✅ Requirements

**Included with ComfyUI:**
- PyTorch (GPU acceleration)
- PIL (image I/O)
- NumPy (array processing)
- imageio (format support)

**Optional for source inputs:**
- [ComfyUI_Marigold- kijai](https://github.com/kijai/ComfyUI-Marigold)
- [ComfyUI_Lotus -kijai](https://github.com/kijai/ComfyUI-Lotus)




