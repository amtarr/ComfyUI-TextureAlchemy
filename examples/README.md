# 🎮 TextureAlchemy - Example Workflows

Ready-to-use ComfyUI workflow JSON files for testing and learning.

---

## 📂 Workflow Files

### 01_quick_start_pbr_extraction.json
**Fastest PBR Material Extraction** - 6 nodes total

**What it does:**
- Loads an image
- Runs Marigold (appearance + lighting)
- Extracts PBR maps (albedo, roughness, metallic, AO)
- Saves all maps with auto-naming
- Previews all saved maps

**Nodes used:**
1. Load Image
2. Marigold Depth Estimation x2
3. PBR Extractor
4. PBR Saver
5. Preview Image

**Perfect for:** First-time testing, quick material generation

**Expected output:** 4 maps (albedo, roughness, metallic, lighting) saved to `ComfyUI/output/pbr_materials/`

---

### 02_seamless_tiling.json
**Tileable Material Creation** - 7 nodes total

**What it does:**
- Loads texture
- Makes it seamlessly tileable
- Extracts PBR maps
- Saves tileable material set

**Nodes used:**
1. Load Image
2. Seamless Tiling Maker
3. Marigold x2
4. PBR Extractor
5. PBR Saver
6. Preview Image

**Perfect for:** Game textures, repeating materials

**Parameters:**
- Tiling method: `blend_edges` (best quality)
- Blend width: `0.1` (10% edge blending)

**Outputs:**
- Seamless tileable texture
- **Edge mask** - White mask showing seam regions (perfect for inpainting cleanup!)

**Expected output:** Seamless tileable PBR material set

**Pro tip:** Use the edge_mask output with inpainting nodes to further clean up any remaining seams!

---

### 03_channel_packing_orm.json
**ORM Texture Packing** - 8 nodes total

**What it does:**
- Extracts PBR maps
- Packs AO, Roughness, Metallic into single RGB texture
- Saves packed ORM texture (game engine format)

**Nodes used:**
1. Load Image
2. Marigold x2
3. PBR Extractor
4. PBR Splitter
5. Channel Packer
6. Save Image
7. Preview Image

**Perfect for:** Game engines (Unity/Unreal), VRAM optimization

**Format:** ORM_packed.png
- Red channel = AO (Ambient Occlusion)
- Green channel = Roughness
- Blue channel = Metallic

**Memory saved:** 66% (3 textures → 1 texture)

---

### 04_color_ramp_stylized.json
**Stylized Material with Custom Colors** - 10 nodes total

**What it does:**
- Extracts PBR material
- Uses AO to create color gradient
- Applies heat map preset
- Replaces albedo with stylized colors
- Saves final material

**Nodes used:**
1. Load Image
2. Marigold x2
3. PBR Extractor
4. PBR Splitter
5. Gradient Map
6. Color Ramp (interactive!)
7. PBR Combiner
8. PBR Saver
9. Preview Image

**Perfect for:** Artistic materials, data visualization, heat maps

**Interactive:** Double-click Color Ramp node to edit gradient visually!

**Parameters:**
- Preset: `heat` (black → red → orange → yellow)
- Gradient range: 0.2 to 0.8
- Can customize colors by clicking on Color Ramp

### 05_seamless_with_inpainting.json
**Seamless Tiling with Edge Mask for Inpainting** - 6 nodes + notes

**What it does:**
- Makes texture seamless
- Generates edge mask showing seam regions
- Provides mask for targeted inpainting cleanup

**Nodes used:**
1. Load Image
2. Seamless Tiling Maker
3. Preview Image x2 (result + mask)
4. Note nodes (documentation)

**Perfect for:** Creating ultra-clean seamless textures with inpainting

**Outputs:**
- **Seamless image** - Ready to use or further process
- **Edge mask** - White where seams are (use for inpainting)

**How the mask helps:**
- Shows exactly where edges/seams are located
- Connect to inpainting nodes (LaMa, MAG, etc.)
- Inpainting targets ONLY the seam areas
- Results in cleaner, more perfect tiling

**Advanced workflow:**
```
Seamless Tiling → image + edge_mask
                    ↓         ↓
                [PBR Extract] + [Inpainting with mask]
                    ↓
              Perfect seamless PBR material
```

### 06_texture_offset_and_tiling.json
**Texture Offset & Tiling Preview** - 8 nodes + notes

**What it does:**
- Makes texture seamless
- Tests seamless quality by offsetting by 50%
- Creates tiled previews (2x2 and 3x3)
- Reveals any remaining seams

**Nodes used:**
1. Load Image
2. Seamless Tiling Maker
3. Texture Offset (shift by 0.5 to test seams)
4. Texture Tiler x2 (2x2 and 3x3 previews)
5. Preview Image x2
6. Note node (documentation)

**Perfect for:** Testing seamless texture quality thoroughly

**How it works:**
- **Upper branch:** Offset by 0.5 (half) shifts seams to center
  - If texture is truly seamless, NO seams will appear
  - Tiles as 3x3 grid for comprehensive view
- **Lower branch:** Direct 2x2 tiling without offset
  - Shows normal tiling appearance

**Why offset by 0.5?**
- Moves edge seams to the CENTER of the image
- Makes ANY remaining seams immediately visible
- Industry-standard technique for testing tileable textures

**Texture Offset features:**
- `offset_x/y`: -1.0 to 1.0 (fraction of image size)
- `rotation`: -360° to 360° (test different orientations)
- `wrap_mode`: 
  - **repeat** - Seamless circular wrapping (uses `torch.roll` for offset, tiles 3x3 for rotation)
  - **clamp** - Extends edge pixels
  - **mirror** - Reflects at boundaries
- `edge_mask_width`: 0.0-0.5 (shows affected/transformed areas)
- **NEW OUTPUT:** `edge_mask` - White mask showing where wrapping/rotation affected the image

**Texture Tiler features:**
- `tile_x/y`: 1-8 (grid size)
- `scale_to_input`: **NEW!** Two modes:
  - **OFF (default):** Output size = input × (tile_x, tile_y) - Creates large preview
  - **ON:** Output size = input size - Each tile scaled down to fit
- Perfect for visualizing repeating patterns at different scales

**Tiler Mode Examples:**
```
Input: 512×512, Tiles: 3×3
  - scale_to_input OFF → Output: 1536×1536 (see all tiles full-size)
  - scale_to_input ON  → Output: 512×512 (see tiling density/frequency)
```

**Advanced uses:**
```
Texture Offset
├─ offset_x: 0.5 → shift right by half
├─ offset_y: 0.5 → shift down by half
├─ rotation: 45 → test at angle
└─ wrap_mode: repeat → seamless wrapping
```

### 07_texture_tiler_modes.json
**Texture Tiler Mode Comparison** - 8 nodes + notes

**What it does:**
- Demonstrates both Texture Tiler modes side-by-side
- Shows difference between large grid vs density preview
- Compares 3×3 and 5×5 tiling at same output size

**Nodes used:**
1. Load Image (seamless texture)
2. Texture Tiler x3 (different configurations)
3. Preview Image x3
4. Note node (documentation)

**Perfect for:** Understanding scale_to_input feature

**Comparison:**
- **Top preview:** 3×3 tiles, scale OFF → 1536×1536 output (large grid)
- **Middle preview:** 3×3 tiles, scale ON → 512×512 output (density view)
- **Bottom preview:** 5×5 tiles, scale ON → 512×512 output (high frequency)

**Use cases for each mode:**

**Scale OFF (default):**
- Check seam quality at full resolution
- Create large texture sheets
- Export high-res tiled versions

**Scale ON:**
- Preview how dense/frequent pattern appears
- Compare different tiling densities (3×3 vs 5×5)
- Test visual impact of scale
- See how pattern looks when tiled smaller

**Example comparison:**
```
Same 512×512 input texture:

3×3, scale OFF → 1536×1536 (see every detail)
3×3, scale ON  → 512×512 (normal density)
5×5, scale ON  → 512×512 (high density/busy pattern)
```

**Why use scale ON?**
- Quickly preview different tiling frequencies
- All previews same size = easy comparison
- Useful for game textures (how does it look at different UV scales?)

### 08_texture_equalizer_test.json
**Texture Equalization - Remove Shadows** - 7 nodes + notes

**What it does:**
- Removes uneven lighting, shadows, and vignetting from textures
- Based on professional Photoshop High Pass + Linear Light technique
- Compares different radius settings side-by-side

**Nodes used:**
1. Load Image (texture with uneven lighting)
2. Texture Equalizer x2 (radius 100 and radius 50, both using overlay method)
3. Preview Image x4 (original + results + average_color debug)
4. Note node (comprehensive documentation)

**Outputs:**
- **image** - Equalized result (shadows removed)
- **average_color** - Debug output showing extracted average tone

**Blend methods:**
- **overlay** - Photoshop standard (default, recommended)
- **soft_light** - Gentle correction
- **linear_light** - Strong correction

**Perfect for:** Cleaning photos with shadows, preparing textures for tiling

**How the technique works:**
Based on [this classic tutorial](https://tolas.wordpress.com/2009/05/26/tutorial-how-to-equalize-textures-in-photoshop/):
1. Extracts average color from texture
2. Applies High Pass filter (isolates details, removes gradients)
3. Uses Linear Light blend mode to normalize lighting
4. Preserves original color hue (optional)

**Parameters to experiment with:**

**Radius (50-150):**
- **50** - Gentle, preserves some gradients
- **100** - Standard correction (default) ⭐
- **150** - Aggressive, removes all gradients

**Strength (0.0-2.0):**
- **0.5** - Subtle correction
- **1.0** - Full correction ⭐
- **1.5** - Extra strong (stylized)

**Preserve Color:**
- **ON** - Keep original hue/saturation (recommended for albedo)
- **OFF** - Full equalization (good for height maps)

**Common problems it fixes:**
- ✓ Photos with directional lighting/shadows
- ✓ Camera vignetting (dark edges)
- ✓ Uneven scanner illumination
- ✓ Baked lighting in textures
- ✓ Gradient lighting from light sources
- ✓ Hot spots and dark patches

**Recommended workflow:**
```
Load Photo (with shadows/uneven lighting)
   ↓
Texture Equalizer
├─ radius: 100
├─ strength: 1.0
└─ preserve_color: ON
   ↓
Seamless Tiling Maker (now works better!)
   ↓
PBR Extractor
   ↓
Clean material, no shadow artifacts!
```

**Pro tips:**
- Apply BEFORE seamless tiling for best results
- Use higher radius (150+) for removing large shadows
- Use lower radius (50-70) to preserve some depth
- Enable preserve_color for albedo/diffuse maps
- Disable preserve_color for grayscale maps

**Troubleshooting:**
- **If output looks black:** 
  - ⭐ **Ensure method is set to "overlay"** (not linear_light)
  - Check console output for value ranges
  - View `average_color` output to see extracted tone
  - Try reducing radius (try 30-50)
  - Try reducing strength (try 0.5-0.7)
- **If output looks washed out:** 
  - Increase radius or strength
  - Try method="linear_light" for stronger correction
- **If colors look wrong:** Enable `preserve_color`
- **If not enough effect:**
  - Increase radius to 150-200
  - Try method="linear_light"

**Reference:** [Tolas' Photoshop Equalization Tutorial](https://tolas.wordpress.com/2009/05/26/tutorial-how-to-equalize-textures-in-photoshop/)

---

## 09_normal_format_validator.json
**Description:** Automatically detect whether a normal map is OpenGL or DirectX format with visual analysis.

**Nodes used:**
1. Load Image (normal map to validate)
2. Normal Format Validator (analyzer)
3. Preview Image x2 (original + 4-panel analysis)
4. Note node (comprehensive documentation)

**Outputs:**
- **visualization** - 4 panels showing:
  1. Original normal map
  2. Green channel isolated
  3. Threshold map (white=up, black=down)
  4. Format indicator (green=OGL, red=DX, yellow=ambiguous)
- **detected_format** - String with format and confidence

**Console output:**
```
📊 GREEN CHANNEL ANALYSIS:
  Pixels > 0.5: 73.2% (up-facing)
  Pixels < 0.5: 26.8% (down-facing)
  Bias: +0.464

🎯 DETECTION RESULT:
  Format: OpenGL
  Confidence: High
```

**How it works:**
- Analyzes green channel distribution
- **OpenGL**: Y points UP → more pixels > 0.5
- **DirectX**: Y points DOWN → more pixels < 0.5
- Provides confidence based on bias strength

**Confidence levels:**
- **High** (>20% bias) - Very clear
- **Medium** (10-20%) - Likely correct
- **Low** (5-10%) - Uncertain
- **Ambiguous** (<5%) - Flat or balanced

**Use cases:**
- Verify normals from unknown sources
- Debug "why do my normals look inverted?"
- QA for asset pipelines
- Educational tool for learning normal map formats

**Pro tips:**
- Check visualization if result is ambiguous
- Flat normal maps may return AMBIGUOUS (this is normal)
- Use with "Normal Format Converter" to fix incorrect formats
- Great for troubleshooting imported assets

---

## 📥 How to Use These Workflows

### Method 1: Drag & Drop
1. Open ComfyUI in your browser
2. Drag the `.json` file directly onto the ComfyUI window
3. Workflow loads automatically!

### Method 2: Load from Menu
1. In ComfyUI, click **"Load"** button (top menu)
2. Navigate to `custom_nodes/ComfyUI_PBR_MaterialProcessor/examples/`
3. Select a workflow JSON file
4. Click **Open**

### Method 3: Copy-Paste
1. Open the `.json` file in a text editor
2. Copy all contents
3. In ComfyUI, press **Ctrl+V**
4. Workflow appears!

---

## ⚙️ Before Running

### Required Nodes
All workflows require these ComfyUI nodes:
- **PBR Material Processor** (this pack) ✅
- **ComfyUI Marigold** (for depth estimation)
- ComfyUI base nodes (Load Image, Save Image, etc.)

Optional (for extended workflows):
- **ComfyUI Lotus** (for normal/height maps)

### Check Your Input
Each workflow starts with **Load Image**. You'll need to:
1. Click on the "Load Image" node
2. Choose your input image from the dropdown
3. Or upload a new image using the upload button

---

## 🎯 Workflow Testing Checklist

### Test 1: Quick Start ✓
```
□ Load workflow: 01_quick_start_pbr_extraction.json
□ Select an input image
□ Click "Queue Prompt"
□ Check console for progress
□ Verify 4 maps saved to output/pbr_materials/
□ Preview shows all maps in one view
```

### Test 2: Seamless Tiling ✓
```
□ Load workflow: 02_seamless_tiling.json
□ Use a texture with visible edges
□ Queue prompt
□ Check if result tiles seamlessly
□ Open saved files in image viewer
□ Tile them side-by-side to verify seamlessness
```

### Test 3: Channel Packing ✓
```
□ Load workflow: 03_channel_packing_orm.json
□ Queue prompt
□ Check saved ORM_packed.png
□ Verify R=AO, G=Roughness, B=Metallic
□ Compare file size: 1 ORM vs 3 separate maps
```

### Test 4: Color Ramp ✓
```
□ Load workflow: 04_color_ramp_stylized.json
□ Queue prompt
□ Check stylized albedo
□ INTERACTIVE: Double-click Color Ramp node
□ Try changing colors (click swatches)
□ Try different presets (heat, rainbow, gold)
□ Queue again to see changes
```

---

## 🔧 Customizing Workflows

### Change Output Location
Find `PBRSaver` node → Change `output_path`:
```
"widgets_values": [
  "my_material",          // base_name
  "pbr_materials",        // ← Change this
  "png",
  "enumerate",
  1
]
```

### Change File Format
In `PBRSaver` or `SaveImage` → Change format:
```
"png"   → Standard, lossless
"jpg"   → Compressed, smaller files
"exr"   → 32-bit precision (for normals/height)
"tiff"  → Professional, 16-bit
```

### Adjust Gamma (if maps look washed out)
Find `PBRExtractor` node → Adjust parameters:
```
"widgets_values": [
  "lighting",   // albedo_source ← Try "appearance" for alternative
  0.45,         // gamma_albedo (affects albedo from both sources)
  2.2,          // gamma_metal_rough ← Adjust 1.8-2.6 if needed
  0.45          // gamma_lighting_ao
]
```

### Change Tiling Method
Find `SeamlessTiling` node → Change method:
```
"widgets_values": [
  "blend_edges",  // ← Try: "mirror" or "offset"
  0.1             // blend_width (for blend_edges mode)
]
```

### Change Color Ramp Preset
Find `ColorRamp` node → Change preset:
```
"widgets_values": [
  "heat",     // ← Try: "rainbow", "gold_metal", "rust", etc.
  "linear",   // interpolation
  "[...]"     // color_stops data
]
```

---

## 📊 Performance Notes

| Workflow | Approx. Time (1024x1024) | VRAM Usage |
|----------|--------------------------|------------|
| 01_quick_start | ~30 seconds | ~4 GB |
| 02_seamless_tiling | ~35 seconds | ~4 GB |
| 03_channel_packing | ~30 seconds | ~4 GB |
| 04_color_ramp_stylized | ~35 seconds | ~4 GB |

*Times based on RTX 3080. Actual times vary by GPU and image size.*

---

## 🐛 Troubleshooting

### "Cannot find node type: PBRExtractor"
**Solution:** Restart ComfyUI after installing PBR Material Processor

### "Cannot find node type: MarigoldDepthEstimation_v2"
**Solution:** Install ComfyUI_Marigold from ComfyUI Manager

### "Output folder not found"
**Solution:** Folders are created automatically. If error persists, manually create `ComfyUI/output/pbr_materials/`

### Workflow loads but nodes are red
**Solution:** Missing required nodes. Check console for which nodes are missing.

### "Expected IMAGE but got PBR_PIPE"
**Solution:** Use PBR Splitter to extract IMAGE from PBR_PIPE if needed

### Maps look washed out
**Solution:** Adjust `gamma_metal_rough` (try 1.8-2.6) in PBR Extractor. Default is 2.2.

### Color Ramp gradient doesn't show
**Solution:** 
1. Restart ComfyUI (widget loads on startup)
2. Clear browser cache (Ctrl+F5)
3. Check browser console (F12) for errors

---

## 🎨 Creating Your Own Workflows

### Tips for Building Workflows:
1. **Start simple:** Begin with quick_start, add nodes gradually
2. **Use PBR_PIPE:** Cleaner than individual connections
3. **Add PBR Pipe Preview:** Debug at any stage
4. **Save incrementally:** Save versions as you build
5. **Test frequently:** Queue prompt after each major change
6. **Document:** Add notes about what each section does

### Common Workflow Pattern:
```
Load Image
   ↓
[Preprocessing] (optional: Seamless Tiling, Scaling, etc.)
   ↓
[AI Generation] (Marigold, Lotus)
   ↓
[PBR Extraction] (PBR Extractor)
   ↓
[Pipe Assembly] (PBR Combiner)
   ↓
[Adjustments] (PBR Pipeline Adjuster, Color tools)
   ↓
[Effects] (Wear, Detail, Mixing - optional)
   ↓
[Output] (PBR Saver, Save Image)
   ↓
[Preview] (Preview Image)
```

---

---

## 🆕 NEW WORKFLOWS (v3.0) - Advanced Features

### 10_fastest_pbr_extraction.json
**Lightning-Fast PBR Extraction** ⚡ **NEW!**

**Nodes:** 6 total  
**Difficulty:** ⭐ Beginner  
**Time:** ~2 seconds

**What it does:** Fastest workflow to extract complete PBR material

**Perfect for:** Learning basics, quick materials, testing

---

### 11_frequency_separation_workflow.json
**Professional Texture Editing** 🔬 **NEW!**

**Nodes:** 7 total  
**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** ~1 second

**What it demonstrates:** Frequency separation - the professional's secret weapon!

**Perfect for:** Retouching, skin/fabric editing, pro workflows

---

### 12_procedural_material_creation.json
**Create Materials from Scratch** 🎲 **NEW!**

**Nodes:** 11 total  
**Difficulty:** ⭐⭐⭐ Advanced  
**Time:** ~3 seconds

**What it demonstrates:** Complete PBR material without any photo!

**Perfect for:** Stylized materials, procedural generation, experimental art

---

### 13_mask_generation_weathering.json
**Mask Generation Showcase** 🎭 **NEW!**

**Nodes:** 10 total  
**Difficulty:** ⭐⭐ Intermediate  
**Time:** <1 second

**What it demonstrates:** All 4 mask generator types with examples

**Perfect for:** Weathering, selective editing, layer masking

---

### 14_blend_modes_showcase.json
**24 Blend Modes Comparison** 🎨 **NEW!**

**Nodes:** 11 total  
**Difficulty:** ⭐⭐ Intermediate  
**Time:** <1 second

**What it demonstrates:** 4 essential blend modes + guide to all 24

**Perfect for:** Learning blending, texture compositing, effects

---

**See `WORKFLOW_INDEX.md` for detailed descriptions of all workflows!**

---

## 📤 Sharing Your Workflows

If you create a great workflow:
1. Save it as JSON
2. Test it with different images
3. Add comments (using Group nodes or notes)
4. Share with the community!

**Workflow naming convention:**
```
##_descriptive_name.json

Examples:
05_weathered_metal.json
06_fabric_detail.json
07_multi_material_mix.json
```

---

## 🔗 Next Steps

After testing these workflows:
1. **Read full docs:** `../README.md` for all node details
2. **Try more workflows:** `../WORKFLOWS.md` for 10 more examples
3. **Quick reference:** `../QUICK_REFERENCE.md` for parameters
4. **Build custom:** Combine techniques from multiple workflows

---

## 📝 Workflow JSON Format

ComfyUI workflows are JSON files containing:
- **Nodes:** Each node's type, position, and parameters
- **Links:** Connections between nodes
- **Groups:** Optional visual grouping
- **Config:** Workflow-specific settings

You can edit them in any text editor, but it's easier to use ComfyUI's visual interface!

---

**Have fun creating amazing PBR materials!** 🎨🚀

For more workflows and examples, see:
- `../WORKFLOWS.md` - Detailed workflow guide
- `../README.md` - Complete node reference
- `../QUICK_REFERENCE.md` - Fast parameter lookup

