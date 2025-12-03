# ⚡ TextureAlchemy - Quick Reference

Fast lookup for all 38 nodes.

**📍 Finding Nodes in ComfyUI:**
Right-click → Add Node → **Texture Alchemist** → Choose subcategory

---

## 📋 Node Categories & Quick Links

All organized under **Texture Alchemist** in ComfyUI:

| Category | Nodes | Use For |
|----------|-------|---------|
| **Core** | 3 nodes | Extract & adjust from AI |
| **Pipeline** | 6 nodes | Bundle, mix, save materials |
| **Normal/Height** | 5 nodes | Process, convert, combine |
| **Color/Effects** | 7 nodes | Recolor, wear, masks |
| **Texture** | 3 nodes | Tile, scale, project |
| **Channel** | 2 nodes | Pack/unpack RGB |

---

## 🔧 Core Processing

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **PBR Extractor** | Marigold x2 | pbr_pipe | Extract all PBR maps |
| **PBR Adjuster** | Individual maps | Adjusted maps | Quick brightness/contrast |
| **AO Approximator** | Height (+normal) | AO map | Generate AO from height |

**Quick Tips:**
- Washed out? ↑ `gamma_appearance` (0.8-1.2)
- Better AO? Use height + normal + 24 samples

---

## 🔗 Pipeline System

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **PBR Combiner** | Maps or pipe | pbr_pipe | Bundle maps into pipe |
| **PBR Pipeline Adjuster** | pbr_pipe | pbr_pipe | Batch adjust all maps |
| **PBR Splitter** | pbr_pipe | Individual maps | Extract maps from pipe |
| **PBR Saver** | pbr_pipe | IMAGE batch | Save + preview all maps |
| **PBR Pipe Preview** | pbr_pipe | pbr_pipe + preview | Debug/check pipeline |
| **PBR Material Mixer** | 2x pbr_pipe | pbr_pipe | Blend materials |

**Quick Tips:**
- Clean workflow: Extractor → Combiner → Adjuster → Saver
- Debug: Add Pipe Preview anywhere
- Weathering: `ao_strength_roughness=0.3`

---

## 🗺️ Normal & Height

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **Normal Processor** | Lotus normal | Normal | Channel control + strength |
| **Normal Combiner** | 2x normals | Combined | Layer detail normals |
| **Normal to Depth** | Normal | Depth | Generate height from normal |
| **Height to Normal** | Height | Normal | Generate normal from height |
| **Normal Converter** | Normal | Normal | OpenGL ↔ DirectX |
| **Normal Validator** | Normal | Viz + String | Detect OGL vs DX format |

**Quick Tips:**
- OpenGL: `invert_green=True`
- DirectX: `invert_green=False`
- Best blend: `mode=reoriented`
- Sharp normals: `method=scharr`

---

## 🎨 Color & Effects

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **Color Ramp** | IMAGE | Colored | Map values to gradient |
| **Simple Recolor** | IMAGE | Colored | Quick 2-color gradient |
| **HSV Adjuster** | IMAGE | Adjusted | Hue/Sat/Value control |
| **Curvature Generator** | Normal/Height | Curvature | Detect edges/crevices |
| **Detail Blender** | Base + Detail | Blended | Add micro-detail |
| **Wear Generator** | Albedo + Curv/AO | Worn | Procedural weathering |
| **Gradient Map** | IMAGE | Mask | Value range selection |

**Quick Tips:**
- Interactive: Color Ramp (click/drag/double-click)
- Color variants: HSV hue_shift
- Realistic wear: Curv + Wear (edge_wear=0.7)
- Masks: Gradient Map → Color Ramp

---

## 🖼️ Texture Utilities

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **Texture Equalizer** | IMAGE | Equalized | Remove shadows/uneven lighting |
| **Square Maker** | IMAGE | Square | Crop or scale to square (9 positions) |
| **Smart Texture Resizer** | IMAGE | Optimized | Auto GPU-friendly resolution |
| **Texture Offset** | IMAGE | Offset + mask | Shift X/Y, rotate, wrap + edge mask |
| **Texture Tiler** | IMAGE | Tiled grid | NxN grid (normal or scaled to input) |
| **Seamless Tiling** | IMAGE | Seamless + mask | Make tileable + edge mask |
| **Texture Scaler** | IMAGE | Scaled | Resize smart (0.125-8x) |
| **Triplanar Projection** | IMAGE | Projected | Remove UV seams |

**Quick Tips:**
- Best quality: `method=blend_edges`
- Pixel art: `method=nearest`
- Photo quality: `method=bicubic` or `lanczos`

---

## 📦 Channel Tools

| Node | Input | Output | Key Use |
|------|-------|--------|---------|
| **Grayscale to Color** | Grayscale | RGB | Convert single → 3 channels |
| **Color to Grayscale** | RGB | Grayscale | Convert 3 → single channel |
| **Channel Packer (RGB)** | 3x grayscale | RGB packed | Save VRAM (ORM format) |
| **Channel Unpacker (RGB)** | RGB | 3x grayscale | Extract channels |
| **Channel Packer (ORMA)** | 4x grayscale | RGBA | ORM + Alpha channel |
| **Channel Packer (RMA)** | 3x grayscale | RGB | Roughness/Metal/AO format |
| **Channel Packer (RMAA)** | 4x grayscale | RGBA | RMA + Alpha channel |
| **Channel Unpacker (RMA)** | RGB | 3x grayscale | Extract RMA channels |

**Quick Tips:**
- **ORM** (Unity/Unreal): AO(R) + Roughness(G) + Metallic(B)
- **RMA** (Alternative): Roughness(R) + Metallic(G) + AO(B)
- **ORMA** (ORM + Alpha): AO(R) + Roughness(G) + Metallic(B) + Alpha(A)
- **RMAA** (RMA + Alpha): Roughness(R) + Metallic(G) + AO(B) + Alpha(A)
- Saves 66% texture memory with RGB packing, 75% with RGBA!
- Use Grayscale↔Color for format conversion

---

## 📊 Parameter Defaults (Copy-Paste Ready)

### PBR Extractor
```
albedo_source: "lighting"
gamma_albedo: 0.45
gamma_metal_rough: 2.2
gamma_lighting_ao: 0.45
```

### PBR Pipeline Adjuster (Realistic)
```
ao_strength_albedo: 1.0
ao_strength_roughness: 0.3
roughness_strength: 1.2
metallic_strength: 1.0
normal_strength: 1.0
albedo_saturation: 1.1
```

### AO Approximator (Quality)
```
radius: 12
strength: 1.2
samples: 24
contrast: 1.0
use_normal: True
```

### Wear Generator (Moderate)
```
wear_strength: 0.5
edge_wear: 0.7
dirt_strength: 0.3
```

### Height to Normal (Sharp)
```
method: "scharr"
strength: 1.5
```

### Seamless Tiling (Best)
```
method: "blend_edges"
blend_width: 0.1
```

---

## 🎯 Common Combinations

### Extract Complete PBR
```
Marigold x2 → PBR Extractor
Lotus x2 → Normal/Height Processor
All → PBR Combiner → PBR Saver
```

### Add Realistic Wear
```
Normal → Curvature Generator ─┐
AO ──────────────────────────┐│
Albedo ──> Wear Generator ───┴┘
```

### Channel Pack for Games
```
AO ──────┐
Roughness ┼─> Channel Packer → ORM.png
Metallic ─┘
```

### Stylized Material
```
Height → Gradient Map → Color Ramp → Albedo
→ HSV Adjuster → PBR Combiner
```

### Multi-Detail Normals
```
Base → Normal Combiner #1 → (+ detail)
     → Normal Combiner #2 → (+ micro)
     → Detail Blender → (+ surface)
```

---

## 🚨 Troubleshooting

| Problem | Solution | Node/Parameter |
|---------|----------|----------------|
| Washed out roughness/metallic | Adjust gamma | `gamma_metal_rough: 1.8-2.6` (default 2.2) |
| Inverted normals | Flip green channel | `invert_normal_green: True` |
| Flat normals | Increase strength | `normal_strength: 1.5` |
| Dark albedo | Check AO strength | `ao_strength_albedo < 1.0` |
| Muted colors | Increase saturation | `albedo_saturation: 1.3` |
| Visible seams | Use seamless tiling | Seamless Tiling Maker |
| Low quality AO | More samples | `samples: 32` |
| Slow Color Ramp | (Should be fast!) | Check GPU, 100x faster in v2.0 |
| Wrong format | Check engine | Normal Converter (DX↔GL) |
| Huge file size | Change format | PNG→JPG or pack channels |

---

## ⌨️ Keyboard Shortcuts (ComfyUI)

| Action | Shortcut |
|--------|----------|
| Add node | Double-click or Space |
| Search nodes | Type to filter |
| Connect nodes | Drag from output to input |
| Duplicate | Ctrl+C, Ctrl+V |
| Delete | Delete or Backspace |
| Queue prompt | Ctrl+Enter |
| Clear queue | Ctrl+Shift+Delete |

---

## 🎨 Color Ramp Controls

| Action | How |
|--------|-----|
| Add stop | Click empty gradient |
| Remove stop | Double-click marker |
| Move stop | Drag triangle |
| Change color | Click color swatch |
| Use preset | Select from dropdown |

---

## 📏 Recommended Resolutions

| Use Case | Resolution | Format |
|----------|------------|--------|
| Web/Mobile | 512x512 | JPG |
| Desktop Games | 1024x1024 or 2048x2048 | PNG |
| AAA Games | 2048x2048 or 4096x4096 | PNG+ORM |
| Archviz | 4096x4096+ | EXR |
| Film/VFX | 8192x8192+ | EXR 32-bit |
| 3D Print | 2048x2048 | PNG 16-bit |

---

## 💾 File Format Guide

| Format | Bit Depth | Compression | Use For |
|--------|-----------|-------------|---------|
| **PNG** | 8/16-bit | Lossless | General, games |
| **JPG** | 8-bit | Lossy | Web, mobile |
| **EXR** | 16/32-bit | Lossless | Film, precision |
| **TIFF** | 16-bit | Lossless | Professional |

**Recommendations:**
- Albedo: PNG or JPG
- Normals: PNG or EXR (16-bit+)
- Height: EXR (32-bit)
- Roughness/Metallic: PNG
- ORM Packed: PNG

---

## 🔢 Value Ranges

| Map Type | Range | Notes |
|----------|-------|-------|
| Albedo | 0.0-1.0 | sRGB, never pure black/white |
| Normal | 0.0-1.0 | Encoded from -1 to +1 |
| Roughness | 0.0-1.0 | 0=mirror, 1=matte |
| Metallic | 0.0 or 1.0 | Usually binary (0 or 1) |
| AO | 0.0-1.0 | 0=occluded, 1=exposed |
| Height | 0.0-1.0 | Relative, can be remapped |
| Curvature | 0.0-1.0 | 0=flat, 1=edge |

---

## 🎯 Performance Tips

1. **Use Pipeline:** Faster than individual connections
2. **Pack Channels:** Saves memory and load time
3. **Scale Smart:** Downscale early, upscale late
4. **Cache Results:** Save intermediate steps
5. **GPU Acceleration:** Most nodes use GPU
6. **Batch Process:** Queue multiple materials
7. **Preview Sparingly:** Use Pipe Preview only when needed
8. **EXR Only When Needed:** PNG is faster to save/load

---

## 🏆 Best Practices

**Workflow:**
1. Extract → Combine → Adjust → Preview → Save
2. Use Pipe Preview to debug
3. Save incremental versions
4. Name materials consistently
5. Organize by project in output folders

**Quality:**
1. Higher samples = better quality (slower)
2. EXR for normals/height
3. Use reference materials
4. Check in target engine
5. Test tile-ability

**Organization:**
1. Consistent naming: `material_map_###`
2. Group by project folder
3. Version control workflows
4. Document special settings
5. Save workflow JSON

---

## 📞 Quick Help

**Issue:** My normals look wrong!
→ Use Normal Converter (DX↔GL)

**Issue:** Maps are too dark!
→ Increase brightness or gamma

**Issue:** Texture doesn't tile!
→ Use Seamless Tiling Maker first

**Issue:** Running out of VRAM!
→ Pack channels (ORM format)

**Issue:** Color Ramp is slow!
→ v2.0 is 100x faster, update!

**Issue:** Need better AO!
→ Use height + normal, 24+ samples

**Issue:** Colors look wrong!
→ Try HSV Adjuster instead of brightness

**Issue:** Want weathered look!
→ Curvature → Wear Generator

---

**Print this page for desk reference!** 📄

For detailed info, see full [README.md](README.md) and [WORKFLOWS.md](WORKFLOWS.md).

