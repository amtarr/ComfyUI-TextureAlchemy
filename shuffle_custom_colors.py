"""
Shuffle Custom Colors
One image input; Autogrow STRING inputs (color0, color1, …) for hex codes.
channel_index selects which color to use for the mask (0 = first, 1 = second, …).
"""

import math
import torch
from comfy_api.latest import io


def _parse_one_hex(hex_str, device, dtype):
    """Parse a single hex code to (r,g,b) tensor, or None if invalid."""
    if hex_str is None or not str(hex_str).strip():
        return None
    part = str(hex_str).strip().lstrip("#")
    if len(part) >= 6:
        try:
            r = int(part[0:2], 16) / 255.0
            g = int(part[2:4], 16) / 255.0
            b = int(part[4:6], 16) / 255.0
            return torch.tensor([r, g, b], device=device, dtype=dtype)
        except ValueError:
            pass
    return None


def _rgb_to_hsv(rgb):
    """rgb: (..., 3). Return (..., 3) HSV in 0-1."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    max_c = torch.max(torch.max(r, g), b)
    min_c = torch.min(torch.min(r, g), b)
    diff = max_c - min_c
    h = torch.zeros_like(max_c)
    mask = diff != 0
    h = torch.where((max_c == r) & mask, ((g - b) / (diff + 1e-7)) % 6.0, h)
    h = torch.where((max_c == g) & mask, ((b - r) / (diff + 1e-7)) + 2.0, h)
    h = torch.where((max_c == b) & mask, ((r - g) / (diff + 1e-7)) + 4.0, h)
    h = h / 6.0
    s = torch.where(max_c != 0, diff / (max_c + 1e-7), torch.zeros_like(max_c))
    v = max_c
    return torch.stack([h, s, v], dim=-1)


def _build_mask(image, target_rgb, tolerance, feather, mode, device, dtype):
    """Build a single-channel mask (B,H,W) from image and target color."""
    target = target_rgb.view(1, 1, 1, 3).to(device=device, dtype=dtype)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    image = image[:, :, :, :3]

    if mode == "rgb":
        diff = image - target
        distance = torch.sqrt((diff * diff).sum(dim=3, keepdim=True))
        distance = distance / math.sqrt(3.0)
    elif mode == "hsv":
        hsv_image = _rgb_to_hsv(image)
        hsv_target = _rgb_to_hsv(target)
        h_diff = torch.abs(hsv_image[:, :, :, 0:1] - hsv_target[:, :, :, 0:1])
        h_diff = torch.min(h_diff, 1.0 - h_diff) * 2.0
        s_diff = torch.abs(hsv_image[:, :, :, 1:2] - hsv_target[:, :, :, 1:2])
        v_diff = torch.abs(hsv_image[:, :, :, 2:3] - hsv_target[:, :, :, 2:3])
        distance = torch.sqrt(h_diff * h_diff + s_diff * s_diff + v_diff * v_diff)
        distance = distance / math.sqrt(3.0)
    else:
        lum_image = 0.299 * image[:, :, :, 0:1] + 0.587 * image[:, :, :, 1:2] + 0.114 * image[:, :, :, 2:3]
        lum_target = 0.299 * target_rgb[0].item() + 0.587 * target_rgb[1].item() + 0.114 * target_rgb[2].item()
        distance = torch.abs(lum_image - lum_target)

    mask = 1.0 - torch.clamp(distance / (tolerance + 1e-7), 0.0, 1.0)
    if feather > 0.0:
        mask = torch.where(
            mask > (1.0 - feather),
            mask,
            torch.where(
                mask < feather,
                torch.zeros_like(mask),
                (mask - feather) / (1.0 - 2.0 * feather),
            ),
        )
    mask = torch.clamp(mask.squeeze(-1), 0.0, 1.0)
    return mask


class ShuffleCustomColors(io.ComfyNode):
    """
    One image input; color0, color1, … are Autogrow STRING inputs (hex codes).
    channel_index selects which color to use for the mask (0 = first, 1 = second, …).
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.String.Input("color", default="#FF0000", tooltip="Hex color, e.g. #FF0000"),
            prefix="color",
            min=1,
            max=50,
        )
        return io.Schema(
            node_id="ShuffleCustomColors",
            display_name="Shuffle Custom Colors",
            category="Texture Alchemist/Masks",
            search_aliases=["shuffle custom colors", "color mask", "custom color mask"],
            inputs=[
                io.Image.Input("image", tooltip="Image to sample for the mask."),
                io.Autogrow.Input("colors", template=autogrow_template),
                io.Int.Input(
                    "channel_index",
                    default=0,
                    min=0,
                    max=99,
                    force_input=True,
                    tooltip="Which color to use for the mask: 0 = first, 1 = second, … (connectable e.g. from LLM).",
                ),
                io.Float.Input(
                    "tolerance",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Color matching tolerance.",
                ),
                io.Float.Input(
                    "feather",
                    default=0.1,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Edge softness.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["rgb", "hsv", "luminance"],
                    default="rgb",
                    tooltip="Color distance mode.",
                ),
            ],
            outputs=[io.Mask.Output()],
        )

    @classmethod
    def execute(
        cls,
        image,
        colors,
        channel_index,
        tolerance,
        feather,
        mode,
    ) -> io.NodeOutput:
        device = image.device
        dtype = image.dtype
        color_list = []

        # Colors from Autogrow string slots (color0, color1, …) — each is a hex string
        if isinstance(colors, dict):
            for k in sorted(colors.keys(), key=lambda x: (len(x), x)):
                v = colors[k]
                parsed = _parse_one_hex(v, device, dtype)
                if parsed is not None:
                    color_list.append(parsed)

        if not color_list:
            # No colors: output empty mask
            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            mask = torch.zeros((b, h, w), device=device, dtype=dtype)
            return io.NodeOutput(mask)

        idx = int(channel_index) if not isinstance(channel_index, (list, tuple)) else int(channel_index[0])
        idx = max(0, min(len(color_list) - 1, idx))
        target_rgb = color_list[idx]

        mask = _build_mask(image, target_rgb, tolerance, feather, mode, device, dtype)
        return io.NodeOutput(mask)


NODE_CLASS_MAPPINGS = {
    "ShuffleCustomColors": ShuffleCustomColors,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShuffleCustomColors": "Shuffle Custom Colors",
}
