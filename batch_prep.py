"""
Batch Images and Prep
Same dynamic image inputs as Comfy core Batch Images (Autogrow: new slot per connection),
plus prep: normalize all images to a common size via scale or crop.
"""

import torch
from comfy_api.latest import io

# Lazy import so we don't depend on comfy path at import time
def _common_upscale(samples, width, height, upscale_method, crop):
    import comfy.utils
    return comfy.utils.common_upscale(samples, width, height, upscale_method, crop)


MODES = [
    "scale_to_custom",
    "crop_to_custom",
    "scale_to_smallest",
    "scale_to_largest",
    "crop_to_smallest",
]
UPSAMPLE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

MAX_RESOLUTION = 16384


def _flatten_images(images_list):
    """Turn list of (B, H, W, C) tensors into list of (1, H, W, C) with possibly different H, W."""
    out = []
    for t in images_list:
        if t is None:
            continue
        if isinstance(t, torch.Tensor):
            for i in range(t.shape[0]):
                out.append(t[i : i + 1])
        else:
            for x in t:
                if x is not None and isinstance(x, torch.Tensor):
                    for i in range(x.shape[0]):
                        out.append(x[i : i + 1])
    return out


def _get_target_size(mode, flat_list, custom_width, custom_height):
    """Compute (target_w, target_h) from mode and image list."""
    if not flat_list:
        return (custom_width, custom_height)
    h_list = [t.shape[1] for t in flat_list]
    w_list = [t.shape[2] for t in flat_list]
    min_h, max_h = min(h_list), max(h_list)
    min_w, max_w = min(w_list), max(w_list)
    if mode == "scale_to_custom" or mode == "crop_to_custom":
        return (custom_width, custom_height)
    if mode == "scale_to_smallest" or mode == "crop_to_smallest":
        return (min_w, min_h)
    if mode == "scale_to_largest":
        return (max_w, max_h)
    return (custom_width, custom_height)


def _scale_image(img, target_w, target_h, method):
    """Scale single image (1, H, W, C) to (target_w, target_h)."""
    x = img.movedim(-1, 1)
    x = _common_upscale(x, target_w, target_h, method, "disabled")
    return x.movedim(1, -1)


def _crop_or_scale_to_target(img, target_w, target_h, scale_method):
    """
    Crop-to behavior: always output (target_w, target_h).
    - If image is bigger than target: center-crop to target.
    - If image is smaller than target: scale up to target (so all batch items match size).
    """
    _, h, w, _ = img.shape
    if w >= target_w and h >= target_h:
        # Bigger or equal: center-crop
        x0 = (w - target_w) // 2
        y0 = (h - target_h) // 2
        return img[:, y0 : y0 + target_h, x0 : x0 + target_w, :]
    else:
        # Smaller in at least one dimension: scale up to target
        x = img.movedim(-1, 1)
        x = _common_upscale(x, target_w, target_h, scale_method, "disabled")
        return x.movedim(1, -1)


def _batch_images_prep(images_dict, mode, width, height, upscale_method):
    """Core logic: collect images from Autogrow dict, normalize size, return single batch."""
    if not images_dict:
        return torch.zeros(0, 512, 512, 3, dtype=torch.float32)
    values = [v for v in images_dict.values() if v is not None]
    if not values:
        return torch.zeros(0, 512, 512, 3, dtype=torch.float32)
    flat = _flatten_images(values)
    if not flat:
        return torch.zeros(0, 512, 512, 3, dtype=torch.float32)
    target_w, target_h = _get_target_size(mode, flat, width, height)
    target_w = max(1, target_w)
    target_h = max(1, target_h)
    max_channels = max(t.shape[-1] for t in flat)
    scale_modes = ("scale_to_custom", "scale_to_smallest", "scale_to_largest")
    results = []
    for single in flat:
        if mode in scale_modes:
            out = _scale_image(single, target_w, target_h, upscale_method)
        else:
            out = _crop_or_scale_to_target(single, target_w, target_h, upscale_method)
        if out.shape[-1] < max_channels:
            out = torch.nn.functional.pad(out, (0, max_channels - out.shape[-1]), mode="constant", value=1.0)
        results.append(out)
    return torch.cat(results, dim=0)


class BatchImagesPrep(io.ComfyNode):
    """
    Same as Comfy Batch Images (dynamically adds image inputs when you connect),
    plus prep: normalize all images to a common size (scale or crop).
    """

    @classmethod
    def define_schema(cls):
        autogrow_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"), prefix="image", min=2, max=50
        )
        return io.Schema(
            node_id="BatchImagesPrep",
            display_name="Batch Images and Prep",
            category="Texture Alchemist",
            search_aliases=["batch images prep", "batch and prep", "batch scale crop"],
            inputs=[
                io.Autogrow.Input("images", template=autogrow_template),
                io.Combo.Input(
                    "mode",
                    options=MODES,
                    default="scale_to_smallest",
                    tooltip="How to normalize sizes: scale/crop to custom, smallest, or largest dimensions",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=1,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Target width (used for scale_to_custom and crop_to_custom)",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=1,
                    max=MAX_RESOLUTION,
                    step=1,
                    tooltip="Target height (used for scale_to_custom and crop_to_custom)",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=UPSAMPLE_METHODS,
                    default="bilinear",
                    tooltip="Interpolation for scaling",
                ),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, mode, width, height, upscale_method) -> io.NodeOutput:
        # Autogrow passes images as dict { "image0": tensor, "image1": tensor, ... }
        if not isinstance(images, dict):
            images = dict(images) if hasattr(images, "items") else {}
        out = _batch_images_prep(images, mode, width, height, upscale_method)
        return io.NodeOutput(out)


# New API node is the class above; expose for registration.
# Legacy NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS are used by __init__
# when this package uses NODE_CLASS_MAPPINGS (we add BatchImagesPrep there).
NODE_CLASS_MAPPINGS = {
    "BatchImagesPrep": BatchImagesPrep,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchImagesPrep": "Batch Images and Prep",
}
