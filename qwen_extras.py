"""
Texture Alchemy- Qwen Image Edit Plus (Batch) – same behaviour as the built-in
Text Encode Qwen Image Edit Plus but accepts a single IMAGE batch and chains
each image in the batch.
"""
import math

import node_helpers
import comfy.utils


class TextEncodeQwenImageEditPlusBatchNode:
    """
    Same as Text Encode Qwen Image Edit Plus but takes one IMAGE batch and
    chains every image in the batch (Picture 1, Picture 2, ...) into one
    conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "vae": ("VAE", {"default": None}),
                "images": ("IMAGE", {"default": None}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_batch"
    CATEGORY = "Conditioning"

    def encode_batch(self, clip, prompt, vae=None, images=None):
        ref_latents = []
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        if images is not None:
            batch_size = images.shape[0]
            for i in range(batch_size):
                image = images[i : i + 1]
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditPlusBatchNode": TextEncodeQwenImageEditPlusBatchNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditPlusBatchNode": "Text Encode Qwen Image Edit Plus (Batch)",
}
