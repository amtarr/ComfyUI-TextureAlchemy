"""
TextureAlchemy
Transform images into professional PBR materials with AI-powered nodes.
Complete workflow suite for extracting, processing, and enhancing textures.
"""

import os

# Core modules
from .pbr_core import NODE_CLASS_MAPPINGS as CORE_CLASS_MAPPINGS
from .pbr_core import NODE_DISPLAY_NAME_MAPPINGS as CORE_DISPLAY_MAPPINGS
from .pbr_pipeline import NODE_CLASS_MAPPINGS as PIPELINE_CLASS_MAPPINGS
from .pbr_pipeline import NODE_DISPLAY_NAME_MAPPINGS as PIPELINE_DISPLAY_MAPPINGS
from .pbr_extractor_node import NODE_CLASS_MAPPINGS as EXTRACTOR_CLASS_MAPPINGS
from .pbr_extractor_node import NODE_DISPLAY_NAME_MAPPINGS as EXTRACTOR_DISPLAY_MAPPINGS

# Map and material utilities
from .map_utils import NODE_CLASS_MAPPINGS as MAP_CLASS_MAPPINGS
from .map_utils import NODE_DISPLAY_NAME_MAPPINGS as MAP_DISPLAY_MAPPINGS
from .normal_utils import NODE_CLASS_MAPPINGS as NORMAL_CLASS_MAPPINGS
from .normal_utils import NODE_DISPLAY_NAME_MAPPINGS as NORMAL_DISPLAY_MAPPINGS
from .height_advanced import NODE_CLASS_MAPPINGS as HEIGHT_ADV_CLASS_MAPPINGS
from .height_advanced import NODE_DISPLAY_NAME_MAPPINGS as HEIGHT_ADV_DISPLAY_MAPPINGS

# Color and effects
from .color_utils import NODE_CLASS_MAPPINGS as COLOR_CLASS_MAPPINGS
from .color_utils import NODE_DISPLAY_NAME_MAPPINGS as COLOR_DISPLAY_MAPPINGS
from .color_advanced import NODE_CLASS_MAPPINGS as COLOR_ADV_CLASS_MAPPINGS
from .color_advanced import NODE_DISPLAY_NAME_MAPPINGS as COLOR_ADV_DISPLAY_MAPPINGS
from .effect_utils import NODE_CLASS_MAPPINGS as EFFECT_CLASS_MAPPINGS
from .effect_utils import NODE_DISPLAY_NAME_MAPPINGS as EFFECT_DISPLAY_MAPPINGS

# Texture processing
from .texture_utils import NODE_CLASS_MAPPINGS as TEXTURE_CLASS_MAPPINGS
from .texture_utils import NODE_DISPLAY_NAME_MAPPINGS as TEXTURE_DISPLAY_MAPPINGS
from .channel_utils import NODE_CLASS_MAPPINGS as CHANNEL_CLASS_MAPPINGS
from .channel_utils import NODE_DISPLAY_NAME_MAPPINGS as CHANNEL_DISPLAY_MAPPINGS

# Advanced tools (NEW!)
from .blend_utils import NODE_CLASS_MAPPINGS as BLEND_CLASS_MAPPINGS
from .blend_utils import NODE_DISPLAY_NAME_MAPPINGS as BLEND_DISPLAY_MAPPINGS
from .mask_generators import NODE_CLASS_MAPPINGS as MASK_CLASS_MAPPINGS
from .mask_generators import NODE_DISPLAY_NAME_MAPPINGS as MASK_DISPLAY_MAPPINGS
from .noise_utils import NODE_CLASS_MAPPINGS as NOISE_CLASS_MAPPINGS
from .noise_utils import NODE_DISPLAY_NAME_MAPPINGS as NOISE_DISPLAY_MAPPINGS
from .detail_utils import NODE_CLASS_MAPPINGS as DETAIL_CLASS_MAPPINGS
from .detail_utils import NODE_DISPLAY_NAME_MAPPINGS as DETAIL_DISPLAY_MAPPINGS
from .analysis_utils import NODE_CLASS_MAPPINGS as ANALYSIS_CLASS_MAPPINGS
from .analysis_utils import NODE_DISPLAY_NAME_MAPPINGS as ANALYSIS_DISPLAY_MAPPINGS
from .material_advanced import NODE_CLASS_MAPPINGS as MATERIAL_ADV_CLASS_MAPPINGS
from .material_advanced import NODE_DISPLAY_NAME_MAPPINGS as MATERIAL_ADV_DISPLAY_MAPPINGS
from .filter_utils import NODE_CLASS_MAPPINGS as FILTER_CLASS_MAPPINGS
from .filter_utils import NODE_DISPLAY_NAME_MAPPINGS as FILTER_DISPLAY_MAPPINGS

# Lighting and relighting (NEW!)
from .relight_utils import NODE_CLASS_MAPPINGS as RELIGHT_CLASS_MAPPINGS
from .relight_utils import NODE_DISPLAY_NAME_MAPPINGS as RELIGHT_DISPLAY_MAPPINGS
from .environment_utils import NODE_CLASS_MAPPINGS as ENVIRONMENT_CLASS_MAPPINGS
from .environment_utils import NODE_DISPLAY_NAME_MAPPINGS as ENVIRONMENT_DISPLAY_MAPPINGS

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    # Core
    **CORE_CLASS_MAPPINGS,
    **PIPELINE_CLASS_MAPPINGS,
    **EXTRACTOR_CLASS_MAPPINGS,
    # Maps
    **MAP_CLASS_MAPPINGS,
    **NORMAL_CLASS_MAPPINGS,
    **HEIGHT_ADV_CLASS_MAPPINGS,
    # Color & Effects
    **COLOR_CLASS_MAPPINGS,
    **COLOR_ADV_CLASS_MAPPINGS,
    **EFFECT_CLASS_MAPPINGS,
    # Texture
    **TEXTURE_CLASS_MAPPINGS,
    **CHANNEL_CLASS_MAPPINGS,
    # Advanced (NEW!)
    **BLEND_CLASS_MAPPINGS,
    **MASK_CLASS_MAPPINGS,
    **NOISE_CLASS_MAPPINGS,
    **DETAIL_CLASS_MAPPINGS,
    **ANALYSIS_CLASS_MAPPINGS,
    **MATERIAL_ADV_CLASS_MAPPINGS,
    **FILTER_CLASS_MAPPINGS,
    # Lighting
    **RELIGHT_CLASS_MAPPINGS,
    **ENVIRONMENT_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Core
    **CORE_DISPLAY_MAPPINGS,
    **PIPELINE_DISPLAY_MAPPINGS,
    **EXTRACTOR_DISPLAY_MAPPINGS,
    # Maps
    **MAP_DISPLAY_MAPPINGS,
    **NORMAL_DISPLAY_MAPPINGS,
    **HEIGHT_ADV_DISPLAY_MAPPINGS,
    # Color & Effects
    **COLOR_DISPLAY_MAPPINGS,
    **COLOR_ADV_DISPLAY_MAPPINGS,
    **EFFECT_DISPLAY_MAPPINGS,
    # Texture
    **TEXTURE_DISPLAY_MAPPINGS,
    **CHANNEL_DISPLAY_MAPPINGS,
    # Advanced (NEW!)
    **BLEND_DISPLAY_MAPPINGS,
    **MASK_DISPLAY_MAPPINGS,
    **NOISE_DISPLAY_MAPPINGS,
    **DETAIL_DISPLAY_MAPPINGS,
    **ANALYSIS_DISPLAY_MAPPINGS,
    **MATERIAL_ADV_DISPLAY_MAPPINGS,
    **FILTER_DISPLAY_MAPPINGS,
    # Lighting
    **RELIGHT_DISPLAY_MAPPINGS,
    **ENVIRONMENT_DISPLAY_MAPPINGS,
}

# Tell ComfyUI where to find our web/JS files
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

