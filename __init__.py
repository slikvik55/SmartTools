#
# __init__.py
#
# This file makes the folder a Python package and tells ComfyUI
# what nodes are available to load from this package.
#

from .SmartResizer import (
    NODE_CLASS_MAPPINGS as SMART_RESIZER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_RESIZER_DISPLAY_MAPPINGS,
)
from .SmartImage import (
    NODE_CLASS_MAPPINGS as SMART_IMAGE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_IMAGE_DISPLAY_MAPPINGS,
)

# Merge both nodes' mappings into a single dict for ComfyUI
NODE_CLASS_MAPPINGS = {
    **SMART_RESIZER_CLASS_MAPPINGS,
    **SMART_IMAGE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SMART_RESIZER_DISPLAY_MAPPINGS,
    **SMART_IMAGE_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
