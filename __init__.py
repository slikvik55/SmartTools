#
# __init__.py
#
# This file makes the folder a Python package and tells ComfyUI
# what nodes are available to load from this package.
#

import os

from .SmartResizer import (
    NODE_CLASS_MAPPINGS as SMART_RESIZER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_RESIZER_DISPLAY_MAPPINGS,
)
from .SmartSave import (
    NODE_CLASS_MAPPINGS as SMART_SAVE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_SAVE_DISPLAY_MAPPINGS,
)

# Merge all nodes' mappings into a single dict for ComfyUI
NODE_CLASS_MAPPINGS = {
    **SMART_RESIZER_CLASS_MAPPINGS,
    **SMART_SAVE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SMART_RESIZER_DISPLAY_MAPPINGS,
    **SMART_SAVE_DISPLAY_MAPPINGS,
}

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
