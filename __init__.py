#
# __init__.py
#
# This file makes the folder a Python package and tells ComfyUI
# what nodes are available to load from this package.
#

import logging
import os

from aiohttp import web

from .SmartResizer import (
    NODE_CLASS_MAPPINGS as SMART_RESIZER_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_RESIZER_DISPLAY_MAPPINGS,
)
from .SmartSave import (
    NODE_CLASS_MAPPINGS as SMART_SAVE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SMART_SAVE_DISPLAY_MAPPINGS,
    smart_save_write_output,
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


def _register_smart_save_route():
    try:
        from server import PromptServer
    except ImportError:
        logging.warning("SmartTools: could not import server; SmartSave HTTP route disabled.")
        return

    @PromptServer.instance.routes.post("/smart_tools/save_image")
    async def smart_save_image_route(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)
        uid = body.get("unique_id")
        if uid is None:
            return web.json_response({"ok": False, "error": "missing unique_id"}, status=400)
        results, err = smart_save_write_output(str(uid))
        if err:
            return web.json_response({"ok": False, "error": err}, status=400)
        return web.json_response({"ok": True, "images": results})


try:
    _register_smart_save_route()
except Exception as e:
    logging.warning("SmartTools: SmartSave route registration failed: %s", e)
