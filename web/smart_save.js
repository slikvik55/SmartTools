import { app } from "/scripts/app.js";

const STYLE_ID = "smart-tools-smart-save-btn-styles";

function ensureSmartSaveButtonStyles() {
    if (document.getElementById(STYLE_ID)) {
        return;
    }
    const style = document.createElement("style");
    style.id = STYLE_ID;
    /* Matches ComfyUI Load Image "choose file to upload" control: flat bar, sharp corners, thin border */
    style.textContent = `
        button.comfy-smart-save-btn {
            display: block;
            width: 100%;
            box-sizing: border-box;
            margin: 6px 0 0 0;
            padding: 7px 10px;
            border-radius: 0;
            border: 1px solid rgba(255, 255, 255, 0.28);
            background: rgba(55, 55, 55, 1);
            color: rgba(255, 255, 255, 0.88);
            font-size: 12px;
            font-family: inherit;
            font-weight: normal;
            text-align: center;
            text-transform: lowercase;
            letter-spacing: 0.01em;
            cursor: pointer;
            line-height: 1.35;
        }
        button.comfy-smart-save-btn:hover {
            background: rgba(62, 62, 62, 1);
            border-color: rgba(255, 255, 255, 0.38);
        }
        button.comfy-smart-save-btn:active {
            background: rgba(48, 48, 48, 1);
        }
    `;
    document.head.appendChild(style);
}

/**
 * Adds a "save image" button to SmartSave nodes (styled like Load Image upload).
 * POSTs to /smart_tools/save_image to write cached tensors to the output folder.
 */
app.registerExtension({
    name: "SmartTools.SmartSave",
    async setup() {
        ensureSmartSaveButtonStyles();
    },
    beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "SmartSave") {
            return;
        }
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated ? onCreated.apply(this, arguments) : undefined;
            ensureSmartSaveButtonStyles();
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "comfy-smart-save-btn";
            btn.textContent = "save image";
            btn.onclick = async () => {
                const uid = String(this.id);
                try {
                    const res = await fetch(
                        new URL("/smart_tools/save_image", window.location.origin),
                        {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ unique_id: uid }),
                        }
                    );
                    let data = {};
                    try {
                        data = await res.json();
                    } catch {
                        /* ignore */
                    }
                    if (!res.ok) {
                        throw new Error(data.error || res.statusText || "Request failed");
                    }
                    const toast = _app?.extensionManager?.toast;
                    if (toast?.add) {
                        toast.add({
                            severity: "success",
                            summary: "Smart Save",
                            detail: "Saved to output folder.",
                            life: 3000,
                        });
                    } else {
                        alert("Saved to output folder.");
                    }
                } catch (e) {
                    const msg = e?.message || String(e);
                    const toast = _app?.extensionManager?.toast;
                    if (toast?.add) {
                        toast.add({
                            severity: "error",
                            summary: "Smart Save",
                            detail: msg,
                            life: 6000,
                        });
                    } else {
                        alert("Smart Save: " + msg);
                    }
                }
            };
            this.addDOMWidget("smart_save_btn", "button", btn, () => {});
            return r;
        };
    },
});
