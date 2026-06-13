import { app } from "../../scripts/app.js";

const HANDLE = 10;
const MIN_SIZE = 10;

const HANDLES = [
    { id: "nw", cx: 0, cy: 0 },
    { id: "n", cx: 0.5, cy: 0 },
    { id: "ne", cx: 1, cy: 0 },
    { id: "e", cx: 1, cy: 0.5 },
    { id: "se", cx: 1, cy: 1 },
    { id: "s", cx: 0.5, cy: 1 },
    { id: "sw", cx: 0, cy: 1 },
    { id: "w", cx: 0, cy: 0.5 },
];

function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
}

/** Square resize: corners use min(w,h); edges keep opposite edge fixed and center on perpendicular axis. */
function squareResizeFromDrag(mode, s, dx, dy) {
    const left = s.x,
        top = s.y,
        right = s.x + s.w,
        bottom = s.y + s.h;
    const cx = s.x + s.w / 2,
        cy = s.y + s.h / 2;

    let nl = left,
        nt = top,
        nr = right,
        nb = bottom;
    if (mode.includes("w")) nl = left + dx;
    if (mode.includes("e")) nr = right + dx;
    if (mode.includes("n")) nt = top + dy;
    if (mode.includes("s")) nb = bottom + dy;

    if (mode === "n") {
        const side = Math.max(MIN_SIZE, bottom - nt);
        return { x: cx - side / 2, y: bottom - side, w: side, h: side };
    }
    if (mode === "s") {
        const side = Math.max(MIN_SIZE, nb - top);
        return { x: cx - side / 2, y: top, w: side, h: side };
    }
    if (mode === "e") {
        const side = Math.max(MIN_SIZE, nr - left);
        return { x: left, y: cy - side / 2, w: side, h: side };
    }
    if (mode === "w") {
        const side = Math.max(MIN_SIZE, right - nl);
        return { x: nr - side, y: cy - side / 2, w: side, h: side };
    }

    let w = nr - nl,
        h = nb - nt;
    if (w < 0) {
        const t = nl;
        nl = nr;
        nr = t;
        w = -w;
    }
    if (h < 0) {
        const t = nt;
        nt = nb;
        nb = t;
        h = -h;
    }
    const side = Math.max(MIN_SIZE, Math.min(w, h));
    if (mode === "se") return { x: nl, y: nt, w: side, h: side };
    if (mode === "nw") return { x: nr - side, y: nb - side, w: side, h: side };
    if (mode === "ne") return { x: nl, y: nb - side, w: side, h: side };
    if (mode === "sw") return { x: nr - side, y: nt, w: side, h: side };
    return { x: nl, y: nt, w: side, h: side };
}

function parseImageWidget(widget) {
    const v = widget?.value;
    if (v && typeof v === "object") {
        return {
            filename: v.filename ?? v.name ?? "",
            subfolder: v.subfolder ?? "",
            type: v.type ?? "input",
        };
    }
    return { filename: String(v ?? ""), subfolder: "", type: "input" };
}

function buildViewUrl(meta) {
    if (!meta.filename) return null;
    return `/view?filename=${encodeURIComponent(meta.filename)}&type=${encodeURIComponent(
        meta.type
    )}&subfolder=${encodeURIComponent(meta.subfolder || "")}&t=${Date.now()}`;
}

function hideIntWidgets(node) {
    for (const name of ["crop_x", "crop_y", "crop_w", "crop_h"]) {
        const w = node.widgets?.find((x) => x.name === name);
        if (w) {
            w.type = "hidden";
            w.computeSize = () => [0, -4];
        }
    }
}

function getCropWidgets(node) {
    return {
        x: node.widgets?.find((w) => w.name === "crop_x"),
        y: node.widgets?.find((w) => w.name === "crop_y"),
        w: node.widgets?.find((w) => w.name === "crop_w"),
        h: node.widgets?.find((w) => w.name === "crop_h"),
    };
}

function setCropFromSelection(node, sel) {
    const c = getCropWidgets(node);
    if (c.x) c.x.value = Math.round(sel.x);
    if (c.y) c.y.value = Math.round(sel.y);
    if (c.w) c.w.value = Math.round(sel.w);
    if (c.h) c.h.value = Math.round(sel.h);
    app.graph.setDirtyCanvas(true, true);
}

function readSelectionFromWidgets(node, imgW, imgH) {
    const c = getCropWidgets(node);
    if (!c.w || !c.h) return null;
    const wv = c.w.value | 0;
    const hv = c.h.value | 0;
    if (wv <= 0 || hv <= 0) {
        return { x: 0, y: 0, w: imgW, h: imgH };
    }
    return {
        x: clamp(c.x.value | 0, 0, imgW - MIN_SIZE),
        y: clamp(c.y.value | 0, 0, imgH - MIN_SIZE),
        w: Math.max(MIN_SIZE, wv),
        h: Math.max(MIN_SIZE, hv),
    };
}

function openCropModal(node, imageWidget) {
    const meta = parseImageWidget(imageWidget);
    const url = buildViewUrl(meta);
    if (!url) {
        alert("Select an image first.");
        return;
    }

    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed; inset: 0; z-index: 100000;
        background: rgba(0,0,0,0.65);
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        font-family: system-ui, Segoe UI, Roboto, Arial, sans-serif;
    `;

    const panel = document.createElement("div");
    panel.style.cssText = `
        background: #1a1a1e;
        border: 1px solid rgba(127,127,127,.35);
        border-radius: 12px;
        width: min(96vw, 1100px);
        height: min(90vh, 820px);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    `;

    const header = document.createElement("div");
    header.style.cssText =
        "display:flex;gap:10px;align-items:center;padding:12px 14px;border-bottom:1px solid rgba(127,127,127,.25);flex-wrap:wrap;";
    header.innerHTML = `
        <span style="opacity:.85;font-size:13px;margin-right:8px;">Crop region (image pixels)</span>
    `;

    const btnFit = document.createElement("button");
    btnFit.textContent = "Fit selection";
    const btnReset = document.createElement("button");
    btnReset.textContent = "Reset selection";
    const btnApply = document.createElement("button");
    btnApply.textContent = "Apply";
    const btnCancel = document.createElement("button");
    btnCancel.textContent = "Cancel";

    [btnFit, btnReset, btnApply, btnCancel].forEach((b) => {
        b.style.cssText =
            "border:1px solid rgba(127,127,127,.35);background:rgba(127,127,127,.15);padding:8px 12px;border-radius:8px;cursor:pointer;color:#eee;";
    });
    btnFit.disabled = true;
    btnReset.disabled = true;
    btnApply.disabled = true;

    const spacer = document.createElement("div");
    spacer.style.flex = "1";
    const lblSquare = document.createElement("label");
    lblSquare.style.cssText =
        "display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#ddd;white-space:nowrap;";
    const chkSquare = document.createElement("input");
    chkSquare.type = "checkbox";
    chkSquare.title = "Keep crop square while resizing from handles";
    lblSquare.appendChild(chkSquare);
    lblSquare.appendChild(document.createTextNode("Square"));

    header.appendChild(btnFit);
    header.appendChild(btnReset);
    header.appendChild(lblSquare);
    header.appendChild(spacer);
    header.appendChild(btnApply);
    header.appendChild(btnCancel);

    const hint = document.createElement("span");
    hint.style.cssText = "opacity:.75;font-size:12px;width:100%;padding:0 14px 8px;";
    hint.textContent =
        "Drag inside to move; handles to resize; enable Square for 1:1 resize; arrows to nudge (Shift = 10 px).";
    panel.appendChild(header);
    panel.appendChild(hint);

    const stage = document.createElement("div");
    stage.style.cssText =
        "position:relative;flex:1;min-height:0;background:#111;padding:14px;display:grid;place-items:center;overflow:hidden;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "max-width:100%;max-height:100%;border-radius:10px;";
    stage.appendChild(canvas);
    panel.appendChild(stage);

    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const ctx = canvas.getContext("2d", { alpha: false });

    let img = null;
    let imgW = 0,
        imgH = 0;
    let scale = 1,
        offsetX = 0,
        offsetY = 0,
        drawW = 0,
        drawH = 0;

    /** @type {{ x: number, y: number, w: number, h: number }} */
    let sel = { x: 0, y: 0, w: 100, h: 100 };

    let dragging = false;
    let dragMode = null;
    let start = null;

    function normalizeSel() {
        if (!img) return;
        if (sel.w < 0) {
            sel.x += sel.w;
            sel.w = -sel.w;
        }
        if (sel.h < 0) {
            sel.y += sel.h;
            sel.h = -sel.h;
        }
        sel.x = clamp(sel.x, 0, imgW - MIN_SIZE);
        sel.y = clamp(sel.y, 0, imgH - MIN_SIZE);
        sel.w = clamp(sel.w, MIN_SIZE, imgW - sel.x);
        sel.h = clamp(sel.h, MIN_SIZE, imgH - sel.y);
    }

    function normalizeSquareSel() {
        let side = Math.max(MIN_SIZE, Math.min(sel.w, sel.h));
        sel.x = clamp(sel.x, 0, imgW - MIN_SIZE);
        sel.y = clamp(sel.y, 0, imgH - MIN_SIZE);
        side = Math.min(side, imgW - sel.x, imgH - sel.y);
        side = Math.max(MIN_SIZE, side);
        sel.x = clamp(sel.x, 0, imgW - side);
        sel.y = clamp(sel.y, 0, imgH - side);
        sel.w = sel.h = Math.min(side, imgW - sel.x, imgH - sel.y);
    }

    function computeFit() {
        const cw = canvas.width,
            ch = canvas.height;
        const s = Math.min(cw / imgW, ch / imgH);
        scale = s;
        drawW = imgW * s;
        drawH = imgH * s;
        offsetX = Math.floor((cw - drawW) / 2);
        offsetY = Math.floor((ch - drawH) / 2);
    }

    function imgToCanvas(p) {
        return { x: offsetX + p.x * scale, y: offsetY + p.y * scale };
    }

    function canvasToImg(p) {
        return { x: (p.x - offsetX) / scale, y: (p.y - offsetY) / scale };
    }

    function getPointerCanvas(e) {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);
        return { x, y };
    }

    function selectionInCanvas() {
        const tl = imgToCanvas({ x: sel.x, y: sel.y });
        return { x: tl.x, y: tl.y, w: sel.w * scale, h: sel.h * scale };
    }

    function hitTestHandle(pc) {
        const r = selectionInCanvas();
        for (const h of HANDLES) {
            const hx = r.x + r.w * h.cx;
            const hy = r.y + r.h * h.cy;
            if (Math.abs(pc.x - hx) <= HANDLE && Math.abs(pc.y - hy) <= HANDLE) return h.id;
        }
        return null;
    }

    function hitTestMove(pc) {
        const r = selectionInCanvas();
        return pc.x >= r.x && pc.x <= r.x + r.w && pc.y >= r.y && pc.y <= r.y + r.h;
    }

    function draw() {
        const cw = canvas.width,
            ch = canvas.height;
        ctx.fillStyle = "#111";
        ctx.fillRect(0, 0, cw, ch);

        if (!img) return;

        ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

        const r = selectionInCanvas();
        ctx.save();
        ctx.fillStyle = "rgba(0,0,0,.45)";
        ctx.beginPath();
        ctx.rect(offsetX, offsetY, drawW, drawH);
        ctx.rect(r.x, r.y, r.w, r.h);
        ctx.fill("evenodd");
        ctx.restore();

        ctx.save();
        ctx.lineWidth = Math.max(2, Math.floor(2 * (canvas.width / 1200)));
        ctx.strokeStyle = "rgba(255,255,255,.95)";
        ctx.strokeRect(r.x, r.y, r.w, r.h);

        ctx.fillStyle = "rgba(255,255,255,.95)";
        for (const h of HANDLES) {
            const hx = r.x + r.w * h.cx;
            const hy = r.y + r.h * h.cy;
            ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
        }

        ctx.fillStyle = "rgba(255,255,255,.9)";
        ctx.font =
            "14px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
        const info = `${Math.round(sel.w)}×${Math.round(sel.h)} @ (${Math.round(sel.x)},${Math.round(sel.y)})`;
        ctx.fillText(info, 14, 22);
        ctx.restore();
    }

    function resizeCanvasToStage() {
        const r = stage.getBoundingClientRect();
        const maxW = Math.max(400, Math.floor(r.width - 28));
        const maxH = Math.max(300, Math.floor(r.height - 28));
        const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
        canvas.width = Math.floor(maxW * dpr);
        canvas.height = Math.floor(maxH * dpr);
        canvas.style.width = maxW + "px";
        canvas.style.height = maxH + "px";
        if (img) computeFit();
        draw();
    }

    function resetSelectionMargin() {
        if (!img) return;
        const margin = 0.08;
        sel.x = Math.round(imgW * margin);
        sel.y = Math.round(imgH * margin);
        sel.w = Math.round(imgW * (1 - 2 * margin));
        sel.h = Math.round(imgH * (1 - 2 * margin));
        normalizeSel();
        draw();
    }

    function fitFullImage() {
        if (!img) return;
        sel.x = 0;
        sel.y = 0;
        sel.w = imgW;
        sel.h = imgH;
        normalizeSel();
        draw();
    }

    function pointerDown(e) {
        if (!img) return;
        canvas.setPointerCapture?.(e.pointerId);
        const pc = getPointerCanvas(e);
        const h = hitTestHandle(pc);
        if (h) {
            dragging = true;
            dragMode = h;
        } else if (hitTestMove(pc)) {
            dragging = true;
            dragMode = "move";
        } else {
            const pi = canvasToImg(pc);
            sel.x = clamp(pi.x, 0, imgW);
            sel.y = clamp(pi.y, 0, imgH);
            sel.w = 1;
            sel.h = 1;
            dragging = true;
            dragMode = "se";
        }
        start = { pc, sel: { ...sel } };
        draw();
    }

    function pointerMove(e) {
        if (!img) return;
        const pc = getPointerCanvas(e);
        if (!dragging) {
            const h = hitTestHandle(pc);
            canvas.style.cursor = h ? "nwse-resize" : hitTestMove(pc) ? "move" : "crosshair";
            return;
        }

        const pi0 = canvasToImg(start.pc);
        const pi = canvasToImg(pc);
        const dx = pi.x - pi0.x;
        const dy = pi.y - pi0.y;

        sel = { ...start.sel };

        if (dragMode === "move") {
            sel.x += dx;
            sel.y += dy;
            normalizeSel();
        } else if (chkSquare.checked) {
            sel = squareResizeFromDrag(dragMode, start.sel, dx, dy);
            normalizeSquareSel();
        } else {
            const left = start.sel.x;
            const top = start.sel.y;
            const right = start.sel.x + start.sel.w;
            const bottom = start.sel.y + start.sel.h;

            let nl = left,
                nt = top,
                nr = right,
                nb = bottom;

            if (String(dragMode).includes("w")) nl = left + dx;
            if (String(dragMode).includes("e")) nr = right + dx;
            if (String(dragMode).includes("n")) nt = top + dy;
            if (String(dragMode).includes("s")) nb = bottom + dy;

            sel.x = nl;
            sel.y = nt;
            sel.w = nr - nl;
            sel.h = nb - nt;
            normalizeSel();
        }
        draw();
    }

    function pointerUp() {
        if (!img) return;
        dragging = false;
        dragMode = null;
        start = null;
        draw();
    }

    function onKeyDown(e) {
        if (!img) return;
        const step = e.shiftKey ? 10 : 1;
        let moved = false;
        if (e.key === "ArrowLeft") {
            sel.x -= step;
            moved = true;
        }
        if (e.key === "ArrowRight") {
            sel.x += step;
            moved = true;
        }
        if (e.key === "ArrowUp") {
            sel.y -= step;
            moved = true;
        }
        if (e.key === "ArrowDown") {
            sel.y += step;
            moved = true;
        }
        if (moved) {
            e.preventDefault();
            normalizeSel();
            draw();
        }
    }

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);

    const ro = new ResizeObserver(() => resizeCanvasToStage());
    ro.observe(stage);

    window.addEventListener("keydown", onKeyDown, { passive: false });

    btnFit.addEventListener("click", fitFullImage);
    btnReset.addEventListener("click", resetSelectionMargin);
    btnApply.addEventListener("click", () => {
        normalizeSel();
        setCropFromSelection(node, sel);
        cleanup();
    });
    btnCancel.addEventListener("click", cleanup);
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) cleanup();
    });

    function cleanup() {
        window.removeEventListener("keydown", onKeyDown);
        ro.disconnect();
        overlay.remove();
    }

    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
        img = im;
        imgW = im.naturalWidth;
        imgH = im.naturalHeight;

        const fromWidgets = readSelectionFromWidgets(node, imgW, imgH);
        if (fromWidgets) {
            sel = fromWidgets;
            normalizeSel();
        } else {
            fitFullImage();
        }

        resizeCanvasToStage();
        setButtons(true);
        draw();
    };
    im.onerror = () => {
        alert("Could not load image preview.");
        cleanup();
    };
    function setButtons(enabled) {
        btnFit.disabled = !enabled;
        btnReset.disabled = !enabled;
        btnApply.disabled = !enabled;
    }

    im.src = url;
}

function attachImageSync(node) {
    const imageWidget = node.widgets?.find((w) => w.name === "image");
    if (!imageWidget) return;

    const onImageMaybeChanged = () => {
        const url = buildViewUrl(parseImageWidget(imageWidget));
        if (!url) return;
        const probe = new Image();
        probe.crossOrigin = "anonymous";
        probe.onload = () => {
            const iw = probe.naturalWidth;
            const ih = probe.naturalHeight;
            setCropFromSelection(node, { x: 0, y: 0, w: iw, h: ih });
        };
        probe.src = url;
    };

    const orig = imageWidget.callback;
    imageWidget.callback = function () {
        orig?.apply(this, arguments);
        onImageMaybeChanged();
    };

    queueMicrotask(onImageMaybeChanged);
}

app.registerExtension({
    name: "SmartTools.SmartLoad",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "SmartLoad") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;

            hideIntWidgets(node);

            const cropWidget = node.addWidget(
                "button",
                "smartload_crop_open",
                "crop",
                () => {
                    const iw = node.widgets?.find((w) => w.name === "image");
                    if (iw) openCropModal(node, iw);
                },
                {
                    serialize: false,
                    canvasOnly: true,
                }
            );
            cropWidget.label = "Crop...";
            cropWidget.serialize = false;

            attachImageSync(node);
            queueMicrotask(() => hideIntWidgets(node));
        };
    },
});
