import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Smart Lora: dual-model (high / low) LoRA node with two independent,
// dynamically managed LoRA lists rendered as custom canvas widgets. All
// per-LoRA state is funnelled into the hidden `lora_config` STRING widget as
// JSON so the backend can apply an arbitrary number of LoRAs per group.
//
// Persistence strategy (see ComfyUI litegraph serialize/configure):
//   - `widget.serialize = false`         -> keep widget out of workflow
//                                           `widgets_values` (avoids the
//                                           full-index/compacted-index hole
//                                           mismatch caused by our UI-only
//                                           widgets sitting above prompt_text).
//   - `widget.options.serialize = false` -> keep widget out of the API prompt.
//   - State that must persist (lora_config, prompt_text) is stored in
//     `node.properties` (round-trips reliably) and bound back to the widgets
//     via `options.property`, while still being sent to the backend because
//     their `options.serialize` is left truthy.

const ROW_H = (window.LiteGraph && LiteGraph.NODE_WIDGET_HEIGHT) || 20;
const MARGIN = 15;
const GAP = 6;
const MIN_WIDTH = 320;
const SEP_H = 9;

let LORA_FILES = [];
let _activePicker = null;

// Shared cache of saved profiles: { profileName: { high, low, prompt_text } }.
let PROFILES = {};
let _profilesLoaded = false;

// ── small drawing/util helpers ──────────────────────────────────────────────
function themeColor(key, fallback) {
    return (window.LiteGraph && LiteGraph[key]) || fallback;
}

function roundRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, r);
        return;
    }
    r = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function baseName(name) {
    if (!name) return "";
    const parts = String(name).split(/[\\/]/);
    return parts[parts.length - 1];
}

function fitText(ctx, text, maxW) {
    if (maxW <= 0) return "";
    if (ctx.measureText(text).width <= maxW) return text;
    let t = text;
    while (t.length > 1 && ctx.measureText(t + "\u2026").width > maxW) {
        t = t.slice(0, -1);
    }
    return t + "\u2026";
}

// ── config / state helpers ──────────────────────────────────────────────────
function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function hideWidget(w) {
    if (!w) return;
    // `hidden` hides DOM widgets (isVisible() checks it); `type === "hidden"`
    // makes both canvas and DOM widgets report zero layout size.
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.draw = () => {};
}

function setOption(w, key, value) {
    if (!w) return;
    w.options = Object.assign({}, w.options);
    w.options[key] = value;
}

function collectGroups(node) {
    const state = { high: [], low: [] };
    for (const w of node.widgets || []) {
        if (!w.__isLoraRow) continue;
        const target = state[w.group];
        if (!target) continue;
        target.push({
            on: w.value.on !== false,
            name: w.value.name || "",
            strength: Number(w.value.strength),
        });
    }
    return state;
}

function writeConfig(node) {
    const state = collectGroups(node);
    const json = JSON.stringify(state);
    node.properties = node.properties || {};
    node.properties.lora_config = json;
    if (node._cfgWidget) {
        node._cfgWidget.value = json;
        node._cfgWidget.callback?.(json);
    }
    node.setDirtyCanvas(true, true);
}

function reorder(node) {
    const rows = (node.widgets || []).filter((w) => w.__isLoraRow);
    const high = rows.filter((r) => r.group === "high");
    const low = rows.filter((r) => r.group === "low");
    const out = [];
    if (node._addHigh) out.push(node._addHigh);
    out.push(...high);
    // Divider between the high and low LoRA sections.
    if (node._sepHighLow) out.push(node._sepHighLow);
    if (node._addLow) out.push(node._addLow);
    out.push(...low);
    // Divider between the low LoRA section and the profiles section.
    if (node._sepLowProfiles) out.push(node._sepLowProfiles);
    // Profile controls sit directly above the prompt text box.
    if (node._profileSelect) out.push(node._profileSelect);
    if (node._profileSave) out.push(node._profileSave);
    if (node._profileUpdate) out.push(node._profileUpdate);
    if (node._profileDelete) out.push(node._profileDelete);
    // Divider between the profiles section and the prompt text.
    if (node._sepProfilesPrompt) out.push(node._sepProfilesPrompt);
    if (node._promptText) out.push(node._promptText);
    for (const w of node.widgets || []) {
        if (!out.includes(w)) out.push(w);
    }
    node.widgets = out;
}

function makeSeparator() {
    return {
        name: "__sep",
        type: "smartlora_sep",
        __isSeparator: true,
        serialize: false,
        options: { serialize: false },
        computeSize(width) {
            return [width || MIN_WIDTH, SEP_H];
        },
        draw(ctx, n, width, y, H) {
            const cy = Math.round(y + H * 0.5) + 0.5;
            ctx.save();
            ctx.strokeStyle = themeColor("WIDGET_OUTLINE_COLOR", "#666");
            ctx.globalAlpha = 0.6;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(MARGIN, cy);
            ctx.lineTo(width - MARGIN, cy);
            ctx.stroke();
            ctx.restore();
        },
    };
}

function fitNode(node) {
    const sz = node.computeSize();
    if (node.size[0] < Math.max(MIN_WIDTH, sz[0])) {
        node.size[0] = Math.max(MIN_WIDTH, sz[0]);
    }
    if (node.size[1] < sz[1]) {
        node.size[1] = sz[1];
    }
    node.setDirtyCanvas(true, true);
}

// ── LoRA name picker (filterable dropdown) ──────────────────────────────────
function closePicker() {
    if (!_activePicker) return;
    document.removeEventListener("pointerdown", _activePicker.outside, true);
    document.removeEventListener("mousedown", _activePicker.outside, true);
    document.removeEventListener("keydown", _activePicker.onKey, true);
    _activePicker.menu.remove();
    _activePicker = null;
}

function openLoraPicker(node, row, event) {
    closePicker();

    const items = ["(None)", ...LORA_FILES];

    const menu = document.createElement("div");
    menu.style.cssText = `
        position: fixed; z-index: 99999;
        background: #1e1e1e; border: 1px solid #555; border-radius: 6px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.5);
        display: flex; flex-direction: column;
        min-width: 260px; max-width: 480px; max-height: 60vh;
        overflow: hidden; font-family: sans-serif;
    `;

    const search = document.createElement("input");
    search.type = "text";
    search.placeholder = "Search LoRAs\u2026";
    search.style.cssText = `
        margin: 6px; padding: 6px 8px; border-radius: 4px;
        border: 1px solid #555; background: #111; color: #eee;
        outline: none; font-size: 13px;
    `;

    const list = document.createElement("div");
    list.style.cssText = "overflow-y: auto; padding: 0 6px 6px;";

    menu.appendChild(search);
    menu.appendChild(list);

    function render(filter) {
        const f = (filter || "").toLowerCase();
        list.innerHTML = "";
        const filtered = items.filter(
            (it) => it === "(None)" || it.toLowerCase().includes(f)
        );
        if (filtered.length === 0) {
            const empty = document.createElement("div");
            empty.textContent = "No matching LoRAs";
            empty.style.cssText = "padding: 8px; color: #777; font-size: 13px;";
            list.appendChild(empty);
            return;
        }
        for (const it of filtered) {
            const isSel =
                (it === "(None)" && !row.value.name) || it === row.value.name;
            const el = document.createElement("div");
            el.textContent = it;
            el.style.cssText = `
                padding: 5px 8px; border-radius: 4px; cursor: pointer;
                color: ${isSel ? "#fff" : "#ccc"};
                background: ${isSel ? "#3a6e57" : "transparent"};
                font-size: 13px; white-space: nowrap;
                overflow: hidden; text-overflow: ellipsis;
            `;
            el.addEventListener("mouseenter", () => {
                if (!isSel) el.style.background = "#333";
            });
            el.addEventListener("mouseleave", () => {
                if (!isSel) el.style.background = "transparent";
            });
            el.addEventListener("click", () => {
                row.value.name = it === "(None)" ? "" : it;
                writeConfig(node);
                closePicker();
            });
            list.appendChild(el);
        }
    }

    render("");
    search.addEventListener("input", () => render(search.value));
    search.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closePicker();
    });

    document.body.appendChild(menu);

    const px = event?.clientX ?? 200;
    const py = event?.clientY ?? 200;
    const rect = menu.getBoundingClientRect();
    let left = px;
    let top = py;
    if (left + rect.width > window.innerWidth) left = window.innerWidth - rect.width - 8;
    if (top + rect.height > window.innerHeight) top = window.innerHeight - rect.height - 8;
    menu.style.left = Math.max(8, left) + "px";
    menu.style.top = Math.max(8, top) + "px";

    search.focus();

    const outside = (e) => {
        if (!menu.contains(e.target)) closePicker();
    };
    const onKey = (e) => {
        if (e.key === "Escape") {
            e.preventDefault();
            closePicker();
        }
    };
    // The canvas uses pointer events and preventDefault()s pointerdown, which
    // suppresses the synthetic mousedown; listen for pointerdown (plus mousedown
    // as a fallback) so clicking anywhere outside the popup closes it.
    setTimeout(() => {
        document.addEventListener("pointerdown", outside, true);
        document.addEventListener("mousedown", outside, true);
        document.addEventListener("keydown", onKey, true);
    }, 0);
    _activePicker = { menu, outside, onKey };
}

function editStrength(node, row, event) {
    const canvas = app.canvas;
    if (!canvas || typeof canvas.prompt !== "function") {
        const v = window.prompt("Strength", String(row.value.strength));
        if (v != null) {
            const n = parseFloat(v);
            if (!isNaN(n)) {
                row.value.strength = n;
                writeConfig(node);
            }
        }
        return;
    }
    canvas.prompt(
        "Strength",
        row.value.strength,
        (v) => {
            const n = parseFloat(v);
            if (!isNaN(n)) {
                row.value.strength = n;
                writeConfig(node);
            }
        },
        event
    );
}

// ── LoRA info modal ─────────────────────────────────────────────────────────
function toast(message) {
    const el = document.createElement("div");
    el.textContent = message;
    el.style.cssText = `
        position: fixed; bottom: 30px; right: 30px; z-index: 100000;
        background: #1a6b4a; color: #fff; padding: 10px 16px;
        border-radius: 8px; font-family: sans-serif; font-size: 13px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4); transition: opacity 0.4s ease;
    `;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.opacity = "0";
        setTimeout(() => el.remove(), 400);
    }, 2200);
}

function copyText(text, label) {
    const value = text == null ? "" : String(text);
    const done = () => toast(`${label} copied`);
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(value).then(done).catch(() => fallbackCopy(value, done));
    } else {
        fallbackCopy(value, done);
    }
}

function fallbackCopy(value, done) {
    const ta = document.createElement("textarea");
    ta.value = value;
    ta.style.cssText = "position:fixed;opacity:0;";
    document.body.appendChild(ta);
    ta.select();
    try {
        document.execCommand("copy");
        done?.();
    } catch (e) {
        /* ignore */
    }
    ta.remove();
}

async function openLoraInfo(row) {
    const name = row.value.name;
    if (!name) {
        toast("Select a LoRA first");
        return;
    }
    let payload = null;
    try {
        const res = await api.fetchApi(
            `/smart_tools/smart_lora/lora_info?name=${encodeURIComponent(name)}`
        );
        if (res.ok) payload = await res.json();
    } catch (e) {
        payload = null;
    }
    if (!payload || !payload.found) {
        toast(`No info JSON found for ${baseName(name)}`);
        return;
    }
    showInfoModal(baseName(name), payload.data || {});
}

function makeCopyButton(label, getText) {
    const btn = document.createElement("button");
    btn.textContent = "Copy";
    btn.style.cssText = `
        padding: 5px 12px; background: #2a9d8f; color: #fff; border: none;
        border-radius: 6px; cursor: pointer; font-size: 12px; flex-shrink: 0;
    `;
    btn.addEventListener("click", () => copyText(getText(), label));
    return btn;
}

function showInfoModal(title, data) {
    const existing = document.getElementById("smart_lora_info_modal");
    if (existing) existing.remove();

    const url = data.url || "";
    const triggerWords = Array.isArray(data.triggerWords)
        ? data.triggerWords
        : data.triggerWords
        ? [String(data.triggerWords)]
        : [];
    const triggerText = triggerWords.join(", ");
    const description = data.description || "";
    const baseModel = data.baseModel || "";

    const overlay = document.createElement("div");
    overlay.id = "smart_lora_info_modal";
    overlay.style.cssText = `
        position: fixed; inset: 0; z-index: 99999;
        background: rgba(0,0,0,0.6);
        display: flex; align-items: center; justify-content: center;
    `;

    const dialog = document.createElement("div");
    dialog.style.cssText = `
        background: #1e2a2a; border: 1px solid #2a9d8f; border-radius: 10px;
        padding: 18px; width: 560px; max-width: 92vw; max-height: 86vh;
        display: flex; flex-direction: column; gap: 12px;
        color: #eee; font-family: sans-serif; box-sizing: border-box;
    `;

    const header = document.createElement("div");
    header.style.cssText =
        "display:flex;align-items:center;justify-content:space-between;gap:10px;";
    const h = document.createElement("h3");
    h.textContent = title;
    h.style.cssText =
        "margin:0;color:#2a9d8f;font-size:16px;word-break:break-all;";
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "\u2715";
    closeBtn.style.cssText =
        "background:none;border:none;color:#aaa;font-size:16px;cursor:pointer;flex-shrink:0;";
    closeBtn.addEventListener("click", () => overlay.remove());
    header.appendChild(h);
    header.appendChild(closeBtn);
    dialog.appendChild(header);

    if (baseModel) {
        const bm = document.createElement("div");
        bm.textContent = `Base model: ${baseModel}`;
        bm.style.cssText = "font-size:12px;color:#9bbdb6;";
        dialog.appendChild(bm);
    }

    const field = (labelText, valueText, opts = {}) => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:5px;";

        const top = document.createElement("div");
        top.style.cssText =
            "display:flex;align-items:center;justify-content:space-between;gap:10px;";
        const lab = document.createElement("span");
        lab.textContent = labelText;
        lab.style.cssText = "font-size:12px;color:#9bbdb6;font-weight:bold;";
        top.appendChild(lab);
        top.appendChild(makeCopyButton(labelText, () => valueText));
        wrap.appendChild(top);

        if (opts.link && valueText) {
            const a = document.createElement("a");
            a.href = valueText;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
            a.textContent = valueText;
            a.style.cssText =
                "font-size:13px;color:#6cc6ff;word-break:break-all;text-decoration:underline;";
            wrap.appendChild(a);
        } else {
            const box = document.createElement("div");
            box.textContent = valueText || "(none)";
            box.style.cssText = `
                font-size:13px;color:${valueText ? "#ddd" : "#777"};
                background:#0d1f1f;border:1px solid #2a5d54;border-radius:6px;
                padding:8px 10px;white-space:pre-wrap;word-break:break-word;
                ${opts.scroll ? "overflow-y:auto;max-height:34vh;" : ""}
            `;
            wrap.appendChild(box);
        }
        return wrap;
    };

    dialog.appendChild(field("Link", url, { link: true }));
    dialog.appendChild(field("Trigger words", triggerText));
    dialog.appendChild(field("Description", description, { scroll: true }));

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    overlay.addEventListener("pointerdown", (e) => {
        if (e.target === overlay) overlay.remove();
    });
    const onKey = (e) => {
        if (e.key === "Escape") {
            overlay.remove();
            document.removeEventListener("keydown", onKey, true);
        }
    };
    document.addEventListener("keydown", onKey, true);
}

// ── Profiles ────────────────────────────────────────────────────────────────
const PROFILE_NONE = "(none)";

async function loadProfiles() {
    try {
        const res = await api.fetchApi("/smart_tools/smart_lora/profiles");
        if (res.ok) {
            const data = await res.json();
            if (data && typeof data === "object") PROFILES = data;
        }
    } catch (e) {
        /* keep whatever we have */
    }
    _profilesLoaded = true;
    return PROFILES;
}

async function persistProfiles() {
    try {
        await api.fetchApi("/smart_tools/smart_lora/profiles", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ profiles: PROFILES }),
        });
    } catch (e) {
        toast("Failed to save profiles");
    }
}

function getState(node) {
    const groups = collectGroups(node);
    return {
        high: groups.high,
        low: groups.low,
        prompt_text: node._promptText ? node._promptText.value || "" : "",
    };
}

function applyProfile(node, profile) {
    if (!profile || typeof profile !== "object") return;
    const cfg = { high: profile.high || [], low: profile.low || [] };
    node.properties = node.properties || {};
    node.properties.lora_config = JSON.stringify(cfg);
    if (node._cfgWidget) node._cfgWidget.value = node.properties.lora_config;
    rebuildFromConfig(node);
    if (node._promptText) {
        node._promptText.value = profile.prompt_text || "";
        node.properties.prompt_text = node._promptText.value;
    }
    writeConfig(node);
    fitNode(node);
}

function refreshProfileCombo(node, selectName) {
    const w = node._profileSelect;
    if (!w) return;
    const names = Object.keys(PROFILES).sort((a, b) =>
        a.toLowerCase().localeCompare(b.toLowerCase())
    );
    w.options = w.options || {};
    w.options.values = [PROFILE_NONE, ...names];
    if (selectName != null) {
        w.value = selectName;
    } else if (!w.options.values.includes(w.value)) {
        w.value = PROFILE_NONE;
    }
    node.setDirtyCanvas(true, true);
}

function promptForName(defaultValue, callback) {
    const canvas = app.canvas;
    if (canvas && typeof canvas.prompt === "function") {
        canvas.prompt("Profile name", defaultValue || "", (v) => callback(v), null);
    } else {
        const v = window.prompt("Profile name", defaultValue || "");
        if (v != null) callback(v);
    }
}

function setupProfileControls(node) {
    node._profileSelect = node.addWidget(
        "combo",
        "Profile",
        PROFILE_NONE,
        (value) => {
            if (value && value !== PROFILE_NONE && PROFILES[value]) {
                applyProfile(node, PROFILES[value]);
            }
        },
        { values: [PROFILE_NONE] }
    );
    node._profileSelect.serialize = false;
    setOption(node._profileSelect, "serialize", false);

    node._profileSave = node.addWidget("button", "Save Profile As...", null, () => {
        const current =
            node._profileSelect.value && node._profileSelect.value !== PROFILE_NONE
                ? node._profileSelect.value
                : "";
        promptForName(current, (rawName) => {
            const name = (rawName || "").trim();
            if (!name) return;
            PROFILES[name] = getState(node);
            persistProfiles();
            refreshProfileCombo(node, name);
            toast(`Profile "${name}" saved`);
        });
    });
    node._profileSave.serialize = false;
    setOption(node._profileSave, "serialize", false);

    node._profileUpdate = node.addWidget("button", "Update Profile", null, () => {
        const name = node._profileSelect.value;
        if (!name || name === PROFILE_NONE) {
            toast("Select a profile to update");
            return;
        }
        PROFILES[name] = getState(node);
        persistProfiles();
        toast(`Profile "${name}" updated`);
    });
    node._profileUpdate.serialize = false;
    setOption(node._profileUpdate, "serialize", false);

    node._profileDelete = node.addWidget("button", "Delete Profile", null, () => {
        const name = node._profileSelect.value;
        if (!name || name === PROFILE_NONE) {
            toast("Select a profile to delete");
            return;
        }
        delete PROFILES[name];
        persistProfiles();
        refreshProfileCombo(node, PROFILE_NONE);
        toast(`Profile "${name}" deleted`);
    });
    node._profileDelete.serialize = false;
    setOption(node._profileDelete, "serialize", false);

    // Populate from the shared cache (fetched once per session).
    if (_profilesLoaded) {
        refreshProfileCombo(node);
    } else {
        loadProfiles().then(() => refreshProfileCombo(node));
    }
}

// ── custom LoRA row widget ──────────────────────────────────────────────────
function createRowRaw(node, group, entry) {
    node._rowSeq = (node._rowSeq || 0) + 1;
    const row = {
        name: `lora_${group}_${node._rowSeq}`,
        type: "smartlora_row",
        group,
        __isLoraRow: true,
        serialize: false,
        options: { serialize: false },
        value: {
            on: entry && entry.on === false ? false : true,
            name: (entry && entry.name) || "",
            strength:
                entry && entry.strength != null ? Number(entry.strength) : 1.0,
        },
        computeSize(width) {
            return [width || MIN_WIDTH, ROW_H];
        },
        draw(ctx, n, width, y, H) {
            this.last_y = y;

            const bg = themeColor("WIDGET_BGCOLOR", "#222");
            const outline = themeColor("WIDGET_OUTLINE_COLOR", "#666");
            const textCol = themeColor("WIDGET_TEXT_COLOR", "#ddd");
            const secondary = themeColor("WIDGET_SECONDARY_TEXT_COLOR", "#999");
            const on = this.value.on !== false;

            const midY = y + H * 0.5;
            const radius = H * 0.5;
            const delW = H;
            const togW = Math.round(H * 1.7);
            const infoW = H;
            const strW = 50;
            const right = width - MARGIN;
            const delX = right - delW;
            const togX = delX - GAP - togW;
            const infoX = togX - GAP - infoW;
            const strX = infoX - GAP - strW;
            const nameX = MARGIN;
            const nameW = Math.max(40, strX - GAP - nameX);

            this._rects = {
                nameX, nameW, strX, strW, infoX, infoW,
                togX, togW, delX, delW, y, H,
            };

            ctx.save();
            ctx.textBaseline = "middle";

            // Name field (combo style)
            ctx.globalAlpha = on ? 1 : 0.5;
            ctx.fillStyle = bg;
            roundRect(ctx, nameX, y, nameW, H, radius);
            ctx.fill();
            ctx.strokeStyle = outline;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = secondary;
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillText("\u25C0", nameX + 9, midY);
            ctx.fillText("\u25B6", nameX + nameW - 9, midY);
            ctx.fillStyle = this.value.name ? textCol : secondary;
            ctx.font = "12px Arial";
            ctx.textAlign = "left";
            const label = this.value.name
                ? baseName(this.value.name)
                : "(click to select)";
            ctx.fillText(fitText(ctx, label, nameW - 34), nameX + 18, midY);

            // Strength field (number style)
            ctx.fillStyle = bg;
            roundRect(ctx, strX, y, strW, H, radius);
            ctx.fill();
            ctx.strokeStyle = outline;
            ctx.stroke();
            ctx.fillStyle = textCol;
            ctx.font = "12px Arial";
            ctx.textAlign = "center";
            ctx.fillText(Number(this.value.strength).toFixed(2), strX + strW * 0.5, midY);

            ctx.globalAlpha = 1;

            // Info button ("i")
            ctx.fillStyle = bg;
            roundRect(ctx, infoX, y, infoW, H, 4);
            ctx.fill();
            ctx.strokeStyle = outline;
            ctx.stroke();
            ctx.fillStyle = textCol;
            ctx.font = "bold italic 13px Georgia, 'Times New Roman', serif";
            ctx.textAlign = "center";
            ctx.fillText("i", infoX + infoW * 0.5, midY + 0.5);

            // Enable toggle (boolean pill)
            ctx.fillStyle = bg;
            roundRect(ctx, togX, y, togW, H, radius);
            ctx.fill();
            ctx.strokeStyle = outline;
            ctx.stroke();
            const knobR = H * 0.5 - 3;
            const knobX = on ? togX + togW - radius : togX + radius;
            ctx.fillStyle = on ? "#4caf50" : "#666";
            ctx.beginPath();
            ctx.arc(knobX, midY, knobR, 0, Math.PI * 2);
            ctx.fill();

            // Delete button
            ctx.fillStyle = "#3a2222";
            roundRect(ctx, delX, y, delW, H, 4);
            ctx.fill();
            ctx.strokeStyle = "#7a3a3a";
            ctx.stroke();
            ctx.fillStyle = "#e06666";
            ctx.font = "bold 13px Arial";
            ctx.textAlign = "center";
            ctx.fillText("\u2715", delX + delW * 0.5, midY + 0.5);

            ctx.restore();
        },
        mouse(event, pos, n) {
            const r = this._rects;
            if (!r) return false;
            const y = this.last_y ?? r.y;
            const lx = pos[0];
            const ly = pos[1];
            if (ly < y || ly > y + r.H) return false;

            // processNodeWidgets fires on both pointer down and up; only act on
            // down so the toggle (and other controls) don't trigger twice.
            const type = event?.type;
            if (type && type !== "pointerdown" && type !== "mousedown") {
                return true;
            }

            const inX = (x0, w) => lx >= x0 && lx <= x0 + w;

            if (inX(r.togX, r.togW)) {
                this.value.on = !this.value.on;
                writeConfig(n);
                return true;
            }
            if (inX(r.delX, r.delW)) {
                deleteRow(n, this);
                return true;
            }
            if (inX(r.infoX, r.infoW)) {
                openLoraInfo(this);
                return true;
            }
            if (inX(r.strX, r.strW)) {
                editStrength(n, this, event);
                return true;
            }
            if (inX(r.nameX, r.nameW)) {
                openLoraPicker(n, this, event);
                return true;
            }
            return true;
        },
    };
    node.widgets = node.widgets || [];
    node.widgets.push(row);
    return row;
}

function addRow(node, group) {
    createRowRaw(node, group, {});
    reorder(node);
    writeConfig(node);
    fitNode(node);
}

function deleteRow(node, row) {
    const idx = node.widgets.indexOf(row);
    if (idx !== -1) node.widgets.splice(idx, 1);
    reorder(node);
    writeConfig(node);
    fitNode(node);
}

function rebuildFromConfig(node) {
    node.widgets = (node.widgets || []).filter((w) => !w.__isLoraRow);
    let cfg = {};
    const raw =
        (node.properties && node.properties.lora_config) ||
        node._cfgWidget?.value ||
        "{}";
    try {
        cfg = JSON.parse(raw) || {};
    } catch (e) {
        cfg = {};
    }
    (cfg.high || []).forEach((e) => createRowRaw(node, "high", e));
    (cfg.low || []).forEach((e) => createRowRaw(node, "low", e));
    // Mirror restored state back into the widget value sent to the backend.
    if (node._cfgWidget) node._cfgWidget.value = JSON.stringify(cfg);
    reorder(node);
    fitNode(node);
}

// ── extension registration ──────────────────────────────────────────────────
app.registerExtension({
    name: "SmartTools.SmartLora",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SmartLora") return;

        const loraFiles =
            nodeData.input?.optional?.lora_files?.[0] ||
            nodeData.input?.required?.lora_files?.[0];
        if (Array.isArray(loraFiles)) {
            LORA_FILES = loraFiles.filter((n) => n && n !== "(no loras found)");
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const node = this;
            node.properties = node.properties || {};

            node._cfgWidget = getWidget(node, "lora_config");
            node._promptText = getWidget(node, "prompt_text");
            const loraFilesWidget = getWidget(node, "lora_files");

            // Keep lora_config / prompt_text out of widgets_values (workflow),
            // persist them via node.properties, but still send to the backend.
            if (node._cfgWidget) {
                node._cfgWidget.serialize = false;
                setOption(node._cfgWidget, "property", "lora_config");
                if (node.properties.lora_config == null) {
                    node.properties.lora_config = node._cfgWidget.value || "{}";
                }
            }
            if (node._promptText) {
                node._promptText.serialize = false;
                setOption(node._promptText, "property", "prompt_text");
                if (node.properties.prompt_text == null) {
                    node.properties.prompt_text = node._promptText.value || "";
                }
            }
            // lora_files is only a data source for the UI: hide it and keep it
            // out of both the workflow and the backend prompt.
            if (loraFilesWidget) {
                loraFilesWidget.serialize = false;
                setOption(loraFilesWidget, "serialize", false);
                hideWidget(loraFilesWidget);
            }
            hideWidget(node._cfgWidget);

            // Native buttons -> guaranteed standard styling. UI-only.
            node._addHigh = node.addWidget("button", "Add Lora (High)", null, () => {
                addRow(node, "high");
            });
            node._addHigh.serialize = false;
            setOption(node._addHigh, "serialize", false);

            node._addLow = node.addWidget("button", "Add Lora (Low)", null, () => {
                addRow(node, "low");
            });
            node._addLow.serialize = false;
            setOption(node._addLow, "serialize", false);

            // Profile selector + Save/Update/Delete controls (UI-only).
            setupProfileControls(node);

            // Dividers separating high / low / profiles / prompt sections.
            node._sepHighLow = makeSeparator();
            node._sepLowProfiles = makeSeparator();
            node._sepProfilesPrompt = makeSeparator();
            node.widgets.push(
                node._sepHighLow,
                node._sepLowProfiles,
                node._sepProfilesPrompt
            );

            // prompt_text is a multiline DOM widget; ComfyUI natively gives it
            // the leftover vertical space (only it grows/shrinks on resize),
            // so no custom computeSize is needed here. Overriding it fights the
            // DOM layout and causes runaway vertical growth.

            // Persist latest widget values into node.properties at save time.
            const origSerialize = node.onSerialize;
            node.onSerialize = function (o) {
                origSerialize?.call(this, o);
                o.properties = o.properties || {};
                if (node._cfgWidget) o.properties.lora_config = node._cfgWidget.value;
                if (node._promptText) o.properties.prompt_text = node._promptText.value;
            };

            rebuildFromConfig(node);

            if (node.size[0] < MIN_WIDTH) node.size[0] = MIN_WIDTH;
            fitNode(node);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            const node = this;
            node._cfgWidget = getWidget(node, "lora_config");
            node._promptText = getWidget(node, "prompt_text");
            // Restore prompt_text from properties if widget binding missed it.
            if (
                node._promptText &&
                node.properties &&
                node.properties.prompt_text != null
            ) {
                node._promptText.value = node.properties.prompt_text;
            }
            rebuildFromConfig(node);
            if (_profilesLoaded) {
                refreshProfileCombo(node);
            } else {
                loadProfiles().then(() => refreshProfileCombo(node));
            }
        };
    },
});
