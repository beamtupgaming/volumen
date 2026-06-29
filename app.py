"""
app.py — PHerc. 1667 Papyrus Reader
Main GUI application (CustomTkinter).
"""

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

import numpy as np
from processing import run_pipeline, find_character_regions, clarify_characters
from ai_analysis import analyze_image, analyze_image_stream, research_character, analyze_section_stream

# ─────────────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.json"
OUTPUT_DIR = Path(__file__).parent / "output"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─────────────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
class ClickableImagePanel(ctk.CTkFrame):
    """
    Image panel backed by a tk.Canvas.
    Features: mouse-wheel zoom (HD crop-then-scale), right-click drag pan,
    semi-transparent region highlights, hover/click interaction.
    Zoom and pan are independent per panel; zoom level can be synchronised
    externally via set_zoom() without triggering the on_zoom_change callback.
    """

    def __init__(self, master, title: str, on_region_click=None,
                 on_zoom_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self._on_region_click = on_region_click
        self._on_zoom_change = on_zoom_change   # callback(float) – no loop risk
        self._regions: list[dict] = []
        self._raw_img = None
        self._zoom: float = 1.0          # multiplier on top of fit-to-panel scale
        self._pan_x: float = 0.0         # in rendered-image pixels
        self._pan_y: float = 0.0
        self._scale: float = 1.0         # fit_scale * zoom (image px → canvas px)
        self._offset_x: int = 0          # canvas offset when image smaller than panel
        self._offset_y: int = 0
        self._hover_idx: int = -1
        self._drag_start: tuple | None = None
        self._photo = None
        # section-select state
        self._mode: str = "click"          # "click" | "select"
        self._sel_start: tuple | None = None
        self._sel_rect_id = None
        self._on_section_select = None

        self.title_label = ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=13, weight="bold")
        )
        self.title_label.pack(pady=(6, 0))

        self._canvas = tk.Canvas(self, bg="#1c1c1c", highlightthickness=0)
        self._canvas.pack(expand=True, fill="both", padx=4, pady=4)
        self._canvas.bind("<Button-1>",        self._on_lmb_press)
        self._canvas.bind("<B1-Motion>",       self._on_lmb_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self._canvas.bind("<Motion>",          self._on_hover)
        self._canvas.bind("<Configure>",       self._on_resize)
        self._canvas.bind("<MouseWheel>",      self._on_mousewheel)
        self._canvas.bind("<Button-3>",        self._drag_begin)
        self._canvas.bind("<B3-Motion>",       self._drag_move)
        self._canvas.bind("<ButtonRelease-3>", self._drag_end)

    # ── public API ────────────────────────────────────────────────────────────
    def set_image(self, img):
        """Display a NumPy image (grayscale or BGR). Pass None to clear."""
        if img is None:
            self._raw_img = None
            self._regions = []
            self._pan_x = self._pan_y = 0.0
            self._canvas.delete("all")
            cw = max(self._canvas.winfo_width(), 100)
            ch = max(self._canvas.winfo_height(), 100)
            self._canvas.create_text(
                cw // 2, ch // 2,
                text="No image loaded", fill="gray", font=("Helvetica", 11),
            )
            return
        self._raw_img = img
        self._render()

    def set_overlay_regions(self, regions: list[dict]):
        """Stamp bounding-box highlights onto the processed panel."""
        self._regions = regions
        self._draw_overlay()

    def set_zoom(self, zoom: float):
        """
        External zoom sync (does NOT fire on_zoom_change to avoid loops).
        Preserves the centre of the current view.
        """
        zoom = max(1.0, min(zoom, 20.0))
        if self._raw_img is not None and self._zoom > 0:
            cw = max(self._canvas.winfo_width(), 100)
            ch = max(self._canvas.winfo_height(), 100)
            orig_h, orig_w = self._raw_img.shape[:2]
            fit_s = min(cw / max(orig_w, 1), ch / max(orig_h, 1))
            old_rs = fit_s * self._zoom
            new_rs = fit_s * zoom
            if old_rs > 0:
                cx_img = (self._pan_x + cw / 2) / old_rs
                cy_img = (self._pan_y + ch / 2) / old_rs
                self._pan_x = cx_img * new_rs - cw / 2
                self._pan_y = cy_img * new_rs - ch / 2
        self._zoom = zoom
        self._render()

    def reset_view(self):
        """Reset to fit-to-panel zoom with no pan."""
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0.0
        self._render()

    def get_zoom(self) -> float:
        return self._zoom
    def set_mode(self, mode: str):
        """Switch between 'click' (character research) and 'select' (drag-section)."""
        self._mode = mode
        self._canvas.configure(cursor="crosshair" if mode == "select" else "")
        if self._sel_rect_id is not None:
            self._canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None
        self._sel_start = None

    def set_section_callback(self, callback):
        """Register callback(crop_img, rect_dict) fired on section release."""
        self._on_section_select = callback
    # ── rendering ─────────────────────────────────────────────────────────────
    def _render(self):
        if self._raw_img is None:
            return
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)

        arr = self._raw_img
        src = (
            Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim == 2
            else Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        )
        orig_w, orig_h = src.size
        fit_s = min(cw / max(orig_w, 1), ch / max(orig_h, 1))
        rs = fit_s * self._zoom          # final render scale
        scaled_w = max(int(orig_w * rs), 1)
        scaled_h = max(int(orig_h * rs), 1)

        # Centering when image is smaller than canvas; pan clamping when larger
        if scaled_w <= cw:
            self._offset_x = (cw - scaled_w) // 2
            self._pan_x = 0.0
        else:
            self._offset_x = 0
            self._pan_x = max(0.0, min(self._pan_x, float(scaled_w - cw)))

        if scaled_h <= ch:
            self._offset_y = (ch - scaled_h) // 2
            self._pan_y = 0.0
        else:
            self._offset_y = 0
            self._pan_y = max(0.0, min(self._pan_y, float(scaled_h - ch)))

        self._scale = rs

        # HD crop-then-scale: work from original pixels for maximum sharpness
        px, py = int(self._pan_x), int(self._pan_y)
        vis_w = min(cw, scaled_w - px)
        vis_h = min(ch, scaled_h - py)
        sx0 = max(0, int(px / rs))
        sy0 = max(0, int(py / rs))
        sx1 = min(orig_w, int((px + vis_w) / rs) + 1)
        sy1 = min(orig_h, int((py + vis_h) / rs) + 1)
        tw, th = max(vis_w, 1), max(vis_h, 1)
        display = src.crop((sx0, sy0, sx1, sy1)).resize((tw, th), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(display)
        self._canvas.delete("all")
        self._canvas.create_image(self._offset_x, self._offset_y, anchor="nw",
                                  image=self._photo)
        self._draw_overlay()

    def _draw_overlay(self):
        self._canvas.delete("overlay")
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        for i, r in enumerate(self._regions):
            cx1 = r["x"] * self._scale - self._pan_x + self._offset_x
            cy1 = r["y"] * self._scale - self._pan_y + self._offset_y
            cx2 = (r["x"] + r["w"]) * self._scale - self._pan_x + self._offset_x
            cy2 = (r["y"] + r["h"]) * self._scale - self._pan_y + self._offset_y
            if cx2 < 0 or cy2 < 0 or cx1 > cw or cy1 > ch:
                continue  # skip off-screen boxes
            color = "#ff9800" if i == self._hover_idx else "#00e676"
            width = 2 if i == self._hover_idx else 1
            self._canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=width,
                fill=color, stipple="gray12",
                tags="overlay",
            )

    # ── event handlers ─────────────────────────────────────────────────────────
    def _on_resize(self, _event):
        self._render()

    def _on_mousewheel(self, event):
        if self._raw_img is None:
            return
        cw = max(self._canvas.winfo_width(), 100)
        ch = max(self._canvas.winfo_height(), 100)
        orig_h, orig_w = self._raw_img.shape[:2]
        fit_s = min(cw / max(orig_w, 1), ch / max(orig_h, 1))
        old_rs = fit_s * self._zoom

        # Image coordinate under cursor before zoom
        ix = (event.x - self._offset_x + self._pan_x) / max(old_rs, 1e-6)
        iy = (event.y - self._offset_y + self._pan_y) / max(old_rs, 1e-6)

        factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        self._zoom = max(1.0, min(self._zoom * factor, 20.0))
        new_rs = fit_s * self._zoom

        # Re-compute offsets at new zoom for accurate pan adjustment
        new_sw = int(orig_w * new_rs)
        new_sh = int(orig_h * new_rs)
        nox = (cw - new_sw) // 2 if new_sw <= cw else 0
        noy = (ch - new_sh) // 2 if new_sh <= ch else 0

        # Zoom toward cursor
        self._pan_x = ix * new_rs - (event.x - nox)
        self._pan_y = iy * new_rs - (event.y - noy)
        self._render()

        if self._on_zoom_change:
            self._on_zoom_change(self._zoom)

    def _drag_begin(self, event):
        self._drag_start = (event.x, event.y, self._pan_x, self._pan_y)
        self._canvas.configure(cursor="fleur")

    def _drag_move(self, event):
        if self._drag_start is None:
            return
        sx, sy, spx, spy = self._drag_start
        self._pan_x = spx - (event.x - sx)
        self._pan_y = spy - (event.y - sy)
        self._render()

    def _drag_end(self, _event):
        self._drag_start = None
        self._canvas.configure(cursor="hand2" if self._hover_idx >= 0 else "")

    def _on_hover(self, event):
        if not self._regions:
            return
        ix = (event.x - self._offset_x + self._pan_x) / max(self._scale, 1e-6)
        iy = (event.y - self._offset_y + self._pan_y) / max(self._scale, 1e-6)
        prev = self._hover_idx
        self._hover_idx = -1
        for i, r in enumerate(self._regions):
            if r["x"] <= ix <= r["x"] + r["w"] and r["y"] <= iy <= r["y"] + r["h"]:
                self._hover_idx = i
                break
        if self._hover_idx != prev:
            self._canvas.configure(cursor="hand2" if self._hover_idx >= 0 else "")
            self._draw_overlay()

    def _on_click(self, event):
        if not self._regions or self._raw_img is None:
            return
        ix = (event.x - self._offset_x + self._pan_x) / max(self._scale, 1e-6)
        iy = (event.y - self._offset_y + self._pan_y) / max(self._scale, 1e-6)
        for i, r in enumerate(self._regions):
            if r["x"] <= ix <= r["x"] + r["w"] and r["y"] <= iy <= r["y"] + r["h"]:
                if self._on_region_click:
                    ih, iw = self._raw_img.shape[:2]
                    pad = 6
                    y1 = max(0, r["y"] - pad)
                    y2 = min(ih, r["y"] + r["h"] + pad)
                    x1 = max(0, r["x"] - pad)
                    x2 = min(iw, r["x"] + r["w"] + pad)
                    self._on_region_click(i, r, self._raw_img[y1:y2, x1:x2])
                return

    # ── left-mouse routing (click vs. rubber-band select) ──────────────────────
    def _on_lmb_press(self, event):
        if self._mode == "select":
            self._sel_start = (event.x, event.y)
            if self._sel_rect_id is not None:
                self._canvas.delete(self._sel_rect_id)
            self._sel_rect_id = self._canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#ffeb3b", width=2, dash=(4, 4), tags="selection",
            )
        else:
            self._on_click(event)

    def _on_lmb_drag(self, event):
        if self._mode == "select" and self._sel_start and self._sel_rect_id:
            sx, sy = self._sel_start
            self._canvas.coords(self._sel_rect_id, sx, sy, event.x, event.y)

    def _on_lmb_release(self, event):
        if self._mode != "select" or self._sel_start is None:
            return
        sx, sy = self._sel_start
        ex, ey = event.x, event.y
        if self._sel_rect_id is not None:
            self._canvas.delete(self._sel_rect_id)
            self._sel_rect_id = None
        self._sel_start = None
        if abs(ex - sx) < 8 or abs(ey - sy) < 8:
            return  # too small — treat as accidental click
        if self._raw_img is None or self._on_section_select is None:
            return
        # Convert canvas rect → image coordinates
        s = max(self._scale, 1e-6)
        cx0, cx1 = min(sx, ex), max(sx, ex)
        cy0, cy1 = min(sy, ey), max(sy, ey)
        ix0 = int((cx0 - self._offset_x + self._pan_x) / s)
        iy0 = int((cy0 - self._offset_y + self._pan_y) / s)
        ix1 = int((cx1 - self._offset_x + self._pan_x) / s)
        iy1 = int((cy1 - self._offset_y + self._pan_y) / s)
        ih, iw = self._raw_img.shape[:2]
        ix0 = max(0, min(ix0, iw - 1))
        iy0 = max(0, min(iy0, ih - 1))
        ix1 = max(ix0 + 1, min(ix1, iw))
        iy1 = max(iy0 + 1, min(iy1, ih))
        crop = self._raw_img[iy0:iy1, ix0:ix1].copy()
        rect = {"x": ix0, "y": iy0, "w": ix1 - ix0, "h": iy1 - iy0}
        self._on_section_select(crop, rect)


# ─────────────────────────────────────────────────────────────────────────────
class SettingsWindow(ctk.CTkToplevel):
    """Modal settings dialog for processing parameters and AI prompts."""

    def __init__(self, master, cfg: dict, on_save):
        super().__init__(master)
        self.title("Settings")
        self.geometry("700x640")
        self.resizable(True, True)
        self.grab_set()
        self._cfg = json.loads(json.dumps(cfg))  # deep copy
        self._on_save = on_save
        self._build_ui()

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self)
        scroll.pack(expand=True, fill="both", padx=12, pady=8)

        # ── Processing parameters ──────────────────────────────────────────
        section = ctk.CTkLabel(scroll, text="Image Processing Parameters",
                               font=ctk.CTkFont(size=14, weight="bold"))
        section.pack(anchor="w", pady=(8, 4))

        p = self._cfg["processing"]
        self._fields: dict[str, ctk.CTkEntry] = {}

        proc_params = [
            ("NLM – Filter Strength (h)", "nlm_h"),
            ("NLM – Template Window", "nlm_template_window"),
            ("NLM – Search Window", "nlm_search_window"),
            ("Bilateral – Diameter (d)", "bilateral_d"),
            ("Bilateral – Sigma Color", "bilateral_sigma_color"),
            ("Bilateral – Sigma Space", "bilateral_sigma_space"),
            ("CLAHE – Clip Limit", "clahe_clip_limit"),
        ]
        for label, key in proc_params:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=120)
            entry.insert(0, str(p.get(key, "")))
            entry.pack(side="left", padx=8)
            self._fields[key] = entry

        # CLAHE tile grid (special: list of 2 ints)
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="CLAHE – Tile Grid (w,h)", width=240, anchor="w").pack(side="left")
        tg = p.get("clahe_tile_grid_size", [8, 8])
        self._tile_grid_w = ctk.CTkEntry(row, width=55)
        self._tile_grid_w.insert(0, str(tg[0]))
        self._tile_grid_w.pack(side="left", padx=(8, 2))
        self._tile_grid_h = ctk.CTkEntry(row, width=55)
        self._tile_grid_h.insert(0, str(tg[1]))
        self._tile_grid_h.pack(side="left")

        # Save intermediate steps toggle
        row2 = ctk.CTkFrame(scroll, fg_color="transparent")
        row2.pack(fill="x", pady=(6, 2))
        ctk.CTkLabel(row2, text="Save Intermediate Steps", width=240, anchor="w").pack(side="left")
        self._save_steps_var = ctk.BooleanVar(value=self._cfg["output"].get("save_intermediate_steps", True))
        ctk.CTkSwitch(row2, variable=self._save_steps_var, text="").pack(side="left", padx=8)

        # ── AI / Ollama settings ───────────────────────────────────────────
        section2 = ctk.CTkLabel(scroll, text="AI / Ollama Settings",
                                font=ctk.CTkFont(size=14, weight="bold"))
        section2.pack(anchor="w", pady=(14, 4))

        ai = self._cfg["ai"]

        row3 = ctk.CTkFrame(scroll, fg_color="transparent")
        row3.pack(fill="x", pady=2)
        ctk.CTkLabel(row3, text="Ollama Model", width=240, anchor="w").pack(side="left")
        self._model_entry = ctk.CTkEntry(row3, width=200)
        self._model_entry.insert(0, ai.get("ollama_model", "llava"))
        self._model_entry.pack(side="left", padx=8)

        row4 = ctk.CTkFrame(scroll, fg_color="transparent")
        row4.pack(fill="x", pady=2)
        ctk.CTkLabel(row4, text="Ollama Host", width=240, anchor="w").pack(side="left")
        self._host_entry = ctk.CTkEntry(row4, width=200)
        self._host_entry.insert(0, ai.get("ollama_host", "http://localhost:11434"))
        self._host_entry.pack(side="left", padx=8)

        ctk.CTkLabel(scroll, text="System Prompt", anchor="w").pack(anchor="w", pady=(10, 2))
        self._system_text = ctk.CTkTextbox(scroll, height=110, wrap="word")
        self._system_text.insert("end", ai.get("system_prompt", ""))
        self._system_text.pack(fill="x")

        ctk.CTkLabel(scroll, text="User Prompt Template", anchor="w").pack(anchor="w", pady=(10, 2))
        self._user_text = ctk.CTkTextbox(scroll, height=80, wrap="word")
        self._user_text.insert("end", ai.get("user_prompt_template", ""))
        self._user_text.pack(fill="x")
        # ── Legibility Refinement ──────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Papyrus Legibility Refinement",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(14, 4))

        ref = self._cfg.get("refinement", {})

        row_re = ctk.CTkFrame(scroll, fg_color="transparent")
        row_re.pack(fill="x", pady=2)
        ctk.CTkLabel(row_re, text="Enable Refinement Step", width=240, anchor="w").pack(side="left")
        self._refine_enabled = ctk.BooleanVar(value=ref.get("enabled", False))
        ctk.CTkSwitch(row_re, variable=self._refine_enabled, text="").pack(side="left", padx=8)

        row_nb = ctk.CTkFrame(scroll, fg_color="transparent")
        row_nb.pack(fill="x", pady=2)
        ctk.CTkLabel(row_nb, text="Normalize Background", width=240, anchor="w").pack(side="left")
        self._norm_bg = ctk.BooleanVar(value=ref.get("normalize_background", False))
        ctk.CTkSwitch(row_nb, variable=self._norm_bg, text="").pack(side="left", padx=8)

        refine_params = [
            ("Background Blur Radius", "bg_blur_radius"),
            ("Gamma Correction", "gamma"),
            ("Unsharp Mask Strength", "unsharp_strength"),
            ("Unsharp Mask Sigma", "unsharp_sigma"),
        ]
        self._refine_fields: dict[str, ctk.CTkEntry] = {}
        for label, key in refine_params:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, width=240, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=120)
            entry.insert(0, str(ref.get(key, "")))
            entry.pack(side="left", padx=8)
            self._refine_fields[key] = entry

        row_bh = ctk.CTkFrame(scroll, fg_color="transparent")
        row_bh.pack(fill="x", pady=2)
        ctk.CTkLabel(row_bh, text="Blackhat Ink Enhancement", width=240, anchor="w").pack(side="left")
        self._blackhat = ctk.BooleanVar(value=ref.get("blackhat_enhance", False))
        ctk.CTkSwitch(row_bh, variable=self._blackhat, text="").pack(side="left", padx=8)

        row_bhk = ctk.CTkFrame(scroll, fg_color="transparent")
        row_bhk.pack(fill="x", pady=2)
        ctk.CTkLabel(row_bhk, text="Blackhat Kernel Size", width=240, anchor="w").pack(side="left")
        self._blackhat_kernel = ctk.CTkEntry(row_bhk, width=120)
        self._blackhat_kernel.insert(0, str(ref.get("blackhat_kernel_size", 7)))
        self._blackhat_kernel.pack(side="left", padx=8)
        # ── Character Clarification ────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Character Clarification (🔬 Clarify button)",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(14, 4))

        c = self._cfg.get("clarification", {})
        clarify_params = [
            ("Stroke Kernel Size",   "stroke_kernel"),
            ("Stroke Weight",        "stroke_weight"),
            ("DoG Sigma (fine)",     "dog_sigma1"),
            ("DoG Sigma (coarse)",   "dog_sigma2"),
            ("DoG Weight",           "dog_weight"),
            ("Unsharp Strength",     "unsharp_strength"),
            ("Unsharp Sigma",        "unsharp_sigma"),
            ("Final CLAHE Clip",     "final_clahe_clip"),
        ]
        self._clarify_fields: dict[str, ctk.CTkEntry] = {}
        for lbl, key in clarify_params:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=lbl, width=240, anchor="w").pack(side="left")
            entry = ctk.CTkEntry(row, width=120)
            entry.insert(0, str(c.get(key, "")))
            entry.pack(side="left", padx=8)
            self._clarify_fields[key] = entry

        row_fc = ctk.CTkFrame(scroll, fg_color="transparent")
        row_fc.pack(fill="x", pady=2)
        ctk.CTkLabel(row_fc, text="Final CLAHE Pass", width=240, anchor="w").pack(side="left")
        self._final_clahe_var = ctk.BooleanVar(value=c.get("final_clahe", True))
        ctk.CTkSwitch(row_fc, variable=self._final_clahe_var, text="").pack(side="left", padx=8)

        row_cm = ctk.CTkFrame(scroll, fg_color="transparent")
        row_cm.pack(fill="x", pady=2)
        ctk.CTkLabel(row_cm, text="Median Cleanup Pass", width=240, anchor="w").pack(side="left")
        self._cleanup_var = ctk.BooleanVar(value=c.get("cleanup_median", True))
        ctk.CTkSwitch(row_cm, variable=self._cleanup_var, text="").pack(side="left", padx=8)

        # Auto-recover if blurry
        row_ar = ctk.CTkFrame(scroll, fg_color="transparent")
        row_ar.pack(fill="x", pady=2)
        ctk.CTkLabel(row_ar, text="Auto-Recover if Blurry", width=240, anchor="w").pack(side="left")
        self._auto_recover_var = ctk.BooleanVar(value=c.get("auto_recover", True))
        ctk.CTkSwitch(row_ar, variable=self._auto_recover_var, text="").pack(side="left", padx=8)

        row_msr = ctk.CTkFrame(scroll, fg_color="transparent")
        row_msr.pack(fill="x", pady=2)
        ctk.CTkLabel(
            row_msr, text="Min Sharpness Ratio (0–1)", width=240, anchor="w"
        ).pack(side="left")
        self._min_sharpness_entry = ctk.CTkEntry(row_msr, width=80)
        self._min_sharpness_entry.insert(0, str(c.get("min_sharpness_ratio", 0.85)))
        self._min_sharpness_entry.pack(side="left", padx=8)
        ctk.CTkLabel(
            row_msr, text="fallback fires when sharpness drops below this fraction",
            text_color="gray", font=ctk.CTkFont(size=10),
        ).pack(side="left", padx=4)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(4, 10))
        ctk.CTkButton(btn_row, text="Save", command=self._save).pack(side="right", padx=4)
        ctk.CTkButton(btn_row, text="Cancel", fg_color="gray30",
                      command=self.destroy).pack(side="right", padx=4)

    def _save(self):
        try:
            p = self._cfg["processing"]
            for key, entry in self._fields.items():
                val = entry.get().strip()
                # Preserve int vs float
                if "." in val:
                    p[key] = float(val)
                else:
                    p[key] = int(val)
            p["clahe_tile_grid_size"] = [
                int(self._tile_grid_w.get()),
                int(self._tile_grid_h.get()),
            ]
            self._cfg["output"]["save_intermediate_steps"] = self._save_steps_var.get()
            self._cfg["ai"]["ollama_model"] = self._model_entry.get().strip()
            self._cfg["ai"]["ollama_host"] = self._host_entry.get().strip()
            self._cfg["ai"]["system_prompt"] = self._system_text.get("1.0", "end").strip()
            self._cfg["ai"]["user_prompt_template"] = self._user_text.get("1.0", "end").strip()
            self._cfg["refinement"] = {
                "enabled":              self._refine_enabled.get(),
                "normalize_background": self._norm_bg.get(),
                "bg_blur_radius":  int(self._refine_fields["bg_blur_radius"].get()),
                "gamma":         float(self._refine_fields["gamma"].get()),
                "unsharp_strength": float(self._refine_fields["unsharp_strength"].get()),
                "unsharp_sigma":    float(self._refine_fields["unsharp_sigma"].get()),
                "blackhat_enhance":     self._blackhat.get(),
                "blackhat_kernel_size": int(self._blackhat_kernel.get()),
            }
            self._cfg["clarification"] = {
                "stroke_kernel":   int(self._clarify_fields["stroke_kernel"].get()),
                "stroke_weight": float(self._clarify_fields["stroke_weight"].get()),
                "dog_sigma1":    float(self._clarify_fields["dog_sigma1"].get()),
                "dog_sigma2":    float(self._clarify_fields["dog_sigma2"].get()),
                "dog_weight":    float(self._clarify_fields["dog_weight"].get()),
                "unsharp_strength": float(self._clarify_fields["unsharp_strength"].get()),
                "unsharp_sigma":    float(self._clarify_fields["unsharp_sigma"].get()),
                "final_clahe":       self._final_clahe_var.get(),
                "final_clahe_clip": float(self._clarify_fields["final_clahe_clip"].get()),
                "final_clahe_tile": self._cfg.get("clarification", {}).get("final_clahe_tile", [4, 4]),
                "cleanup_median":    self._cleanup_var.get(),
                "auto_recover":         self._auto_recover_var.get(),
                "min_sharpness_ratio":  float(self._min_sharpness_entry.get()),
            }
        except ValueError as e:
            messagebox.showerror("Invalid value", str(e), parent=self)
            return

        self._on_save(self._cfg)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
class CharacterDetailWindow(ctk.CTkToplevel):
    """Popup: AI-streamed identification, translation, morphology, and cross-references."""

    def __init__(self, master, region_img: np.ndarray, region_idx: int, cfg: dict):
        super().__init__(master)
        self.title(f"Character Research — Region {region_idx + 1}")
        self.geometry("620x700")
        self.resizable(True, True)
        self._cfg = cfg
        self._region_img = region_img
        self._build_ui()
        self.after(200, self._start_research)

    def _build_ui(self):
        # ── Cropped region preview ──────────────────────────────────────
        preview_frame = ctk.CTkFrame(self, height=130)
        preview_frame.pack(fill="x", padx=10, pady=(10, 4))
        preview_frame.pack_propagate(False)
        ctk.CTkLabel(
            preview_frame, text="Detected Region (4× zoom)",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(4, 2))
        self._preview_lbl = ctk.CTkLabel(preview_frame, text="")
        self._preview_lbl.pack(expand=True)
        self._show_preview()

        # ── AI research output ──────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="AI Research  ·  Greek Text · Translation · Morphology · Cross-References",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self._text_box = ctk.CTkTextbox(
            self, wrap="word", font=ctk.CTkFont(family="Courier New", size=12)
        )
        self._text_box.pack(expand=True, fill="both", padx=10, pady=(0, 4))
        self._text_box.insert("end", "Connecting to AI model…\n")
        self._text_box.configure(state="disabled")

        # ── Status + copy ───────────────────────────────────────────────
        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.pack(fill="x", padx=10, pady=(0, 8))
        self._status_var = tk.StringVar(value="Waiting…")
        ctk.CTkLabel(
            bot, textvariable=self._status_var,
            text_color="gray", font=ctk.CTkFont(size=10),
        ).pack(side="left")
        ctk.CTkButton(
            bot, text="Copy Text", width=90, fg_color="gray30",
            command=self._copy_text,
        ).pack(side="right")

    def _show_preview(self):
        arr = self._region_img
        if arr is None:
            return
        pil_img = (
            Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim == 2
            else Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        )
        w = min(pil_img.width * 4, 560)
        h = min(pil_img.height * 4, 90)
        pil_img = pil_img.resize((max(w, 1), max(h, 1)), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self._preview_lbl.configure(image=self._preview_photo, text="")

    def _start_research(self):
        self._status_var.set("Streaming AI research…")

        def _token(t):
            self.after(0, lambda t=t: self._append(t))

        def _done():
            self.after(0, lambda: self._status_var.set("Analysis complete."))

        threading.Thread(
            target=research_character,
            args=(self._region_img, self._cfg, _token, _done),
            daemon=True,
        ).start()

    def _append(self, token: str):
        self._text_box.configure(state="normal")
        self._text_box.insert("end", token)
        self._text_box.see("end")
        self._text_box.configure(state="disabled")

    def _copy_text(self):
        text = self._text_box.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status_var.set("Copied to clipboard.")


# ─────────────────────────────────────────────────────────────────────────────
class SectionContextWindow(ctk.CTkToplevel):
    """
    Streams a full 7-heading contextual AI analysis of a user-drawn papyrus section:
    Layout · Transcription · Translation · Text Type · Damage · Scribal Features · Context
    """

    def __init__(self, master, section_img: np.ndarray, rect: dict, cfg: dict):
        super().__init__(master)
        self.title(
            f"Section Context  —  {rect['w']}×{rect['h']} px  @  ({rect['x']}, {rect['y']})"
        )
        self.geometry("700x740")
        self.resizable(True, True)
        self._cfg = cfg
        self._section_img = section_img
        self._rect = rect
        self._build_ui()
        self.after(200, self._start_analysis)

    def _build_ui(self):
        # ── Section preview ──────────────────────────────────────────────────
        preview_frame = ctk.CTkFrame(self, height=160)
        preview_frame.pack(fill="x", padx=10, pady=(10, 4))
        preview_frame.pack_propagate(False)
        ctk.CTkLabel(
            preview_frame,
            text=(
                f"Selected Section  ·  {self._rect['w']}×{self._rect['h']} px  ·  "
                f"origin ({self._rect['x']}, {self._rect['y']})"
            ),
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(4, 2))
        self._preview_lbl = ctk.CTkLabel(preview_frame, text="")
        self._preview_lbl.pack(expand=True)
        self._show_preview()

        # ── AI analysis output ──────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="AI Context Analysis  ·  Layout · Transcription · Translation · Damage",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(6, 2))

        self._text_box = ctk.CTkTextbox(
            self, wrap="word", font=ctk.CTkFont(family="Courier New", size=12)
        )
        self._text_box.pack(expand=True, fill="both", padx=10, pady=(0, 4))
        self._text_box.insert("end", "Connecting to AI model…\n")
        self._text_box.configure(state="disabled")

        # ── Bottom bar ─────────────────────────────────────────────────────
        bot = ctk.CTkFrame(self, fg_color="transparent")
        bot.pack(fill="x", padx=10, pady=(0, 8))
        self._status_var = tk.StringVar(value="Waiting…")
        ctk.CTkLabel(
            bot, textvariable=self._status_var,
            text_color="gray", font=ctk.CTkFont(size=10),
        ).pack(side="left")
        ctk.CTkButton(
            bot, text="Save Report", width=100, fg_color="gray30",
            command=self._save_report,
        ).pack(side="right", padx=(4, 0))
        ctk.CTkButton(
            bot, text="Copy Text", width=90, fg_color="gray30",
            command=self._copy_text,
        ).pack(side="right", padx=4)

    def _show_preview(self):
        arr = self._section_img
        if arr is None or arr.size == 0:
            return
        pil_img = (
            Image.fromarray(arr, mode="L").convert("RGB")
            if arr.ndim == 2
            else Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        )
        pil_img.thumbnail((650, 130), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(pil_img)
        self._preview_lbl.configure(image=self._preview_photo, text="")

    def _start_analysis(self):
        self._status_var.set("Streaming AI context analysis…")

        def _token(t):
            self.after(0, lambda t=t: self._append(t))

        def _done():
            self.after(0, lambda: self._status_var.set("Analysis complete."))

        threading.Thread(
            target=analyze_section_stream,
            args=(self._section_img, self._rect, self._cfg, _token, _done),
            daemon=True,
        ).start()

    def _append(self, token: str):
        self._text_box.configure(state="normal")
        self._text_box.insert("end", token)
        self._text_box.see("end")
        self._text_box.configure(state="disabled")

    def _copy_text(self):
        text = self._text_box.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status_var.set("Copied to clipboard.")

    def _save_report(self):
        text = self._text_box.get("1.0", "end").strip()
        if not text or text.startswith("Connecting"):
            messagebox.showinfo("Nothing to save", "Wait for the analysis to complete.",
                                parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Save section report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            parent=self,
        )
        if path:
            divider = "─" * 60
            header = (
                f"Section Context Report\n"
                f"Region: {self._rect['w']}×{self._rect['h']} px "
                f"@ ({self._rect['x']}, {self._rect['y']})\n"
                f"{divider}\n\n"
            )
            Path(path).write_text(header + text, encoding="utf-8")
            self._status_var.set(f"Saved: {Path(path).name}")


# ─────────────────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Volumen — Ancient Papyrus Reader")
        self.geometry("1200x780")
        self.minsize(900, 620)
        self._cfg = load_config()
        self._image_path: str | None = None
        self._pipeline_img = None   # immutable output of run_pipeline
        self._processed_img = None  # current displayed image (may be clarified)
        self._shared_zoom: float = 1.0
        self._section_mode_active: bool = False
        self._build_ui()

    # ─── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        # Top toolbar
        toolbar = ctk.CTkFrame(self, height=50, corner_radius=0)
        toolbar.pack(fill="x", side="top")
        toolbar.pack_propagate(False)

        ctk.CTkButton(toolbar, text="Open Image", width=130,
                      command=self._open_image).pack(side="left", padx=8, pady=8)
        ctk.CTkButton(toolbar, text="⚙  Settings", width=120, fg_color="gray30",
                      command=self._open_settings).pack(side="left", padx=4, pady=8)
        self._process_btn = ctk.CTkButton(toolbar, text="▶  Process", width=120,
                                          state="disabled", command=self._run_processing)
        self._process_btn.pack(side="left", padx=4, pady=8)
        self._analyze_btn = ctk.CTkButton(toolbar, text="🔍  AI Analyze", width=130,
                                          state="disabled", fg_color="#1a6b3c",
                                          hover_color="#145530",
                                          command=self._run_analysis)
        self._analyze_btn.pack(side="left", padx=4, pady=8)
        self._detect_btn = ctk.CTkButton(
            toolbar, text="✦  Detect Characters", width=160,
            state="disabled", fg_color="#4a2080", hover_color="#361660",
            command=self._run_detect_characters,
        )
        self._detect_btn.pack(side="left", padx=4, pady=8)
        self._clarify_btn = ctk.CTkButton(
            toolbar, text="🔬  Clarify", width=110,
            state="disabled", fg_color="#5c3d00", hover_color="#422b00",
            command=self._run_clarify,
        )
        self._clarify_btn.pack(side="left", padx=4, pady=8)
        self._section_btn = ctk.CTkButton(
            toolbar, text="⬚  Section", width=110,
            state="disabled", fg_color="#1a4d6e", hover_color="#143a52",
            command=self._toggle_section_mode,
        )
        self._section_btn.pack(side="left", padx=4, pady=8)
        self._file_label = ctk.CTkLabel(toolbar, text="No file loaded", text_color="gray",
                                        font=ctk.CTkFont(size=11))
        self._file_label.pack(side="left", padx=12)

        # View controls row (zoom + save processed)
        view_row = ctk.CTkFrame(self, height=34, corner_radius=0, fg_color="#161616")
        view_row.pack(fill="x", side="top")
        view_row.pack_propagate(False)
        ctk.CTkLabel(view_row, text="Zoom:", font=ctk.CTkFont(size=11),
                     text_color="gray").pack(side="left", padx=(10, 2), pady=4)
        ctk.CTkButton(view_row, text="−", width=28, height=24,
                      command=self._zoom_out).pack(side="left", padx=1, pady=4)
        self._zoom_label = ctk.CTkLabel(
            view_row, text="1.0×", width=46, font=ctk.CTkFont(size=11)
        )
        self._zoom_label.pack(side="left", padx=2, pady=4)
        ctk.CTkButton(view_row, text="+", width=28, height=24,
                      command=self._zoom_in).pack(side="left", padx=1, pady=4)
        ctk.CTkButton(view_row, text="Fit", width=36, height=24, fg_color="gray30",
                      command=self._zoom_fit).pack(side="left", padx=(4, 14), pady=4)
        ctk.CTkLabel(view_row, text="Scroll wheel to zoom · Right-drag to pan",
                     font=ctk.CTkFont(size=10), text_color="gray40").pack(side="left", pady=4)
        ctk.CTkButton(
            view_row, text="💾  Save Processed", width=150, height=24,
            fg_color="gray30", command=self._save_processed_image,
        ).pack(side="right", padx=8, pady=4)

        # Main content
        paned = ctk.CTkFrame(self)
        paned.pack(expand=True, fill="both", padx=6, pady=(0, 4))

        # Left: image panels
        left = ctk.CTkFrame(paned)
        left.pack(side="left", expand=True, fill="both")

        images_frame = ctk.CTkFrame(left)
        images_frame.pack(expand=True, fill="both", padx=4, pady=4)
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)

        self._panel_original = ClickableImagePanel(
            images_frame, "Original",
            on_zoom_change=self._on_panel_zoom_change,
        )
        self._panel_original.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)

        self._panel_processed = ClickableImagePanel(
            images_frame, "Processed — click a highlighted region to research it",
            on_region_click=self._on_character_click,
            on_zoom_change=self._on_panel_zoom_change,
        )
        self._panel_processed.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        # Register section-select callbacks on both panels
        self._panel_original.set_section_callback(self._on_section_selected)
        self._panel_processed.set_section_callback(self._on_section_selected)

        # Right: AI results
        right = ctk.CTkFrame(paned, width=380)
        right.pack(side="right", fill="both", padx=(0, 4), pady=4)
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="AI Analysis – Transcription & Translation",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=8, pady=(6, 2))

        self._result_textbox = ctk.CTkTextbox(right, wrap="word", font=ctk.CTkFont(size=12))
        self._result_textbox.pack(expand=True, fill="both", padx=6, pady=(0, 4))
        self._result_textbox.insert("end", "Run 'AI Analyze' after processing to see results here.")
        self._result_textbox.configure(state="disabled")

        # Save result button
        save_row = ctk.CTkFrame(right, fg_color="transparent")
        save_row.pack(fill="x", padx=6, pady=(0, 6))
        ctk.CTkButton(save_row, text="Save Analysis", fg_color="gray30",
                      command=self._save_analysis).pack(side="right")

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        status_bar = ctk.CTkLabel(self, textvariable=self._status_var,
                                  anchor="w", font=ctk.CTkFont(size=11),
                                  text_color="gray")
        status_bar.pack(fill="x", side="bottom", padx=10, pady=(0, 4))

    # ─── Actions ────────────────────────────────────────────────────────────
    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select papyrus image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._image_path = path
        self._file_label.configure(text=Path(path).name, text_color="white")
        self._status("Image loaded.")
        self._process_btn.configure(state="normal")
        self._analyze_btn.configure(state="disabled")
        self._detect_btn.configure(state="disabled")
        self._clarify_btn.configure(state="disabled")
        self._section_btn.configure(state="normal")
        if self._section_mode_active:
            self._section_mode_active = False
            self._section_btn.configure(text="⬚  Section", fg_color="#1a4d6e")
            self._panel_original.set_mode("click")
            self._panel_processed.set_mode("click")
        self._pipeline_img = None
        self._processed_img = None
        self._panel_processed.set_overlay_regions([])

        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.after(100, lambda: self._panel_original.set_image(raw))
        self._panel_processed.set_image(None)
        self._set_result_text("Run 'Process' then 'AI Analyze'.")

    def _open_settings(self):
        SettingsWindow(self, self._cfg, on_save=self._on_settings_saved)

    def _on_settings_saved(self, new_cfg: dict):
        self._cfg = new_cfg
        save_config(new_cfg)
        self._status("Settings saved.")

    def _run_processing(self):
        if not self._image_path:
            return
        self._process_btn.configure(state="disabled")
        self._analyze_btn.configure(state="disabled")
        self._clarify_btn.configure(state="disabled")
        self._status("Processing…")

        def _worker():
            try:
                final, steps = run_pipeline(
                    self._image_path,
                    self._cfg,
                    progress_callback=lambda m: self.after(0, lambda m=m: self._status(m)),
                    output_dir=str(OUTPUT_DIR),
                )
                self._pipeline_img = final
                self._processed_img = final
                self.after(0, lambda: self._panel_processed.set_image(final))
                self.after(0, lambda: self._process_btn.configure(state="normal"))
                self.after(0, lambda: self._analyze_btn.configure(state="normal"))
                self.after(0, lambda: self._detect_btn.configure(state="normal"))
                self.after(0, lambda: self._clarify_btn.configure(state="normal"))
                self.after(0, lambda: self._status(
                    "Processing complete. Click '🔬 Clarify', '❖ Detect Characters', or 'AI Analyze'."
                ))
            except Exception as e:
                self.after(0, lambda: self._status(f"Processing error: {e}"))
                self.after(0, lambda: self._process_btn.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_analysis(self):
        if self._processed_img is None:
            messagebox.showwarning("No processed image", "Please process an image first.")
            return
        self._analyze_btn.configure(state="disabled")
        self._process_btn.configure(state="disabled")
        self._detect_btn.configure(state="disabled")
        self._set_result_text("")  # clear for live streaming
        self._status("Streaming AI analysis…")

        def _token(t):
            self.after(0, lambda t=t: self._append_result(t))

        def _done():
            self.after(0, lambda: self._analyze_btn.configure(state="normal"))
            self.after(0, lambda: self._process_btn.configure(state="normal"))
            self.after(0, lambda: self._detect_btn.configure(state="normal"))
            self.after(0, lambda: self._status("AI analysis complete."))

        threading.Thread(
            target=analyze_image_stream,
            args=(self._processed_img, self._cfg, _token, _done),
            daemon=True,
        ).start()

    def _save_analysis(self):
        text = self._result_textbox.get("1.0", "end").strip()
        if not text or text.startswith("Run "):
            messagebox.showinfo("Nothing to save", "No analysis results to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save analysis",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self._status(f"Analysis saved to {Path(path).name}")

    def _run_detect_characters(self):
        if self._processed_img is None:
            messagebox.showwarning("No processed image", "Please process an image first.")
            return
        self._detect_btn.configure(state="disabled")
        self._status("Detecting character regions…")

        def _worker():
            regions = find_character_regions(self._processed_img)
            self.after(0, lambda: self._panel_processed.set_overlay_regions(regions))
            self.after(0, lambda: self._detect_btn.configure(state="normal"))
            self.after(0, lambda: self._status(
                f"{len(regions)} regions detected. Click any green box to research it."
            ))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_clarify(self):
        if self._pipeline_img is None:
            messagebox.showwarning("No processed image", "Process an image first.")
            return
        self._clarify_btn.configure(state="disabled")
        self._status("Clarifying characters…")

        def _worker():
            c_cfg = self._cfg.get("clarification", {})
            result, stats = clarify_characters(self._pipeline_img, c_cfg)
            self._processed_img = result
            sb = stats["sharpness_before"]
            sa = stats["sharpness_after"]
            pct = ((sa - sb) / max(sb, 1e-6)) * 100
            if stats["fallback_used"]:
                msg = (
                    f"🔄 Blurry result detected — auto-recovered with blur-safe fallback. "
                    f"Sharpness: {sb:.0f} → {sa:.0f} "
                    f"({'\u2191' if sa > sb else '\u2193'}{abs(pct):.0f}%)"
                )
            else:
                arrow = "\u2191" if sa > sb else ("\u2193" if sa < sb else "=")
                msg = (
                    f"Clarification complete. Sharpness: {sb:.0f} → {sa:.0f} "
                    f"({arrow}{abs(pct):.0f}%) · Re-run Detect Characters to refresh."
                )
            self.after(0, lambda: self._panel_processed.set_image(result))
            self.after(0, lambda: self._panel_processed.set_overlay_regions([]))
            self.after(0, lambda: self._clarify_btn.configure(state="normal"))
            self.after(0, lambda m=msg: self._status(m))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_character_click(self, idx: int, region: dict, region_img: np.ndarray):
        CharacterDetailWindow(self, region_img, idx, self._cfg)

    def _toggle_section_mode(self):
        self._section_mode_active = not self._section_mode_active
        if self._section_mode_active:
            self._section_btn.configure(text="⬚  Section ●", fg_color="#8b2000")
            self._panel_original.set_mode("select")
            self._panel_processed.set_mode("select")
            self._status(
                "Section mode ON — drag a yellow rectangle on either panel to analyse a section."
            )
        else:
            self._section_btn.configure(text="⬚  Section", fg_color="#1a4d6e")
            self._panel_original.set_mode("click")
            self._panel_processed.set_mode("click")
            self._status("Section mode OFF.")

    def _on_section_selected(self, section_img: np.ndarray, rect: dict):
        SectionContextWindow(self, section_img, rect, self._cfg)

    def _zoom_in(self):
        self._set_zoom(self._shared_zoom * 1.4)

    def _zoom_out(self):
        self._set_zoom(self._shared_zoom / 1.4)

    def _zoom_fit(self):
        self._shared_zoom = 1.0
        self._zoom_label.configure(text="1.0×")
        self._panel_original.reset_view()
        self._panel_processed.reset_view()

    def _set_zoom(self, zoom: float):
        self._shared_zoom = max(1.0, min(zoom, 20.0))
        self._zoom_label.configure(text=f"{self._shared_zoom:.1f}×")
        self._panel_original.set_zoom(self._shared_zoom)
        self._panel_processed.set_zoom(self._shared_zoom)

    def _on_panel_zoom_change(self, zoom: float):
        """One panel scrolled — sync the other and update label."""
        self._shared_zoom = max(1.0, min(zoom, 20.0))
        self._zoom_label.configure(text=f"{self._shared_zoom:.1f}×")
        # Use set_zoom (no callback) to avoid infinite loop
        self._panel_original.set_zoom(self._shared_zoom)
        self._panel_processed.set_zoom(self._shared_zoom)

    def _save_processed_image(self):
        if self._processed_img is None:
            messagebox.showinfo("No processed image", "Process an image first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=[
                ("PNG (lossless)", "*.png"),
                ("TIFF", "*.tif"),
                ("JPEG", "*.jpg"),
                ("All files", "*.*"),
            ],
        )
        if path:
            success = cv2.imwrite(path, self._processed_img)
            if success:
                self._status(f"Saved: {Path(path).name}")
            else:
                messagebox.showerror("Save failed", f"Could not write to:\n{path}")

    # ─── Helpers ────────────────────────────────────────────────────────────
    def _status(self, msg: str):
        self._status_var.set(msg)

    def _append_result(self, token: str):
        self._result_textbox.configure(state="normal")
        self._result_textbox.insert("end", token)
        self._result_textbox.see("end")
        self._result_textbox.configure(state="disabled")

    def _set_result_text(self, text: str):
        self._result_textbox.configure(state="normal")
        self._result_textbox.delete("1.0", "end")
        if text:
            self._result_textbox.insert("end", text)
        self._result_textbox.configure(state="disabled")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
