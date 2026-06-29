"""
processing.py — Papyrus image processing pipeline.

Steps:
  1. Load image
  2. Convert to grayscale
  3. Denoise  (Non-Local Means)
  4. Smooth   (Bilateral Filter)
  5. Enhance contrast (CLAHE)
  6. Save output
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Raises FileNotFoundError if path does not exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"OpenCV could not decode the file: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to 8-bit grayscale."""
    if img.ndim == 2:
        return img  # Already grayscale
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, h: float, template_window: int, search_window: int) -> np.ndarray:
    """Apply Non-Local Means denoising."""
    return cv2.fastNlMeansDenoising(
        gray,
        h=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


def smooth(denoised: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Apply Bilateral Filter smoothing while preserving edges."""
    return cv2.bilateralFilter(
        denoised,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space,
    )


def enhance_contrast(smoothed: np.ndarray, clip_limit: float, tile_grid_size: tuple) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tuple(tile_grid_size),
    )
    return clahe.apply(smoothed)


def run_pipeline(
    image_path: str,
    cfg: dict,
    progress_callback=None,
    output_dir: str = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Run the full processing pipeline.

    Returns
    -------
    final : np.ndarray
        The contrast-enhanced, ready-for-segmentation image.
    steps : dict[str, np.ndarray]
        Intermediate images keyed by step name.
    """
    p_cfg = cfg.get("processing", {})
    out_cfg = cfg.get("output", {})

    def _report(msg: str):
        if progress_callback:
            progress_callback(msg)

    steps: dict[str, np.ndarray] = {}

    _report("Loading image…")
    raw = load_image(image_path)
    steps["1_raw"] = raw.copy()

    _report("Converting to grayscale…")
    gray = to_grayscale(raw)
    steps["2_grayscale"] = gray.copy()

    _report("Denoising (Non-Local Means)…")
    denoised = denoise(
        gray,
        h=p_cfg.get("nlm_h", 10),
        template_window=p_cfg.get("nlm_template_window", 7),
        search_window=p_cfg.get("nlm_search_window", 21),
    )
    steps["3_denoised"] = denoised.copy()

    _report("Smoothing (Bilateral Filter)…")
    smoothed = smooth(
        denoised,
        d=p_cfg.get("bilateral_d", 9),
        sigma_color=p_cfg.get("bilateral_sigma_color", 75),
        sigma_space=p_cfg.get("bilateral_sigma_space", 75),
    )
    steps["4_smoothed"] = smoothed.copy()

    _report("Enhancing contrast (CLAHE)…")
    final = enhance_contrast(
        smoothed,
        clip_limit=p_cfg.get("clahe_clip_limit", 3.0),
        tile_grid_size=p_cfg.get("clahe_tile_grid_size", [8, 8]),
    )
    steps["5_enhanced"] = final.copy()

    r_cfg = cfg.get("refinement", {})
    if r_cfg.get("enabled", False):
        _report("Refining for legibility…")
        final = refine_for_legibility(final, r_cfg)
        steps["6_refined"] = final.copy()

    if out_cfg.get("save_intermediate_steps", False) and output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        base = Path(image_path).stem
        for name, img in steps.items():
            cv2.imwrite(str(out_path / f"{base}_{name}.png"), img)
        _report(f"Intermediate images saved to '{output_dir}'.")

    _report("Processing complete.")
    return final, steps


def find_character_regions(
    enhanced_img: np.ndarray,
    min_area: int = 80,
    max_area_ratio: float = 0.15,
) -> list[dict]:
    """
    Detect candidate word/character regions via morphological grouping of ink marks.

    Parameters
    ----------
    enhanced_img : np.ndarray
        8-bit grayscale output of the CLAHE step.
    min_area : int
        Minimum bounding-box area (pixels²) to keep.
    max_area_ratio : float
        Reject boxes larger than this fraction of the total image area.

    Returns
    -------
    List of {'x', 'y', 'w', 'h'} dicts in approximate reading order.
    """
    # Binarise: ink appears dark on the lighter papyrus background after CLAHE
    _, thresh = cv2.threshold(
        enhanced_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Horizontal dilation groups individual character strokes into word-level blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 4))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = enhanced_img.shape[:2]
    max_area = h_img * w_img * max_area_ratio

    regions: list[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect > 25 or aspect < 0.1:
            continue
        regions.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    # Natural reading order: row-by-row, then left-to-right within each row
    line_h = max(int(h_img * 0.04), 1)
    regions.sort(key=lambda r: (r["y"] // line_h, r["x"]))
    return regions


# ───────────────────────────────────────────────────────────────
# Legibility Refinement Functions
# ───────────────────────────────────────────────────────────────

def normalize_background(img: np.ndarray, blur_radius: int) -> np.ndarray:
    """
    Divide each pixel by a heavily blurred background estimate.
    Removes uneven illumination common in papyrus photography.
    """
    k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    k = max(k, 3)
    background = cv2.GaussianBlur(img, (k, k), sigmaX=k / 3.0)
    norm = img.astype(np.float32) / (background.astype(np.float32) + 1.0) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via a lookup table."""
    if abs(gamma - 1.0) < 0.01:
        return img
    lut = np.array(
        [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, lut)


def unsharp_mask(img: np.ndarray, strength: float, sigma: float) -> np.ndarray:
    """Sharpen ink strokes using an unsharp mask."""
    if strength < 0.01:
        return img
    sigma = max(sigma, 0.1)
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def blackhat_enhance(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Morphological black-hat: accentuates dark ink strokes against
    the lighter papyrus background, then blends back into the image.
    """
    k = max(kernel_size, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.addWeighted(img, 1.0, bh, 2.0, 0)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def refine_for_legibility(img: np.ndarray, r_cfg: dict) -> np.ndarray:
    """
    Apply optional papyrus-specific legibility refinements in order:
      1. Background normalisation (remove uneven illumination)
      2. Gamma correction
      3. Unsharp masking (stroke sharpening)
      4. Black-hat morphological ink enhancement
    """
    if r_cfg.get("normalize_background", False):
        img = normalize_background(img, int(r_cfg.get("bg_blur_radius", 51)))

    gamma = float(r_cfg.get("gamma", 1.0))
    img = gamma_correct(img, gamma)

    strength = float(r_cfg.get("unsharp_strength", 0.0))
    sigma = float(r_cfg.get("unsharp_sigma", 1.0))
    img = unsharp_mask(img, strength, sigma)

    if r_cfg.get("blackhat_enhance", False):
        img = blackhat_enhance(img, int(r_cfg.get("blackhat_kernel_size", 7)))

    return img


# ───────────────────────────────────────────────────────────────
# Character Clarification
# ───────────────────────────────────────────────────────────────

def measure_sharpness(img: np.ndarray) -> float:
    """
    Measure image sharpness via Laplacian variance.
    Higher = sharper / more detail. Returns 0.0 for empty images.
    """
    if img is None or img.size == 0:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _clarify_safe_fallback(img: np.ndarray, c_cfg: dict) -> np.ndarray:
    """
    Blur-safe sharpening fallback used when the main clarification pass
    degraded sharpness. Avoids all Gaussian-blur-heavy operations
    (DoG, median) and instead uses:
      • Morphological gradient subtraction  – darkens ink edges
      • Strong unsharp mask (tight σ=0.5)   – recovers fine stroke detail
      • Elevated tight CLAHE               – restores local contrast
    """
    result = img.copy()

    # Morphological gradient (dilation − erosion) gives an edge map;
    # subtracting it from the image darkens ink strokes without blurring.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
    result = np.clip(
        result.astype(np.int16) - gradient.astype(np.int16), 0, 255
    ).astype(np.uint8)

    # Strong unsharp mask with very tight sigma (0.5) targets fine detail
    um_s = min(float(c_cfg.get("unsharp_strength", 0.8)) * 2.0, 4.0)
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=0.5)
    result = np.clip(
        cv2.addWeighted(result, 1.0 + um_s, blurred, -um_s, 0), 0, 255
    ).astype(np.uint8)

    # Tight CLAHE with elevated clip limit
    clip = min(float(c_cfg.get("final_clahe_clip", 4.0)) * 1.4, 12.0)
    tile = tuple(int(x) for x in c_cfg.get("final_clahe_tile", [4, 4]))
    result = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(result)

    return result


def clarify_characters(img: np.ndarray, c_cfg: dict) -> tuple[np.ndarray, dict]:
    """
    Multi-pass character clarification pipeline with automatic sharpness recovery.

    Passes (applied in sequence):
      1. Black-hat stroke isolation  – pulls dark ink strokes forward
      2. DoG ridge sharpening       – emphasises strokes at character-width scale
      3. Unsharp mask               – general high-frequency edge boost
      4. Tight CLAHE                – squeezes out remaining micro-contrast
      5. Median cleanup             – removes sharpening halos and salt-pepper noise

    If the result is blurrier than the input (Laplacian variance ratio below
    min_sharpness_ratio) and auto_recover is enabled, the blur-safe fallback
    is applied instead.

    Returns
    -------
    result : np.ndarray
    stats  : dict with keys
               sharpness_before  – Laplacian variance of the input
               sharpness_after   – Laplacian variance of the returned image
               fallback_used     – True if the safe fallback was applied
               fallback_sharpness – variance after fallback, or None
    """
    sharpness_before = measure_sharpness(img)
    result = img.copy()

    # ── 1. Black-hat stroke isolation ─────────────────────────────────────
    k = int(c_cfg.get("stroke_kernel", 5))
    k = k if k % 2 == 1 else k + 1
    k = max(k, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
    sw = float(c_cfg.get("stroke_weight", 1.5))
    if sw > 0.0:
        result = np.clip(
            cv2.addWeighted(result, 1.0, blackhat, sw, 0), 0, 255
        ).astype(np.uint8)

    # ── 2. DoG ridge sharpening ────────────────────────────────────────
    s1 = max(0.1, float(c_cfg.get("dog_sigma1", 1.0)))
    s2 = max(s1 + 0.1, float(c_cfg.get("dog_sigma2", 2.0)))
    dw = float(c_cfg.get("dog_weight", 1.0))
    if dw > 0.0:
        b1 = cv2.GaussianBlur(result, (0, 0), sigmaX=s1).astype(np.int16)
        b2 = cv2.GaussianBlur(result, (0, 0), sigmaX=s2).astype(np.int16)
        dog = (b1 - b2) * dw
        result = np.clip(result.astype(np.int16) - dog, 0, 255).astype(np.uint8)

    # ── 3. Unsharp mask ──────────────────────────────────────────────
    um_s = float(c_cfg.get("unsharp_strength", 0.8))
    um_sig = max(0.1, float(c_cfg.get("unsharp_sigma", 1.0)))
    if um_s > 0.0:
        blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=um_sig)
        result = np.clip(
            cv2.addWeighted(result, 1.0 + um_s, blurred, -um_s, 0), 0, 255
        ).astype(np.uint8)

    # ── 4. Tight CLAHE ───────────────────────────────────────────────
    if c_cfg.get("final_clahe", True):
        clip = float(c_cfg.get("final_clahe_clip", 4.0))
        tile = tuple(int(x) for x in c_cfg.get("final_clahe_tile", [4, 4]))
        result = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(result)

    # ── 5. Median cleanup ────────────────────────────────────────────
    if c_cfg.get("cleanup_median", True):
        result = cv2.medianBlur(result, 3)

    # ── Auto-recovery: if result is blurrier, apply blur-safe fallback ──
    sharpness_after = measure_sharpness(result)
    fallback_used = False
    fallback_sharpness = None

    min_ratio = float(c_cfg.get("min_sharpness_ratio", 0.85))
    if (
        c_cfg.get("auto_recover", True)
        and sharpness_before > 0.0
        and sharpness_after / sharpness_before < min_ratio
    ):
        fallback = _clarify_safe_fallback(img, c_cfg)
        fallback_sharpness = measure_sharpness(fallback)
        fallback_used = True
        result = fallback
        sharpness_after = fallback_sharpness

    return result, {
        "sharpness_before":    sharpness_before,
        "sharpness_after":     sharpness_after,
        "fallback_used":       fallback_used,
        "fallback_sharpness":  fallback_sharpness,
    }
