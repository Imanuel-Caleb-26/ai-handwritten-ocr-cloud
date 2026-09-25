"""
Image preprocessing pipeline for handwritten OCR.
Handles diverse handwriting inputs: photos, scans, notes.
"""

import cv2
import numpy as np
from PIL import Image


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct tilt in handwritten page.
    Uses Hough line detection to estimate skew angle.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    # Edge detect
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return image  # No lines found, return as-is

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 < angle < 30:  # Only near-horizontal lines
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image  # Already straight enough

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """
    Remove uneven lighting and background variations.
    Uses morphological top-hat to normalize background.
    """
    # Large kernel for background estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    # Subtract background
    diff  = cv2.absdiff(gray, bg)
    # Invert so text is bright
    normalized = 255 - diff
    # Normalize to [0, 255]
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full image preprocessing pipeline for handwriting OCR.

    Steps:
    1. Load & standardize resolution
    2. Deskew (correct tilt)
    3. Normalize background
    4. CLAHE contrast enhancement
    5. Bilateral filter (preserve edges)
    6. Adaptive threshold → binary image

    Returns a binary (0/255) threshold image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # ── 1. Standardize resolution ──
    h, w = img.shape[:2]
    target_w = 2000
    if w != target_w:
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # ── 2. Deskew ──
    img = deskew(img)

    # ── 3. Grayscale ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 4. Background normalization ──
    gray = normalize_background(gray)

    # ── 5. Bilateral filter (noise reduction, edge preserving) ──
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # ── 6. CLAHE for contrast ──
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(filtered)

    # ── 7. Adaptive threshold ──
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 12,
    )

    # ── 8. Morphological cleanup ──
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


def get_line_segments(thresh_img: np.ndarray):
    """
    Horizontal Projection Segmentation with improved robustness.

    Pipeline:
    1. Horizontal dilation to bridge character gaps.
    2. Compute horizontal projection (row sums).
    3. Gaussian smooth projection curve.
    4. Find content regions (peaks in projection).
    5. Merge tiny fragments; filter noise.

    Returns (upper_bounds, lower_bounds) as pixel row indices.
    """
    h, w = thresh_img.shape[:2]

    # ── Horizontal dilation ──
    # Width = 10% of image width to bridge word gaps
    dil_w = max(50, int(w * 0.10))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_w, 1))
    dilated = cv2.dilate(thresh_img, kernel_h, iterations=1)

    # ── Horizontal projection ──
    projection = np.sum(dilated, axis=1).astype(float)

    # ── Gaussian smoothing (replaces box filter) ──
    win_size = max(5, int(h * 0.015))
    if win_size % 2 == 0:
        win_size += 1
    sigma = win_size / 3.0
    projection_smooth = cv2.GaussianBlur(
        projection.reshape(1, -1).astype(np.float32),
        (1, win_size),
        sigma,
    ).reshape(-1)

    # ── Normalize ──
    max_val = np.max(projection_smooth)
    if max_val == 0:
        return [], []
    projection_norm = projection_smooth / max_val

    # ── Content detection ──
    gap_threshold = 0.02  # slightly higher than before to filter noise
    is_content = projection_norm > gap_threshold

    upper_bounds = []
    lower_bounds = []
    in_line = False

    for i, val in enumerate(is_content):
        if not in_line and val:
            upper_bounds.append(i)
            in_line = True
        elif in_line and not val:
            lower_bounds.append(i)
            in_line = False

    if in_line:
        lower_bounds.append(h)

    # ── Refine: merge fragments, filter noise ──
    refined_u = []
    refined_l = []

    min_line_height = max(10, int(h * 0.015))   # at least 1.5% of page height
    max_gap_to_merge = max(8,  int(h * 0.025))  # merge gaps < 2.5% of page height

    for u, l in zip(upper_bounds, lower_bounds):
        # Skip tiny noise blobs
        if (l - u) < min_line_height:
            continue

        if not refined_u:
            refined_u.append(u)
            refined_l.append(l)
        else:
            gap = u - refined_l[-1]
            if gap < max_gap_to_merge:
                # Merge with previous line
                refined_l[-1] = l
            else:
                refined_u.append(u)
                refined_l.append(l)

    return refined_u, refined_l
