"""
water.py

Pipeline 1: classical CV heuristic for detecting water terrain in
artistic/fantasy-style landscape (map) images.

Steps:
    - color prior: wide HSV hue-band threshold for blue/teal/cyan, unioned
      with a Lab b*-channel threshold (blueness independent of lightness,
      to catch near-black deep water and pale, low-saturation shallows
      that the HSV band alone would miss)
    - texture prior: local standard deviation filter — water tends to be
      smooth or finely/repetitively textured (ripples) relative to
      terrain's rougher brush texture, so low local std is kept
    - combine: AND the two masks, then morphological close+open and
      connected-component area filtering to keep only large, coherent
      blobs (water bodies are contiguous, not scattered pixels)
    - edge cue (optional/debug only): Canny edges, useful for visually
      checking how well the mask boundary lines up with painted
      coastline/riverbank strokes
"""

import os

import cv2
import numpy as np

OUTPUT_DIR = 'water_output'


def color_prior(image, hue_min=75, hue_max=150, sat_min=35, val_min=30, lab_b_max=118):
    """
    Wide color threshold for "water-like" blue/teal/cyan hues.

    Two independent criteria are OR'd together to cast a wide net:
    an HSV hue band catches saturated water colors at any brightness,
    while a Lab b* threshold catches blueness independent of lightness
    (HSV hue is unstable at low saturation/value, which is exactly where
    near-black deep water and pale, washed-out shallows live).

    Args:
        image (numpy.ndarray): BGR image.
        hue_min (int): lower bound of the OpenCV hue band (0-179) to keep.
        hue_max (int): upper bound of the OpenCV hue band (0-179) to keep.
        sat_min (int): minimum saturation (0-255) required for the hue
            band to count, to exclude near-gray pixels.
        val_min (int): minimum value/brightness (0-255) required for the
            hue band to count, to exclude near-black noise where hue is
            meaningless.
        lab_b_max (int): Lab b* is stored with a +128 offset in OpenCV, so
            128 is neutral and lower values are bluer; pixels at or below
            this are counted regardless of lightness or saturation.

    Returns:
        mask (numpy.ndarray): uint8 mask, 255 where the color prior holds.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hsv_mask = (h >= hue_min) & (h <= hue_max) & (s >= sat_min) & (v >= val_min)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    _, _, b = cv2.split(lab)
    lab_mask = b <= lab_b_max

    mask = hsv_mask | lab_mask
    return (mask.astype(np.uint8) * 255)


def local_std(gray, ksize=9):
    """
    Local standard deviation of a grayscale image via box-filtered
    moments: std = sqrt(E[x^2] - E[x]^2), computed over a ksize x ksize
    window centered on each pixel.

    Args:
        gray (numpy.ndarray): single-channel image.
        ksize (int): side length of the square averaging window.

    Returns:
        std (numpy.ndarray): float32 array, same shape as gray.
    """
    gray = gray.astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(ksize, ksize))
    mean_sq = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(ksize, ksize))
    variance = np.clip(mean_sq - mean * mean, 0, None)  # clip: box-filter rounding can push this slightly negative
    return np.sqrt(variance)


def texture_prior(image, ksize=9, std_max=18):
    """
    Keeps pixels in locally smooth (low local-variance) regions, to
    separate "smooth/finely-patterned blue" (water) from "rough blue"
    (e.g. blue-tinted mountains, shadows, brush-textured terrain).

    Args:
        image (numpy.ndarray): BGR image.
        ksize (int): local-std window size, forwarded to local_std.
        std_max (int): maximum local std (0-255 grayscale scale) counted
            as "smooth enough" to be water.

    Returns:
        mask (numpy.ndarray): uint8 mask, 255 where the texture prior holds.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = local_std(gray, ksize)
    mask = std <= std_max
    return (mask.astype(np.uint8) * 255)


def clean_mask(mask, close_ksize=15, min_area_frac=0.001):
    """
    Morphological closing (to bridge small gaps, e.g. reflections or
    ripple highlights breaking up a water body) followed by opening (to
    strip thin spurs and small speckle), then connected-component area
    filtering to drop anything too small to be a real water body.

    Args:
        mask (numpy.ndarray): uint8 binary mask (0/255).
        close_ksize (int): side length of the elliptical structuring
            element used for closing and opening.
        min_area_frac (float): minimum blob area, as a fraction of total
            image pixels, required to survive filtering.

    Returns:
        cleaned (numpy.ndarray): uint8 binary mask (0/255).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = min_area_frac * mask.shape[0] * mask.shape[1]

    cleaned = np.zeros_like(opened)
    for label in range(1, num_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def edge_cue(image, low=50, high=150):
    """
    Canny edge map, for visually checking whether the water mask boundary
    lines up with painted coastline/riverbank strokes. Not used to modify
    the mask itself — see module docstring.

    Args:
        image (numpy.ndarray): BGR image.
        low (int): Canny lower hysteresis threshold.
        high (int): Canny upper hysteresis threshold.

    Returns:
        edges (numpy.ndarray): uint8 single-channel edge map.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    return cv2.Canny(blurred, low, high)


def detect_water(image, debug=False):
    """
    Runs the full classical-CV water detection pipeline: color prior AND
    texture prior, then morphological cleanup and connected-component
    area filtering.

    Args:
        image (numpy.ndarray): BGR image.
        debug (bool): when True, writes intermediate masks, the Canny edge
            cue, and a contour overlay to disk. When False (the default)
            nothing touches disk.

    Returns:
        mask (numpy.ndarray): uint8 binary mask (0/255), 255 where water
            was detected.
    """
    color = color_prior(image)
    texture = texture_prior(image)
    combined = cv2.bitwise_and(color, texture)
    mask = clean_mask(combined)

    if debug:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_color_mask.jpg'), color)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_texture_mask.jpg'), texture)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_combined_raw.jpg'), combined)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_mask.jpg'), mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_edges.jpg'), edge_cue(image))

        overlay = image.copy()
        tint = np.full_like(overlay, (255, 180, 0))  # BGR: cyan-orange tint on water pixels
        water_pixels = mask > 0
        overlay[water_pixels] = cv2.addWeighted(overlay, 0.5, tint, 0.5, 0)[water_pixels]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'water_overlay.jpg'), overlay)

    return mask


# Running this file directly detects water in a single image and writes
# debug images (masks, edge cue, contour overlay) to disk, e.g.:
#   python water.py --image map2
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect water regions in a map image.")
    parser.add_argument("--image", default="map1", help="image filename, without extension, in the working directory (default: map1)")
    args = parser.parse_args()

    image = cv2.imread(f'{args.image}.jpg')
    if image is None:
        raise FileNotFoundError(f"Could not read '{args.image}.jpg'")

    mask = detect_water(image, debug=True)
    coverage = (mask > 0).mean()
    print(f"image={args.image} water_coverage={coverage:.3f}")
