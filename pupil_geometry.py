"""
Pupil geometry extraction: connected component analysis, ellipse fitting,
geometric prior filtering, and diameter computation from binary masks.
"""
import numpy as np
import cv2
import config


def extract_pupil_from_mask(binary_mask):
    """
    Extract pupil ellipse from a binary mask using connected component analysis
    and geometric prior filtering.

    Args:
        binary_mask: np.ndarray (H, W), uint8, 1=pupil, 0=background

    Returns:
        dict with keys:
            'found': bool - whether a valid pupil was found
            'mask': np.ndarray - refined mask (only the selected region)
            'ellipse': tuple or None - ((cx, cy), (major, minor), angle)
            'diameter': float or NaN - mean of major and minor axes
            'center': tuple or None - (cx, cy)
            'area': float - pixel area of the selected region
    """
    result = {
        'found': False,
        'mask': np.zeros_like(binary_mask),
        'ellipse': None,
        'diameter': float('nan'),
        'center': None,
        'area': 0.0,
    }

    mask = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_PUPIL_AREA or area > config.MAX_PUPIL_AREA:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < config.MIN_CIRCULARITY:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (minor_ax, major_ax), angle = ellipse
        if minor_ax == 0:
            continue
        aspect_ratio = major_ax / minor_ax
        if aspect_ratio > config.MAX_ASPECT_RATIO:
            continue
        candidates.append({
            'contour': cnt,
            'area': area,
            'circularity': circularity,
            'ellipse': ellipse,
            'aspect_ratio': aspect_ratio,
        })

    if not candidates:
        return result

    best = max(candidates, key=lambda c: c['area'])
    ellipse = best['ellipse']
    (cx, cy), (minor_ax, major_ax), angle = ellipse

    refined_mask = np.zeros_like(binary_mask)
    cv2.drawContours(refined_mask, [best['contour']], -1, 1, thickness=cv2.FILLED)

    result['found'] = True
    result['mask'] = refined_mask
    result['ellipse'] = ellipse
    result['diameter'] = (major_ax + minor_ax) / 2.0
    result['center'] = (cx, cy)
    result['area'] = best['area']

    return result
