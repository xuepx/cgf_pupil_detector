"""
Confidence scoring for pupil detection output.

Computes four geometric factors to assess whether the traditional method's
output is trustworthy. Each factor and the combined score are in [0, 1].

Factors:
  - circularity:    4*pi*area/perimeter^2  (1 = perfect circle)
  - aspect_ratio:   minor/major axis ratio  (1 = circle)
  - area_score:     log-normal around expected size (1 = ideal size)
  - convex_ratio:   contour_area / hull_area (1 = convex shape)
  - combined:       AUROC-guided weighted sum
"""
import numpy as np
import cv2
import config

# Expected pupil area for area_score (log-normal, empirical from LPW)
EXPECTED_AREA_LOG_MU = np.log(3000.0)
EXPECTED_AREA_LOG_SIGMA = 1.0


def compute_confidence(result, gray_frame=None):
    """
    Compute multi-factor confidence score from a detection result dict.

    Args:
        result: dict from extract_pupil_from_mask / method_xxx
                keys: found, mask, ellipse, diameter, center, area
        gray_frame: (unused, reserved for future use)

    Returns:
        dict with keys: found, circularity, aspect_ratio, area_score,
                        convex_ratio, combined
    """
    zero = {
        'found': 0.0, 'circularity': 0.0, 'aspect_ratio': 0.0,
        'area_score': 0.0, 'convex_ratio': 0.0, 'combined': 0.0,
    }
    if not result.get('found', False):
        return zero

    mask_255 = (result['mask'] > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return zero

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 1:
        return zero

    # Factor 1: Circularity
    perimeter = cv2.arcLength(cnt, True)
    circularity = min((4.0 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0.0

    # Factor 2: Aspect ratio (minor/major, 1 = circle)
    aspect_ratio = 0.0
    if result.get('ellipse') is not None:
        (cx, cy), (ax1, ax2), angle = result['ellipse']
        minor_ax, major_ax = min(ax1, ax2), max(ax1, ax2)
        if major_ax > 0:
            aspect_ratio = minor_ax / major_ax

    # Factor 3: Area score (log-normal)
    if area > 0:
        log_area = np.log(area)
        area_score = float(np.exp(
            -((log_area - EXPECTED_AREA_LOG_MU) ** 2) / (2 * EXPECTED_AREA_LOG_SIGMA ** 2)
        ))
    else:
        area_score = 0.0

    # Factor 4: Convex hull ratio
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convex_ratio = min(float(area / hull_area), 1.0) if hull_area > 0 else 0.0

    # Combined (AUROC-guided weights)
    factors = {
        'circularity': circularity,
        'aspect_ratio': aspect_ratio,
        'area_score': area_score,
        'convex_ratio': convex_ratio,
    }
    combined = sum(factors[k] * config.CONFIDENCE_WEIGHTS[k] for k in config.CONFIDENCE_WEIGHTS)

    return {'found': 1.0, **factors, 'combined': combined}
