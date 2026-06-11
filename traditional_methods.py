"""
Traditional binarization methods for pupil detection.
Each method takes a preprocessed grayscale frame and returns a result dict.
"""
import numpy as np
import cv2
import config
from pupil_geometry import extract_pupil_from_mask


def _morphological_postprocess(binary_img):
    """Morphological open (denoise) + close (fill holes). Input/output: uint8 (0/255)."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
    )
    processed = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=2)
    return processed


def _to_pupil_mask(binary_255):
    """Convert 0/255 binary image to 0/1 mask with morphological filtering."""
    processed = _morphological_postprocess(binary_255)
    return (processed > 0).astype(np.uint8)


def method_otsu(gray_frame):
    """Global Otsu thresholding."""
    _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return extract_pupil_from_mask(_to_pupil_mask(binary))


def method_adaptive_mean(gray_frame):
    """Adaptive thresholding using mean of local neighborhood."""
    binary = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        blockSize=config.ADAPTIVE_BLOCK_SIZE, C=config.ADAPTIVE_C
    )
    return extract_pupil_from_mask(_to_pupil_mask(binary))


def method_adaptive_gaussian(gray_frame):
    """Adaptive thresholding using Gaussian-weighted sum."""
    binary = cv2.adaptiveThreshold(
        gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=config.ADAPTIVE_BLOCK_SIZE, C=config.ADAPTIVE_C
    )
    return extract_pupil_from_mask(_to_pupil_mask(binary))


def method_triangle(gray_frame):
    """Triangle thresholding algorithm."""
    _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    return extract_pupil_from_mask(_to_pupil_mask(binary))


def method_multi_otsu(gray_frame):
    """Multi-level Otsu (3 classes), take the darkest class as pupil."""
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(gray_frame, classes=3)
        binary = ((gray_frame < thresholds[0]) * 255).astype(np.uint8)
    except Exception:
        _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return extract_pupil_from_mask(_to_pupil_mask(binary))


METHODS = {
    "Otsu": method_otsu,
    "Adaptive_Mean": method_adaptive_mean,
    "Adaptive_Gaussian": method_adaptive_gaussian,
    "Triangle": method_triangle,
    "Multi_Otsu": method_multi_otsu,
}
