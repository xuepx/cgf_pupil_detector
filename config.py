"""
Configuration for CGF (Confidence-Guided Fallback) pupil detection.
"""
import os

# --- Paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LPW Dataset ---
LPW_DIR = os.environ.get("LPW_DIR", os.path.join(PROJECT_DIR, "LPW"))

# --- RITNet Model ---
RITNET_WEIGHTS = os.environ.get(
    "RITNET_WEIGHTS",
    os.path.join(PROJECT_DIR, "best_model_finetune_v2_ep122_miou9524.pkl")
)
RITNET_INPUT_SIZE = (320, 240)  # (W, H)
RITNET_NUM_CLASSES = 2

# --- Preprocessing (must match RITNet training) ---
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE = (8, 8)
GAMMA = 0.8

# --- Geometry priors for pupil filtering ---
MIN_PUPIL_AREA = 100        # px^2 (at original resolution)
MAX_PUPIL_AREA = 50000      # px^2
MIN_CIRCULARITY = 0.5       # contour circularity threshold
MAX_ASPECT_RATIO = 2.5      # major/minor axis ratio

# --- Traditional method parameters ---
MORPH_KERNEL_SIZE = 5       # morphological kernel
ADAPTIVE_BLOCK_SIZE = 51    # must be odd
ADAPTIVE_C = 10             # constant subtracted from mean/gaussian

# --- CGF parameters ---
CGF_THRESHOLD_ACCURACY = 0.8   # accuracy mode threshold
CGF_THRESHOLD_STABILITY = 0.7  # stability mode threshold

# --- Confidence weights (AUROC-guided) ---
CONFIDENCE_WEIGHTS = {
    'circularity':  0.30,
    'aspect_ratio': 0.10,
    'area_score':   0.10,
    'convex_ratio': 0.50,
}
