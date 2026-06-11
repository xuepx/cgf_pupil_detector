"""
Confidence-Guided Fallback (CGF) Pupil Detector.

Main entry point for the CGF framework. Combines traditional detection
(Adaptive Mean) with geometric confidence scoring and conditional RITNet
fallback for robust real-time pupil localization.

Usage:
    from cgf_detector import CGFDetector

    detector = CGFDetector(weights_path="model.pkl", threshold=0.8)
    result = detector.detect(gray_frame)
    # result['center']    -> (cx, cy) pupil center
    # result['confidence'] -> float [0,1]
    # result['method']    -> "traditional" or "RITNet"
"""
import numpy as np
import cv2
import config
from traditional_methods import method_adaptive_mean
from confidence_score import compute_confidence
from pupil_geometry import extract_pupil_from_mask


class CGFDetector:
    """
    Confidence-Guided Fallback pupil detector.

    Pipeline:
      1. Preprocess frame (Gamma + CLAHE)
      2. Run Adaptive Mean traditional detection
      3. Compute geometric confidence score
      4. If confidence >= threshold: use traditional result
         Else: invoke RITNet deep segmentation
    """

    def __init__(self, weights_path=None, device=None, threshold=None):
        """
        Args:
            weights_path: path to fine-tuned RITNet .pkl weights
            device: 'cuda' or 'cpu' (auto-detected if None)
            threshold: confidence threshold tau (default: 0.8 accuracy mode)
        """
        if threshold is None:
            threshold = config.CGF_THRESHOLD_ACCURACY
        self.threshold = threshold

        # Preprocessing tables
        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** config.GAMMA)).astype(np.uint8)
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE
        )

        # Lazy-load RITNet
        self._ritnet = None
        self._weights_path = weights_path
        self._device = device

    def _ensure_ritnet(self):
        """Lazy-load RITNet model on first fallback."""
        if self._ritnet is None:
            from ritnet_inference import get_ritnet_model
            self._ritnet = get_ritnet_model(
                weights_path=self._weights_path, device=self._device
            )

    def preprocess(self, gray_frame):
        """Apply Gamma + CLAHE preprocessing."""
        img = cv2.LUT(gray_frame, self.gamma_table)
        img = self.clahe.apply(img)
        return img

    def detect(self, gray_frame):
        """
        Run CGF detection on a single grayscale frame.

        Args:
            gray_frame: np.ndarray (H, W), uint8 grayscale image

        Returns:
            dict with keys:
                'found': bool
                'center': (cx, cy) or None
                'diameter': float or NaN
                'ellipse': ellipse tuple or None
                'mask': binary mask or None
                'confidence': float [0, 1]
                'confidence_factors': dict of individual factors
                'method': 'traditional' or 'RITNet'
        """
        preprocessed = self.preprocess(gray_frame)

        # Step 1: Traditional detection
        trad_result = method_adaptive_mean(preprocessed)

        # Step 2: Confidence scoring
        conf = compute_confidence(trad_result, gray_frame)
        conf_score = conf['combined']

        # Step 3: Decision
        if trad_result['found'] and conf_score >= self.threshold:
            # High confidence: use traditional result
            return {
                'found': True,
                'center': trad_result['center'],
                'diameter': trad_result['diameter'],
                'ellipse': trad_result['ellipse'],
                'mask': trad_result['mask'],
                'confidence': conf_score,
                'confidence_factors': {
                    k: conf[k] for k in ['circularity', 'aspect_ratio', 'area_score', 'convex_ratio']
                },
                'method': 'traditional',
            }

        # Step 4: Low confidence or no detection -> RITNet fallback
        self._ensure_ritnet()
        ritnet_mask = self._ritnet.predict_mask(gray_frame)
        ritnet_result = extract_pupil_from_mask(ritnet_mask)

        return {
            'found': ritnet_result['found'],
            'center': ritnet_result['center'],
            'diameter': ritnet_result['diameter'],
            'ellipse': ritnet_result['ellipse'],
            'mask': ritnet_result['mask'],
            'confidence': conf_score,
            'confidence_factors': {
                k: conf[k] for k in ['circularity', 'aspect_ratio', 'area_score', 'convex_ratio']
            },
            'method': 'RITNet',
        }
