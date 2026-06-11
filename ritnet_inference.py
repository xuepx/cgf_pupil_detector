"""
RITNet inference wrapper.
Loads the fine-tuned DenseNet2D model and produces binary pupil masks.
"""
import numpy as np
import cv2
import torch
from densenet import DenseNet2D
import config


class RITNetInference:
    """Wrapper for RITNet inference on single frames."""

    def __init__(self, weights_path=None, device=None):
        if weights_path is None:
            weights_path = config.RITNET_WEIGHTS
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_w, self.input_h = config.RITNET_INPUT_SIZE

        self.model = DenseNet2D(
            in_channels=1, out_channels=config.RITNET_NUM_CLASSES,
            channel_size=32, dropout=False, prob=0
        )
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_SIZE
        )
        self.gamma_table = (255.0 * (np.linspace(0, 1, 256) ** config.GAMMA)).astype(np.uint8)

    def preprocess(self, gray_frame):
        """Preprocess a grayscale frame for RITNet input. Returns tensor [1,1,H,W]."""
        resized = cv2.resize(gray_frame, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.LUT(resized, self.gamma_table)
        img = self.clahe.apply(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def predict_mask(self, gray_frame):
        """Run inference on a single grayscale frame. Returns binary mask (same size as input)."""
        orig_h, orig_w = gray_frame.shape[:2]
        tensor = self.preprocess(gray_frame).to(self.device)
        output = self.model(tensor)
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if (orig_w, orig_h) != (self.input_w, self.input_h):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return pred


def get_ritnet_model(weights_path=None, device=None):
    """Factory function to get a RITNet inference instance."""
    return RITNetInference(weights_path=weights_path, device=device)
