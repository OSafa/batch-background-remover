"""AI Background Removal Engine using BiRefNet with CUDA acceleration."""

import os
from typing import Optional, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Global cached model instance
_CACHED_MODEL = None
_CACHED_MODEL_NAME = None
_CACHED_DEVICE = None

# ZhengPeng7/BiRefNet preserves the full human figure (hair, face, neck, torso, clothing)
DEFAULT_PORTRAIT_MODEL = "ZhengPeng7/BiRefNet"


def get_default_device() -> str:
    """Returns 'cuda' if NVIDIA GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_name: str = DEFAULT_PORTRAIT_MODEL,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Loads and caches the BiRefNet model.
    
    Args:
        model_name: Hugging Face model repository id.
        device: 'cuda' or 'cpu'. Defaults to auto-detect.
        dtype: Precision (torch.float32 for maximum numerical stability with deform_conv2d).
        
    Returns:
        Loaded PyTorch model in evaluation mode.
    """
    global _CACHED_MODEL, _CACHED_MODEL_NAME, _CACHED_DEVICE

    if device is None:
        device = get_default_device()

    if _CACHED_MODEL is not None and _CACHED_MODEL_NAME == model_name and _CACHED_DEVICE == device:
        return _CACHED_MODEL

    from transformers import AutoModelForImageSegmentation

    print(f"Loading background removal model '{model_name}' on {device.upper()}...")

    if dtype is None:
        dtype = torch.float32

    model = AutoModelForImageSegmentation.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    )

    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_MODEL_NAME = model_name
    _CACHED_DEVICE = device

    return model


class BackgroundRemover:
    """Wraps BiRefNet inference and provides clean image-to-mask processing."""

    def __init__(
        self,
        model_name: str = DEFAULT_PORTRAIT_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or get_default_device()
        self.model_name = model_name
        self.model = load_model(model_name=self.model_name, device=self.device)

        # Standard normalization for BiRefNet (ImageNet statistics)
        self.image_size = (1024, 1024)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def extract_mask(self, image: Image.Image) -> np.ndarray:
        """
        Runs background segmentation and returns the alpha matte [0.0, 1.0].
        
        Args:
            image: PIL Image (RGB).
            
        Returns:
            2D numpy array of shape (original_h, original_w) with values in [0.0, 1.0].
        """
        orig_w, orig_h = image.size
        rgb_image = image.convert("RGB")

        # Pre-process
        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        
        # Match model dtype (float32)
        model_dtype = next(self.model.parameters()).dtype
        input_tensor = input_tensor.to(dtype=model_dtype)

        # Model inference
        preds = self.model(input_tensor)
        
        # BiRefNet returns list/tuple of intermediate masks, final high-res mask is [-1]
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        else:
            pred = preds

        pred = pred.sigmoid().squeeze().cpu().float()

        # Resize matte back to original image size
        to_pil = transforms.ToPILImage()
        mask_pil = to_pil(pred).resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

        # Convert to numpy float [0.0, 1.0]
        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        return mask_np
