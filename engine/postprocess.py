"""Image post-processing: Auto-crop, Scaling, Alpha Feathering, and Defringing."""

from typing import Tuple
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter


def auto_crop_person(
    image: Image.Image,
    mask: np.ndarray,
    margin: int = 8,
    threshold: float = 0.03,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Tightly crops the image and mask around the person's bounding box.
    
    Args:
        image: PIL Image (RGB or RGBA).
        mask: 2D numpy array [0.0, 1.0] representing the alpha matte.
        margin: Extra padding pixels around the bounding box.
        threshold: Alpha threshold to consider a pixel as foreground.
        
    Returns:
        Tuple of (cropped_image, cropped_mask).
    """
    h, w = mask.shape
    fg_indices = np.argwhere(mask > threshold)

    if len(fg_indices) == 0:
        # No subject detected; return original
        return image, mask

    min_y = int(np.min(fg_indices[:, 0]))
    max_y = int(np.max(fg_indices[:, 0]))
    min_x = int(np.min(fg_indices[:, 1]))
    max_x = int(np.max(fg_indices[:, 1]))

    # Apply margin with clamping
    crop_x1 = max(0, min_x - margin)
    crop_y1 = max(0, min_y - margin)
    crop_x2 = min(w, max_x + margin + 1)
    crop_y2 = min(h, max_y + margin + 1)

    cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_image, cropped_mask


def scale_to_max_height(
    image: Image.Image,
    mask: np.ndarray,
    max_height: int = 340,
    force_multiple_of_4: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Resizes image and mask so height equals max_height (or is clamped to max_height),
    preserving aspect ratio and aligning width to a multiple of 4 for BC7 compression.
    
    Args:
        image: PIL Image.
        mask: 2D float numpy array [0.0, 1.0].
        max_height: Maximum height in pixels (default 340).
        force_multiple_of_4: Ensure width is divisible by 4 (BC7 requirement).
        
    Returns:
        Tuple of (scaled_image, scaled_mask).
    """
    w, h = image.size

    if h == 0 or w == 0:
        return image, mask

    # Proportional scaling to target max_height
    scale = max_height / float(h)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))

    if force_multiple_of_4:
        # Align width to nearest multiple of 4
        new_w = max(4, int(round(new_w / 4.0)) * 4)

    # High-quality Lanczos resampling for image
    scaled_image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # High-quality bilinear/bicubic resampling for mask
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = np.clip(scaled_mask, 0.0, 1.0)

    return scaled_image, scaled_mask


def defringe_rgb(
    rgb_arr: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """
    Decontaminates color fringing from background bleed at semi-transparent boundaries.
    Inpaints colors from solid foreground into semi-transparent edge pixels.
    """
    # Solid foreground mask
    solid_mask = (mask > 0.85).astype(np.uint8) * 255
    # Pixels where color may be contaminated by background
    needs_inpaint = ((mask > 0.01) & (mask <= 0.85)).astype(np.uint8) * 255

    # If almost nothing to inpaint or no solid foreground, return original
    if np.sum(needs_inpaint) == 0 or np.sum(solid_mask) == 0:
        return rgb_arr

    # Telea inpainting on RGB channels using solid foreground as source
    # The mask for cv2.inpaint indicates pixels that need to be reconstructed
    inpaint_mask = ((mask <= 0.85) & (mask > 0.0)).astype(np.uint8) * 255
    
    # Fast color diffusion/dilation for large boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_rgb = rgb_arr.copy()
    
    # Blend inpainted color near edges to preserve interior details
    try:
        inpainted = cv2.inpaint(rgb_arr, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        # Smoothly blend between inpainted RGB (at extreme edges) and original RGB (near solid core)
        blend_factor = np.clip((mask - 0.1) / 0.75, 0.0, 1.0)[:, :, np.newaxis]
        defringed = (rgb_arr * blend_factor + inpainted * (1.0 - blend_factor)).astype(np.uint8)
        return defringed
    except Exception:
        return rgb_arr


def soften_and_defringe(
    image: Image.Image,
    mask: np.ndarray,
    feather_radius: float = 1.5,
    defringe: bool = True,
) -> Image.Image:
    """
    Smooths person border using Gaussian alpha feathering and removes background color halos.
    
    Args:
        image: PIL Image (RGB).
        mask: 2D numpy array [0.0, 1.0].
        feather_radius: Gaussian sigma for softening edges (default 1.5).
        defringe: Whether to decontaminate edge RGB color halo.
        
    Returns:
        PIL Image in RGBA mode with softened border.
    """
    # Ensure RGB
    rgb_img = image.convert("RGB")
    rgb_np = np.array(rgb_img)

    # 1. Edge softening via Gaussian filter on alpha mask
    if feather_radius > 0.05:
        # Gaussian smoothing
        soft_mask = gaussian_filter(mask.astype(np.float32), sigma=feather_radius)
        soft_mask = np.clip(soft_mask, 0.0, 1.0)
    else:
        soft_mask = np.clip(mask.astype(np.float32), 0.0, 1.0)

    # 2. Defringe / Color decontamination
    if defringe:
        final_rgb = defringe_rgb(rgb_np, soft_mask)
    else:
        final_rgb = rgb_np

    # 3. Assemble RGBA channels
    alpha_channel = (soft_mask * 255.0).round().astype(np.uint8)
    rgba_np = np.dstack([final_rgb, alpha_channel])

    return Image.fromarray(rgba_np, mode="RGBA")
