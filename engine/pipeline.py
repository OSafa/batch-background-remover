"""Complete portrait background removal and DDS export pipeline with AI Torso Cropping."""

from pathlib import Path
from typing import Optional, Union, Tuple, List
from PIL import Image
import numpy as np

from engine.remover import BackgroundRemover, DEFAULT_PORTRAIT_MODEL
from engine.cropper import TorsoCropper
from engine.postprocess import auto_crop_person, scale_to_max_height, soften_and_defringe
from engine.dds_exporter import export_to_bc7_dds


class PortraitPipeline:
    """Full processing pipeline for portrait background removal, torso cropping, and DDS export."""

    def __init__(
        self,
        model_name: str = DEFAULT_PORTRAIT_MODEL,
        device: Optional[str] = None,
    ):
        self.remover = BackgroundRemover(model_name=model_name, device=device)
        self.cropper = TorsoCropper(device=device)

    def process_image(
        self,
        image_input: Union[str, Path, Image.Image],
        max_height: int = 340,
        margin: int = 8,
        feather_radius: float = 1.5,
        defringe: bool = True,
        output_dds_path: Optional[Union[str, Path]] = None,
        generate_mipmaps: bool = False,
        crop_box: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        auto_torso_crop: bool = True,
        torso_preset: str = "waist",
    ) -> Tuple[Image.Image, np.ndarray, Optional[str], List[int]]:
        """
        Executes the full pipeline on a single image.
        
        Returns:
            Tuple of (final_rgba_pil_image, raw_mask, saved_dds_path_or_none, crop_box_used).
        """
        if isinstance(image_input, (str, Path)):
            orig_pil = Image.open(str(image_input)).convert("RGB")
        else:
            orig_pil = image_input.convert("RGB")

        orig_w, orig_h = orig_pil.size

        # 1. Torso Pre-Crop
        actual_crop_box = [0, 0, orig_w, orig_h]

        if crop_box is not None and len(crop_box) == 4:
            # Explicit user crop box from UI
            x1, y1, x2, y2 = [int(v) for v in crop_box]
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            x2 = max(x1 + 1, min(orig_w, x2))
            y2 = max(y1 + 1, min(orig_h, y2))
            actual_crop_box = [x1, y1, x2, y2]
            working_img = orig_pil.crop(actual_crop_box)
        elif auto_torso_crop and torso_preset != "none":
            # AI auto-detect torso
            torso_info = self.cropper.detect_torso(orig_pil, preset=torso_preset)
            if torso_info.get("is_full_body", False) or torso_preset != "waist":
                actual_crop_box = torso_info["crop_box"]
                working_img = orig_pil.crop(actual_crop_box)
            else:
                working_img = orig_pil
        else:
            working_img = orig_pil

        # 2. Extract Alpha Matte using BiRefNet
        mask = self.remover.extract_mask(working_img)

        # 3. Post-processing
        final_rgba, saved_dds = self.postprocess_from_mask(
            pil_img=working_img,
            mask=mask,
            max_height=max_height,
            margin=margin,
            feather_radius=feather_radius,
            defringe=defringe,
            output_dds_path=output_dds_path,
            generate_mipmaps=generate_mipmaps,
        )

        return final_rgba, mask, saved_dds, actual_crop_box

    def postprocess_from_mask(
        self,
        pil_img: Image.Image,
        mask: np.ndarray,
        max_height: int = 340,
        margin: int = 8,
        feather_radius: float = 1.5,
        defringe: bool = True,
        output_dds_path: Optional[Union[str, Path]] = None,
        generate_mipmaps: bool = False,
    ) -> Tuple[Image.Image, Optional[str]]:
        """
        Fast sub-millisecond postprocessing given an already extracted alpha mask.
        Used for instant live updates when the user adjusts the height limit or sliders.
        """
        # Auto-crop tightly to person bounding box with margin
        cropped_img, cropped_mask = auto_crop_person(pil_img, mask, margin=margin)

        # Proportional scale to max_height with width % 4 alignment
        scaled_img, scaled_mask = scale_to_max_height(
            cropped_img,
            cropped_mask,
            max_height=max_height,
            force_multiple_of_4=True,
        )

        # Soften person border & Defringe
        final_rgba = soften_and_defringe(
            scaled_img,
            scaled_mask,
            feather_radius=feather_radius,
            defringe=defringe,
        )

        # Export to BC7 DDS if requested
        saved_dds = None
        if output_dds_path:
            saved_dds = export_to_bc7_dds(
                final_rgba,
                str(output_dds_path),
                generate_mipmaps=generate_mipmaps,
            )

        return final_rgba, saved_dds
