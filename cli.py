"""Command-line interface for AI portrait background removal, torso cropping, and BC7 DDS conversion."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from engine.remover import DEFAULT_PORTRAIT_MODEL

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def collect_images(input_path: Path) -> List[Path]:
    """Collects all valid image files from an input path or directory."""
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTENSIONS:
            return [input_path]
        else:
            print(f"Warning: {input_path.name} does not have a supported image extension.")
            return []
    elif input_path.is_dir():
        found = []
        for ext in VALID_EXTENSIONS:
            found.extend(input_path.glob(f"*{ext}"))
            found.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(list(set(found)))
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract torso portrait using YOLO-Pose, remove background with BiRefNet, and export as BC7 DDS."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to an input image or folder of images.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Destination directory for output .dds files.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=340,
        help="Maximum height in pixels for the output portrait (default: 340).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=8,
        help="Padding margin around person bounding box in pixels (default: 8).",
    )
    parser.add_argument(
        "--feather",
        type=float,
        default=1.5,
        help="Gaussian blur sigma for smoothing person border (default: 1.5).",
    )
    parser.add_argument(
        "--no-torso-crop",
        action="store_true",
        help="Disable automatic AI torso cropping on full-body photos.",
    )
    parser.add_argument(
        "--torso-preset",
        type=str,
        default="waist",
        choices=["waist", "mid_thigh", "bust", "none"],
        help="Torso framing preset: 'waist' (default), 'mid_thigh', 'bust', or 'none'.",
    )
    parser.add_argument(
        "--no-defringe",
        action="store_true",
        help="Disable color defringing/decontamination.",
    )
    parser.add_argument(
        "--mipmaps",
        action="store_true",
        help="Generate mipmaps for the DDS file (default: false, single mip).",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save a transparent PNG alongside the DDS file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_PORTRAIT_MODEL,
        help=f"Hugging Face model ID (default: '{DEFAULT_PORTRAIT_MODEL}').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Auto-detects by default.",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        print(f"No valid images found in '{input_path}'.")
        sys.exit(0)

    print(f"==================================================")
    print(f"  AI Portrait Background Remover & BC7 DDS Exporter")
    print(f"==================================================")
    print(f"  Model:         {args.model}")
    print(f"  Torso Pre-Crop:{not args.no_torso_crop} (Preset: {args.torso_preset})")
    print(f"  Found {len(images)} image(s) to process.")
    print(f"  Max Height:    {args.max_height} px (width aligned to % 4)")
    print(f"  Crop Margin:   {args.margin} px")
    print(f"  Edge Feather:  {args.feather} px")
    print(f"  Defringe:      {not args.no_defringe}")
    print(f"  DDS Mipmaps:   {args.mipmaps}")
    print(f"  Output Dir:    {output_dir}")
    print(f"==================================================")

    # Initialize pipeline
    from engine.pipeline import PortraitPipeline
    pipeline = PortraitPipeline(model_name=args.model, device=args.device)

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, img_file in enumerate(images, 1):
        rel_time = time.time()
        print(f"[{idx}/{len(images)}] Processing {img_file.name}...", end="", flush=True)

        try:
            target_dds = output_dir / f"{img_file.stem}.dds"
            final_img, mask, dds_path, crop_box = pipeline.process_image(
                image_input=img_file,
                max_height=args.max_height,
                margin=args.margin,
                feather_radius=args.feather,
                defringe=not args.no_defringe,
                output_dds_path=target_dds,
                generate_mipmaps=args.mipmaps,
                auto_torso_crop=not args.no_torso_crop,
                torso_preset=args.torso_preset,
            )

            if args.save_png:
                target_png = output_dir / f"{img_file.stem}.png"
                final_img.save(target_png, format="PNG")

            w, h = final_img.size
            elapsed = time.time() - rel_time
            print(f" Done ({w}x{h} px, {elapsed:.2f}s) -> {target_dds.name}")
            success_count += 1

        except Exception as e:
            print(f" FAILED: {e}")
            fail_count += 1

    total_time = time.time() - start_time
    print(f"\nFinished processing {len(images)} images in {total_time:.2f}s.")
    print(f"Success: {success_count}, Failures: {fail_count}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
