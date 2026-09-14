"""Verification test for AI Torso Cropping + Background Removal + BC7 DDS pipeline."""

from pathlib import Path
from PIL import Image
from engine.pipeline import PortraitPipeline
import struct


def verify_dds(dds_path: Path):
    with open(dds_path, "rb") as f:
        data = f.read(148)
    assert data[:4] == b"DDS ", "Not a DDS file"
    header = data[4:128]
    height, width, pitch, depth, mip_count = struct.unpack("<5I", header[8:28])
    assert width % 4 == 0, f"Width {width} not divisible by 4"
    assert height <= 340, f"Height {height} > 340"
    assert mip_count == 1, f"Mip count {mip_count} != 1"
    print(f"Verified DDS: {dds_path.name} ({width}x{height} px, mip={mip_count})")


def main():
    print("=== Testing Torso Cropping & Background Removal Pipeline ===")
    test_img = Path("test_output/bus.jpg")
    assert test_img.exists(), "bus.jpg not found"

    pipeline = PortraitPipeline()

    # 1. Test auto-torso crop on full body
    out_dds_auto = Path("test_output/bus_torso_waist.dds")
    rgba_auto, mask_auto, dds_auto, crop_box_auto = pipeline.process_image(
        image_input=test_img,
        max_height=340,
        output_dds_path=out_dds_auto,
        auto_torso_crop=True,
        torso_preset="waist",
    )

    print(f"Auto-detected Torso Crop Box: {crop_box_auto}")
    print(f"Resulting Cutout size: {rgba_auto.size}")
    assert rgba_auto.height <= 340
    assert rgba_auto.width % 4 == 0
    rgba_auto.save("test_output/bus_torso_waist.png")
    verify_dds(out_dds_auto)

    # 2. Test manual custom crop box
    out_dds_manual = Path("test_output/bus_torso_manual.dds")
    custom_box = [60, 390, 200, 600]
    rgba_manual, mask_manual, dds_manual, crop_box_manual = pipeline.process_image(
        image_input=test_img,
        max_height=340,
        output_dds_path=out_dds_manual,
        crop_box=custom_box,
    )

    print(f"Manual Crop Box used: {crop_box_manual}")
    print(f"Resulting Manual Cutout size: {rgba_manual.size}")
    assert crop_box_manual == custom_box
    assert rgba_manual.height <= 340
    rgba_manual.save("test_output/bus_torso_manual.png")
    verify_dds(out_dds_manual)

    print("\nALL TORSO CROPPING PIPELINE TESTS PASSED!")


if __name__ == "__main__":
    main()
