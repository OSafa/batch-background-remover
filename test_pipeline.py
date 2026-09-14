"""Full automated verification test for BiRefNet-portrait and BC7 DDS pipeline."""

import sys
import struct
from pathlib import Path
from PIL import Image, ImageDraw


def create_synthetic_portrait(output_path: Path):
    """Creates a synthetic portrait image of a person against a textured background."""
    img = Image.new("RGB", (600, 800), color=(180, 210, 235))
    draw = ImageDraw.Draw(img)

    # Background texture stripes
    for y in range(0, 800, 20):
        draw.line([(0, y), (600, y)], fill=(170, 200, 225), width=2)

    # Person: Shoulders / Body
    draw.polygon([(150, 800), (220, 500), (380, 500), (450, 800)], fill=(40, 60, 100))
    # Person: Neck
    draw.rectangle([(275, 450), (325, 520)], fill=(220, 180, 150))
    # Person: Face / Head
    draw.ellipse([(230, 280), (370, 470)], fill=(230, 190, 160))
    # Person: Hair
    draw.ellipse([(220, 250), (380, 360)], fill=(50, 30, 20))
    # Person: Eyes
    draw.ellipse([(265, 360), (285, 375)], fill=(30, 30, 30))
    draw.ellipse([(315, 360), (335, 375)], fill=(30, 30, 30))
    # Person: Smile
    draw.arc([(275, 390), (325, 420)], start=0, end=180, fill=(180, 50, 50), width=3)

    img.save(output_path)
    print(f"Created synthetic portrait: {output_path} ({img.size})")


def verify_dds_header(dds_path: Path, expected_max_height: int = 340):
    """Parses DDS header to confirm format and mipmap count."""
    with open(dds_path, "rb") as f:
        data = f.read(148)

    magic = data[:4]
    assert magic == b"DDS ", f"Invalid DDS magic: {magic}"

    header = data[4:128]
    height, width, pitch, depth, mip_count = struct.unpack("<5I", header[8:28])
    pf_flags, pf_fourcc = struct.unpack("<2I", header[76:84])

    print(f"DDS Header Info ({dds_path.name}):")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Mipmap count: {mip_count}")
    print(f"  Width is multiple of 4: {width % 4 == 0}")
    print(f"  Height <= {expected_max_height}: {height <= expected_max_height}")

    assert width % 4 == 0, f"Width {width} is not a multiple of 4!"
    assert height <= expected_max_height, f"Height {height} exceeds {expected_max_height}!"
    assert mip_count == 1, f"Expected 1 mipmap, got {mip_count}"

    # Check DX10 extended header if fourcc is DX10 (0x30315844)
    if pf_fourcc == 0x30315844:
        dx10_header = data[128:148]
        dxgi_format = struct.unpack("<I", dx10_header[:4])[0]
        print(f"  DX10 DXGI Format: {dxgi_format} (98 = BC7_UNORM, 99 = BC7_UNORM_SRGB)")
        assert dxgi_format in (98, 99), f"Unexpected DXGI format: {dxgi_format}"

    print(f"DDS header verification PASSED for {dds_path.name}.")


def main():
    print("=== Step 1: Testing Pipeline with BiRefNet-portrait ===")
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)

    input_img_path = test_dir / "portrait_sample.png"
    create_synthetic_portrait(input_img_path)

    from engine.pipeline import PortraitPipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pipeline = PortraitPipeline(device=device)

    # 1. Default height (340px) test
    output_dds_340 = test_dir / "portrait_340h.dds"
    final_rgba, mask, saved_dds = pipeline.process_image(
        image_input=input_img_path,
        max_height=340,
        margin=8,
        feather_radius=1.5,
        defringe=True,
        output_dds_path=output_dds_340,
        generate_mipmaps=False,
    )

    print(f"Processed RGBA image size (340h): {final_rgba.size}")
    assert final_rgba.mode == "RGBA", "Output image is not RGBA"
    assert final_rgba.height <= 340, f"Height {final_rgba.height} > 340"
    assert final_rgba.width % 4 == 0, f"Width {final_rgba.width} not divisible by 4"
    assert output_dds_340.exists(), "Output DDS was not created"
    verify_dds_header(output_dds_340, expected_max_height=340)

    # Save PNG cutout for visual inspection
    final_rgba.save(test_dir / "portrait_cutout_340h.png")

    # 2. Dynamic height limit adjustment test (e.g. user changing to 256px in Web UI)
    print("\n=== Step 2: Testing Live Height Limit Change (from 340px -> 256px) ===")
    output_dds_256 = test_dir / "portrait_256h.dds"
    final_rgba_256, saved_dds_256 = pipeline.postprocess_from_mask(
        pil_img=Image.open(input_img_path).convert("RGB"),
        mask=mask,
        max_height=256,
        margin=8,
        feather_radius=1.5,
        defringe=True,
        output_dds_path=output_dds_256,
        generate_mipmaps=False,
    )

    print(f"Processed RGBA image size (256h): {final_rgba_256.size}")
    assert final_rgba_256.height <= 256, f"Height {final_rgba_256.height} > 256"
    assert final_rgba_256.width % 4 == 0, f"Width {final_rgba_256.width} not divisible by 4"
    assert output_dds_256.exists(), "256h DDS was not created"
    verify_dds_header(output_dds_256, expected_max_height=256)

    print("\nALL TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
