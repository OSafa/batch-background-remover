"""DDS Exporter module using Microsoft DirectXTex texconv.exe."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image


def get_texconv_path() -> str:
    """Find the path to texconv.exe."""
    base_dir = Path(__file__).parent.parent
    local_tool = base_dir / "tools" / "texconv.exe"
    if local_tool.exists():
        return str(local_tool)
    
    # Fallback to system PATH
    system_tool = os.environ.get("TEXCONV_PATH")
    if system_tool and os.path.exists(system_tool):
        return system_tool

    return "texconv.exe"


def export_to_bc7_dds(
    image: Image.Image,
    output_path: str,
    generate_mipmaps: bool = False,
    texconv_path: str | None = None,
) -> str:
    """
    Exports a PIL RGBA Image to a BC7 compressed .dds file.
    
    Args:
        image: PIL Image in 'RGBA' mode.
        output_path: Full destination path for the .dds file.
        generate_mipmaps: Whether to generate mipmaps (default False: 1 level).
        texconv_path: Optional custom path to texconv.exe.
    
    Returns:
        The absolute path to the generated .dds file.
    """
    if texconv_path is None:
        texconv_path = get_texconv_path()

    output_path_obj = Path(output_path).resolve()
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output has .dds extension
    final_dds_path = output_path_obj.with_suffix(".dds")

    # Save to a temporary PNG first with full alpha channel
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_png = Path(temp_dir) / "source.png"
        image.save(temp_png, format="PNG")

        # texconv command:
        # -f BC7_UNORM: format
        # -m 1: 1 mipmap level (no mipmaps) unless requested
        # -y: overwrite
        # -o <out_dir>: output directory
        mip_arg = "0" if generate_mipmaps else "1"
        cmd = [
            texconv_path,
            "-f", "BC7_UNORM",
            "-m", mip_arg,
            "-y",
            "-o", temp_dir,
            str(temp_png),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"texconv failed with return code {result.returncode}:\n{result.stderr}\n{result.stdout}"
            )

        generated_dds = Path(temp_dir) / "source.dds"
        if not generated_dds.exists():
            raise FileNotFoundError(
                f"Expected DDS file at {generated_dds} was not created. Output:\n{result.stdout}"
            )

        # Move to destination (shutil.move handles cross-drive moves safely)
        if final_dds_path.exists():
            final_dds_path.unlink()
        shutil.move(str(generated_dds), str(final_dds_path))

    return str(final_dds_path)
