"""FastAPI backend server for AI Portrait Background Remover & BC7 DDS Exporter."""

import io
import os
import sys
import uuid
import json
import base64
import zipfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

from engine.pipeline import PortraitPipeline
from engine.remover import DEFAULT_PORTRAIT_MODEL

app = FastAPI(title="AI Portrait Matting & BC7 DDS Exporter")

# Directories
BASE_DIR = Path(__file__).parent.resolve()
WEB_DIR = BASE_DIR / "web"
EXPORTS_DIR = BASE_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

# Pipeline instance
pipeline_instance: Optional[PortraitPipeline] = None

# In-memory item cache for live real-time slider and crop adjustments
# item_id -> { "orig_img": Image, "working_img": Image, "mask": np.ndarray, "stem": str, "crop_box": [x1, y1, x2, y2] }
IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}


def get_pipeline() -> PortraitPipeline:
    global pipeline_instance
    if pipeline_instance is None:
        pipeline_instance = PortraitPipeline(model_name=DEFAULT_PORTRAIT_MODEL)
    return pipeline_instance


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = WEB_DIR / "index.html"
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/system_info")
async def system_info():
    import torch
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "cuda_available": cuda_available,
        "device_name": gpu_name,
        "model": DEFAULT_PORTRAIT_MODEL,
        "has_pose_model": True,
    }


@app.post("/api/detect_crop")
async def detect_crop(
    file: Optional[UploadFile] = File(None),
    item_id: Optional[str] = Form(None),
    preset: str = Form("waist"),
):
    """
    Detects torso crop box using YOLO-Pose before processing.
    Accepts either an uploaded file or an item_id.
    """
    pipeline = get_pipeline()
    pil_img = None

    if item_id and item_id in IMAGE_CACHE and "orig_img" in IMAGE_CACHE[item_id]:
        pil_img = IMAGE_CACHE[item_id]["orig_img"]
    elif file is not None:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        if item_id:
            stem = Path(file.filename).stem if file.filename else f"portrait_{item_id}"
            IMAGE_CACHE[item_id] = {
                "orig_img": pil_img,
                "stem": stem,
            }
    else:
        raise HTTPException(status_code=400, detail="Must provide either file or valid item_id")

    res = pipeline.cropper.detect_torso(pil_img, preset=preset)
    if item_id and item_id in IMAGE_CACHE:
        IMAGE_CACHE[item_id]["crop_box"] = res["crop_box"]

    return res


@app.post("/api/process_single")
async def process_single(
    file: UploadFile = File(...),
    max_height: int = Form(340),
    margin: int = Form(8),
    feather_radius: float = Form(1.5),
    defringe: bool = Form(True),
    generate_mipmaps: bool = Form(False),
    output_folder: str = Form(""),
    item_id: str = Form(""),
    auto_torso_crop: bool = Form(True),
    torso_preset: str = Form("waist"),
    crop_box_json: str = Form(""),
):
    try:
        pipeline = get_pipeline()

        # Read uploaded image
        contents = await file.read()
        orig_img = Image.open(io.BytesIO(contents)).convert("RGB")

        if not item_id:
            item_id = str(uuid.uuid4())[:8]

        stem = Path(file.filename).stem if file.filename else f"portrait_{item_id}"
        dds_filename = f"{stem}_{max_height}h.dds"
        cached_dds_path = EXPORTS_DIR / dds_filename

        saved_local_path = None
        if output_folder and output_folder.strip():
            dest_dir = Path(output_folder.strip()).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved_local_path = dest_dir / f"{stem}.dds"

        target_dds = saved_local_path if saved_local_path else cached_dds_path

        # Parse manual crop box if passed
        parsed_crop_box = None
        if crop_box_json and crop_box_json.strip():
            try:
                parsed_crop_box = json.loads(crop_box_json)
            except Exception:
                pass

        # Run pipeline
        final_rgba, mask, dds_path, actual_crop_box = pipeline.process_image(
            image_input=orig_img,
            max_height=max_height,
            margin=margin,
            feather_radius=feather_radius,
            defringe=defringe,
            output_dds_path=target_dds,
            generate_mipmaps=generate_mipmaps,
            crop_box=parsed_crop_box,
            auto_torso_crop=auto_torso_crop,
            torso_preset=torso_preset,
        )

        # Cache images and mask for instant adjustments
        working_img = orig_img.crop(actual_crop_box)
        IMAGE_CACHE[item_id] = {
            "orig_img": orig_img,
            "working_img": working_img,
            "mask": mask,
            "stem": stem,
            "crop_box": actual_crop_box,
        }

        if saved_local_path and saved_local_path.exists():
            import shutil
            shutil.copy2(saved_local_path, cached_dds_path)

        w, h = final_rgba.size
        orig_w, orig_h = orig_img.size
        cutout_b64 = image_to_base64(final_rgba, format="PNG")
        alpha = final_rgba.split()[-1]
        mask_b64 = image_to_base64(alpha, format="PNG")

        normalized_crop_box = [
            round(actual_crop_box[0] / orig_w, 4),
            round(actual_crop_box[1] / orig_h, 4),
            round(actual_crop_box[2] / orig_w, 4),
            round(actual_crop_box[3] / orig_h, 4),
        ]

        return {
            "status": "success",
            "item_id": item_id,
            "cutout_data_url": cutout_b64,
            "mask_data_url": mask_b64,
            "width": w,
            "height": h,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "crop_box": actual_crop_box,
            "normalized_crop_box": normalized_crop_box,
            "dds_filename": dds_filename,
            "dds_download_url": f"/api/download_dds/{dds_filename}",
            "saved_local_path": str(saved_local_path) if saved_local_path else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ReprocessRequest(BaseModel):
    item_id: str
    max_height: int = 340
    margin: int = 8
    feather_radius: float = 1.5
    defringe: bool = True
    generate_mipmaps: bool = False
    output_folder: str = ""
    crop_box: Optional[List[int]] = None


@app.post("/api/reprocess")
async def reprocess(req: ReprocessRequest):
    """
    Sub-millisecond live update when changing height, sliders, or crop box.
    """
    if req.item_id not in IMAGE_CACHE:
        raise HTTPException(status_code=404, detail="Item not found in cache. Process it first.")

    try:
        pipeline = get_pipeline()
        cached = IMAGE_CACHE[req.item_id]
        orig_img = cached["orig_img"]
        stem = cached["stem"]
        orig_w, orig_h = orig_img.size

        # If user altered the crop box or working_img not yet processed
        if req.crop_box is not None and len(req.crop_box) == 4:
            new_box = [
                max(0, min(orig_w - 1, int(req.crop_box[0]))),
                max(0, min(orig_h - 1, int(req.crop_box[1]))),
                max(int(req.crop_box[0]) + 1, min(orig_w, int(req.crop_box[2]))),
                max(int(req.crop_box[1]) + 1, min(orig_h, int(req.crop_box[3]))),
            ]
            if new_box != cached.get("crop_box") or "mask" not in cached:
                working_img = orig_img.crop(new_box)
                mask = pipeline.remover.extract_mask(working_img)
                cached["working_img"] = working_img
                cached["mask"] = mask
                cached["crop_box"] = new_box
        elif "mask" not in cached:
            working_img = orig_img
            mask = pipeline.remover.extract_mask(working_img)
            cached["working_img"] = working_img
            cached["mask"] = mask
            cached["crop_box"] = [0, 0, orig_w, orig_h]

        working_img = cached["working_img"]
        mask = cached["mask"]
        actual_crop_box = cached.get("crop_box", [0, 0, orig_w, orig_h])

        dds_filename = f"{stem}_{req.max_height}h.dds"
        cached_dds_path = EXPORTS_DIR / dds_filename

        saved_local_path = None
        if req.output_folder and req.output_folder.strip():
            dest_dir = Path(req.output_folder.strip()).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            saved_local_path = dest_dir / f"{stem}.dds"

        target_dds = saved_local_path if saved_local_path else cached_dds_path

        final_rgba, saved_dds = pipeline.postprocess_from_mask(
            pil_img=working_img,
            mask=mask,
            max_height=req.max_height,
            margin=req.margin,
            feather_radius=req.feather_radius,
            defringe=req.defringe,
            output_dds_path=target_dds,
            generate_mipmaps=req.generate_mipmaps,
        )

        if saved_local_path and saved_local_path.exists():
            import shutil
            shutil.copy2(saved_local_path, cached_dds_path)

        w, h = final_rgba.size
        cutout_b64 = image_to_base64(final_rgba, format="PNG")
        alpha = final_rgba.split()[-1]
        mask_b64 = image_to_base64(alpha, format="PNG")

        normalized_crop_box = [
            round(actual_crop_box[0] / orig_w, 4),
            round(actual_crop_box[1] / orig_h, 4),
            round(actual_crop_box[2] / orig_w, 4),
            round(actual_crop_box[3] / orig_h, 4),
        ]

        return {
            "status": "success",
            "item_id": req.item_id,
            "cutout_data_url": cutout_b64,
            "mask_data_url": mask_b64,
            "width": w,
            "height": h,
            "crop_box": actual_crop_box,
            "normalized_crop_box": normalized_crop_box,
            "dds_filename": dds_filename,
            "dds_download_url": f"/api/download_dds/{dds_filename}",
            "saved_local_path": str(saved_local_path) if saved_local_path else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download_dds/{filename}")
async def download_dds(filename: str):
    dds_file = EXPORTS_DIR / filename
    if not dds_file.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(dds_file),
        filename=filename,
        media_type="application/octet-stream",
    )


class ZipExportRequest(BaseModel):
    filenames: List[str]


@app.post("/api/export_batch_zip")
async def export_batch_zip(req: ZipExportRequest):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for fname in req.filenames:
            file_path = EXPORTS_DIR / fname
            if file_path.exists():
                zip_file.write(file_path, arcname=fname)

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=portraits_bc7_batch.zip"},
    )


def start():
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    start()
