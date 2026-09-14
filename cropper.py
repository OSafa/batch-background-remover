"""AI Torso Detection and Cropping Engine using YOLO-Pose keypoints."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch

_CACHED_YOLO_MODEL = None


def get_yolo_pose_model(model_name: str = "yolo11n-pose.pt", device: Optional[str] = None):
    """Loads and caches the YOLO-Pose model on the target device."""
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is not None:
        return _CACHED_YOLO_MODEL

    from pathlib import Path
    from ultralytics import YOLO

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading YOLO-Pose model ({model_name}) on {device.upper()}...")

    # Resolve local weights in project root if available
    base_dir = Path(__file__).parent.parent
    local_weights = base_dir / model_name
    target_weights = str(local_weights) if local_weights.exists() else model_name

    model = YOLO(target_weights)
    _CACHED_YOLO_MODEL = model
    return model


class TorsoCropper:
    """Anatomically pinpoints upper body / torso using 17 pose keypoints."""

    def __init__(self, model_name: str = "yolo11n-pose.pt", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_yolo_pose_model(model_name=model_name, device=self.device)

    def detect_torso(
        self,
        image: Image.Image,
        preset: str = "waist",
        confidence_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Detects if lower body is present and computes an anatomical torso crop box.
        
        Args:
            image: Input PIL Image.
            preset: 'waist', 'mid_thigh', 'bust', or 'none'.
            confidence_threshold: Keypoint detection confidence threshold.
            
        Returns:
            dict containing:
                - is_full_body: bool (True if hips/legs are visible)
                - crop_box: [x1, y1, x2, y2] in pixel coords
                - normalized_box: [nx1, ny1, nx2, ny2] in [0, 1] relative coords
                - preset_used: str
        """
        w, h = image.size

        if preset == "none":
            return {
                "is_full_body": False,
                "crop_box": [0, 0, w, h],
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
                "preset_used": "none",
            }

        # Run inference
        results = self.model(image, device=self.device, verbose=False)
        if not results or len(results) == 0 or results[0].keypoints is None:
            # Fallback to full image if no pose detected
            return {
                "is_full_body": False,
                "crop_box": [0, 0, w, h],
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
                "preset_used": preset,
            }

        res = results[0]
        if res.keypoints.data is None or len(res.keypoints.data) == 0:
            return {
                "is_full_body": False,
                "crop_box": [0, 0, w, h],
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
                "preset_used": preset,
            }

        # If multiple persons detected, take the largest bounding box (main subject)
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        best_idx = 0
        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

        # 17 keypoints: [x, y, conf]
        kpts = res.keypoints.data[best_idx].cpu().numpy()

        # Keypoint indices
        # 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
        # 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
        # 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle

        # Check visibility of lower body
        hips_visible = (kpts[11, 2] > confidence_threshold) or (kpts[12, 2] > confidence_threshold)
        knees_visible = (kpts[13, 2] > confidence_threshold) or (kpts[14, 2] > confidence_threshold)
        ankles_visible = (kpts[15, 2] > confidence_threshold) or (kpts[16, 2] > confidence_threshold)

        is_full_body = bool(knees_visible or ankles_visible or (hips_visible and (kpts[11, 1] < h * 0.75 or kpts[12, 1] < h * 0.75)))

        # Collect valid head keypoints
        head_indices = [0, 1, 2, 3, 4]
        valid_head_pts = [kpts[i] for i in head_indices if kpts[i, 2] > confidence_threshold]

        # Collect shoulder keypoints
        shoulder_pts = [kpts[i] for i in [5, 6] if kpts[i, 2] > confidence_threshold]

        # If shoulders or head aren't clearly detected, return full image
        if not shoulder_pts and not valid_head_pts:
            return {
                "is_full_body": False,
                "crop_box": [0, 0, w, h],
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
                "preset_used": preset,
            }

        # Shoulder metrics
        if shoulder_pts:
            shoulder_y_avg = float(np.mean([pt[1] for pt in shoulder_pts]))
            shoulder_width = abs(kpts[5, 0] - kpts[6, 0]) if len(shoulder_pts) == 2 else w * 0.3
            shoulder_center_x = float(np.mean([pt[0] for pt in shoulder_pts]))
        else:
            shoulder_y_avg = h * 0.3
            shoulder_width = w * 0.3
            shoulder_center_x = w * 0.5

        # Determine top of head (combining YOLO detector box + anatomical keypoints + headroom)
        # 1. Keypoint anatomical projection:
        # Anatomically, distance from eyes to the skull crown is roughly 0.85-1.0x distance from shoulders to eyes,
        # or ~0.45x shoulder width.
        if valid_head_pts:
            face_y_min = min(pt[1] for pt in valid_head_pts)
            dist_shoulder_face = max(shoulder_y_avg - face_y_min, 30.0)
            kpt_head_top = face_y_min - max(dist_shoulder_face * 0.85, shoulder_width * 0.4)
        elif shoulder_pts:
            kpt_head_top = shoulder_y_avg - max(shoulder_width * 0.75, h * 0.2)
        else:
            kpt_head_top = 0.0

        # 2. YOLO Object Detection Bounding Box:
        # The YOLO person detector explicitly bounds the top of the hair, hats, and head.
        if boxes is not None and len(boxes) > best_idx:
            box_y1 = float(boxes[best_idx, 1])
            # Take the higher (smaller y) bound between detector box and keypoint projection
            head_top = min(box_y1, kpt_head_top)
        else:
            head_top = kpt_head_top

        # 3. Headroom Padding:
        # Provide proportional breathing room above the hair so the head is never clipped
        head_height = max(shoulder_y_avg - head_top, 40.0)
        head_margin = max(25.0, head_height * 0.15)
        crop_y1 = max(0, int(head_top - head_margin))

        # Determine bottom of torso based on preset
        hip_pts = [kpts[i] for i in [11, 12] if kpts[i, 2] > confidence_threshold]
        if hip_pts:
            hip_y_avg = float(np.mean([pt[1] for pt in hip_pts]))
            torso_length = max(hip_y_avg - shoulder_y_avg, 50.0)
        else:
            # Approximate torso length as 1.5x shoulder width
            torso_length = shoulder_width * 1.5
            hip_y_avg = shoulder_y_avg + torso_length

        if preset == "bust":
            # Mid-chest (45% down from shoulder to hip)
            crop_y2 = int(shoulder_y_avg + torso_length * 0.45)
        elif preset == "mid_thigh":
            # Mid-thigh
            knee_pts = [kpts[i] for i in [13, 14] if kpts[i, 2] > confidence_threshold]
            if knee_pts:
                knee_y_avg = float(np.mean([pt[1] for pt in knee_pts]))
                crop_y2 = int(hip_y_avg + (knee_y_avg - hip_y_avg) * 0.5)
            else:
                crop_y2 = int(hip_y_avg + torso_length * 0.6)
        else:  # 'waist' default
            # Just below hips/waist
            crop_y2 = int(hip_y_avg + torso_length * 0.15)

        crop_y2 = min(h, max(crop_y1 + 100, crop_y2))

        # Horizontal bounds: include upper body width with margin, arms/sleeves and YOLO bounding box
        upper_body_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        valid_x = [kpts[i, 0] for i in upper_body_indices if (kpts[i, 2] > confidence_threshold and kpts[i, 1] <= crop_y2 + 20)]

        if boxes is not None and len(boxes) > best_idx:
            box_x1 = float(boxes[best_idx, 0])
            box_x2 = float(boxes[best_idx, 2])
            body_min_x = min(valid_x + [box_x1]) if valid_x else box_x1
            body_max_x = max(valid_x + [box_x2]) if valid_x else box_x2
        elif valid_x:
            body_min_x = min(valid_x)
            body_max_x = max(valid_x)
        else:
            body_min_x = shoulder_center_x - shoulder_width * 0.8
            body_max_x = shoulder_center_x + shoulder_width * 0.8

        body_width = body_max_x - body_min_x
        pad_x = max(body_width * 0.12, 30.0)
        center_x = (body_min_x + body_max_x) / 2.0
        half_w = (body_width / 2.0) + pad_x

        crop_x1 = max(0, int(center_x - half_w))
        crop_x2 = min(w, int(center_x + half_w))

        # Ensure valid box
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return {
                "is_full_body": False,
                "crop_box": [0, 0, w, h],
                "normalized_box": [0.0, 0.0, 1.0, 1.0],
                "preset_used": preset,
            }

        return {
            "is_full_body": is_full_body,
            "crop_box": [crop_x1, crop_y1, crop_x2, crop_y2],
            "normalized_box": [
                round(crop_x1 / w, 4),
                round(crop_y1 / h, 4),
                round(crop_x2 / w, 4),
                round(crop_y2 / h, 4),
            ],
            "preset_used": preset,
        }
