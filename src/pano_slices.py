# src/pano_slices.py
# Slices 360 images into crops

from pathlib import Path
from typing import List, Dict
import cv2
import math

def slice_equirectangular(images: List[Dict], fov_half_angle: float = 25.0) -> List[Dict]:
    """
    Given a list of image metadata dicts (with 'path' and optionally 'camera_type'),
    create cropped 90° horizontal slices from 360° equirectangular images.

    Returns updated list of image dicts (including new 'path' for each slice).
    """
    sliced = []
    for img in images:
        path = Path(img["path"])
        cam_type = img.get("camera_type", "").lower()

        # Skip non-360s
        if "spherical" not in cam_type and "360" not in cam_type:
            sliced.append(img)
            continue

        if not path.exists():
            print(f"[WARN] missing file: {path}")
            continue

        im = cv2.imread(str(path))
        if im is None:
            print(f"[WARN] failed to read {path}")
            continue

        h, w = im.shape[:2]
        slice_w = w // 4  # four 90° slices (roughly quarter turns)
        for i in range(4):
            x1, x2 = i * slice_w, (i + 1) * slice_w
            crop = im[:, x1:x2]
            slice_path = path.with_name(f"{path.stem}_slice{i}.jpg")
            cv2.imwrite(str(slice_path), crop)

            new_entry = dict(img)
            new_entry["path"] = str(slice_path)
            new_entry["slice_index"] = i
            sliced.append(new_entry)

    return sliced
