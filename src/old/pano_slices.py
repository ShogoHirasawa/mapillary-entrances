# src/pano_slices.py
# Slices 360 images into crops

from pathlib import Path
from typing import List, Dict
import cv2

def slice_equirectangular(images: List[Dict], fov_half_angle: float = 45.0) -> List[Dict]:
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
        
        # Get original compass angle (default to 0 if missing)
        original_bearing = img.get("compass_angle") or 0.0
        
        for i in range(4):
            x1, x2 = i * slice_w, (i + 1) * slice_w
            crop = im[:, x1:x2]
            slice_path = path.with_name(f"{path.stem}_slice{i}.jpg")
            cv2.imwrite(str(slice_path), crop)

            new_entry = dict(img)
            new_entry["path"] = str(slice_path)
            new_entry["slice_index"] = i
            # Slices are no longer 360° panoramas, but preserve camera_type for reference
            new_entry["is_360"] = False
            # Update compass_angle for each slice: slice i represents bearing at i*90° offset
            # For equirectangular: slice 0 = original, slice 1 = +90°, slice 2 = +180°, slice 3 = +270°
            new_entry["compass_angle"] = (original_bearing + (i * 90.0)) % 360.0
            sliced.append(new_entry)

    return sliced
