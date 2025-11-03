# Mapillary fetch + 360 slicing + JSON writing

import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from .utils import *
from .pano_slices import slice_equirectangular

def _ensure_dir(path: Path):
    # create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)


def fetch_and_slice_for_building(
    building_row,
    radius_m: int,
    min_capture_date: Optional[str],
    apply_fov: bool,
    max_images_per_building: int,
    prefer_360: bool,
    fov_half_angle: float,
) -> List[Dict]:
    """
    Fetch Mapillary images near a building centroid and optionally slice panoramas.
    Returns list of metadata dicts (and saves images locally).
    """
    lat, lon = building_row["lat"], building_row["lon"]
    building_id = building_row["id"]
    token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("MAPILLARY_ACCESS_TOKEN missing")

    out_dir = Path(f"results/buildings/{building_id}")
    _ensure_dir(out_dir)

    t0 = time.perf_counter()
    print(f"Building {building_id}, ({lat:.6f},{lon:.6f}), radius {radius_m} m")

    # fetch images using actual mly_utils signature
    imgs = fetch_images(
        token=token,
        fields=[
            "id",
            "computed_geometry",
            "captured_at",
            "compass_angle",
            "thumb_1024_url",
            "camera_type",
        ],
        prefer_360=prefer_360,
        min_capture_date_filter=min_capture_date,
        radius_m=radius_m,
        lat=lat,
        lon=lon,
    )

    if not imgs:
        print("  ↳ no images nearby")
        return []

    # cap results
    imgs = imgs[:max_images_per_building]

    saved = []
    for img in imgs:
        img_id = img.get("id")
        img_path = out_dir / f"{img_id}.jpg"
        download_image(img.get("thumb_1024_url"), img_path)
        saved.append({
            "id": img_id,
            "path": str(img_path),
            "coordinates": img.get("computed_geometry", {}).get("coordinates"),
            "compass_angle": img.get("compass_angle"),
            "camera_type": img.get("camera_type"),
            "captured_at": img.get("captured_at"),
        })

    # slice panoramas if enabled
    if prefer_360:
        n_before = len(saved)
        saved = slice_equirectangular(saved, fov_half_angle=fov_half_angle)
        print(f"Building {building_id} ({n_before} originals, {len(saved)} kept 360-slices)")
    else:
        print(f"Building {building_id} ({len(saved)} images)")

    tlog("Building done", t0)
    return saved


def _extract_lon_lat(rec, fallback_lon, fallback_lat):
    """
    Try multiple fields to recover (lon, lat). Return floats.
    Priority:
      1) rec['lon'], rec['lat']
      2) rec['lng'], rec['lat']               # sometimes 'lng' is used
      3) rec['image_lon'], rec['image_lat']   # upstream variants
      4) rec['orig_lon'], rec['orig_lat']     # originals before slicing
      5) rec['coordinates'] = [lon, lat]
      6) fallback (building center)
    """
    def _ok(a, b):
        return isinstance(a, (int, float)) and isinstance(b, (int, float))

    cand_pairs = [
        ("lon", "lat"),
        ("lng", "lat"),
        ("image_lon", "image_lat"),
        ("orig_lon", "orig_lat"),
    ]
    for lo_k, la_k in cand_pairs:
        lo, la = rec.get(lo_k), rec.get(la_k)
        if _ok(lo, la):
            return float(lo), float(la)

    coords = rec.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        lo, la = coords[0], coords[1]
        if _ok(lo, la):
            return float(lo), float(la)

    # last resort – building center (ensures numbers, never nulls)
    return float(fallback_lon), float(fallback_lat)


def write_candidates_json(building_row, best_place, saved, out_dir=None):
    """
    Write candidates.json for a building and return [(evan_image_dict, wall_pair), ...].

    - building_row: pandas Series with at least id, lat, lon, wkt
    - best_place: dict or None
    - saved: list of dicts produced by fetch_and_slice_for_building()
      each item typically has:
        {
          "id": <image id>,
          "path": <slice or original path>,
          "lon": <float>,
          "lat": <float>,
          "compass_angle": <float or None>,
          "camera_type": "spherical" | "perspective" | None,
          "captured_at": <int or None>,
          "slice_index": <int or None>
        }
    - out_dir: optional override; if None, derived from first saved path
    """

    if out_dir is None:
        # use dir of first saved item if available, otherwise a sensible default
        if saved and "path" in saved[0]:
            out_dir = os.path.dirname(saved[0]["path"])
        else:
            out_dir = os.path.join("results", "buildings", str(building_row["id"]))
    os.makedirs(out_dir, exist_ok=True)

    # prefer column 'wkt', fallbacks included for safety
    building_wkt = (
        building_row.get("wkt")
        or building_row.get("geom_wkt")
        or building_row.get("geometry_wkt")
    )
    if not building_wkt:
        raise ValueError("write_candidates_json: building_row has no WKT field ('wkt').")

    building_rec = {
        "id": str(building_row["id"]),
        "lon": float(building_row["lon"]),
        "lat": float(building_row["lat"]),
        "wkt": str(building_wkt),
    }

    # place
    place_rec = None
    if best_place:
        place_rec = {
            "place_id": best_place.get("place_id"),
            "name": best_place.get("name"),
            "categories": best_place.get("categories"),
            "lon": best_place.get("lon"),
            "lat": best_place.get("lat"),
            "inside": bool(best_place.get("inside", False)),
            "dist_m": float(best_place.get("dist_m", 0.0)),
        }


    images_out = []
    # we’ll use the building center as a hard fallback to avoid nulls
    b_lon, b_lat = float(building_rec["lon"]), float(building_rec["lat"])

    for r in saved:
        img_path = r.get("path")
        if not img_path:
            continue

        lon, lat = _extract_lon_lat(r, b_lon, b_lat)

        images_out.append({
            "id": r.get("id"),
            "path": img_path,
            "coordinates": [lon, lat],
            "compass_angle": r.get("compass_angle"),
            "camera_type": r.get("camera_type"),
            "captured_at": r.get("captured_at"),
            "slice_index": r.get("slice_index"),
        })
    # polygon + walls
    polygon_xy = polygon_vertices_from_wkt(building_wkt, drop_last=True)
    walls = polygon_walls_from_wkt(building_wkt)

    # write JSON
    record = {
        "building": building_rec,
        "place": place_rec,
        "images": images_out,
        "polygon": polygon_xy,
        "walls": walls,
    }

    out_path = os.path.join(out_dir, "candidates.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {out_path}", flush=True)


    # image dicts: {"image_path": <jpg>, "compass_angle": <float>, "coordinates": [lon,lat]}
    image_dicts = []
    for r in images_out:
        if not r.get("path"):
            continue
        image_dicts.append({
            "image_path": r["path"],
            "compass_angle": r.get("compass_angle"),
            "coordinates": r.get("coordinates"),  # [lon,lat]
        })

    pairs = []
    for img in image_dicts:
        for wall in walls:
            pairs.append((img, wall))

    return pairs

