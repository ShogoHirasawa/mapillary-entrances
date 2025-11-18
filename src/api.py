# --- new file: src/api.py ---
# interface for buildings/images/places near a (lat, lon) point. uses same pipeline
# pieces but returns results in-memory as a dict 

# example usage


from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import math
import pandas as pd

from .utils import polygon_vertices_from_wkt
from .buildings import get_buildings
from .places import join_buildings_places
from .selection import select_best_place_for_building
from .imagery import fetch_and_slice_for_building, _extract_lon_lat
from .sources import resolve_sources

def _point_bbox(lon: float, lat: float, meters: float = 80.0) -> Dict[str, float]:
    # simple meters to deg approximation for small areas
    dy = meters / 111_320.0
    dx = meters / (111_320.0 * math.cos(math.radians(lat)))
    return {"xmin": lon - dx, "ymin": lat - dy, "xmax": lon + dx, "ymax": lat + dy}

def _nearest_building(bdf: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    if bdf.empty:
        raise RuntimeError("No buildings in bbox")
    d2 = (bdf["lat"] - lat)**2 + (bdf["lon"] - lon)**2
    return bdf.loc[d2.idxmin()]

def get_buildings_and_imagery_in_radius(
    lat: float,
    lon: float,
    search_radius_m: int,
    place_radius_m: int,
    max_images_total: int,
    min_capture_date: Optional[str],
    prefer_360: bool,
    src_mode: str,
) -> Dict[str, Any]:
    """
    Multi-building + shared imagery adapter for inference.py

    - Finds *all* buildings within search_radius_m of (lat, lon)
    - Joins each with its best nearby place (optional)
    - Fetches a *single shared* imagery set centered on (lat, lon)
    - Returns:
        {
          "input_coordinates": [lon, lat],
          "building_polygons": {building_id: [[lon,lat], ...], ...},
          "building_walls": {building_id: [[[lon,lat],[lon,lat]], ...], ...},
          "places": {building_id: place_info or None, ...},
          "image_dicts": [ {...}, {...}, ... ]   # all images in radius
        }
    """

    # define bounding box and load building and place data
    bbox = _point_bbox(lon, lat, meters=search_radius_m)
    b_src, p_src = resolve_sources(bbox, src_mode)
    bdf = get_buildings(bbox, b_src, limit_hint=200)
    if bdf is None or len(bdf) == 0:
        print("[WARN] No buildings found in radius.")
        return {
            "input_coordinates": [lon, lat],
            "building_polygons": {},
            "building_walls": {},
            "places": {},
            "image_dicts": []
        }

    print(f"[INFO] Found {len(bdf)} buildings within {search_radius_m} m")

    # join once with places
    links = join_buildings_places(bdf, bbox, p_src, radius_m=place_radius_m)

    building_polygons = {}
    building_walls = {}
    building_places = {}

    for _, b in bdf.iterrows():
        bid = b["id"]

        # polygon + wall segments
        polygon = polygon_vertices_from_wkt(b["wkt"])  # [[lon,lat], ...]
        building_polygons[bid] = polygon

        walls = []
        if len(polygon) >= 2:
            for i in range(len(polygon)):
                a = polygon[i]
                bpt = polygon[(i + 1) % len(polygon)]
                walls.append([a, bpt])
        building_walls[bid] = walls

        # best place (if any)
        best_place = None
        if "building_id" in links.columns:
            subset = links[links["building_id"] == bid]
            if len(subset) > 0:
                best_place = select_best_place_for_building(
                    subset,
                    building_id=bid,
                    max_dist_m=place_radius_m,
                )
        building_places[bid] = best_place or None

    # fetch imagery *once* for the entire area around (lat, lon)
    print(f"[INFO] Fetching imagery around ({lat:.6f}, {lon:.6f}) within {search_radius_m} m")

    temp_building = {
        "id": "shared_area",
        "lat": lat,
        "lon": lon,
        "wkt": None, # unused
    }

    saved = fetch_and_slice_for_building(
        temp_building,
        radius_m=search_radius_m,
        min_capture_date=min_capture_date,
        max_images_per_building=max_images_total,
        prefer_360=prefer_360,
    )

    if not saved:
        print("[WARN] No imagery fetched for area.")
        saved = []

    # build unified image metadata list
    image_data: List[Dict[str, Any]] = []
    for rec in saved:
        lon_rec, lat_rec = _extract_lon_lat(rec, lon, lat)
        image_data.append({
            "image_path": rec.get("path") or rec.get("jpg_path"),
            "compass_angle": rec.get("compass_angle"),
            "coordinates": [lon_rec, lat_rec],
            "is_360": rec.get("is_360", False),
            "camera_type": rec.get("camera_type"),
        })

    # return unified dictionary
    return {
        "input_coordinates": [lon, lat],
        "building_polygons": building_polygons,
        "building_walls": building_walls,
        "places": building_places,
        "image_dicts": image_data,
    }
