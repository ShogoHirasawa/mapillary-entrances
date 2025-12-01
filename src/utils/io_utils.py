# io_utils.py
# pipeline agnostic utilities

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any
import time
import json
import webbrowser
import urllib.parse

def _ensure_dir(path: Path):
    # create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)

def tlog(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[{t1 - t0:6.2f}s] {label}", flush=True)
    return t1

def parse_bbox_string(bbox_str: str) -> Dict[str, float]:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError('bbox must be "xmin,ymin,xmax,ymax"')
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}

def write_geojson_for_verification(
    building_entrances: List[Dict[str, Any]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    output_dir: Path,
    output_name: str,
):
    """
    Generate GeoJSON and automatically open it in a browser window on GeoJSON.io
    """

    features = []

    # building polygons (blue)
    for bid, polygon in buildings_lat_lon.items():
        if polygon and polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Building {bid}",
                "stroke": "#1f77b4",
                "stroke-width": 2,
                "fill": "#1f77b4",
                "fill-opacity": 0.1,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon],
            }
        })

    # predicted entrance points (red stars)
    for dic in building_entrances:
        bid = dic.get("bid")
        entrance = dic.get("entrance")
        if not bid or not entrance or len(entrance) != 2:
            continue

        lon, lat = float(entrance[0]), float(entrance[1])

        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Predicted Entrance for Building {bid}",
                "marker-color": "#ff0000",
                "marker-symbol": "star",
                "marker-size": "medium",
                "source_image": Path(dic.get("image_path", "")).name,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            }
        })

    # place points (green circles)
    for bid, place in place_names.items():
        if not isinstance(place, dict):
            continue
        lon = place.get("lon")
        lat = place.get("lat")
        if lon is None or lat is None:
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "name": place.get("name", ""),
                "marker-color": "#00cc00",
                "marker-symbol": "circle",
                "marker-size": "small",
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            }
        })

    # build GeoJSON
    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }

    # write to disk
    out_path = Path(output_dir) / "geojsons" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"[OK] Wrote GeoJSON file: {out_path.resolve()}")

    # encode and open in GeoJSON.io
    raw = json.dumps(geojson_data)
    encoded = urllib.parse.quote(raw)
    url = f"https://geojson.io/#data=data:application/json,{encoded}"
    webbrowser.open(url)