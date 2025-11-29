# io_utils.py
# pipeline agnostic utilities

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Any
import time
import json

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
'''
def write_geojson_for_verification(
    building_entrances: Dict[str, Tuple[float, float]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    image_detections: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
    output_name: str,
):
    """
    Write a GeoJSON file containing:
      - Building polygons (blue)
      - Predicted entrance points (red stars)
      - Place points (green circles)
      - Image detection coordinates (orange diamonds)

    """
    features = []

    # building polygons (blue)
    for bid, polygon in buildings_lat_lon.items():
        # ensure ring is closed
        if polygon and polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Building {bid}",
                "stroke": "#1f77b4",
                "stroke-width": 2,
                "fill": "#1f77b4",
                "fill-opacity": 0.1
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        })

    # entrance detections (red stars)
    for bid, entrance_coords in building_entrances.items():
        lon, lat = entrance_coords
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Predicted Entrance for Building {bid}",
                "marker-color": "#ff0000",
                "marker-symbol": "star",
                "marker-size": "medium"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        })

    # place points (green circles)
    if place_names:
        for bid, place in place_names.items():
            if not isinstance(place, dict):
                continue
            lon = place.get("lon")
            lat = place.get("lat")
            if lon is not None and lat is not None:
                features.append({
                    "type": "Feature",
                    "properties": {
                        "name": f"Place {place.get('place_id', '')} ({place.get('name', '')})",
                        "marker-color": "#00cc00",
                        "marker-symbol": "circle",
                        "marker-size": "small"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)]
                    }
                })

    # iage detections (orange diamonds)
    if image_detections:
        for img_path, lonlat in image_detections.items():
            lon, lat = float(lonlat[0]), float(lonlat[1])
            features.append({
                "type": "Feature",
                "properties": {
                    "name": f"Detection from {Path(img_path).name}",
                    "marker-color": "#ffa500",
                    "marker-symbol": "diamond",
                    "marker-size": "small"
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })

    # write file
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path = Path(output_dir) / Path("geojsons") /Path(output_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"[OK] Wrote GeoJSON verification file: {out_path.resolve()}")
    print("→ Open this file at https://geojson.io to visualize results.")

def write_geojson_for_verification(
    building_entrances: List[Dict[str, Any]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    image_detections: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
    output_name: str,
):
    """
    Write a GeoJSON file containing:
      - Building polygons (blue)
      - Predicted entrance points (red stars)
      - Place points (green circles)
      - Image detection coordinates (orange diamonds)
    """

    features = []

    # 1. Building polygons (blue)
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

    # 2. Predicted entrance points (red stars)
    # The structure is now a LIST, so we pull lon/lat from each dict
    for dic in building_entrances:
        bid = dic.get("bid")
        entrance = dic.get("entrance")

        # Defensive: skip invalid or incomplete entries
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
                "snapped": dic.get("snapped", False),  # still preserved
                "source_image": Path(dic.get("image_path", "")).name,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            }
        })

    # 3. Place points (green circles)
    if place_names:
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
                    "name": f"Place {place.get('place_id','')} ({place.get('name','')})",
                    "marker-color": "#00cc00",
                    "marker-symbol": "circle",
                    "marker-size": "small",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                }
            })

    # 4. Image door detections (orange diamonds)
    # (same structure, unchanged logic)
    if image_detections:
        for img_path, lonlat in image_detections.items():
            lon, lat = float(lonlat[0]), float(lonlat[1])
            
            features.append({
                "type": "Feature",
                "properties": {
                    "name": f"Detection from {Path(img_path).name}",
                    "marker-color": "#ffa500",
                    "marker-symbol": "diamond",
                    "marker-size": "small",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }
            })

    # 5. Build FeatureCollection and write out
    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path = Path(output_dir) / "geojsons" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"[OK] Wrote GeoJSON verification file: {out_path.resolve()}")
    print("→ Open this file at https://geojson.io to visualize results.")
'''
def write_geojson_for_verification(
    building_entrances: List[Dict[str, Any]],
    buildings_lat_lon: Dict[str, List[List[float]]],
    place_names: Dict[str, Dict[str, Any]],
    image_detections: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
    output_name: str,
):
    """
    Write a GeoJSON file and also generate a CSV containing predicted entrance points.
    """

    # -------------------- Build GeoJSON Features --------------------
    features = []

    # 1. Building polygons (blue)
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

    # 2. Predicted entrance points (red stars)
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
                "snapped": dic.get("snapped", False),
                "source_image": Path(dic.get("image_path", "")).name,
            },
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            }
        })

    # 3. Place metadata points (green)
    if place_names:
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
                    "name": f"Place {place.get('place_id','')} ({place.get('name','')})",
                    "marker-color": "#00cc00",
                    "marker-symbol": "circle",
                    "marker-size": "small",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                }
            })

    # 4. Raw image detections (orange diamonds)
    if image_detections:
        for img_path, lonlat in image_detections.items():
            lon, lat = float(lonlat[0]), float(lonlat[1])

            features.append({
                "type": "Feature",
                "properties": {
                    "name": f"Detection from {Path(img_path).name}",
                    "marker-color": "#ffa500",
                    "marker-symbol": "diamond",
                    "marker-size": "small",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }
            })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features,
    }

    # -------------------- Write GeoJSON --------------------
    out_path = Path(output_dir) / "geojsons" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    print(f"[OK] Wrote GeoJSON verification file: {out_path.resolve()}")

    # -------------------- Create CSV --------------------
    csv_rows = []

    for dic in building_entrances:
        bid = dic.get("bid")
        entrance = dic.get("entrance")

        if not bid or not entrance or len(entrance) != 2:
            continue

        csv_rows.append({
            "building_id": bid,
            "predicted_lat": entrance[1],
            "predicted_lon": entrance[0],
            "source_image": dic.get("image_path"),
            "snapped": dic.get("snapped", False),
        })

    if csv_rows:
        df = pd.DataFrame(csv_rows)
        csv_path = Path(output_dir) / "geojsons" / "predicted_entrances.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] Wrote CSV: {csv_path.resolve()} ({len(df)} rows)")
    else:
        print("[WARN] No valid entrance rows for CSV.")