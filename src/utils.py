# Misc helpers

import time, math
from typing import Dict, List
from shapely import wkt as _wkt
from shapely.geometry import Polygon, MultiPolygon


def parse_bbox_string(bbox_str: str) -> Dict[str, float]:
    parts = [float(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError('bbox must be "xmin,ymin,xmax,ymax"')
    return {"xmin": parts[0], "ymin": parts[1], "xmax": parts[2], "ymax": parts[3]}

def deg_per_meter(lat_deg: float):
    dlat = 1.0 / 111_320.0
    dlon = dlat / max(0.01, abs(math.cos(math.radians(lat_deg))))
    return dlat, dlon

def tlog(label: str, t0: float) -> float:
    t1 = time.perf_counter()
    print(f"[{t1 - t0:6.2f}s] {label}", flush=True)
    return t1

def polygon_vertices_from_wkt(wkt: str, drop_last: bool = True) -> List[List[float]]:
    """
    Convert WKT (Polygon or MultiPolygon) to a flat list of [x, y] vertices for the
    exterior ring. Drops the duplicated closing vertex by default.
      - Returns coordinates in (lon, lat) order (i.e., x=lon, y=lat).
    """
    g = _wkt.loads(wkt)
    if isinstance(g, MultiPolygon):
        # pick the largest polygon by area
        g = max(g.geoms, key=lambda p: p.area)
    if not isinstance(g, Polygon):
        raise ValueError(f"Expected Polygon/MultiPolygon, got {g.geom_type}")

    coords = list(g.exterior.coords)
    if drop_last and len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    # ensure floats (lon, lat)
    return [[float(x), float(y)] for (x, y) in coords]


def polygon_walls_from_wkt(wkt: str) -> List[List[List[float]]]:
    """
    Convert a Polygon/MultiPolygon WKT into a list of wall segments.

    Each wall is represented as [[lon1, lat1], [lon2, lat2]].
    Automatically closes the polygon (i.e., connects last→first).
    """
    verts = polygon_vertices_from_wkt(wkt, drop_last=True)
    walls: List[List[List[float]]] = []
    if len(verts) < 2:
        return walls
    for i in range(len(verts)):
        a = verts[i]
        b = verts[(i + 1) % len(verts)]  # wrap around
        walls.append([a, b])
    return walls
