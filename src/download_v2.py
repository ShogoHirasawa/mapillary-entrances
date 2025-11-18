# src/download_v2.py

import math
import argparse
from pathlib import Path
from shapely.geometry import box
from src.db import open_duckdb
from src.config import RELEASE

S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

EARTH_RADIUS_M = 6378137.0

def radius_to_bbox(lat: float, lon: float, radius_m: float):
    """
    Convert a (lat, lon, radius_m) into a bbox = (xmin, ymin, xmax, ymax)
    using a local equirectangular approximation.
    """
    delta_lat = (radius_m / EARTH_RADIUS_M) * (180 / math.pi)
    delta_lon = (radius_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))) * (180 / math.pi)
    return lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat

def bbox_to_wkt(b):
    xmin, ymin, xmax, ymax = b
    return box(xmin, ymin, xmax, ymax).wkt

def _try(con, stmt: str):
    try:
        con.execute(stmt)
    except Exception as e:
        print(f"[warn] skipped setting: {stmt} ({e})")

def download_overture_radius(lat: float, lon: float, radius_m: int,
                             out_buildings: str = "data/buildings_radius.parquet",
                             out_places: str = "data/places_radius.parquet"):
    """
    Download Overture buildings & places within *radius_m* of (lat, lon)
    and save to local parquet files.
    """
    bbox = radius_to_bbox(lat, lon, radius_m)
    xmin, ymin, xmax, ymax = bbox
    aoi_wkt = bbox_to_wkt(bbox)

    out_b = Path(out_buildings); out_b.parent.mkdir(parents=True, exist_ok=True)
    out_p = Path(out_places);    out_p.parent.mkdir(parents=True, exist_ok=True)

    con = open_duckdb()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")

    _try(con, "SET s3_region='us-west-2'")
    _try(con, "SET s3_url_style='path'")
    _try(con, "SET s3_endpoint='s3.us-west-2.amazonaws.com'")
    _try(con, "SET enable_object_cache=true")
    _try(con, "SET http_keep_alive=true")
    _try(con, "SET http_timeout=30")
    _try(con, "SET threads=8")

    con.execute(f"CREATE OR REPLACE TABLE aoi AS SELECT ST_GeomFromText('{aoi_wkt}') AS g;")

    bfilter = f"""
      struct_extract(bbox,'xmax') >= {xmin} AND
      struct_extract(bbox,'xmin') <= {xmax} AND
      struct_extract(bbox,'ymax') >= {ymin} AND
      struct_extract(bbox,'ymin') <= {ymax}
    """

    # BUILDINGS
    con.execute(f"""
      COPY (
        WITH src AS (
          SELECT id, geometry, bbox
          FROM read_parquet('{S3_BUILDINGS}')
          WHERE {bfilter}
        )
        SELECT
          id,
          ST_AsText(geometry) AS wkt,
          ST_X(ST_Centroid(geometry)) AS lon,
          ST_Y(ST_Centroid(geometry)) AS lat
        FROM src
        WHERE ST_Intersects(geometry, (SELECT g FROM aoi))
      ) TO '{out_b.as_posix()}' (FORMAT 'PARQUET');
    """)

    # PLACES
    con.execute(f"""
      COPY (
        WITH src AS (
          SELECT id, geometry, names, categories, bbox
          FROM read_parquet('{S3_PLACES}')
          WHERE {bfilter}
        )
        SELECT id, geometry, names, categories, bbox
        FROM src
        WHERE ST_Intersects(geometry, (SELECT g FROM aoi))
      ) TO '{out_p.as_posix()}' (FORMAT 'PARQUET');
    """)

    con.close()
    print(f"✓ wrote {out_b}")
    print(f"✓ wrote {out_p}")

