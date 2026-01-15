from pathlib import Path
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_RELEASE = "2025-12-17.0"  # fallback so the app still runs

def get_latest_overture_release(bucket="overturemaps-us-west-2", prefix="release/") -> str:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    releases = [
        cp["Prefix"].replace(prefix, "").strip("/")
        for page in pages
        for cp in page.get("CommonPrefixes", [])
    ]
    if not releases:
        raise RuntimeError("No Overture releases found")
    return sorted(releases)[-1]

def resolve_overture_release() -> str:
    try:
        return get_latest_overture_release()
    except Exception as e:
        print(f"[WARN] Could not fetch latest Overture release ({e}). Falling back to {DEFAULT_RELEASE}.")
        return DEFAULT_RELEASE

RELEASE = resolve_overture_release()

EARTH_RADIUS_M = 6378137

# Local parquet file paths
LOCAL_BUILDINGS = Path("../data/buildings_local.parquet")
LOCAL_PLACES    = Path("../data/places_local.parquet")

# S3 paths for downloading Overture data
S3_BUILDINGS = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=buildings/type=building/*"
S3_PLACES    = f"s3://overturemaps-us-west-2/release/{RELEASE}/theme=places/type=place/*"

_CAT_PRIOR = {
    "art_museum": 1.2,
    "museum": 1.1,
    "library": 1.1,
    "school": 1.1,
    "university": 1.1,
    "hotel": 1.05,
    "restaurant": 1.05,
}