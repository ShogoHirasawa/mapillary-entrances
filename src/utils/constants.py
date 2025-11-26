from pathlib import Path

RELEASE = "2025-10-22.0"

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