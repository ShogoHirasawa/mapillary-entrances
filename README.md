# mapillary-entrances

Julien Howard and Evan Rantala

This project integrates **Overture Buildings** (footprints) and **Overture Places** (points of interest) with **Mapillary** street-level imagery to detect and map building entrances.  
The pipeline combines:

- Overture → DuckDB geospatial queries  
- Mapillary imagery (360° preferred)  
- View filtering (distance + heading/FOV logic)  
- YOLOv8 entrance detection  
- Geometry-based triangulation to estimate entrance coordinates  

All core Overture/Mapillary processing is implemented in `src/api.py`.  
The full end-to-end pipeline is exposed through a **single CLI entrypoint**: `src.pipeline`.

## Project Structure

src/
  api.py            → Overture + Places + Mapillary pipeline (programmatic API)
  pipeline.py       → Unified end-to-end CLI runner (download → imagery → inference)
  
  # Overture data pipeline
  buildings.py      → Overture Buildings helpers (footprints, polygons, walls)
  places.py         → Overture Places helpers (optional metadata)
  download_v2.py    → Download Overture Buildings/Places using DuckDB + S3
  sources.py        → Resolves S3/local source paths
  db.py             → DuckDB connection + parquet querying helpers
  config.py         → Centralized constants (paths, S3 buckets, configs)
  
  # Imagery and geometry
  imagery.py        → Mapillary API queries + filtering logic (360 preference)
  selection.py      → Image–wall pairing and FOV/heading filters
  inference_v2.py   → YOLOv8 inference + entrance triangulation
  utils.py          → Common utilities (logging, geometry helpers, etc.)

docs/
  ARCHITECTURE.md
  pilot_area.md
  places_matching.md

data/               → Downloaded Overture extracts (buildings + places)
results/            → Per-building output folders:
                       candidates.json, raw images, _vis images, entrance results
---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/project-terraforma/mapillary-entrances
cd mapillary-entrances
```
### 2. (Recommended) Create a virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
### 3. Environment variables
Option A -- Export it in your shell
```bash
export MAPILLARY_ACCESS_TOKEN=YOUR_TOKEN_HERE
```
Option B -- Create a .env file (recommended)
``` bash
MAPILLARY_ACCESS_TOKEN=YOUR_TOKEN_HERE
```

## Running the pipeline
To run the pipeline, use `src.pipeline`

Running `src.pipeline` performs the full workflow automatically:

1. **Download Overture data**  
   Uses `download_v2` to fetch Buildings (and optional Places) for the area around `--input_point` and `--search_radius`.

2. **Collect Mapillary imagery**  
   Gathers nearby images, prefers 360° when available, and filters them by distance, heading, and FOV so only building-facing images are used.

3. **Run YOLOv8 inference**  
   Passes the selected images to `inference_v2`, which detects potential entrances and saves optional visualizations.

4. **Compute final entrance location**  
   Uses building geometry + detection rays to estimate a single entrance coordinate per building.

All results are written to `results/buildings/<building_id>/`.

Example Usage:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -m src.pipeline \
  --input_point="47.610,-122.341" \
  --search_radius=100 \
  --place_radius=100 \
  --max_images=50 \
  --prefer_360 \
  --src_mode=local \
  --model="yolo_weights_500_image_set.pt" \
  --device="cpu" \
  --conf=0.60 \
  --iou=0.50 \
  --save-vis="./outputs/visualizations"
```
# Notes on arguments

- `--input_point="lat,lon"` → Coordinates to seed the search  
- `--search_radius` → Radius to collect buildings + imagery  
- `--place_radius` → Radius around each building to search for place  
- `--max_images` → Limit on total Mapillary images  
- `--prefer_360` → Prefer panorama imagery when available  
- `--model` → YOLOv8 model weights  
- `--conf`, `--iou` → YOLO thresholds  
- `--save-vis` → Folder to save annotated images 

# Output Structure

Each processed building produces:

results/buildings/<building_id>/
├── candidates.json
├── <image_id>.jpg
├── <image_id>_vis.jpg                (if --save-vis is enabled)
└── model_predictions.json            (optional)

Where `candidates.json` contains:

- building metadata  
- place metadata (optional)  
- Mapillary image metadata  
- building polygon + wall geometry  
- YOLO detection results  
- triangulated entrance coordinate estimate  
