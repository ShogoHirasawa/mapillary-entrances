import argparse, os, sys, subprocess, shutil
from pathlib import Path

def _run(cmd: list[str], env=None):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)

def main():
    ap = argparse.ArgumentParser(description="One-shot pipeline: download Overture bbox → run Mapillary slicer")
    ap.add_argument("--bbox", required=True, help='xmin,ymin,xmax,ymax (lon/lat)')
    # download step outputs (defaults match what cli_plumbing expects locally)
    ap.add_argument("--out-buildings", default="data/buildings_soma.parquet")
    ap.add_argument("--out-places",   default="data/places_soma.parquet")
    ap.add_argument("--skip-download", action="store_true", help="Skip Overture download; reuse existing parquet files")
    # slicer params
    ap.add_argument("--radius-m", type=int, default=120)
    ap.add_argument("--place-radius-m", type=int, default=60)
    ap.add_argument("--limit-buildings", type=int, default=10)
    ap.add_argument("--max-images-per-building", type=int, default=8)
    ap.add_argument("--min-capture-date", type=str, default=None)
    ap.add_argument("--prefer-360", action="store_true")
    ap.add_argument("--no-fov", action="store_true")
    ap.add_argument("--fov-half-angle", type=float, default=25.0)
    args = ap.parse_args()

    # ensure output dirs exist
    Path(args.out_buildings).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_places).parent.mkdir(parents=True, exist_ok=True)

    # step 1: download (unless skipped)
    if not args.skip_download:
        _run([
            sys.executable, "-m", "src.download_bbox_extract",
            f"--bbox={args.bbox}",
            "--out-buildings", args.out_buildings,
            "--out-places",    args.out_places,
        ])


    # sanity check presence
    if not Path(args.out_buildings).exists() or not Path(args.out_places).exists():
        print("ERROR: expected local extracts missing. Re-run without --skip-download or check paths.", file=sys.stderr)
        sys.exit(2)

    # step 2: slicer pipeline (always runs in local mode, using those parquet files)
    # NOTE: sources.py should already resolve to these default paths; if your config
    # points somewhere else, just pass the same filenames above so they match.
    cmd = [
        sys.executable, "-m", "src.cli_plumbing",
        f"--bbox={args.bbox}",
        "--radius-m", str(args.radius_m),
        "--place-radius-m", str(args.place_radius_m),
        "--limit-buildings", str(args.limit_buildings),
        "--max-images-per-building", str(args.max_images_per_building),
        "--src-mode", "local",
        "--fov-half-angle", str(args.fov_half_angle),
    ]
    if args.min_capture_date:
        cmd += ["--min-capture-date", args.min_capture_date]
    if args.prefer_360:
        cmd += ["--prefer-360"]
    if args.no_fov:
        cmd += ["--no-fov"]

    # pass through current env (needs MAPILLARY_TOKEN set)
    env = os.environ.copy()
    if not env.get("MAPILLARY_TOKEN"):
        print("WARNING: MAPILLARY_TOKEN not set; imagery requests will fail with 401.", file=sys.stderr)

    _run(cmd, env=env)

if __name__ == "__main__":
    main()
