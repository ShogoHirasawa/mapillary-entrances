# src/selection.py
import math
from typing import Any, Dict, Optional
import pandas as pd

# Simple category prior — tweak as you like
_CAT_PRIOR = {
    "art_museum": 1.2,
    "museum": 1.1,
    "library": 1.1,
    "school": 1.1,
    "university": 1.1,
    "hotel": 1.05,
    "restaurant": 1.05,
}

def _cat_weight(categories: Any) -> float:
    """
    categories is typically a dict like:
      {"primary": "art_museum", "alternate": ["museum", ...]}
    but may be None / string / JSON; be defensive.
    """
    try:
        primary = None
        if isinstance(categories, dict):
            primary = categories.get("primary") or None
        # fallbacks — very conservative
        if primary and isinstance(primary, str):
            return float(_CAT_PRIOR.get(primary, 1.0))
    except Exception:
        pass
    return 1.0

def _extract_place_name(names_obj) -> Optional[str]:
    # Overture 'names' field can look like: {"primary":{"en":"Foo"},"alternate":{...}}
    if not isinstance(names_obj, dict):
        return None
    primary = names_obj.get("primary")
    if isinstance(primary, dict) and primary:
        # take any language (prefer 'en' if present)
        return primary.get("en") or next(iter(primary.values()))
    # fallbacks
    alt = names_obj.get("alternate")
    if isinstance(alt, dict) and alt:
        return alt.get("en") or next(iter(alt.values()))
    return None


def _as_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        return float(val) if val is not None else float(default)
    except Exception:
        return float(default)

def _as_bool(row: pd.Series, key: str) -> bool:
    return bool(row.get(key, False))

def _geom_weight(row) -> float:
    w = 0.0
    # keep overlap/inside bonuses if they exist; ignore if missing
    if "inside" in row and row["inside"]:
        w += 0.6
    if "overlap_ratio" in row and row["overlap_ratio"] and row["overlap_ratio"] > 0:
        w += min(0.5, float(row["overlap_ratio"]))
    # inverse distance bonus (cap at 0.4)
    if "dist_m" in row and row["dist_m"] is not None:
        w += max(0.0, 0.4 - 0.4 * min(1.0, float(row["dist_m"]) / 10.0))
    return w

def select_best_place_for_building(
    links_df,
    building_id,
    max_dist_m: float = 60.0,
    hard_max_dist_m: Optional[float] = None,
):
    """
    Pick the best place row for a building.

    - hard_max_dist_m: if set, rows with dist_m > this are dropped entirely.
                       Pass your CLI's --place-radius-m here for a strict cap.
    Returns a dict with name already extracted, or None.
    """
    if links_df is None or len(links_df) == 0:
        return None

    df = links_df[links_df["building_id"] == building_id].copy()
    if df.empty:
        return None

    # hard cap: drop anything beyond this distance
    cap = hard_max_dist_m if hard_max_dist_m is not None else max_dist_m
    if "dist_m" in df.columns:
        df = df[df["dist_m"].notna() & (df["dist_m"] <= cap)]

    if df.empty:
        return None

    # score & pick
    scores = []
    for _, r in df.iterrows():
        s = _geom_weight(r) + _cat_weight(r.get("categories", {}) or {})
        scores.append(s)
    df = df.assign(_score=scores).sort_values("_score", ascending=False)

    top = df.iloc[0].to_dict()

    # return clean dict w/ human-readable name
    return {
        "place_id": top.get("place_id"),
        "name": _extract_place_name(top.get("names")),
        "categories": top.get("categories"),
        "lon": top.get("place_lon"),
        "lat": top.get("place_lat"),
        "inside": bool(top.get("inside", False)),
        "dist_m": float(top.get("dist_m")) if top.get("dist_m") is not None else None,
    }