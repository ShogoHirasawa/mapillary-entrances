import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2
import pyproj

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

#----------------------------------
# Basic helper functions

# shared local projection utilities
def make_local_proj(lat0: float, lon0: float):
    # create a local Azimuthal Equidistant projection centered at (lat0, lon0)
    return pyproj.Proj(proj="aeqd", lat_0=lat0, lon_0=lon0,
                       ellps="WGS84", units="m")

def to_local_xy(lat: float, lon: float, proj_local):
    # convert (lat, lon) to local (x, y) meters using the provided projection
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_local, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.array([x, y], dtype=float)


# ----------------------------------
# Step 1:
# Image quality filters

def is_sharp(img: np.ndarray, thresh: float = 100.0) -> bool:
    # check image sharpness using the variance of the Laplacian
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var() > thresh


def has_enough_resolution(img: np.ndarray, min_width: int = 300, min_height: int = 300) -> bool:
    # check if image meets minimum width and height requirements
    h, w = img.shape[:2]
    return (w >= min_width) and (h >= min_height)


def is_well_exposed(img: np.ndarray, dark_thresh: float = 0.05, bright_thresh: float = 0.95) -> bool:
    # check image exposure based on the proportion of very dark or very bright pixels
    if img.ndim > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = img.flatten() / 255.0
    return ((flat < dark_thresh).mean() < 0.5 and
            (flat > bright_thresh).mean() < 0.5)


def filter_images_by_quality(
    images : List,
    sharpness_thresh: float = 100.0,
    min_width: int = 300,
    min_height: int = 300,
    dark_thresh: float = 0.05,
    bright_thresh: float = 0.95
) -> List:
    """Apply quality filters to list of images pulled from mapillary, removing low-quality ones.
    Returns the list of images that passed quality filters.
    """
    good_images = []
    for img_path in images:
        img = cv2.imread(img_path["image_path"])
        if (has_enough_resolution(img, min_width, min_height) and
            is_sharp(img, sharpness_thresh) and
            is_well_exposed(img, dark_thresh, bright_thresh)):
            good_images.append(img)
    return good_images

'''
#Segmentation (classical CV) ------------------------
def image_segmentation(
    img: np.ndarray,
    facade_tau: float = 0.035,
    min_roi_area_frac: float = 0.01,
    ground_band_frac: float = 0.35,
    visualize: bool = False
) -> Tuple[np.ndarray, float, List[Tuple[int,int,int,int]]]:
    """
    Lightweight segmentation & ROI proposal for façades/doors using classical CV.

    Returns:
      - mask (uint8, 0/255): façade-like areas near ground (candidate regions)
      - facade_presence_score: fraction of façade-like pixels in a center band
      - rois: list of (x1,y1,x2,y2) boxes to run the detector on

    How it works (simple & fast):
      1) Smooth + grayscale
      2) Vertical structure (Sobel-X) + Canny edges
      3) Focus on bottom "ground band" where doors meet ground
      4) Morphological cleanup → connected components → ROI boxes
      5) facade_presence_score = % of strong edges in center band (coarse gate)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gentle denoise to reduce texture noise
    gray_blur = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    # vertical structure (doors & façade seams have verticals)
    sobelx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(np.abs(sobelx))
    sobelx_norm = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX)

    # edges
    edges = cv2.Canny(gray_blur, 50, 150)

    # emphasize places with both vertical structure and edges
    mix = cv2.addWeighted(sobelx_norm, 0.6, edges, 0.4, 0)

    # keep a horizontal "center band" to compute a façade presence score
    band_w1, band_w2 = int(0.25 * w), int(0.75 * w)
    band = mix[:, band_w1:band_w2]
    facade_presence_score = (band > 64).mean()

    # restrict to ground band (bottom part of image)
    gb_y = int((1.0 - ground_band_frac) * h)
    ground_band = np.zeros_like(mix, dtype=np.uint8)
    ground_band[gb_y:, :] = 255

    cand = cv2.bitwise_and(mix, ground_band)
    # binarize
    _, binmask = cv2.threshold(cand, 64, 255, cv2.THRESH_BINARY)

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binmask = cv2.morphologyEx(binmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binmask = cv2.dilate(binmask, kernel, iterations=1)

    # find connected components as ROIs
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: List[Tuple[int,int,int,int]] = []
    min_area = int(min_roi_area_frac * w * h)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area >= min_area and bh >= int(0.1 * h):  # tall-ish
            # clip to image bounds
            x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
            rois.append((x1, y1, x2, y2))

    # optional visualization mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in rois:
        mask[y1:y2, x1:x2] = 255

    if visualize:
        dbg = img.copy()
        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("seg_rois", dbg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask, float(facade_presence_score), rois

'''

# --------------------------------------------------------------
# Step 2:
# Get image(s) pointing at candidate wall
# After filtering images for quality, these functions correspond to the secondary filter for direction

def nearest_point_on_segment(C, A, B):
    '''
    This function finds the projection proportion of AC on AB (t)
    The returned point E is the closest point along AB to C
    '''
    AB = B - A 
    AC = C - A 
    d = AB @ AB 
    if d == 0:  #if A and B are the same point
        return A, 0.0
    t = (AC @ AB) / d 
    t = max(0.0, min(1.0, t))
    E = A + t * AB
    return E

# used for difference between compas angle and wall normal vector
def angular_difference(a,b):
    return abs((a - b + 180) % 360 - 180)

def outward_normal(A, B, G, E):
    '''
    Find normal vector for line AB
    Use unit vector representation of AB to find two normal vectors
    Return the normal vector that points away from center (G)
    '''
    line = (B - A)
    length = np.linalg.norm(line)
    if length == 0: # when A = B
        return np.array([0.0, 0.0])
    t_hat = line / length
    n1 = np.array([-t_hat[1],  t_hat[0]]) #(x,y) -> (-y, x)
    n2 = np.array([ t_hat[1], -t_hat[0]]) #(x,y) -> (y, -x)
    if n1 @ (E - G) > 0: # E-G is vector pointing towards camera
        return n1
    else:
        return n2
    
def image_points_toward_wall(
    cam_lat, cam_lon, cam_bearing_deg,
    edge1_lat, edge1_lon, edge2_lat, edge2_lon,
    centroid_lat, centroid_lon, proj_local,
    fov_half_angle_deg=45.0
) -> tuple[bool, dict]:
    """
    Determine if a camera image points toward a wall segment, using consistent
    local projection geometry (same as extract_bbox_coordinates).
    """
    # convert all points to local (meters)
    C = to_local_xy(cam_lat, cam_lon, proj_local)
    A = to_local_xy(edge1_lat, edge1_lon, proj_local)
    B = to_local_xy(edge2_lat, edge2_lon, proj_local)
    G = to_local_xy(centroid_lat, centroid_lon, proj_local)

    # nearest point and vector math identical
    E = nearest_point_on_segment(C, A, B)
    v = E - C

    # bearing from camera to wall
    bearing_cam_to_wall = (math.degrees(math.atan2(v[0], v[1])) + 360) % 360
    delta = angular_difference(bearing_cam_to_wall, cam_bearing_deg)
    facing_ok = (delta <= fov_half_angle_deg)

    # outward normal, frontness same as before
    n_hat = outward_normal(A, B, G, E)
    frontness = float(n_hat @ (C - E))
    front_ok = (frontness > 0)

    keep = bool(facing_ok and front_ok)
    score = math.cos(math.radians(delta)) * max(0.0, frontness) / (np.linalg.norm(C - E) + 1e-6)

    return keep, {
        "delta_deg": float(delta),
        "frontness": frontness,
        "distance_m": float(np.linalg.norm(C - E)),
        "score": float(score)
    }


#--------------------------------------



#-----------------------------
# Step 3:
# Vision model functions
# Run model on desired images to detect entrance

def load_yolo_model(model_path: str, device: Optional[str] = None):
    """
    Load YOLOv8 model. `model_path` should point to the weights from:
      https://github.com/sayedmohamedscu/YOLOv8-Door-detection-for-visually-impaired-people
    """
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed. `pip install ultralytics`")

    model = YOLO(model_path)  # ultralytics auto-selects device unless specified
    if device is not None:
        # The ultralytics API selects device at predict() time; we’ll pass device then.
        model.overrides = model.overrides or {}
        model.overrides['device'] = device
    return model


def run_yolo_on_image(
    model,
    img: np.ndarray,
    conf_thr: float = 0.35,
    iou_thr: float = 0.5,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Run YOLO on a full image and return list of detections:
      [{ 'conf': float, 'bbox': (x1,y1,x2,y2), 'cls_id': int, 'cls_name': str }]
    """
    # Ultralytics expects BGR numpy or path
    results = model.predict(source=img, conf=conf_thr, iou=iou_thr, verbose=False, device=device)
    dets: List[Dict] = []
    if not results:
        return dets

    res = results[0]
    names = res.names
    if res.boxes is None:
        return dets

    for b in res.boxes:
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        conf = float(b.conf[0].cpu().numpy())
        cls_id = int(b.cls[0].cpu().numpy()) if b.cls is not None else -1
        cls_name = names.get(cls_id, str(cls_id))
        dets.append({"conf": conf, "bbox": tuple(xyxy.tolist()), "cls_id": cls_id, "cls_name": cls_name})
    return dets

'''
def run_yolo_on_rois(
    model,
    img: np.ndarray,
    rois: List[Tuple[int,int,int,int]],
    conf_thr: float = 0.35,
    iou_thr: float = 0.5,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Run YOLO only inside ROI crops, map detections back to full-image coords.
    Returns same dict format as run_yolo_on_image.
    """
    out: List[Dict] = []
    for (x1, y1, x2, y2) in rois:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        dets = run_yolo_on_image(model, crop, conf_thr=conf_thr, iou_thr=iou_thr, device=device)
        for d in dets:
            bx1, by1, bx2, by2 = d["bbox"]
            # map back to full image coords
            d["bbox"] = (bx1 + x1, by1 + y1, bx2 + x2, by2 + y2)
            out.append(d)
    return out
'''

# visualize detections
def _draw_dets(img: np.ndarray, dets: List[Dict], color=(0, 200, 0)) -> np.ndarray:
    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        conf = d["conf"]
        label = d.get("cls_name", "obj")
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def validate_folder_with_seg_and_yolo(
    images_dir: Path,
    yolo_weights_path: str,
    facade_tau: float = 0.01,
    use_rois: bool = False,
    conf_thr: float = 0.35,
    iou_thr: float = 0.5,
    device: Optional[str] = None,
    save_vis_dir: Optional[str] = None
)-> Tuple:
    """
    - Run quality filters
    - Optionally run segmentation to compute facade_presence_score and ROIs
    - Run YOLO (on ROIs or full image)
    - Save/print results for quick sanity check
    - Return image with bbox around detected doors as dictionary
    """
    # Load YOLO door model
    model = load_yolo_model(yolo_weights_path, device=device)

    if save_vis_dir:
        Path(save_vis_dir).mkdir(parents=True, exist_ok=True)


    #image_with_detection = Dict()
    
    img = cv2.imread(str(images_dir), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Could not read {images_dir}")
        return None
    '''
    mask, facade_score, rois = image_segmentation(img, facade_tau=facade_tau)
    print(f"[{p.name}] facade_presence_score={facade_score:.3f}  ROIs={len(rois)}")

    if facade_score < facade_tau:
        print("rejected (low facade score)")
        continue
    '''
    #if use_rois and rois:
        #dets = run_yolo_on_rois(model, img, rois, conf_thr=conf_thr, iou_thr=iou_thr, device=device)
    #else:
    dets = run_yolo_on_image(model, img, conf_thr=conf_thr, iou_thr=iou_thr, device=device)
    # Keep only door class if model has multiple classes; otherwise keep all
    door_dets = [d for d in dets if d.get("cls_name", "").lower().find("door") != -1 or True]

    print(f"detections: {len(door_dets)}")
    #assume for now there will only be one door detection in an image

    if save_vis_dir:
        vis = _draw_dets(img, door_dets)
        # draw ROI boxes
        '''
        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        '''
        out_path = Path(save_vis_dir) / f"{images_dir.stem}_vis.jpg"
        cv2.imwrite(str(out_path), vis)

    if len(door_dets) == 0:
        return None
    else:
        return door_dets[0]["bbox"] # (x1, y1, x2, y2)


#---------------------------------
# Step 4
# Coordinate Extraction Functions
# Given an image with a detected entrance, these functions will find the entrance point to the building


def intersect_ray_segment(ray_origin, direction, A, B):
    """
    Ray: C + t*d, t>=0 ;  Segment: A + u*(B-A), u in [0,1], t is positive scalar
    Solve C + t d = A + u v  =>  t d - u v = (A - C)
    Returns (hit_point, hit_bool)
    """
    AB = B - A
    M = np.stack([direction, -AB], axis=1)  # 2x2 matrix with direction vector and -AB as columns
    rhs = (A - ray_origin)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]

    if abs(det) < 1e-9: # no intersection point
        return None, False
    inv = np.array([[ M[1,1], -M[0,1] ],
                    [ -M[1,0], M[0,0] ]]) / det
    
    t, u = (inv @ rhs)
    if t is not None and u is not None and (t >= 0) and (0 <= u <= 1):
        hit = ray_origin + t*direction
        return hit, True
    return None, False

def horizontal_fov_to_fx(img_w, hfov_deg):
    # pinhole: fx = (W/2)/tan(hfov/2)
    return (img_w * 0.5) / math.tan((math.pi*(hfov_deg * 0.5))/180)

def pixel_to_yaw_offset_deg(u, img_w, fx):
    # optical center at img_w/2
    cx = img_w * 0.5
    yaw_rad = math.atan2((u - cx), fx)  # horizontal angle from camera forward axis
    return (180*yaw_rad)/math.pi

def bbox_bottom_center_xy(bbox_xywh, img_w, img_h):
    """
    bbox_xywh: [xc, yc, w, h] ; can be normalized (<=1.0) or pixels
    Returns pixel coords (u, v) bottom-center of the box.
    """
    xc, yc, bw, bh = bbox_xywh
    if max(xc, yc, bw, bh) <= 1.5:  # assume normalized
        xc, yc, bw, bh = xc*img_w, yc*img_h, bw*img_w, bh*img_h
    u = xc
    v = yc + 0.5*bh  # bottom center
    return float(u), float(v)


# given entrance bbox from model, convert center floor point (entrance) to lat/lon
# yolo bbox format is: [class_id x_center y_center width height]  (often normalized)

def extract_bbox_coordinates(
    image_dict: Dict,
    wall_AB_latlon: Tuple[float, float, float, float],
    centroid_lat: float,
    centroid_lon: float,
    hfov_deg: float = 45.0
) -> Tuple[float, float]:
    """
    Compute the geographic (lat, lon) of a detected entrance from a YOLO bbox,
    reusing geometric helper functions for clarity and consistency.
    """

    # parse inputs
    A_lat, A_lon, B_lat, B_lon = wall_AB_latlon
    cam_lat, cam_lon = image_dict["coordinates"]
    cam_bearing_deg = image_dict["compass_angle"]
    bbox = image_dict["bbox"]

    # load image to get size
    img = cv2.imread(image_dict["image_path"])
    if img is None:
        raise FileNotFoundError(f"Could not read {image_dict['image_path']}")
    img_h, img_w = img.shape[:2]

    # compute bottom-center pixel of bbox
    x1, y1, x2, y2 = bbox
    # normalize
    bbox_xywh = [
        (x1 + x2) / 2.0 / img_w,
        (y1 + y2) / 2.0 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h
    ]
    u, _ = bbox_bottom_center_xy(bbox_xywh, img_w, img_h)

    # convert pixel offset to world yaw
    fx = horizontal_fov_to_fx(img_w, hfov_deg)
    yaw_off_deg = pixel_to_yaw_offset_deg(u, img_w, fx)
    world_yaw_deg = (cam_bearing_deg + yaw_off_deg) % 360.0
    theta = math.radians(world_yaw_deg)
    d = np.array([math.sin(theta), math.cos(theta)], dtype=float)

    # local projection (Azimuthal Equidistant around building centroid)
    proj_local = pyproj.Proj(
        proj="aeqd", lat_0=centroid_lat, lon_0=centroid_lon,
        ellps="WGS84", units="m"
    )
    to_local = pyproj.Transformer.from_crs("EPSG:4326", proj_local, always_xy=True)
    to_geo = pyproj.Transformer.from_crs(proj_local, "EPSG:4326", always_xy=True)

    # convert to local XY coordinates
    Cx, Cy = to_local.transform(cam_lon, cam_lat)
    Ax, Ay = to_local.transform(A_lon, A_lat)
    Bx, By = to_local.transform(B_lon, B_lat)
    C, A, B = np.array([Cx, Cy]), np.array([Ax, Ay]), np.array([Bx, By])

    # ray–segment intersection
    hit, ok = intersect_ray_segment(C, d, A, B)
    if not ok:
        hit, _ = nearest_point_on_segment(C, A, B)

    # convert back to geographic coordinates
    lon_e, lat_e = to_geo.transform(hit[0], hit[1])
    return float(lat_e), float(lon_e)


def run_inference(data, fov_half_angle, 
                  yolo_weights, facade_tau, use_rois, 
                  conf, iou, device, save_vis):
    
    # data will be dictionary returned from api.py get_building_package_for_point

    if not _HAS_ULTRALYTICS:
        raise SystemExit("ERROR: ultralytics not installed. Try: pip install ultralytics")

    building_id = data["building_id"]
    centroid_lat, centroid_lon = data["building_center"][0], data["building_center"][1]
    all_images = data["image_dicts"]
    place = data["place"]
    if place is None:
        print("No matching place found")
    wall_points_lat_lon = data["walls"]

    # all_images : [ {"image_path": ./img, "compass_angle": angle, "coordinates": [lat,lon]}, {...} ]
    all_images = filter_images_by_quality(all_images, sharpness_thresh = 100.0, 
                                        min_width = 300, min_height= 300, 
                                        dark_thresh = 0.05, bright_thresh = 0.95)

    # Main Loop:
    # Filter candidate images to point at current wall 
    # Run loop until entrance has been found
    # Edge case to consider: finding an entrance on multiple walls

    potential_entrance = None
    proj_local = make_local_proj(centroid_lat, centroid_lon)
    for wall in wall_points_lat_lon:
        edge1_lat, edge1_lon, edge2_lat, edge2_lon = wall[0][0], wall[0][1], wall[1][0], wall[1][1]
        max_frontness = 0
        best_img = {} #best image will be the single dictionary with image path, coordinates, and compass angle
        for img in all_images:
            cam_lat, cam_lon, cam_bearing_deg = img["coordinates"][0], img["coordinates"][1], img["compass_angle"]
            check, img_scores = image_points_toward_wall(
                                cam_lat, cam_lon, cam_bearing_deg,
                                edge1_lat, edge1_lon, edge2_lat, edge2_lon,
                                centroid_lat, centroid_lon,
                                proj_local, fov_half_angle
                                )
            if check:
                if img_scores["frontness"] > max_frontness:
                    max_frontness = img_scores["frontness"]
                    best_img = img
                    best_wall = wall

        # no images facing towards wall so move to next wall
        if len(best_img) == 0:
            continue

        # best_img = {"image_path": ./img, "compass_angle": angle, "coordinates": [lat,lon]}
    
        # I'm going to ignore multiple entrance points edge case for now and stop when one is found
        detection = validate_folder_with_seg_and_yolo(
                    best_img['image_path'], yolo_weights,
                    facade_tau, use_rois, conf, iou, device, save_vis)
        
        if detection is not None:
            # found a entrance so end loop and extract entrance
            break
    if detection is None:
        print("No entrances detected!")
        
    else:
        best_img["bbox"] = detection #detection is (x1, y1, x2, y2)
        wall_tuple = (best_wall[0][0], best_wall[0][1], best_wall[1][0], best_wall[1][1])
        entrance_point = extract_bbox_coordinates(best_img, wall_tuple,
                            centroid_lat, centroid_lon, hfov_deg=45.0)
        print(f"Entrance Found at ({entrance_point[0]}, {entrance_point[1]})")