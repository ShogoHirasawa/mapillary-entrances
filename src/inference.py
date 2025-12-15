from src.utils.inference_utils import *
from src.utils.mapillary_utils import _is_360
from collections import defaultdict
from src.utils.geo_utils import _haversine
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False


def run_inference(data, yolo_weights, conf, iou, device, save_vis):

    if not _HAS_ULTRALYTICS:
        raise SystemExit("ERROR: ultralytics not installed. Try: pip install ultralytics")

    centroid_lon, centroid_lat = data['input_coordinates'][0], data['input_coordinates'][1]
    all_images = data['image_dicts']

    # image_compass_angles -> {img_path : angle, ...}
    image_compass_angles = {}
    for img_path in all_images:
        # build dictionary of all compass angles to be easily accessed later
        path = img_path['image_path']
        if _is_360(img_path):
            image_compass_angles[path] = "360"
        else:
            image_compass_angles[path] = img_path['compass_angle']
    
    place_names = data['places']
    
    # buildings : {building_id: [(wall_point_lon, wall_point_lat), ...], building_id: [...]}
    buildings_lat_lon = data['building_polygons']

    # perform basic filtering based on brightness/quality
    print(f"All images pre image filtering by quality: {len(all_images)}")
    all_images = filter_images_by_quality(all_images, sharpness_thresh = 100.0, 
                                        dark_thresh = 0.05, bright_thresh = 0.95)
    print(f" All images Dictionary post filtering by quality: {len(all_images)}")


    # create full x,y approximation of images and building polygons coordinates
    # use input coordinates as reference (0,0)
    proj_local = make_local_proj(centroid_lat, centroid_lon)
    
    # convert all building polygons to local coordinates
    # buildings_xy -> {building_id_i : [(wall_point_1_x, wall_point_1_y), (wall_point_2_x, wall_point_2_y), ...], building_id_j : {...}}
    buildings_xy = {}
    for id in buildings_lat_lon.keys():
        polygon_xy = [to_local_xy(wall_tuple[0], wall_tuple[1], proj_local) for wall_tuple in buildings_lat_lon[id]]
        buildings_xy[id] = polygon_xy


    # convert all image coordinates to local coordinates
    # images_xy -> {image_path : (x, y), image_path : (x, y), ...}
    images_xy = {}
    for img in all_images:
        images_xy[img['image_path']] = to_local_xy(img['coordinates'][0], img['coordinates'][1], proj_local)

    # main loop
    # run model on each image get entrance detections
    # for each entrance detected, map x,y approximation to associated building polygon, 
    # entrances are matched with nearest building polygon, iff the image was pointing towards that building
    # convert entrance point back to lon,lat, attach to entrances dictionary with associated building id
    # building_entrances -> {building_id : (lon,lat), ...}

    #building_entrances = {}
    building_entrances = []
    for img in all_images:
        path = img['image_path']
        detections = validate_folder_with_seg_and_yolo(path, yolo_weights, conf, iou, device, save_vis)
        # when no detections are made, move to next image
        if detections is None:
            continue
        if len(detections) > 1:
            print(f"Found {len(detections)} in image: {path}")
        entrances_xy = []
        for d in detections:
            C, dir_xy = extract_bbox_coordinates(img, d, proj_local, get_fov_half_angle(img))
            bid, candidates = match_entrance_to_building(
                (C, dir_xy), buildings_xy, max_range_m=60.0
            )

            if bid is None or not candidates:
                continue

            best = select_exterior_seg(candidates, C)
            if best is None:
                continue

            hit_xy = best["hit"]
            seg = best["segment"]

            print("Building matched")
            entrances_xy.append({"camera_xy":C,"bid":bid, "image_path":path, "hit":hit_xy, "wall_segment": (seg[0].tolist(), seg[1].tolist())})
        building_entrances.extend(entrances_xy)

    print(f"len(building_entrances) : {len(building_entrances)}")

    if len(building_entrances) > 0:
        building_entrances = clamp_entrance_to_hit_segment(proj_local, building_entrances)
        
        groups = defaultdict(list)
        for d in building_entrances:
            bid = d.get("bid")
            ent = d.get("entrance")
            if bid and ent and len(ent) == 2:
                groups[bid].append(d)

        # remove duplicates within each building group
        deduped = []
        for bid, items in groups.items():
            clusters = []
            for it in items:
                lon, lat = it["entrance"]
                placed = False
                for cl in clusters:
                    _, (lon0, lat0) = cl[0]
                    if _haversine(lat, lon, lat0, lon0) < 5.0:
                        cl.append((it, (lon, lat)))
                        placed = True
                        break
                if not placed:
                    clusters.append([(it, (lon, lat))])

            # from each cluster, keep mean lon/lat
            for cl in clusters:
                lons = [pt[1][0] for pt in cl]
                lats = [pt[1][1] for pt in cl]
                lon_mean = float(np.mean(lons))
                lat_mean = float(np.mean(lats))

                # build a clean entrance record
                rep = {
                    "bid": bid,
                    "entrance": (lon_mean, lat_mean),
                    "snapped": any(it.get("snapped", False) for (it, _) in cl),
                    "image_path": cl[0][0].get("image_path"),
                    "hit": cl[0][0].get("hit"),
                    "wall_segment": cl[0][0].get("wall_segment"),
                }
                deduped.append(rep)

        building_entrances = deduped
        

    return building_entrances, buildings_lat_lon, place_names