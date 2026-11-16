origin = np.asarray(scene.sensor_origin)
target_points = scene.points(point_ids=disputed_ids)
target_xyz = np.asarray(target_points.xyz)
upper_ids = disputed_ids[target_local[:, 2] > 0.6]
upper_height = float(np.median(target_xyz[target_local[:, 2] > 0.6, 2]))

def upper_row_continuity(cid, points):
    ids = np.asarray(points.point_ids)
    xyz = np.asarray(points.xyz)
    local = np.asarray(geom.to_box_coordinates(points, box0))
    base = np.asarray(scene.labels_for(ids, "KEEP"))
    preview = np.asarray(scene.labels_for(ids, cid))
    delta = xyz-origin
    ranges = np.linalg.norm(delta, axis=1)
    azimuth = np.arctan2(delta[:, 1], delta[:, 0])
    elevation = np.arctan2(delta[:, 2], np.linalg.norm(delta[:, :2], axis=1))
    disputed = np.isin(ids, disputed_ids)
    # Height-selected context is fixed across candidates and not constrained to the box side.
    high = ((np.abs(xyz[:, 2]-upper_height) < 0.10) &
            (np.linalg.norm(xyz[:, :2]-target_xyz[:, :2].mean(axis=0), axis=1) < 3.0))
    selected = np.flatnonzero(high)
    selected = selected[np.argsort(azimuth[selected])]
    rows = []
    for i in selected:
        rows.append({"id": int(ids[i]), "S0": int(base[i]),
                     "preview": int(preview[i]), "disputed": bool(disputed[i]),
                     "box_local_xyz": local[i].tolist(), "range_m": float(ranges[i]),
                     "azimuth_deg": float(np.degrees(azimuth[i])),
                     "elevation_deg": float(np.degrees(elevation[i]))})
    edges = []
    for a, b in zip(selected[:-1], selected[1:]):
        edges.append({
            "ids": [int(ids[a]), int(ids[b])],
            "original_labels": [int(base[a]), int(base[b])],
            "preview_labels": [int(preview[a]), int(preview[b])],
            "touches_disputed": bool(disputed[a] or disputed[b]),
            "xyz_gap_m": float(np.linalg.norm(xyz[a]-xyz[b])),
            "range_jump_m": float(abs(ranges[a]-ranges[b])),
            "azimuth_gap_deg": float(abs(np.degrees(azimuth[a]-azimuth[b])))
        })
    fixed_high_truck = high & (base == 9) & ~disputed
    refidx = np.flatnonzero(fixed_high_truck)
    queries = np.flatnonzero(np.isin(ids, upper_ids))
    nearest = []
    for i in queries:
        row = {"id": int(ids[i]), "preview": int(preview[i])}
        if len(refidx):
            distances = np.linalg.norm(xyz[refidx]-xyz[i], axis=1)
            j = refidx[np.argmin(distances)]
            row.update({"nearest_same_height_truck_id": int(ids[j]),
                        "distance_m": float(np.min(distances)),
                        "box_local_offset": (local[i]-local[j]).tolist()})
        nearest.append(row)
    return {"height_band_global_z_m": [upper_height-0.10, upper_height+0.10],
            "upper_row_points": rows, "adjacent_observed_row_pairs": edges,
            "nearest_same_height_truck": nearest,
            "fixed_high_truck_ids": ids[refidx].tolist()}

comparison3 = diagnose.compare(
    "Is the disputed upper row laterally contiguous with unchanged truck returns at the same height?",
    upper_row_continuity, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT", "SENSOR_POSE", "S0", "D0", "CANDIDATES"],
    assumptions=["Height band is shared and does not establish instance identity.",
                 "Azimuth ordering uses actual sensor pose, not assumed ring IDs.",
                 "A continuous observed row can still cross a semantic boundary."])
print("E3", comparison3["evidence_id"], "source_checks", comparison3.get("source_checks"))
for cid in cids:
    rr = comparison3["results"][cid]
    print(cid, json.dumps(rr["result"] if cid == "KEEP" and rr["status"] == "OK"
                         else {"status": rr["status"],
                               "nearest_same_height_truck": rr.get("result", {}).get("nearest_same_height_truck"),
                               "error": rr.get("error")}, indent=2))

# Matched BEV previews distinguish lateral offset from a mere height gap.
camera_bev = viz.make_camera(look_at=box0.center, eye_direction=(0, 0, 1),
                             up=box0.rotation[:, 0].tolist(),
                             span_m=7.0, image_size=850)
for cid in ["KEEP", "DET_TO_SEG:0"]:
    v = viz.render(wide, camera_bev, box_ids=(), color_mode="semantic",
                   labels=scene.labels_for(wide.point_ids, cid),
                   highlight_ids=disputed_ids,
                   title=cid + ": same wider support, top view")
    show(v, "Matched top view; highlights are the identical eight disputed original points.")

