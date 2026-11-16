lower_height = float(np.median(scene.points(
    point_ids=[19989, 20021, 20053, 20085]).xyz[:, 2]))

def vertical_range_profile(cid, points):
    ids = np.asarray(points.point_ids)
    xyz = np.asarray(points.xyz)
    local = np.asarray(geom.to_box_coordinates(points, box0))
    base = np.asarray(scene.labels_for(ids, "KEEP"))
    preview = np.asarray(scene.labels_for(ids, cid))
    d = xyz-origin
    rng = np.linalg.norm(d, axis=1)
    horizontal_rng = np.linalg.norm(d[:, :2], axis=1)
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arctan2(d[:, 2], horizontal_rng)
    disputed = np.isin(ids, disputed_ids)
    lower = (np.abs(xyz[:, 2]-lower_height) < 0.10) & ~disputed
    upper = ((np.abs(xyz[:, 2]-upper_height) < 0.10) &
             (np.linalg.norm(xyz[:, :2]-target_xyz[:, :2].mean(axis=0), axis=1) < 3.0))
    lowidx = np.flatnonzero(lower)
    upidx = np.flatnonzero(upper)
    pairs = []
    for i in upidx:
        angle_diff = np.abs(np.arctan2(np.sin(az[lowidx]-az[i]),
                                       np.cos(az[lowidx]-az[i])))
        if len(lowidx) == 0:
            continue
        j = lowidx[np.argmin(angle_diff)]
        if np.degrees(np.min(angle_diff)) > 0.12:
            continue
        pairs.append({
            "upper_id": int(ids[i]), "lower_id": int(ids[j]),
            "disputed_upper": bool(disputed[i]),
            "S0_labels": [int(base[i]), int(base[j])],
            "preview_labels": [int(preview[i]), int(preview[j])],
            "azimuth_gap_deg": float(np.degrees(np.min(angle_diff))),
            "elevation_gap_deg": float(np.degrees(el[i]-el[j])),
            "signed_range_difference_m": float(rng[i]-rng[j]),
            "signed_horizontal_range_difference_m": float(horizontal_rng[i]-horizontal_rng[j]),
            "box_local_offset_upper_minus_lower": (local[i]-local[j]).tolist()
        })
    control = [p for p in pairs if not p["disputed_upper"] and p["S0_labels"] == [9, 9]]
    targets = [p for p in pairs if p["disputed_upper"]]
    def summary(rows):
        if not rows:
            return {"count": 0}
        return {
            "count": len(rows),
            "horizontal_range_abs_difference_quantiles_m": np.quantile(
                [abs(p["signed_horizontal_range_difference_m"]) for p in rows],
                [0, .5, .9, 1]).tolist(),
            "elevation_gap_quantiles_deg": np.quantile(
                [p["elevation_gap_deg"] for p in rows], [0, .5, 1]).tolist(),
            "pair_ids": [[p["upper_id"], p["lower_id"]] for p in rows]
        }
    return {
        "lower_height_band_m": [lower_height-.10, lower_height+.10],
        "upper_height_band_m": [upper_height-.10, upper_height+.10],
        "matching_rule": "nearest azimuth in common lower-height band; maximum 0.12 degrees",
        "target_pairs": targets,
        "unchanged_truck_control_summary": summary(control),
        "unchanged_truck_control_pairs": control,
        "unmatched_disputed_upper_ids": [
            int(i) for i in upper_ids if int(i) not in {p["upper_id"] for p in targets}]
    }

comparison4 = diagnose.compare(
    "Do matched-azimuth vertical pairs indicate continuous depth on the disputed side?",
    vertical_range_profile, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT", "SENSOR_POSE", "S0", "D0", "CANDIDATES"],
    assumptions=["Matching uses observed angles, not confirmed laser-channel identities.",
                 "Similar depth across heights supports a surface continuation, not confirmed object identity.",
                 "Controls and disputed points use identical height bands and angular tolerance."])
print("evidence_id", comparison4["evidence_id"])
print("source_checks", comparison4.get("source_checks"))
for cid in cids:
    rr = comparison4["results"][cid]
    if rr["status"] != "OK":
        print(cid, rr)
    else:
        r = rr["result"]
        if cid == "KEEP":
            print(cid, json.dumps(r, indent=2))
        else:
            print(cid, {
                "target_pair_ids": [[p["upper_id"], p["lower_id"]] for p in r["target_pairs"]],
                "preview_pair_labels": [p["preview_labels"] for p in r["target_pairs"]],
                "unmatched_disputed_upper_ids": r["unmatched_disputed_upper_ids"]
            })

