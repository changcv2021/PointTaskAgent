sensor = np.asarray(scene.sensor_origin, dtype=float)

def grazing_return_profile(cid, points):
    labels = np.asarray(scene.labels_for(points.point_ids, cid))
    original = np.asarray(points.semantic_labels)
    rays = points.xyz-sensor
    ranges = np.linalg.norm(rays, axis=1)
    azimuth = np.arctan2(rays[:,1], rays[:,0])
    elevation = np.arctan2(rays[:,2], np.linalg.norm(rays[:,:2], axis=1))
    outputs = []
    for pid in affected:
        i = int(np.flatnonzero(points.point_ids == pid)[0])
        offsets = points.xyz-points.xyz[i]
        dxy = np.linalg.norm(offsets[:,:2], axis=1)
        road_ix = np.flatnonzero(
            (original == 10) & (points.point_ids != pid)
            & (dxy <= 1.25) & (np.abs(offsets[:,2]) < 0.4))
        A = np.column_stack([offsets[road_ix,:2], np.ones(len(road_ix))])
        coef = np.linalg.lstsq(A, offsets[road_ix,2], rcond=None)[0]
        normal = np.array([-coef[0], -coef[1], 1.0])
        road_at_target_xy = points.xyz[i,2]+coef[2]
        plane_constant = road_at_target_xy-coef[:2]@points.xyz[i,:2]
        directions = rays/ranges[:,None]
        predicted_ground_range = (plane_constant-normal@sensor)/(directions@normal)
        heights = offsets[:,2]-offsets[:,:2]@coef[:2]-coef[2]
        da = np.arctan2(np.sin(azimuth-azimuth[i]), np.cos(azimuth-azimuth[i]))
        de = elevation-elevation[i]
        profiles = []
        for tolerance in (0.001, 0.002):
            ix = np.flatnonzero((np.abs(da) <= 0.10) & (np.abs(de) <= tolerance))
            ix = ix[np.argsort(da[ix])]
            profiles.append({
                "elevation_tolerance_rad": tolerance,
                "returns": [{
                    "point_id": int(points.point_ids[j]),
                    "azimuth_offset_deg": float(np.degrees(da[j])),
                    "elevation_offset_deg": float(np.degrees(de[j])),
                    "range_m": float(ranges[j]),
                    "return_short_of_ground_m": float(predicted_ground_range[j]-ranges[j]),
                    "height_above_road_m": float(heights[j]),
                    "preview_label": int(labels[j])
                } for j in ix]
            })
        outputs.append({
            "target_id": int(pid), "assigned_label": int(labels[i]),
            "target_elevation_deg": float(np.degrees(elevation[i])),
            "road_anchor_ids": points.point_ids[road_ix].tolist(),
            "angular_profiles": profiles
        })
    return outputs

comparison3 = diagnose.compare(
    "Is there a road-to-car range transition at the changed low-elevation return?",
    grazing_return_profile, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "SENSOR_POSE", "S0", "CANDIDATES"],
    assumptions=["Similar global elevation is a diagnostic subset, not a confirmed sensor ring.",
                 "Ray-plane intersection is a local road hypothesis, not synthetic observed geometry.",
                 "Near-contact or mixed returns may remain ambiguous."]
)
print("evidence_id", comparison3["evidence_id"])
print("source_checks", comparison3["source_checks"])
for cid in cids:
    result = comparison3["results"][cid]
    print(cid, result["status"],
          result["result"][0]["assigned_label"] if result["status"] == "OK" else result["error"])
print("Shared profiles with KEEP labels:", comparison3["results"]["KEEP"]["result"])

