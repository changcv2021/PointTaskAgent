sensor = np.asarray(scene.sensor_origin, dtype=float)
ray_support = scene.points(bounds=[target_xyz - 1.0, target_xyz + 1.0])

def ray_surface_test(cid, points):
    xyz = np.asarray(points.xyz)
    ids = np.asarray(points.point_ids)
    original = np.asarray(scene.labels_for(ids, "KEEP"))
    preview = np.asarray(scene.labels_for(ids, cid))
    j = int(np.flatnonzero(ids == 2020)[0])
    relative = xyz - sensor
    ranges = np.linalg.norm(relative, axis=1)
    directions = relative / ranges[:, None]
    target_ray = directions[j]
    angles = np.arccos(np.clip(directions @ target_ray, -1, 1))
    records = []
    for cls in (5, 6):
        eligible = np.flatnonzero(
            (original == cls) & (~np.isin(ids, targets))
        )
        ordered = eligible[np.argsort(angles[eligible])]
        nearest = ordered[:5]
        fits = []
        for k in (12, 24):
            ix = ordered[:k]
            center = xyz[ix].mean(axis=0)
            eigenvalues, vectors = np.linalg.eigh(np.cov((xyz[ix] - center).T))
            normal = vectors[:, 0]
            denom = float(target_ray @ normal)
            predicted_range = (
                float((center - sensor) @ normal / denom)
                if abs(denom) > 1e-8 else None
            )
            intersection = (
                sensor + predicted_range * target_ray
                if predicted_range is not None else None
            )
            residuals = (xyz[ix] - center) @ normal
            fits.append({
                "k_actual": len(ix), "fit_point_ids": ids[ix].tolist(),
                "max_angular_distance_deg": float(np.degrees(angles[ix].max())),
                "neighbor_plane_rms_m": float(np.sqrt(np.mean(residuals**2))),
                "eigenvalues_m2": eigenvalues.tolist(),
                "ray_normal_dot": denom,
                "plane_predicted_range_m": predicted_range,
                "target_minus_plane_range_m": (
                    float(ranges[j] - predicted_range)
                    if predicted_range is not None else None
                ),
                "intersection_to_nearest_fit_return_m": (
                    float(np.linalg.norm(xyz[ix] - intersection, axis=1).min())
                    if intersection is not None else None
                ),
                "intersection_to_fit_center_m": (
                    float(np.linalg.norm(intersection - center))
                    if intersection is not None else None
                )
            })
        records.append({
            "reference_S0_class": cls,
            "nearest_angular_returns": [
                {"id": int(ids[i]), "angle_deg": float(np.degrees(angles[i])),
                 "range_m": float(ranges[i]),
                 "target_minus_return_range_m": float(ranges[j] - ranges[i])}
                for i in nearest
            ],
            "plane_extrapolations": fits
        })
    return {
        "target_id": 2020, "target_range_m": float(ranges[j]),
        "original_label": int(original[j]), "preview_label": int(preview[j]),
        "surface_context": records
    }

comparison3 = diagnose.compare(
    "Which neighboring measured surface is consistent with the disputed return along its actual sensor ray?",
    ray_surface_test, candidate_ids=cids, point_ids=ray_support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "CANDIDATES", "SENSOR_POSE"],
    assumptions=[
        "Planar ray intersections are diagnostic extrapolations, not synthetic observed points.",
        "S0 classes identify competing context only; neither class subset is a confirmed instance.",
        "All changed targets are excluded from every reference fit for every candidate.",
        "This exclusion does not establish exclusion from candidate generation."
    ]
)
print("E3", comparison3["evidence_id"])
print("RESULT", comparison3["results"]["KEEP"])
print("PREVIEW_LABELS", {
    cid: result["result"]["preview_label"]
    if result["status"] == "OK" else result["error"]
    for cid, result in comparison3["results"].items()
})
print("SOURCE_CHECKS", comparison3["source_checks"])

