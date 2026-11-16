print("E1_SOURCE_CHECKS", comparison1["source_checks"])
target_xyz = scene.points(point_ids=targets).xyz[0]
wide_support = scene.points(bounds=[target_xyz - 0.8, target_xyz + 0.8])

def surface_separation_test(cid, points):
    ids = np.asarray(points.point_ids)
    xyz = np.asarray(points.xyz)
    original = np.asarray(scene.labels_for(ids, "KEEP"))
    preview = np.asarray(scene.labels_for(ids, cid))
    j = int(np.flatnonzero(ids == 2020)[0])
    distances = np.linalg.norm(xyz - xyz[j], axis=1)
    order = np.argsort(distances)
    order = order[ids[order] != 2020]
    fits = []
    for k in (8, 20, 40):
        ix = order[:k]
        center = xyz[ix].mean(axis=0)
        ev, vectors = np.linalg.eigh(np.cov((xyz[ix] - center).T))
        normal = vectors[:, 0]
        residuals = (xyz[ix] - center) @ normal
        fits.append({
            "k": k, "fit_point_ids": ids[ix].tolist(),
            "furthest_fit_distance_m": float(distances[ix].max()),
            "eigenvalues_m2": ev.tolist(),
            "target_plane_offset_m": float(abs((xyz[j] - center) @ normal)),
            "neighbor_plane_rms_m": float(np.sqrt(np.mean(residuals**2))),
            "neighbor_max_plane_offset_m": float(np.max(np.abs(residuals))),
            "normal": normal.tolist()
        })
    nearest_by_original_class = {}
    for cls in (5, 6):
        ix = np.flatnonzero((original == cls) & (ids != 2020))
        ix = ix[np.argsort(distances[ix])[:5]]
        nearest_by_original_class[str(cls)] = [
            {"id": int(ids[i]), "distance_m": float(distances[i]),
             "relative_xyz": (xyz[i] - xyz[j]).tolist()}
            for i in ix
        ]
    return {
        "target_id": 2020, "original_label": int(original[j]),
        "preview_label": int(preview[j]),
        "label_blind_surface_fits": fits,
        "nearest_by_original_class": nearest_by_original_class,
        "changed_ids_on_common_support": ids[preview != original].tolist()
    }

comparison2 = diagnose.compare(
    "Is the disputed return close to a label-blind local surface and separated from other pedestrian returns?",
    surface_separation_test, candidate_ids=cids,
    point_ids=wide_support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "CANDIDATES"],
    assumptions=[
        "Plane fits describe local measured geometry, not completed surfaces or instances.",
        "Classes contextualize the same fixed geometry; they are not independent votes.",
        "All candidates use identical support and neighborhood definitions."
    ]
)
print("E2", comparison2["evidence_id"])
print("E2_RESULTS", {
    cid: result["result"] if result["status"] == "OK" else result["error"]
    for cid, result in comparison2["results"].items()
})
print("E2_SOURCE_CHECKS", comparison2["source_checks"])

view_points = wide_support[
    np.linalg.norm(wide_support.xyz - target_xyz, axis=1) <= 0.7
]
camera2 = viz.make_camera(
    look_at=target_xyz, eye_direction=(0, 0, 1), up=(0, 1, 0),
    projection="orthographic", span_m=1.5, image_size=800
)
for cid in ("KEEP", "TRACK_TO_SEG:0"):
    view = viz.render(
        view_points, camera2, box_ids=(), color_mode="semantic",
        labels=scene.labels_for(view_points.point_ids, cid),
        highlight_ids=targets, title=cid + " | point 2020 | identical BEV support"
    )
    show(view, "Same current returns and camera; highlighted point 2020. "
         "TRACK_TO_SEG:1 has the identical executable label change.")

