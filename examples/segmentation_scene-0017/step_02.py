# Compactly expose the previous comparison and its dependency checks.
r1 = comparison1["results"]["KEEP"]["result"]
print("E1 compact", {
    "id": comparison1["evidence_id"],
    "candidate_changes": {c: comparison1["results"][c]["result"]["changed_ids"] for c in cids},
    "nearest_distances_m": [r["nearest_distance_m"] for r in r1["local_surface_measurements"]],
    "source_checks": comparison1.get("source_checks")
})

box0 = scene.reference_boxes[0]
xyz0 = np.asarray(support.xyz)
wide = scene.points(bounds=[(xyz0.min(axis=0)-3.0).tolist(),
                            (xyz0.max(axis=0)+3.0).tolist()])
target_local = np.asarray(geom.to_box_coordinates(
    scene.points(point_ids=disputed_ids), box0))
upper_target = target_local[target_local[:, 2] > np.median(target_local[:, 2])-0.15]
side_y = float(np.median(target_local[:, 1]))
upper_z = float(np.median(upper_target[:, 2]))
x_limits = [float(target_local[:, 0].min()-0.6),
            float(target_local[:, 0].max()+0.6)]

def contextual_surface_test(cid, points):
    ids = np.asarray(points.point_ids)
    xyz = np.asarray(points.xyz)
    local = np.asarray(geom.to_box_coordinates(points, box0))
    base = np.asarray(scene.labels_for(ids, "KEEP"))
    labels = np.asarray(scene.labels_for(ids, cid))
    disputed = np.isin(ids, disputed_ids)
    queries = np.flatnonzero(disputed)
    fixed_truck = (base == 9) & ~disputed
    fixed_manmade = (base == 14) & ~disputed
    side_support = (fixed_truck &
                    (np.abs(local[:, 1]-side_y) < 0.25) &
                    (local[:, 0] >= x_limits[0]) &
                    (local[:, 0] <= x_limits[1]))
    idx = np.flatnonzero(side_support)
    out = {"support_count": len(ids), "disputed_ids": ids[queries].tolist(),
           "disputed_preview_labels": labels[queries].tolist(),
           "side_support_ids": ids[idx].tolist(),
           "side_support_box_local_xyz": local[idx].tolist()}
    if len(idx) >= 6:
        ref = local[idx]
        mean = ref.mean(axis=0)
        _, sv, axes = np.linalg.svd(ref-mean, full_matrices=False)
        normal = axes[-1]
        out["side_plane"] = {
            "normal_box_local": normal.tolist(), "singular_values": sv.tolist(),
            "fixed_abs_residual_quantiles_m":
                np.quantile(np.abs((ref-mean)@normal), [0.5, 0.9, 1]).tolist(),
            "disputed_signed_residual_m": ((local[queries]-mean)@normal).tolist()
        }
    else:
        out["side_plane"] = {"limitation": "fewer than six fixed side-support points"}
    near_upper_row = ((np.abs(local[:, 1]-side_y) < 0.25) &
                      (np.abs(local[:, 2]-upper_z) < 0.15) & ~disputed)
    rowidx = np.flatnonzero(near_upper_row)
    out["upper_row_context"] = [
        {"id": int(ids[i]), "local_xyz": local[i].tolist(),
         "S0_label": int(base[i]), "preview_label": int(labels[i]),
         "inside_source_box": bool(geom.points_in_box(xyz[i:i+1], box0)[0])}
        for i in rowidx
    ]
    nearest = []
    for i in queries:
        row = {"id": int(ids[i])}
        for name, mask in [("truck", fixed_truck), ("manmade", fixed_manmade)]:
            j = np.flatnonzero(mask)
            if len(j):
                dist = np.linalg.norm(xyz[j]-xyz[i], axis=1)
                selected = np.argsort(dist)[:3]
                row[name] = [{"id": int(ids[j[k]]), "distance_m": float(dist[k]),
                              "local_xyz": local[j[k]].tolist()} for k in selected]
            else:
                row[name] = []
        nearest.append(row)
    out["nearest_unchanged_semantic_context"] = nearest
    return out

comparison2 = diagnose.compare(
    "Does wider context support a vehicle-side sampling gap or a distinct manmade continuation?",
    contextual_surface_test, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "D0", "CANDIDATES"],
    assumptions=["Side slab is fixed from common disputed geometry, not candidate labels.",
                 "Plane support can include different objects; residuals are not instance confirmation.",
                 "No observed continuation is not proof of empty space."])
print("E2 source_checks", comparison2.get("source_checks"))
for cid in cids:
    rr = comparison2["results"][cid]
    if rr["status"] == "OK":
        result = rr["result"]
        print(cid, json.dumps(result if cid == "KEEP" else {
            "disputed_ids": result["disputed_ids"],
            "disputed_preview_labels": result["disputed_preview_labels"]}, indent=2))
    else:
        print(cid, rr)

# A box-side elevation retains vertical separation while revealing external context.
cam_side = viz.make_camera(
    look_at=box0.center,
    eye_direction=box0.rotation[:, 1].tolist(),
    up=(0, 0, 1), span_m=10.0, image_size=1000)
view2 = viz.render(wide, cam_side, box_ids=(), color_mode="semantic",
                   labels=scene.labels_for(wide.point_ids, "KEEP"),
                   highlight_ids=disputed_ids,
                   title="Wider current scan: side elevation, exact disputed points")
show(view2, "Same current LiDAR, wider contextual support; no inferred surfaces.")

