def surface_continuity(cid, points):
    preview = np.asarray(scene.labels_for(points.point_ids, cid))
    original = np.asarray(points.semantic_labels)
    outputs = []
    for pid in affected:
        i = int(np.flatnonzero(points.point_ids == pid)[0])
        q = points.xyz[i]
        offsets = points.xyz-q
        dxy = np.linalg.norm(offsets[:,:2], axis=1)
        d3 = np.linalg.norm(offsets, axis=1)
        fits = []
        for radius in (0.75, 1.25, 2.0):
            ix = np.flatnonzero((original == 10) & (points.point_ids != pid)
                                & (dxy <= radius) & (np.abs(offsets[:,2]) < 0.4))
            if len(ix) < 6:
                fits.append({"radius": radius, "n": len(ix), "insufficient": True})
                continue
            A = np.column_stack([offsets[ix,:2], np.ones(len(ix))])
            coef, _, rank, singular = np.linalg.lstsq(A, offsets[ix,2], rcond=None)
            residuals = offsets[ix,2]-A@coef
            car_ix = np.flatnonzero((original == 3) & (d3 <= 0.6)
                                    & (points.point_ids != pid))
            car_height = offsets[car_ix,2] - (
                offsets[car_ix,:2]@coef[:2]+coef[2])
            fits.append({
                "radius": radius, "road_anchor_ids": points.point_ids[ix].tolist(),
                "rank": int(rank), "design_singular_values": singular.tolist(),
                "road_slope_xy": coef[:2].tolist(),
                "target_height_above_road_m": float(-coef[2]),
                "road_residual_rmse_m": float(np.sqrt(np.mean(residuals**2))),
                "road_residual_range_m": [float(residuals.min()), float(residuals.max())],
                "nearby_car_ids": points.point_ids[car_ix].tolist(),
                "nearby_car_heights_above_road_m": car_height.tolist()
            })
        car_planes = []
        for radius in (0.4, 0.6, 0.9):
            ix = np.flatnonzero((original == 3) & (d3 <= radius)
                                & (points.point_ids != pid))
            if len(ix) < 5:
                continue
            x = offsets[ix]
            center = x.mean(axis=0)
            _, singular, vt = np.linalg.svd(x-center, full_matrices=False)
            normal = vt[-1]
            residuals = (x-center)@normal
            car_planes.append({
                "radius": radius, "anchor_ids": points.point_ids[ix].tolist(),
                "normal_xyz": normal.tolist(), "singular_values": singular.tolist(),
                "target_plane_distance_m": float(abs(center@normal)),
                "anchor_plane_rmse_m": float(np.sqrt(np.mean(residuals**2)))
            })
        outputs.append({"point_id": int(pid), "assigned_label": int(preview[i]),
                        "road_plane_tests": fits, "car_plane_tests": car_planes})
    return outputs

comparison2 = diagnose.compare(
    "Does the changed point lie on the locally observed road or car surface?",
    surface_continuity, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "CANDIDATES"],
    assumptions=["S0 anchor labels may be wrong.",
                 "Local planes approximate real returns, not completed object geometry.",
                 "No anchor is known to have been excluded from candidate generation."]
)
print(comparison2["evidence_id"], comparison2["source_checks"])
print("Shared geometry; candidate labels:",
      {cid: comparison2["results"][cid]["result"][0]["assigned_label"] for cid in cids})
print("Surface measurements:", comparison2["results"]["KEEP"]["result"])

distance_to_target = np.linalg.norm(support.xyz-targets.xyz[0], axis=1)
view_points = scene.points(point_ids=support.point_ids[distance_to_target <= 1.8])
eye_side = source_box.rotation[:,1] + 0.25*source_box.rotation[:,0] + np.array([0.,0.,0.18])
camera_side = viz.make_camera(
    look_at=targets.xyz[0]+np.array([0.,0.,0.45]),
    eye_direction=eye_side, span_m=3.3, image_size=850
)
views2 = []
for cid in ("KEEP", "TRACK_TO_SEG:0"):
    view = viz.render(
        view_points, camera_side, box_ids=(), color_mode="semantic",
        labels=scene.labels_for(view_points.point_ids, cid),
        highlight_ids=affected, title=f"{cid}: highlighted point 13105"
    )
    views2.append(view)
    show(view, "Identical real points, camera and scale; highlighted target at the road/car boundary.")
camera_top = viz.make_camera(
    look_at=targets.xyz[0], eye_direction=(0,0,1), up=(1,0,0),
    span_m=3.8, image_size=850
)
top_view = viz.render(
    view_points, camera_top, box_ids=(), color_mode="semantic",
    labels=scene.labels_for(view_points.point_ids, "KEEP"), highlight_ids=affected,
    title="KEEP top view: target and competing surfaces"
)
show(top_view, "Additional view of the same scan, not independent evidence.")

