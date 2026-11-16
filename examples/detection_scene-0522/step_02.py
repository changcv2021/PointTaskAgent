def surface_constraints(cid, points):
    ref = scene.box("KEEP")
    b = scene.box(cid)
    qref = geom.to_box_coordinates(points, ref)
    q = geom.to_box_coordinates(points, b)
    half = b.size_lwh / 2
    body = np.isin(points.semantic_labels, scene.semantic_indices_for_detection_class(ref.class_name))
    # Fixed shared subsets: candidate membership does not define these patches.
    end = body & (qref[:,0] < -2.6) & (qref[:,2] > -1.2)
    side = body & (qref[:,0] >= -2.6) & (qref[:,1] > .85) & (qref[:,2] > -1.2)
    high = body & (qref[:,2] > 1.1)
    road = (points.semantic_labels == 10)
    def describe(mask, residual):
        ids = points.point_ids[mask]
        return {"n": int(mask.sum()), "point_ids": ids.tolist(),
                "reference_extent": [qref[mask].min(0).tolist(), qref[mask].max(0).tolist()] if mask.any() else None,
                "residual_q10_q50_q90": np.quantile(residual[mask], [.1,.5,.9]).tolist() if mask.any() else None,
                "outside_by_over_10cm_ids": points.point_ids[mask & (residual < -.10)].tolist()}
    out = {
        "end_inward_distance": describe(end, q[:,0] + half[0]),
        "side_inward_distance": describe(side, half[1] - q[:,1]),
        "high_top_clearance": describe(high, half[2] - q[:,2]),
        "road_below_bottom_clearance": describe(road, -half[2] - q[:,2]),
    }
    # Expose spatially extended versus isolated disagreements, not just counts.
    for name, mask, residual, split_axis in [
        ("end_by_height", end, q[:,0]+half[0], 2),
        ("side_by_length", side, half[1]-q[:,1], 0)]:
        edges = np.linspace(qref[mask,split_axis].min(), qref[mask,split_axis].max()+1e-6, 5) if mask.any() else []
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = mask & (qref[:,split_axis]>=lo) & (qref[:,split_axis]<hi)
            bins.append({"reference_interval": [float(lo),float(hi)],
                         "n": int(m.sum()),
                         "median_inward_distance": float(np.median(residual[m])) if m.any() else None,
                         "point_ids": points.point_ids[m].tolist()})
        out[name] = bins
    inside = geom.points_in_box(points, b)
    orig_inside = geom.points_in_box(points, ref)
    for name, m in [("gained", inside & ~orig_inside), ("lost", orig_inside & ~inside)]:
        out[name] = {"point_ids": points.point_ids[m].tolist(),
                     "labels_counts": {str(int(l)): int((points.semantic_labels[m]==l).sum())
                                      for l in np.unique(points.semantic_labels[m])}}
    return out

comparison = diagnose.compare(
    "Do fixed visible end/side patches and vertical limits contradict any proposed center?",
    surface_constraints, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=[
        "Truck labels are class evidence, not confirmed instance membership.",
        "Fixed end and side subsets are proposed body patches; inward offset can reflect non-box-shaped bodywork.",
        "High returns constrain clipping, not an inferred unseen roof.",
        "Road clearance uses actual nearby road returns, not a completed ground surface."
    ])
print(comparison)
# Highlight the exact shared patch points for ownership and surface-shape inspection.
patch_mask = (np.isin(support.semantic_labels, truck_classes) &
              (((local[:,0] < -2.6) & (local[:,2] > -1.2)) |
               ((local[:,0] >= -2.6) & (local[:,1] > .85) & (local[:,2] > -1.2))))
patch_ids = support.point_ids[patch_mask]
for cid in ["KEEP", "SEG_TO_DET:11"]:
    view = viz.render(support, cam_face, box_ids=[cid], color_mode="semantic",
                      highlight_ids=patch_ids, title=cid + ": fixed end and side patches")
    show(view, "Identical camera and highlighted original patch IDs; inspect body ownership and boundary cuts.")

