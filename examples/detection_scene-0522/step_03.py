print("E2 source checks", comparison["source_checks"])
for cid, entry in comparison["results"].items():
    r = entry["result"]
    print(cid, {k: r[k]["residual_q10_q50_q90"] for k in
                ["end_inward_distance","side_inward_distance","high_top_clearance","road_below_bottom_clearance"]},
          "gained", r["gained"]["labels_counts"], "lost", r["lost"]["labels_counts"])

def lower_boundary_test(cid, points):
    ref = scene.box("KEEP")
    b = scene.box(cid)
    qr = geom.to_box_coordinates(points, ref)
    qb = geom.to_box_coordinates(points, b)
    road = points.semantic_labels == 10
    low_body = (points.semantic_labels == 9) & (qr[:,2] < -1.1)
    # Compare a box's bottom at the XY of each actual return; no plane completion.
    normal = b.rotation[:,2]
    bottom_center = b.center - normal * b.size_lwh[2]/2
    bottom_at_xy = bottom_center[2] - ((points.xyz[:,:2]-bottom_center[:2]) @ normal[:2])/normal[2]
    out = {}
    for name, mask in [("road", road), ("low_body", low_body)]:
        footprint = np.all(np.abs(qb[:,:2]) <= b.size_lwh[:2]/2, axis=1)
        rows = []
        for i in np.flatnonzero(mask):
            rows.append({"point_id": int(points.point_ids[i]),
                         "ref_local_xyz": np.round(qr[i],3).tolist(),
                         "global_z": float(points.xyz[i,2]),
                         "bottom_minus_return_global_z": float(bottom_at_xy[i]-points.xyz[i,2]),
                         "in_xy_footprint": bool(footprint[i])})
        out[name] = rows
    return out

lower_cmp = diagnose.compare(
    "Does a small lowering fit actual low-body and road returns without cutting into coherent pavement?",
    lower_boundary_test, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=["Road class does not guarantee pavement identity; inspect row geometry.",
                 "Bottom plane evaluated only at actual return XY; no inferred free space.",
                 "Low-body points may be chassis or wheels, not a flat full-box bottom."])
for cid, entry in lower_cmp["results"].items():
    r = entry["result"]
    print("LOWER", cid)
    for name in ["road","low_body"]:
        rows = r[name]
        footprint_rows = [x for x in rows if x["in_xy_footprint"]]
        print(name, "n", len(rows), "footprint_ids", [x["point_id"] for x in footprint_rows],
              "all_bottom_minus_return_q", np.quantile([x["bottom_minus_return_global_z"] for x in rows],[0,.1,.5,.9,1]).tolist(),
              "footprint_bottom_minus_return_q", np.quantile([x["bottom_minus_return_global_z"] for x in footprint_rows],[0,.5,1]).tolist() if footprint_rows else None)
print("KEEP actual lower rows", lower_cmp["results"]["KEEP"]["result"])
low_ids = support.point_ids[(support.semantic_labels == 10) | ((support.semantic_labels == 9) & (local[:,2] < -1.1))]
cam_elev = viz.make_camera(look_at=base.center, eye_direction=tuple(base.rotation[:,1]),
                           up=(0,0,1), span_m=10, image_size=900)
for cid in ["KEEP", "TRACK_TO_DET:14"]:
    vv = viz.render(support, cam_elev, box_ids=[cid], color_mode="semantic",
                    highlight_ids=low_ids, title=cid + ": pavement and low body")
    show(vv, "Identical side-elevation camera and low-return IDs; no new observation source.")

