wide = scene.points(bounds=[corners.min(0)-2.0, corners.max(0)+2.0])
gained14 = comparison["results"]["TRACK_TO_DET:14"]["result"]["gained"]["point_ids"]
low_row_ids = [row["point_id"] for row in lower_cmp["results"]["KEEP"]["result"]["low_body"]
               if row["in_xy_footprint"] and row["bottom_minus_return_global_z"] > 0]
low_row = scene.points(point_ids=low_row_ids)
road_w = wide.select(wide.semantic_labels == 10)
upper_w = wide.select((wide.semantic_labels == 9) & (wide.xyz[:,2] > .35))
details = []
for i, pid in enumerate(low_row.point_ids):
    p = low_row.xyz[i]
    dxy = np.linalg.norm(road_w.xyz[:,:2] - p[:2], axis=1)
    j = np.argsort(dxy)[:3]
    db = np.linalg.norm(upper_w.xyz-p, axis=1)
    jb = np.argsort(db)[:3]
    details.append({
        "point_id": int(pid), "xyz": p.tolist(),
        "truck_probability": float(low_row.semantic_probs[i,9]),
        "nearest_road": [{"point_id": int(road_w.point_ids[k]), "xy_distance": float(dxy[k]),
                         "row_minus_road_z": float(p[2]-road_w.xyz[k,2])} for k in j],
        "nearest_upper_truck": [{"point_id": int(upper_w.point_ids[k]), "distance": float(db[k])}
                                for k in jb]
    })
e_ownership = report("Low row ownership: actual road/body neighbors, without surface completion",
                     details, point_ids=wide.point_ids,
                     evidence_sources=["LIDAR_CURRENT","S0","D0"],
                     candidate_ids=cids)
print(e_ownership, details)
print("E3 source checks", lower_cmp["source_checks"])
# Zoom the low structure; retain nearby original geometry rather than projecting inferred surfaces.
qb_w = geom.to_box_coordinates(wide, base)
zoom = wide.select((qb_w[:,0] > -3.2) & (qb_w[:,0] < 1.3) &
                   (np.abs(qb_w[:,1]) < 2.4) & (wide.xyz[:,2] < 1.2))
cam_low = viz.make_camera(look_at=low_row.xyz.mean(0)+np.array([0,0,.25]),
                          eye_direction=np.array(scene.sensor_origin)-base.center+np.array([0,0,12]),
                          span_m=5.5, image_size=900)
for cid in ["KEEP", "TRACK_TO_DET:14"]:
    vl = viz.render(zoom, cam_low, box_ids=[cid], color_mode="semantic",
                    highlight_ids=low_row.point_ids,
                    title=cid + ": low row and actual nearby pavement")
    show(vl, "Same close-up camera and original low-row IDs; pavement/body ownership remains a hypothesis until inspected.")

