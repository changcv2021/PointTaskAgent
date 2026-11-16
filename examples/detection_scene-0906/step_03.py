low_ids = np.array([e["point_id"] for e in exceptions
                    if e["keep_local_xyz"][2] < -1.0], dtype=int)
ground_shared = np.isin(wlab,[10,12]) & (wpr.max(axis=1)>=0.6)
ground_xyz = wxyz[ground_shared]
ground_point_ids = wide.point_ids[ground_shared]

local_context = []
for pid in low_ids:
    ii = int(np.flatnonzero(wide.point_ids == pid)[0])
    horizontal = np.linalg.norm(ground_xyz[:,:2]-wxyz[ii,:2],axis=1)
    order = np.argsort(horizontal)[:6]
    local_context.append({
        "truck_point_id": int(pid), "truck_xyz": wxyz[ii].tolist(),
        "nearest_ground": [
            {"point_id": int(ground_point_ids[j]),
             "xy_distance_m": float(horizontal[j]),
             "xyz": ground_xyz[j].tolist(),
             "truck_minus_ground_z_m": float(wxyz[ii,2]-ground_xyz[j,2])}
            for j in order]
    })

# Expose all nearby high returns, regardless of winning semantic label.
near_footprint = np.all(np.abs(wq[:,:2]) <= np.asarray(keepbox.size_lwh)[:2]/2 + 0.35,axis=1)
upper = near_footprint & (wq[:,2] > 0.45)
upper_records = [{
    "point_id": int(wide.point_ids[i]), "keep_local_xyz": wq[i].tolist(),
    "label": scene.class_names[int(wlab[i])],
    "truck_probability": float(wpr[i,9])
} for i in np.flatnonzero(upper)]
print("LOW_TRUCK_LOCAL_GROUND", local_context)
print("UPPER_NEAR_FOOTPRINT", upper_records)

def local_measured_ground(cid, points):
    b = scene.box(cid)
    pp = np.asarray(points.xyz)
    lab = np.asarray(points.semantic_labels)
    pr = np.asarray(points.semantic_probs)
    ggmask = np.isin(lab,[10,12]) & (pr.max(axis=1)>=0.6)
    gg = pp[ggmask]
    gids = points.point_ids[ggmask]
    half = np.asarray(b.size_lwh)/2
    low = np.isin(points.point_ids,low_ids)
    lower_face = b.center - half[2]*b.rotation[:,2]
    normal = b.rotation[:,2]
    records = []
    for i in np.flatnonzero(low):
        xy_dist = np.linalg.norm(gg[:,:2]-pp[i,:2],axis=1)
        jj = np.argsort(xy_dist)[:6]
        # The retained tilted lower face evaluated at the measured XY.
        lower_z = lower_face[2] - (
            normal[0]*(pp[i,0]-lower_face[0]) +
            normal[1]*(pp[i,1]-lower_face[1]))/normal[2]
        ground_lower_z = lower_face[2] - (
            normal[0]*(gg[jj,0]-lower_face[0]) +
            normal[1]*(gg[jj,1]-lower_face[1]))/normal[2]
        records.append({
            "truck_point_id": int(points.point_ids[i]),
            "truck_z_minus_lower_face_m": float(pp[i,2]-lower_z),
            "nearest_ground_ids": gids[jj].tolist(),
            "nearest_ground_xy_distances_m": xy_dist[jj].tolist(),
            "lower_face_minus_observed_ground_z_m": (ground_lower_z-gg[jj,2]).tolist()
        })
    return {"local_lower_face_checks": records}

comparison3 = diagnose.compare(
    "At the coherent excluded low returns, how does each lower face relate to nearby observed ground?",
    local_measured_ground, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=[
        "Ground labels are uncertain; distances and actual original IDs are exposed.",
        "Nearby-ground checks do not establish unobserved roof height or full truck center.",
        "No ground plane is extrapolated in this comparison."
    ]
)
for cid, entry in comparison3["results"].items():
    print(cid, entry["status"], entry.get("result"), entry.get("error"))
print("SOURCE_CHECKS", comparison3["source_checks"])
context3_eid = report(
    "Actual local ground and all high footprint returns",
    {"low_truck_ground": local_context, "upper_returns": upper_records},
    point_ids=wide.point_ids, evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    candidate_ids=cids
)
local_highlights = set(low_ids.tolist())
for item in local_context:
    local_highlights.update(v["point_id"] for v in item["nearest_ground"])
bev_camera = viz.make_camera(
    look_at=keepbox.center, eye_direction=(0,0,1),
    up=keepbox.rotation[:,0].tolist(), span_m=12., image_size=900)
view3 = viz.render(
    wide, bev_camera, box_ids=cids, color_mode="semantic",
    highlight_ids=np.array(sorted(local_highlights),dtype=int),
    title="Actual low truck / nearest ground relationships in plan")
show(view3, "Highlights identify the low truck returns and their measured nearest ground context; no fitted plane is rendered.")

