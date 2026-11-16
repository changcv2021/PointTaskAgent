wide = scene.points(bounds=[
    (allcorners.min(axis=0) - [4.0,4.0,0.7]).tolist(),
    (allcorners.max(axis=0) + [4.0,4.0,0.7]).tolist()
])
wxyz = np.asarray(wide.xyz)
wlab = np.asarray(wide.semantic_labels)
wpr = np.asarray(wide.semantic_probs)
keepbox = scene.box("KEEP")
wq = (wxyz - keepbox.center) @ keepbox.rotation
# Shared surface definition: sensor-facing side, away from ends and lowest returns.
side_mask = ((wlab == 9) & (wpr[:,9] >= 0.5) &
             (wq[:,1] > 0.75) & (np.abs(wq[:,0]) < 2.6) &
             (wq[:,2] > -0.70))
side_ids = wide.point_ids[side_mask].copy()
ground_models = {}
for class_id in [10,12]:
    gm = (wlab == class_id) & (wpr[:,class_id] >= 0.6)
    for box in boxes.values():
        lq = (wxyz - box.center) @ box.rotation
        gm &= ~np.all(np.abs(lq[:,:2]) <= np.asarray(box.size_lwh)[:2]/2 + 0.2, axis=1)
    gg = wxyz[gm]
    info = {"n": len(gg), "point_ids": wide.point_ids[gm].tolist(), "usable": False}
    if len(gg) >= 20:
        AA = np.c_[gg[:,:2]-anchor_xy, np.ones(len(gg))]
        ff = np.linalg.lstsq(AA,gg[:,2],rcond=None)[0]
        aa = np.ones(len(gg),dtype=bool)
        for _ in range(5):
            rr = gg[:,2] - AA @ ff
            mm = np.median(rr[aa])
            mad = np.median(np.abs(rr[aa]-mm))
            aa = np.abs(rr-mm) <= max(0.08,3*1.4826*mad)
            if aa.sum() < 20:
                break
            ff = np.linalg.lstsq(AA[aa],gg[aa,2],rcond=None)[0]
        if aa.sum() >= 20:
            info.update({
                "usable": True, "coeff": ff.tolist(), "n_inliers": int(aa.sum()),
                "inlier_ids": wide.point_ids[gm][aa].tolist(),
                "xy_extent": np.ptp(gg[aa,:2],axis=0).tolist(),
                "residual_q05_q50_q95": np.quantile(gg[aa,2]-AA[aa]@ff,[.05,.5,.95]).tolist()
            })
    ground_models[scene.class_names[class_id]] = info

excluded_ids = np.array(comparison1["results"]["KEEP"]["result"]["truck_outside_point_ids"], dtype=int)
exceptions = []
for pid in excluded_ids:
    idx = int(np.flatnonzero(wide.point_ids == pid)[0])
    dist = np.linalg.norm(wxyz-wxyz[idx],axis=1)
    near = np.argsort(dist)[1:7]
    exceptions.append({
        "point_id": int(pid), "keep_local_xyz": wq[idx].tolist(),
        "s0_truck_probability": float(wpr[idx,9]),
        "nearest_point_ids": wide.point_ids[near].tolist(),
        "nearest_distances_m": dist[near].tolist(),
        "nearest_labels": wlab[near].tolist(),
        "nearest_truck_probs": wpr[near,9].tolist()
    })
print("WIDE_COUNTS", len(wxyz), {scene.class_names[int(k)]:int(v) for k,v in zip(*np.unique(wlab,return_counts=True))})
print("GROUND_MODELS", ground_models)
print("EXCLUDED_POINTS", exceptions)
context_eid = report(
    "Separate observed ground groups and KEEP-excluded return neighborhoods",
    {"ground_models": ground_models, "excluded_points": exceptions,
     "fixed_side_ids": side_ids.tolist(), "fixed_side_n": len(side_ids)},
    point_ids=wide.point_ids, evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    candidate_ids=cids
)

def visible_side_and_ground(cid, points):
    b = scene.box(cid)
    pp = np.asarray(points.xyz)
    qq = (pp-b.center) @ b.rotation
    half = np.asarray(b.size_lwh)/2
    side = np.isin(points.point_ids,side_ids)
    result = {
        "side_n": int(side.sum()),
        "side_point_ids": points.point_ids[side].tolist(),
        "positive_width_face_residual_q05_q50_q95_m":
            np.quantile(qq[side,1]-half[1],[.05,.5,.95]).tolist() if side.any() else [],
        "ground": {}
    }
    bottom = np.array([[x,y,-1] for x in [-1,1] for y in [-1,1]]) * half
    bottom_global = bottom @ b.rotation.T + b.center
    for name, model in ground_models.items():
        if not model["usable"]:
            result["ground"][name] = {"usable": False}
            continue
        gp = pp[np.isin(points.point_ids,model["inlier_ids"])]
        cf = np.asarray(model["coeff"])
        predicted = np.c_[bottom_global[:,:2]-anchor_xy,np.ones(4)] @ cf
        nearest = np.sqrt(((bottom_global[:,None,:2]-gp[None,:,:2])**2).sum(axis=2)).min(axis=1)
        result["ground"][name] = {
            "bottom_corner_clearance_m": (bottom_global[:,2]-predicted).tolist(),
            "corner_nearest_ground_xy_distance_m": nearest.tolist(),
            "ground_residual_q05_q50_q95_m": model["residual_q05_q50_q95"]
        }
    lab = np.asarray(points.semantic_labels)
    pr = np.asarray(points.semantic_probs)
    ground_under = (np.isin(lab,[10,12]) & (pr.max(axis=1)>=0.6) &
                    np.all(np.abs(qq[:,:2])<=half[:2],axis=1))
    result["direct_ground_inside_footprint"] = {
        "point_ids": points.point_ids[ground_under].tolist(),
        "ground_height_above_bottom_local_m": (qq[ground_under,2]+half[2]).tolist()
    }
    return result

comparison2 = diagnose.compare(
    "Does the measured positive-width side or separately fitted ground support each translation?",
    visible_side_and_ground, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=[
        "The fixed side subset is defined in D0 coordinates and shared across candidates; it samples a face, not a full center.",
        "Road and sidewalk are fitted separately; nearest-ground distances expose extrapolation.",
        "Semantic probabilities do not establish truck instance membership or independent validation."
    ]
)
print("COMPARISON2", comparison2)
side_camera = viz.make_camera(
    look_at=keepbox.center,
    eye_direction=keepbox.rotation[:,1].tolist(),
    up=(0,0,1), span_m=9.0, image_size=900
)
view2 = viz.render(wide,side_camera,box_ids=cids,color_mode="semantic",
                   highlight_ids=excluded_ids,title="Sensor-facing side; highlighted KEEP-excluded truck labels")
show(view2, "Same real support and camera for all boxes; highlights are the five truck-labeled points outside KEEP.")

