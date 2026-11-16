# Compactly expose the previously executed evidence, including dependency checks.
for cid in ids:
    r = comparison1["results"][cid]["result"]
    print(cid, {
        "inside_pedestrian_n": len(r["pedestrian_inside_ids"]),
        "top_clipped": r["pedestrian_top_clipped_ids"],
        "bottom_clipped": r["pedestrian_bottom_clipped_ids"],
        "horizontal_clipped": r["pedestrian_horizontal_clipped_ids"],
        "ground_inside": r["ground_inside_ids"],
        "ground_clearance": r["ground"].get("bottom_corner_clearances")
    })
print("E1 source_checks:", comparison1.get("source_checks"))

changed_ids = sorted(set().union(*[
    set(comparison1["results"][cid]["result"][key])
    for cid in ids
    for key in ["pedestrian_top_clipped_ids",
                "pedestrian_bottom_clipped_ids",
                "pedestrian_horizontal_clipped_ids"]
]))
xyz2 = np.asarray(support.xyz)
pids2 = np.asarray(support.point_ids)
coef1 = fit1[0]
ground_heights2 = np.column_stack([xyz2[:,:2]-anchor[:2], np.ones(len(xyz2))]) @ coef1
height2 = xyz2[:,2] - ground_heights2
body_ids = pids2[ped1 & (height2 > 0.15)]
body_points = scene.points(point_ids=body_ids)
changed_points = scene.points(point_ids=changed_ids)
neighbor_details = []
for j, pid in enumerate(changed_points.point_ids):
    d = np.linalg.norm(body_points.xyz-changed_points.xyz[j], axis=1)
    order = np.argsort(d)[:4]
    where = int(np.flatnonzero(pids2 == pid)[0])
    neighbor_details.append({
        "point_id": int(pid),
        "xyz": changed_points.xyz[j].tolist(),
        "S0_label": int(changed_points.semantic_labels[j]),
        "height_above_E1_ground": float(height2[where]),
        "nearest_elevated_pedestrian_ids": body_points.point_ids[order].tolist(),
        "nearest_elevated_pedestrian_distances": d[order].tolist()
    })
detail_eid = report(
    "Changed-return heights and proximity to fixed elevated support",
    neighbor_details, point_ids=sorted(set(changed_ids) | set(body_ids.tolist())),
    evidence_sources=["LIDAR_CURRENT","S0","D0"],
    candidate_ids=ids
)
print(detail_eid, neighbor_details)

def local_ground_test(cid, points):
    xyz = np.asarray(points.xyz)
    pid = np.asarray(points.point_ids)
    xy = xyz[:,:2]-anchor[:2]
    radius = np.linalg.norm(xy, axis=1)
    box = scene.box(cid)
    half = np.asarray(box.size_lwh)/2
    signs = np.array([[x,y,-1] for x in [-1,1] for y in [-1,1]])
    bottom = (signs*half) @ np.asarray(box.rotation).T + np.asarray(box.center)
    output = {}
    # Fixed geometric low-return subsets, without using S0 labels.
    for lo, hi in [(0.35,1.1),(0.75,1.6),(1.1,2.5)]:
        m = (radius > lo) & (radius < hi) & (xyz[:,2] < -0.15)
        gi = np.flatnonzero(m)
        key = str((lo,hi))
        if len(gi) < 12:
            output[key] = {"status":"insufficient", "ids":pid[gi].tolist()}
            continue
        A = np.column_stack([xy[gi], np.ones(len(gi))])
        z = xyz[gi,2]
        active = np.ones(len(gi),dtype=bool)
        for _ in range(6):
            coef = np.linalg.lstsq(A[active],z[active],rcond=None)[0]
            res = z-A@coef
            med = np.median(res)
            mad = np.median(np.abs(res-med))
            updated = np.abs(res-med) <= max(0.035,3*1.4826*mad)
            if updated.sum() < 8:
                break
            active = updated
        coef = np.linalg.lstsq(A[active],z[active],rcond=None)[0]
        res = z[active]-A[active]@coef
        predictions = np.column_stack([bottom[:,:2]-anchor[:2],np.ones(4)]) @ coef
        output[key] = {
            "support_ids":pid[gi[active]].tolist(),
            "plane":coef.tolist(),
            "xy_span":np.ptp(xyz[gi[active],:2],axis=0).tolist(),
            "residual_quantiles":np.quantile(res,[0,0.1,0.5,0.9,1]).tolist(),
            "bottom_corner_clearances":(bottom[:,2]-predictions).tolist()
        }
    return output

comparison2 = diagnose.compare(
    "Does the bottom-face conclusion survive near/far geometric low-return ground fits?",
    local_ground_test, candidate_ids=ids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT","D0","CANDIDATES"],
    assumptions=["Low-return rings may include object feet; residuals and support are exposed.",
                 "Ground fits are conditional local approximations, not proof of free space."]
)
print("E2:", comparison2["evidence_id"])
for cid in ids:
    print(cid, {
        k: {kk:vv for kk,vv in v.items() if kk != "support_ids"}
        for k,v in comparison2["results"][cid]["result"].items()
    })
print("E2 source_checks:", comparison2.get("source_checks"))

bev_camera = viz.make_camera(
    look_at=anchor, eye_direction=(0,0,1), up=(0,1,0),
    projection="orthographic", span_m=3.8, image_size=900
)
bev2 = viz.render(
    display_points, bev_camera, box_ids=ids, color_mode="semantic",
    highlight_ids=changed_ids, title="Shared BEV: actual changed-membership returns highlighted"
)
show(bev2, "BEV of all candidates on identical original points; semantic classes are not instances.")
sensor_direction = np.asarray(scene.sensor_origin)-anchor
oblique_camera = viz.make_camera(
    look_at=anchor, eye_direction=sensor_direction, up=(0,0,1),
    projection="orthographic", span_m=3.0, image_size=900
)
oblique2 = viz.render(
    display_points, oblique_camera, box_ids=["KEEP","SEG_TO_DET:5"],
    color_mode="semantic", highlight_ids=changed_ids,
    title="Sensor-facing view: KEEP and small SEG shift"
)
show(oblique2, "Inspect low and lateral highlighted returns in the same scan, not an additional observation.")

