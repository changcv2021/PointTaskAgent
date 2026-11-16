cids = list(scene.candidate_ids)
catalog = scene.evidence_catalog()
provenance = {cid: scene.candidate_provenance(cid) for cid in cids}
print("CATALOG", catalog)
print("PROVENANCE", provenance)

boxes = {cid: scene.box(cid) for cid in cids}
corners = {}
for cid, b in boxes.items():
    signs = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    corners[cid] = (signs * np.asarray(b.size_lwh)/2) @ b.rotation.T + b.center
allcorners = np.concatenate(list(corners.values()))
support = scene.points(bounds=[
    (allcorners.min(axis=0) - [2.0,2.0,0.6]).tolist(),
    (allcorners.max(axis=0) + [2.0,2.0,0.6]).tolist()
])
xyz = np.asarray(support.xyz)
labels = np.asarray(support.semantic_labels)
probs = np.asarray(support.semantic_probs)
truck_mask = (labels == 9) & (probs[:,9] >= 0.5)
ground_mask = np.isin(labels, [10,11,12]) & (probs.max(axis=1) >= 0.6)
outside_footprints = np.ones(len(xyz), dtype=bool)
for b in boxes.values():
    q = (xyz - b.center) @ b.rotation
    outside_footprints &= ~np.all(np.abs(q[:,:2]) <= np.asarray(b.size_lwh)[:2]/2 + 0.2, axis=1)
ground_fit_mask = ground_mask & outside_footprints
anchor_xy = np.asarray(boxes["KEEP"].center)[:2].copy()
plane = None
plane_info = {"usable": False, "n_available": int(ground_fit_mask.sum())}
ground_ids = support.point_ids[ground_fit_mask]
if ground_fit_mask.sum() >= 20:
    g = xyz[ground_fit_mask]
    A = np.c_[g[:,:2] - anchor_xy, np.ones(len(g))]
    fit = np.linalg.lstsq(A, g[:,2], rcond=None)[0]
    active = np.ones(len(g), dtype=bool)
    for _ in range(4):
        residual = g[:,2] - A @ fit
        median = np.median(residual[active])
        mad = np.median(np.abs(residual[active] - median))
        active = np.abs(residual - median) <= max(0.08, 3.0*1.4826*mad)
        if active.sum() < 20:
            break
        fit = np.linalg.lstsq(A[active], g[active,2], rcond=None)[0]
    if active.sum() >= 20:
        plane = fit.copy()
        residual = g[active,2] - A[active] @ plane
        ground_ids = support.point_ids[ground_fit_mask][active]
        plane_info = {
            "usable": True, "n_inliers": int(active.sum()),
            "coefficients_z_ax_by_c": plane.tolist(),
            "anchor_xy": anchor_xy.tolist(),
            "residual_quantiles_m": np.quantile(residual,[0.05,0.5,0.95]).tolist(),
            "xy_extent_m": np.ptp(g[active,:2],axis=0).tolist(),
            "inlier_point_ids": ground_ids.tolist()
        }
print("SUPPORT", len(xyz), "SEMANTIC_COUNTS",
      {scene.class_names[int(k)]: int(v) for k,v in zip(*np.unique(labels,return_counts=True))})
print("GROUND", plane_info)
env_eid = report(
    "Common support and external semantic-ground fit", plane_info,
    point_ids=support.point_ids, evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    candidate_ids=cids
)

def ground_and_boundaries(cid, points):
    b = scene.box(cid)
    p = np.asarray(points.xyz)
    lab = np.asarray(points.semantic_labels)
    pr = np.asarray(points.semantic_probs)
    vehicle = (lab == 9) & (pr[:,9] >= 0.5)
    q = (p - b.center) @ b.rotation
    half = np.asarray(b.size_lwh)/2
    inside = np.all(np.abs(q) <= half, axis=1)
    signed_axes = np.abs(q) - half
    sensor_local = (np.asarray(scene.sensor_origin) - b.center) @ b.rotation
    out = {
        "center": np.asarray(b.center).tolist(),
        "sensor_local": sensor_local.tolist(),
        "truck_n": int(vehicle.sum()),
        "truck_inside_n": int((vehicle & inside).sum()),
        "truck_outside_point_ids": points.point_ids[vehicle & ~inside].tolist(),
        "truck_local_xyz_quantiles_m": (
            np.quantile(q[vehicle],[0.05,0.5,0.95],axis=0).tolist()
            if vehicle.any() else []),
        "truck_outside_each_axis_point_ids": {
            axis: points.point_ids[vehicle & (signed_axes[:,j] > 0.10)].tolist()
            for j,axis in enumerate(["length","width","height"])
        }
    }
    if plane is not None:
        bottom = corners[cid][np.array([False,True]*4) == False]
        clearance = bottom[:,2] - (np.c_[bottom[:,:2]-anchor_xy,np.ones(len(bottom))] @ plane)
        out["bottom_corner_ground_clearance_m"] = clearance.tolist()
    return out

comparison1 = diagnose.compare(
    "Do translations cut credible truck returns or materially cross measured ground?",
    ground_and_boundaries, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","SENSOR_POSE","D0","CANDIDATES"],
    assumptions=[
        "S0 truck labels are class evidence, not instance identity.",
        "Ground plane uses confident ground labels outside all candidate footprints.",
        "Box corners include the retained D0 tilt; no shape correction is allowed.",
        "Current scan and semantics may have been used by candidate generators."
    ]
)
print("COMPARISON", comparison1)
camera1 = viz.preset_camera("complementary", points=support, box_ids=cids, image_size=900)
view1 = viz.render(support, camera1, box_ids=cids, color_mode="semantic",
                   highlight_ids=ground_ids, title="Shared support; highlighted ground-fit returns")
show(view1, "All candidate boxes on the same current scan; highlighted points define the tentative ground fit.")

