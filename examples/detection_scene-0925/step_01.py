print("Catalog:", scene.evidence_catalog())
print("Provenance:", {cid: scene.candidate_provenance(cid) for cid in scene.candidate_ids})
ids = list(scene.candidate_ids)
b0 = scene.box("KEEP")
centers = np.array([scene.box(cid).center for cid in ids])
support = scene.points(bounds=[
    [float(centers[:,0].min()-2.5), float(centers[:,1].min()-2.5), float(b0.center[2]-1.5)],
    [float(centers[:,0].max()+2.5), float(centers[:,1].max()+2.5), float(b0.center[2]+1.7)]
])
ped_classes = scene.semantic_indices_for_detection_class(b0.class_name)
anchor = np.array(b0.center, dtype=float)

def ground_and_masks(points):
    xyz = np.asarray(points.xyz)
    xy = xyz[:,:2] - anchor[:2]
    radius = np.linalg.norm(xy, axis=1)
    labels = np.asarray(points.semantic_labels)
    ground_mask = np.isin(labels, [10,11,12,13]) & (radius > 0.75) & (radius < 2.5)
    gi = np.flatnonzero(ground_mask)
    fit = None
    if len(gi) >= 12:
        A = np.column_stack([xy[gi], np.ones(len(gi))])
        z = xyz[gi,2]
        active = np.ones(len(gi), dtype=bool)
        for _ in range(6):
            coef = np.linalg.lstsq(A[active], z[active], rcond=None)[0]
            residual = z - A @ coef
            median = np.median(residual)
            mad = np.median(np.abs(residual-median))
            active = np.abs(residual-median) <= max(0.06, 3.0*1.4826*mad)
            if active.sum() < 8:
                break
        if active.sum() >= 8:
            coef = np.linalg.lstsq(A[active], z[active], rcond=None)[0]
            residual = z-A@coef
            fit = (coef, gi[active], residual[active])
    ped = np.isin(labels, ped_classes) & (radius < 1.2)
    return fit, ped, ground_mask

def placement_test(cid, points):
    box = scene.box(cid)
    xyz = np.asarray(points.xyz)
    pid = np.asarray(points.point_ids)
    local = geom.to_box_coordinates(points, box)
    half = np.asarray(box.size_lwh)/2
    margins = half - np.abs(local)
    inside = np.all(margins >= 0, axis=1)
    fit, ped, ground_mask = ground_and_masks(points)
    result = {
        "center": np.asarray(box.center).tolist(),
        "pedestrian_support_ids": pid[ped].tolist(),
        "pedestrian_xyz_quantiles": np.quantile(xyz[ped], [0,0.1,0.5,0.9,1], axis=0).tolist() if ped.any() else [],
        "pedestrian_inside_ids": pid[ped & inside].tolist(),
        "pedestrian_top_clipped_ids": pid[ped & (local[:,2] > half[2])].tolist(),
        "pedestrian_bottom_clipped_ids": pid[ped & (local[:,2] < -half[2])].tolist(),
        "pedestrian_horizontal_clipped_ids": pid[ped & np.any(margins[:,:2]<0, axis=1)].tolist(),
        "ground_inside_ids": pid[ground_mask & inside].tolist(),
        "fixed_height_bands": {}
    }
    for lo, hi in [(-0.5,0.0),(0.0,0.5),(0.5,1.0),(1.0,1.7)]:
        m = ped & (xyz[:,2]>=lo) & (xyz[:,2]<hi)
        result["fixed_height_bands"][str((lo,hi))] = {
            "ids": pid[m].tolist(),
            "inside_ids": pid[m & inside].tolist(),
            "min_face_margins_xyz": margins[m].min(axis=0).tolist() if m.any() else None
        }
    signs = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    corners = signs*half @ np.asarray(box.rotation).T + np.asarray(box.center)
    result["box_global_z_range"] = [float(corners[:,2].min()),float(corners[:,2].max())]
    if fit is not None:
        coef, gi, residual = fit
        bottom = corners[signs[:,2] == -1]
        ground_z = np.column_stack([bottom[:,:2]-anchor[:2], np.ones(4)]) @ coef
        result["ground"] = {
            "plane_z_equals_a_dx_b_dy_c": coef.tolist(),
            "support_ids": pid[gi].tolist(),
            "residual_quantiles": np.quantile(residual,[0,0.1,0.5,0.9,1]).tolist(),
            "xy_span": np.ptp(xyz[gi,:2], axis=0).tolist(),
            "bottom_corner_clearances": (bottom[:,2]-ground_z).tolist()
        }
    else:
        result["ground"] = {"status":"insufficient robust surrounding support",
                            "available_ids":pid[ground_mask].tolist()}
    return result

comparison1 = diagnose.compare(
    "Do center proposals recover upright returns without upper clipping or unsupported ground penetration?",
    placement_test, candidate_ids=ids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=["S0 pedestrian labels are not instances.",
                 "Ground surface extrapolation is local and conditional on fit support.",
                 "No generator-excluded points or independent observations are claimed."]
)
print(comparison1)
fit1, ped1, ground1 = ground_and_masks(support)
display_mask = np.linalg.norm(support.xyz[:,:2]-anchor[:2], axis=1) < 1.65
display_points = support[display_mask]
profile_camera = viz.make_camera(
    look_at=anchor, eye_direction=(1,0,0), up=(0,0,1),
    projection="orthographic", span_m=3.8, image_size=900
)
profile1 = viz.render(
    display_points, profile_camera, box_ids=ids, color_mode="semantic",
    highlight_ids=support.point_ids[ped1 & display_mask],
    title="Shared global-Y / height profile; S0 pedestrian support highlighted"
)
show(profile1, "Same real points and camera for all candidates; ground and upright support remain provisional.")

