from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
focus_xyz = scene.points(point_ids=focus_ids).xyz[0].copy()
wide = scene.points(bounds=[
    (focus_xyz - np.array([1.9, 1.9, 0.8])).tolist(),
    (focus_xyz + np.array([1.9, 1.9, 1.0])).tolist()
])

def surface_and_stack(cid, points):
    delta = points.xyz - focus_xyz
    radial = np.linalg.norm(delta[:, :2], axis=1)
    labels = np.asarray(scene.labels_for(points.point_ids, cid))
    core = radial <= 0.22
    fits = []
    for radius in [0.8, 1.2, 1.8]:
        mask = (radial >= 0.28) & (radial <= radius)
        xy = delta[mask, :2]
        z = delta[mask, 2]
        if len(z) < 6:
            fits.append({"radius": radius, "n": len(z), "reliable": False})
            continue
        A = np.column_stack([xy, np.ones(len(xy))])
        init = np.linalg.lstsq(A, z, rcond=None)[0]
        fit = least_squares(lambda b: A @ b - z, init,
                            loss="soft_l1", f_scale=0.025)
        residual = z - A @ fit.x
        hull = ConvexHull(xy)
        inside_hull = bool(np.all(hull.equations[:, -1] <= 1e-9))
        fits.append({
            "radius": radius, "n": len(z),
            "fit_point_ids": points.point_ids[mask].tolist(),
            "slopes_xy": fit.x[:2].tolist(),
            "focus_height_above_fit_m": float(-fit.x[2]),
            "absolute_residual_quantiles_m": np.quantile(
                np.abs(residual), [0.5, 0.9, 1.0]).tolist(),
            "xy_covariance_eigenvalues": np.linalg.eigvalsh(np.cov(xy.T)).tolist(),
            "focus_inside_support_convex_hull": inside_hull
        })
    core_rows = [
        {"id": int(points.point_ids[i]), "delta_xyz": delta[i].tolist(),
         "preview_label": int(labels[i])}
        for i in np.flatnonzero(core)
    ]
    noncore_indices = np.flatnonzero(~core)
    order = noncore_indices[np.argsort(
        np.linalg.norm(delta[noncore_indices], axis=1))[:8]]
    return {
        "focus_label": int(labels[np.flatnonzero(points.point_ids == focus_ids[0])[0]]),
        "surrounding_surface_fits": fits,
        "core_returns": core_rows,
        "nearest_noncore_returns": [
            {"id": int(points.point_ids[i]), "delta_xyz": delta[i].tolist(),
             "distance_m": float(np.linalg.norm(delta[i])),
             "preview_label": int(labels[i])} for i in order]
    }

comparison2 = diagnose.compare(
    "Does surrounding geometry separate the disputed compact stack from local ground?",
    surface_and_stack, candidate_ids=cids, point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "CANDIDATES"],
    assumptions=[
        "Annular fitting excludes the focal footprint by geometry, not known generator membership.",
        "A single sloped plane may not describe uneven terrain; expose residuals and radius sensitivity.",
        "A compact stack alone does not confirm an instance or distinguish every manmade object."
    ]
)
print("Evidence:", comparison2["evidence_id"])
print("Source checks:", comparison2["source_checks"])
for cid, result in comparison2["results"].items():
    if result["status"] != "OK":
        print(cid, result)
        continue
    compact = result["result"].copy()
    compact["surrounding_surface_fits"] = [
        {k: v for k, v in f.items() if k != "fit_point_ids"}
        for f in compact["surrounding_surface_fits"]
    ]
    print(cid, compact)

direction = np.asarray(scene.sensor_origin) - focus_xyz
direction[2] = 0.0
camera2 = viz.make_camera(
    look_at=focus_xyz + np.array([0., 0., 0.15]),
    eye_direction=direction, up=(0, 0, 1),
    projection="orthographic", span_m=2.5, image_size=850)
render_support = wide[np.linalg.norm(wide.xyz[:, :2] - focus_xyz[:2], axis=1) <= 1.1]
views2 = []
for cid in ["KEEP", "DET_TO_SEG:0"]:
    view = viz.render(
        render_support, camera2, box_ids=(), color_mode="semantic",
        labels=scene.labels_for(render_support.point_ids, cid),
        highlight_ids=focus_ids, title=cid + ": disputed point highlighted")
    views2.append(view)
    show(view, "Same real points and side camera; DET_TO_SEG:1 has the same edit as DET_TO_SEG:0.")

