roof_ids = np.array([r["point_id"] for r in upper_records], dtype=int)
roof_points = scene.points(point_ids=roof_ids)
roof_local = (np.asarray(roof_points.xyz)-keepbox.center) @ keepbox.rotation
roof_A = np.c_[roof_local[:,:2],np.ones(len(roof_local))]
roof_coeff = np.linalg.lstsq(roof_A,roof_local[:,2],rcond=None)[0]
roof_residual = roof_local[:,2]-roof_A@roof_coeff
roof_info = {
    "point_ids": roof_ids.tolist(),
    "n": len(roof_ids),
    "local_xy_extent_m": np.ptp(roof_local[:,:2],axis=0).tolist(),
    "local_z_min_max_m": [float(roof_local[:,2].min()),float(roof_local[:,2].max())],
    "plane_local_z_ax_by_c": roof_coeff.tolist(),
    "max_abs_plane_residual_m": float(np.max(np.abs(roof_residual))),
    "plane_normal_angle_from_box_up_deg": float(
        np.degrees(np.arctan(np.linalg.norm(roof_coeff[:2])))),
    "original_truck_probabilities": np.asarray(roof_points.semantic_probs)[:,9].tolist()
}
print("UPPER_PATCH_GEOMETRY",roof_info)
roof_eid = report(
    "Geometry of six measured upper truck returns",roof_info,point_ids=roof_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0"],candidate_ids=cids
)

def upper_and_lower_faces(cid, points):
    b = scene.box(cid)
    qq = (np.asarray(points.xyz)-b.center) @ b.rotation
    half = np.asarray(b.size_lwh)/2
    roof = np.isin(points.point_ids,roof_ids)
    low = np.isin(points.point_ids,low_ids)
    result = {
        "upper_point_ids": points.point_ids[roof].tolist(),
        "upper_signed_top_face_residual_m": (qq[roof,2]-half[2]).tolist(),
        "upper_horizontal_inside": np.all(np.abs(qq[roof,:2])<=half[:2],axis=1).tolist(),
        "upper_point_ids_above_top_by_over_1cm":
            points.point_ids[roof & (qq[:,2]>half[2]+0.01)].tolist(),
        "low_point_ids": points.point_ids[low].tolist(),
        "low_signed_bottom_face_residual_m": (qq[low,2]+half[2]).tolist(),
        "side_face_residual_q05_q50_q95_m": np.quantile(
            qq[np.isin(points.point_ids,side_ids),1]-half[1],[.05,.5,.95]).tolist()
    }
    return result

comparison4 = diagnose.compare(
    "Which translations place both measured upper and coherent low truck surfaces within retained-height boundaries?",
    upper_and_lower_faces,candidate_ids=cids,point_ids=wide.point_ids,
    evidence_sources=["LIDAR_CURRENT","S0","D0","CANDIDATES"],
    assumptions=[
        "Six upper points form a potential roof patch, not automatically a complete-object centroid.",
        "Same upper, lower and side original point IDs are used for every candidate.",
        "Planarity, semantics and boundary relationships are reused-source evidence, not independent predictions.",
        "Unobserved roof extensions and retained D0 size/tilt remain uncertain."
    ]
)
print("FACE_COMPARISON",comparison4)
roof_camera = viz.make_camera(
    look_at=keepbox.center,
    eye_direction=(keepbox.rotation[:,1]+0.35*keepbox.rotation[:,0]+0.7*np.array([0,0,1])).tolist(),
    up=(0,0,1),span_m=8.0,image_size=900)
view4 = viz.render(
    support,roof_camera,box_ids=cids,color_mode="semantic",
    highlight_ids=roof_ids,title="Measured upper patch highlighted; all retained-height boxes")
show(view4,"Six original upper truck returns highlighted, with identical support and camera for all candidates.")

