print("catalog", scene.evidence_catalog())
cids = list(scene.candidate_ids)
for cid in cids:
    print("provenance", cid, scene.candidate_provenance(cid))
base = scene.box("KEEP")
corners = []
for cid in cids:
    b = scene.box(cid)
    signs = np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    corners.append((signs * b.size_lwh / 2) @ b.rotation.T + b.center)
corners = np.concatenate(corners)
support = scene.points(bounds=[corners.min(0)-0.8, corners.max(0)+0.8])
local = geom.to_box_coordinates(support, base)
truck_classes = scene.semantic_indices_for_detection_class(base.class_name)
print("base", base.center, base.size_lwh, base.rotation)
print("sensor_local", (np.array(scene.sensor_origin)-base.center) @ base.rotation)
print("local_center_offsets", {cid: ((scene.box(cid).center-base.center) @ base.rotation).tolist() for cid in cids})
stats = {}
for label in np.unique(support.semantic_labels):
    m = support.semantic_labels == label
    stats[str(label)] = {"class": scene.class_names[str(int(label))],
                         "n": int(m.sum()),
                         "local_q10_q50_q90": np.quantile(local[m], [.1,.5,.9], axis=0).tolist()}
e_initial = report("Shared crop geometry and semantic distributions", stats,
                   point_ids=support.point_ids,
                   evidence_sources=["LIDAR_CURRENT","S0","D0"],
                   candidate_ids=cids)
print(e_initial, stats, "truck_classes", truck_classes, "crop_n", len(support.point_ids))
cam_bev = viz.make_camera(look_at=base.center, eye_direction=(0,0,1), up=tuple(base.rotation[:,0]),
                          span_m=12, image_size=900)
v_bev = viz.render(support, cam_bev, box_ids=cids, color_mode="semantic",
                   title="Shared real crop: candidate centers and semantic context")
show(v_bev, "All candidates on identical BEV support; box colors are in the legend.")
cam_face = viz.make_camera(look_at=base.center,
                           eye_direction=np.array(scene.sensor_origin)-base.center+np.array([0,0,7]),
                           span_m=11, image_size=900)
v_face = viz.render(support, cam_face, box_ids=["KEEP"], color_mode="semantic",
                    title="KEEP: sensor-side oblique view")
show(v_face, "Same current returns, a sensor-side view to identify actual sampled surfaces.")

