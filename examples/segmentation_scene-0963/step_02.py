print("Catalog:", scene.evidence_catalog())
cids = ["KEEP", "DET_TO_SEG:0", "DET_TO_SEG:1"]
for cid in cids:
    print("Provenance:", cid, scene.candidate_provenance(cid))

support = scene.initial_points()
box = scene.reference_boxes[0]
focus_ids = np.unique(np.concatenate([
    np.asarray(scene.patch(cid).point_ids, dtype=int)
    for cid in cids if cid != "KEEP"
]))
print("Focus IDs:", focus_ids.tolist())

def inspect_disagreement(cid, points):
    original = np.asarray(scene.labels_for(points.point_ids, "KEEP"))
    preview = np.asarray(scene.labels_for(points.point_ids, cid))
    local = geom.to_box_coordinates(points, box)
    rows = []
    for pid in focus_ids:
        i = int(np.flatnonzero(points.point_ids == pid)[0])
        distance = np.linalg.norm(points.xyz - points.xyz[i], axis=1)
        neighbors = np.argsort(distance)[:16]
        rows.append({
            "point_id": int(pid),
            "xyz": points.xyz[i].tolist(),
            "box_local": local[i].tolist(),
            "original_label": int(original[i]),
            "preview_label": int(preview[i]),
            "S0_probs": points.semantic_probs[i].tolist(),
            "neighbors": [
                {"id": int(points.point_ids[j]),
                 "distance": float(distance[j]),
                 "delta_xyz": (points.xyz[j] - points.xyz[i]).tolist(),
                 "box_local": local[j].tolist(),
                 "S0_label": int(original[j]),
                 "preview_label": int(preview[j])}
                for j in neighbors
            ]
        })
    changed = preview != original
    return {
        "changed_ids": points.point_ids[changed].tolist(),
        "old_labels": original[changed].tolist(),
        "new_labels": preview[changed].tolist(),
        "focus": rows
    }

comparison1 = diagnose.compare(
    "What exact edits occur, and what real returns surround the disputed points?",
    inspect_disagreement, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["CANDIDATES", "S0", "D0", "LIDAR_CURRENT"],
    assumptions=["Neighborhood labels are semantic predictions, not instance identities.",
                 "No exact generator point membership is available."]
)
print(comparison1)

local_record = report(
    "Original local geometry for choosing a surface diagnostic",
    [{"id": int(pid), "relative_xyz": np.round(xyz - box.center, 4).tolist(),
      "S0_label": int(label)}
     for pid, xyz, label in zip(support.point_ids, support.xyz,
                               scene.labels_for(support.point_ids, "KEEP"))],
    point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "D0", "S0"], candidate_ids=cids
)
print("Local geometry evidence:", local_record)
print("Local rows:", [
    (int(pid), np.round(xyz - box.center, 4).tolist(), int(label))
    for pid, xyz, label in zip(support.point_ids, support.xyz,
                              scene.labels_for(support.point_ids, "KEEP"))
])

