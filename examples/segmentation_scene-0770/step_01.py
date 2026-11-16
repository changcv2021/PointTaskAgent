import numpy as np

cids = list(scene.candidate_ids)
catalog = scene.evidence_catalog()
provenance = {cid: scene.candidate_provenance(cid) for cid in cids}
print("catalog", catalog)
print("provenance", provenance)
patches = {cid: scene.patch(cid) for cid in cids if cid != "KEEP"}
affected = np.unique(np.concatenate([p.point_ids for p in patches.values()])).astype(int)
print("patch effects", {
    cid: {"point_ids": p.point_ids.tolist(), "new_labels": p.new_labels.tolist(),
          "region_point_ids": p.region_point_ids.tolist()}
    for cid, p in patches.items()
})
targets = scene.points(point_ids=affected)
support = scene.points(bounds=[targets.xyz.min(axis=0)-3.0,
                              targets.xyz.max(axis=0)+3.0])
source_box = scene.box("KEEP")

def inspect_disagreement(cid, points):
    labels = np.asarray(scene.labels_for(points.point_ids, cid))
    original = np.asarray(points.semantic_labels)
    rows = []
    for pid in affected:
        i = int(np.flatnonzero(points.point_ids == pid)[0])
        distances = np.linalg.norm(points.xyz-points.xyz[i], axis=1)
        order = np.argsort(distances)
        neighbors = [int(j) for j in order if j != i][:16]
        neighborhoods = {}
        for radius in (0.15, 0.30, 0.60, 1.0):
            ix = np.flatnonzero((distances > 0) & (distances <= radius))
            vals, counts = np.unique(labels[ix], return_counts=True)
            neighborhoods[str(radius)] = {
                "point_ids": points.point_ids[ix].tolist(),
                "label_counts": {str(int(v)): int(n) for v, n in zip(vals, counts)},
                "z_range": [float(points.xyz[ix,2].min()), float(points.xyz[ix,2].max())] if len(ix) else None
            }
        rows.append({
            "point_id": int(pid), "xyz": points.xyz[i].tolist(),
            "original_label": int(original[i]), "preview_label": int(labels[i]),
            "S0_probabilities": np.asarray(points.semantic_probs[i]).tolist(),
            "source_box_local_xyz": geom.to_box_coordinates(points.xyz[i:i+1], source_box)[0].tolist(),
            "nearest_neighbors": [
                {"point_id": int(points.point_ids[j]), "distance": float(distances[j]),
                 "offset_xyz": (points.xyz[j]-points.xyz[i]).tolist(),
                 "original_label": int(original[j]), "preview_label": int(labels[j])}
                for j in neighbors
            ],
            "neighborhoods": neighborhoods
        })
    return {"support_count": len(points.point_ids), "targets": rows}

comparison1 = diagnose.compare(
    "Identify the actual changed label and common nearby observed surfaces.",
    inspect_disagreement, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["CANDIDATES", "S0", "LIDAR_CURRENT", "T0"],
    assumptions=["Neighborhood labels are semantic predictions, not confirmed instances.",
                 "The source box is a proposal; containment does not establish ownership."]
)
print(comparison1)

