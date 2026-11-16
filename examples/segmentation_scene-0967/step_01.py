from scipy.spatial import cKDTree

cids = ["KEEP", "TRACK_TO_SEG:0", "TRACK_TO_SEG:1"]
print("CATALOG", scene.evidence_catalog())
print("PROVENANCE", {cid: scene.candidate_provenance(cid) for cid in cids})

support = scene.initial_points()
patches = {cid: scene.patch(cid) for cid in cids if cid != "KEEP"}
targets = np.unique(np.concatenate([p.point_ids for p in patches.values()])).astype(int)
print("PATCHES", {
    cid: {"point_ids": p.point_ids.tolist(), "new_labels": p.new_labels.tolist()}
    for cid, p in patches.items()
})

def attachment_test(cid, points):
    xyz = np.asarray(points.xyz)
    ids = np.asarray(points.point_ids)
    original = np.asarray(scene.labels_for(ids, "KEEP"))
    preview = np.asarray(scene.labels_for(ids, cid))
    tree = cKDTree(xyz)
    changed = np.flatnonzero(preview != original)
    rows = []
    for pid in targets:
        j = int(np.flatnonzero(ids == pid)[0])
        distances, indices = tree.query(xyz[j], k=min(21, len(ids)))
        indices = np.asarray(indices, dtype=int)
        distances = np.asarray(distances)
        neighbor_indices = indices[ids[indices] != pid]
        center = xyz[neighbor_indices].mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.cov((xyz[neighbor_indices] - center).T)
        )
        radius_results = []
        for radius in (0.06, 0.12, 0.25, 0.4):
            ix = np.asarray(tree.query_ball_point(xyz[j], radius), dtype=int)
            ix = ix[ids[ix] != pid]
            classes, counts = np.unique(original[ix], return_counts=True)
            radius_results.append({
                "radius_m": radius, "point_ids": ids[ix].tolist(),
                "original_class_counts": {
                    str(int(k)): int(v) for k, v in zip(classes, counts)
                },
                "z_extent_m": float(np.ptp(xyz[ix, 2])) if len(ix) else None
            })
        probs = np.asarray(points.semantic_probs[j])
        rows.append({
            "target_id": int(pid), "xyz": xyz[j].tolist(),
            "original_label": int(original[j]), "preview_label": int(preview[j]),
            "S0_probabilities": probs.tolist(),
            "nearest_points": [
                {"id": int(ids[i]), "distance_m": float(d),
                 "relative_xyz": (xyz[i] - xyz[j]).tolist(),
                 "S0_label": int(original[i]), "preview_label": int(preview[i])}
                for d, i in zip(distances, indices)
            ],
            "neighbor_covariance_eigenvalues_m2": eigenvalues.tolist(),
            "least_variance_direction": eigenvectors[:, 0].tolist(),
            "fixed_radius_neighborhoods": radius_results
        })
    return {
        "changed_ids": ids[changed].tolist(),
        "label_transitions": [
            [int(ids[i]), int(original[i]), int(preview[i])] for i in changed
        ],
        "targets": rows
    }

comparison1 = diagnose.compare(
    "What exact labels change, and what label-blind geometry surrounds the same target points?",
    attachment_test, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "CANDIDATES"],
    assumptions=[
        "Fixed neighborhoods are context, not confirmed object instances.",
        "S0 labels and probabilities are reused generator evidence.",
        "No exact generator point membership or held-out subset is known."
    ]
)
print(comparison1)

