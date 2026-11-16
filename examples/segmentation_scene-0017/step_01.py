import numpy as np
import json

cids = ["KEEP", "DET_TO_SEG:0", "DET_TO_SEG:1"]
print("catalog", scene.evidence_catalog())
print("provenance", {cid: scene.candidate_provenance(cid) for cid in cids})
support = scene.initial_points()
patches = {cid: scene.patch(cid) for cid in cids if cid != "KEEP"}
disputed_ids = np.unique(np.concatenate([p.point_ids for p in patches.values()]))
print("patches", {cid: {"point_ids": p.point_ids.tolist(),
                            "new_labels": p.new_labels.tolist()}
                  for cid, p in patches.items()})

def surface_disagreement(cid, points):
    ids = np.asarray(points.point_ids)
    xyz = np.asarray(points.xyz)
    base = np.asarray(scene.labels_for(ids, "KEEP"))
    labels = np.asarray(scene.labels_for(ids, cid))
    disputed = np.isin(ids, disputed_ids)
    truck_ids = scene.semantic_indices_for_detection_class("truck")
    fixed = np.isin(base, truck_ids) & ~disputed
    qidx = np.flatnonzero(disputed)
    ridx = np.flatnonzero(fixed)
    local = geom.to_box_coordinates(points, scene.reference_boxes[0])
    out = {
        "support_count": len(ids), "fixed_truck_ids": ids[ridx].tolist(),
        "changed_ids": ids[labels != base].tolist(),
        "disputed_labels": labels[qidx].tolist(),
        "disputed_original_labels": base[qidx].tolist(),
        "disputed_box_local_xyz": np.asarray(local)[qidx].tolist(),
        "disputed_ids": ids[qidx].tolist(),
    }
    probs = np.asarray(points.semantic_probs)
    out["disputed_original_probabilities"] = [
        {"id": int(ids[i]), "original_class": int(base[i]),
         "p_original": float(probs[i, int(base[i])]),
         "p_truck": float(probs[i, truck_ids].sum())} for i in qidx
    ]
    if len(ridx) < 4:
        out["surface_test"] = "insufficient fixed truck support"
        return out
    ref = xyz[ridx]
    rd = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    np.fill_diagonal(rd, np.inf)
    out["fixed_truck_nn_spacing_quantiles_m"] = np.quantile(
        rd.min(axis=1), [0.1, 0.5, 0.9]).tolist()
    centered = ref - ref.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    ref_residual = np.abs(centered @ normal)
    out["fixed_truck_global_plane"] = {
        "singular_values": singular.tolist(), "normal_global": normal.tolist(),
        "reference_abs_residual_quantiles_m":
            np.quantile(ref_residual, [0.5, 0.9, 1.0]).tolist(),
        "disputed_abs_residual_m":
            np.abs((xyz[qidx] - ref.mean(axis=0)) @ normal).tolist()
    }
    rows = []
    for i in qidx:
        distances = np.linalg.norm(ref - xyz[i], axis=1)
        order = np.argsort(distances)[:min(8, len(ref))]
        neighbors = ref[order]
        mean = neighbors.mean(axis=0)
        _, sv, axes = np.linalg.svd(neighbors - mean, full_matrices=False)
        plane_normal = axes[-1]
        rows.append({
            "id": int(ids[i]), "nearest_fixed_truck_id": int(ids[ridx[order[0]]]),
            "nearest_distance_m": float(distances[order[0]]),
            "neighbor_ids": ids[ridx[order]].tolist(),
            "neighbor_max_distance_m": float(distances[order[-1]]),
            "neighbor_singular_values": sv.tolist(),
            "neighbor_plane_rms_m":
                float(np.sqrt(np.mean(((neighbors-mean) @ plane_normal)**2))),
            "query_plane_abs_residual_m": float(abs((xyz[i]-mean) @ plane_normal)),
            "query_minus_nearest_box_local_xyz":
                (np.asarray(local)[i] - np.asarray(local)[ridx[order[0]]]).tolist()
        })
    out["local_surface_measurements"] = rows
    return out

comparison1 = diagnose.compare(
    "Are the disputed labels identical across patches, and do disputed points continue fixed truck surfaces?",
    surface_disagreement, candidate_ids=cids, point_ids=support.point_ids,
    evidence_sources=["LIDAR_CURRENT", "S0", "D0", "CANDIDATES"],
    assumptions=[
        "Fixed S0 truck labels are semantic hypotheses, not confirmed instance membership.",
        "Local plane residuals require adequate two-dimensional support; singular values expose degeneracy.",
        "No exact generator membership or held-out independence is known."
    ])
print(json.dumps(comparison1, indent=2))
camera1 = viz.preset_camera("complementary", points=support, image_size=850)
view1 = viz.render(support, camera1, box_ids=(), color_mode="semantic",
                   labels=scene.labels_for(support.point_ids, "KEEP"),
                   highlight_ids=disputed_ids,
                   title="KEEP: exact disputed original points highlighted")
show(view1, "Complementary view of the same scan; highlights identify the exact patch targets.")

