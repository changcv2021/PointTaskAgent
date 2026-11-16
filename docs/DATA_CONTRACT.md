# Data and source contract

Each `prepared/<sample_token>/observations.json` contains the public scene metadata,
LiDAR-to-global transform, baseline detections, past-only tracking evidence, existing
candidates, episode definitions and class mappings. `observations.npz` contains
unaltered current-scan global `xyz`, `intensity`, LitePT `labels` and `probabilities`,
and candidate patch arrays `<array_key>_indices` / `<array_key>_labels`.
Point indices always refer to the original current scan. No GT belongs in these files.

The scene API exposes only the current episode's allowed candidates and evidence.
For detection, candidates preserve size, rotation, class, score, velocity and other
baseline attributes; the admitted operation is `REFINE_CENTER`. For segmentation,
the action is a complete frozen `PATCH_LABELS` proposal. KEEP is always available.
The agent diagnoses and chooses these proposals; it does not invent new candidate
parameters or run an unconstrained detection/segmentation network.

`configs/selection_100.json` fixes one episode in each of 100 distinct scenes,
50 detection and 50 segmentation. Selection used public predictions and seeded
ordering; it was not filtered by GT outcomes. The seven `examples/` were selected
after offline evaluation for explanation and are not an unbiased new test split.

Source lineage:

| Producer | Recorded source |
|---|---|
| LION-Mamba / LitePT baseline adapters | `reproduction/upstream/quartz_v1/infer.py` |
| Public input manifest | `reproduction/upstream/quartz_v1/public_manifest.py` |
| Strict input/history windows | `reproduction/upstream/quartz_flow_v2/prepare_windows.py` |
| Online reset-per-window MCTrack | `reproduction/upstream/quartz_flow_v2/window_tracking.py` |
| Semantic/track candidate proposals | `src/candidate_operators.py` and upstream generator sources |
| Center-only rebase and frozen joint pools | `reproduction/upstream/quartz_nds_v3/`, `quartz_joint_v4/`, `cross_task_tools/` |
| Point-array and candidate packing | `reproduction/upstream/preparation/prepare.py` and `shared/` |
| No-GT sample selection | `src/prepare_selection.py` |

The four `candidate_operators.py` entry-point families are extracted numerical
primitives. They do not on their own recreate IDs, provenance, center-only rebasing
or the full frozen cache. Use the historical producers and protocol files for those
details. Standalone extraction is validated only to the extent stated in the receipt.

NuScenes trainval LiDAR samples/sweeps, metadata and lidarseg GT are obtained under
the dataset's own terms. GT files are configured only in the offline evaluator.
Backbone repositories/checkpoints are external; use the recorded source commits,
configs and environment locks. Do not include data, models or credentials in a public repository.
