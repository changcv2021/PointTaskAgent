# Running the curated coding core

## Environment

Use Linux with bubblewrap (`bwrap`) and unprivileged user namespaces enabled,
Python 3.10, PyTorch, a matching PyTorch3D build, and `requirements.txt`.
Backbone environments are separate; historical locks and source commits are in
`reproduction/backbones`. Do not install all three backbone locks into one environment.
The validation receipt records versions actually used in the packaging replay.

On an HPC login node, only edit/read small files and prepare submissions.
Run imports, rendering, inference, evaluation, and large transfers in a compute allocation.
Choose partition/account/QOS and resources from your own live cluster configuration.

## Prepared inputs

Obtain the same frozen nuScenes validation observations described in
`DATA_CONTRACT.md`. Each selected token needs `observations.json` and
`observations.npz`; their SHA-256 values must match `configs/selection_100.json`.
Large data and weights are external, never automatically downloaded by this package.

```bash
python scripts/init_workspace.py --workspace /work/coding_run \
  --prepared-root /data/prepared \
  --dataset-root /data/nuscenes --offline-gt /data/offline/private_gt.json
export PTA_RUN_DIR=/work/coding_run
export PYTHONPATH="$PWD/src"
python scripts/verify_package.py
python scripts/check_runtime.py
```

`init_workspace.py` refuses to overwrite an existing run config. It never resets a
ledger or authorizes paid calls by default. To enable a future paid reproduction,
use `--allow-api` when creating a fresh workspace, or deliberately set `authorized`
to true in that workspace configuration. No such run was launched while packaging.

## API and sandbox configuration

Supply `PTA_API_BASE_URL` as the HTTPS Responses API base URL and `PTA_API_KEY`
through your secret manager or shell environment. Do not put real values in this
repository. The release keeps the recorded `gpt-6.1-sol` / `high` model guards;
an account must actually provide that model/deployment for a paid rerun.

If required, set `PTA_BWRAP`, `PTA_PYTHON`, `PTA_RUNTIME_ROOTS` and
`PTA_SANDBOX_PYTHONPATH`. The latter two are colon-separated **software-only** paths.
Never mount a project root, data root, home directory or an evaluation/GT directory
as a software root. The kernel clears inherited environment variables and passes no API key.
Only source, one frame's prepared observations and its output directory enter the worker.

## Paid run and evaluation

After explicitly enabling and configuring a paid run, the serial example is:

```bash
python src/run_selected.py --shards 1 --shard 0 --workers 1
python src/evaluate_selected.py
```

Use `--pilot` for the first two selected cases; finished decisions are reused by the
full run. Independent `--shard K --shards N` processes may share one workspace and
one locked durable budget ledger. Errors count toward the five-code-attempt cap.
Initial planning and JSON-only finalization are model calls but not code execution.
Pricing constants are historical conservative accounting assumptions, not a current
provider quote or invoice; verify the rate ceiling for a newly authorized run.
Never delete the budget ledger to resume a failed experiment.

Recovery scripts preserve the original experiment's narrow conditions. Run
`scripts/recover_truncated.py` only for pre-execution truncations it recognizes.
`scripts/finalize_interrupted_pilot.py` applies only to the documented four-step
pilot interruption; it is not a generic retry command.

GT is used only by the offline evaluator. It reports mAP/NDS on 50 detection frames
and mIoU/mAcc on 50 different segmentation frames. These are **subset** results,
not a full nuScenes validation leaderboard score. Detection's candidate-selection
objective is a lexicographic matching objective, not per-frame AP/NDS.

## Full upstream preparation

`reproduction/upstream` records the producer chain: baseline predictions → strict
past-only MCTrack windows (at most 0.5 s / 10 sweeps) → candidate pools → prepared
observations. Its `${PROJECT_ROOT}` / `${HISTORICAL_WORKSPACE}` strings are explicit
placeholders, not a working turnkey pipeline. Match class IDs, global coordinates,
frozen thresholds and hashes before accepting newly prepared data as a reproduction.
Do not add future tracking states or GT to candidate generation.
