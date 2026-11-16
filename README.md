# PointTaskAgent

Code-driven cross-task verification for 3D point-cloud perception.

[中文说明](README_CN.md) · [Reproduction](docs/REPRODUCTION.md) · [Data contract](docs/DATA_CONTRACT.md) · [Recorded examples](examples/SEVEN_OPTIMAL_CASES_STEP_BY_STEP.md)

PointTaskAgent reads frozen detection, segmentation and past-only tracking evidence,
writes Python diagnostic programs, executes them in a persistent isolated worker,
inspects measurements and real point-cloud views, and revises its diagnostics before
selecting an existing correction or **KEEP**. Ground truth is used only for offline
evaluation, not for candidate generation or agent inference.

This repository contains the curated **GPT-6.1 Sol / high** coding experiment:
100 examples from 100 distinct nuScenes scenes, with 50 detection and 50 segmentation
examples and at most five code attempts per example, including errors. Planning and
JSON-only finalization are model calls but do not execute additional diagnostic code.

## Method components

- Frozen perception inputs: **LION-Mamba**, **LitePT-S** and online **MCTrack**.
- Cross-task candidates: `TRACK_TO_DET`, `SEG_TO_DET`, `DET_TO_SEG`, `TRACK_TO_SEG`.
- Generated Python diagnostics with persistent variables, numerical measurements,
  verified point indices, real point-cloud rendering and execution feedback.
- Existing-box center refinement or a complete existing segmentation patch;
  the agent selects a candidate and does not synthesize arbitrary boxes or labels.
- A shared, durable API budget ledger and a five-attempt code limit.
- Offline mAP, NDS, mIoU and mAcc evaluation, plus beneficial/harmful/neutral outcomes.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Agent, isolated worker, scene/diagnostic APIs, renderer, operator primitives and evaluator |
| `prompts/` | Recorded planning and task prompts |
| `configs/` | Disabled-by-default API configuration, class mapping and frozen 100-scene selection |
| `scripts/` | Workspace initialization, offline checks and recorded recovery procedures |
| `reproduction/backbones/` | Backbone configurations, historical environment locks and source commits |
| `reproduction/upstream/` | Historical producer sources for baselines, tracking windows and prepared observations |
| `reproduction/results/` | Complete 100-example statistics, including unsuccessful selections and KEEP |
| `examples/` | Seven beneficial, oracle-qualified selections with recorded code, traces and decisions |
| `validation/` | Static checks and recorded-code replay receipts |
| `third_party/` | Attribution and license for the SeeGround excerpt |

## Reproduction scope

The portable coding core consumes prepared `observations.json` and
`observations.npz` files. **This is not a validated one-command raw-data-to-results
pipeline.** The historical upstream sources contain explicit path placeholders and
require configuration. NuScenes data, prepared arrays, model repositories/weights,
Python environments and credentials are external to this repository.

See [the reproduction guide](docs/REPRODUCTION.md) for environment requirements,
workspace initialization, software-only sandbox mounts and explicit API opt-in.
API credentials are supplied through environment variables. No model calls are
made by the package verification scripts. Historical pricing assumptions must be
checked for any newly authorized paid run.

The reported detection metrics cover 50 detection-example frames; segmentation
metrics cover 50 different segmentation-example frames. They are subset results,
not full nuScenes validation leaderboard scores. The seven explanatory examples
were chosen after offline evaluation and are not an unbiased new test split.

## Validation

The curated core passed Python syntax checks and replayed the recorded diagnostic
programs for detection `scene-0522` and segmentation `scene-0017`. All 14 rendered
views matched the original PNG hashes. The sandbox probes and budget-reservation
guard passed. This validation did not rerun the language model, backbone inference
or the full dataset evaluation; see [runtime receipt](validation/runtime.json).

Source hashes and extraction notes are in `SOURCE_MANIFEST.json`; published file
hashes are in `FILE_MANIFEST.json`. The public copy is based on the curated export
and preserves the original research directory separately.

## License and attribution

A license for the project's original code has not yet been selected; see
[LICENSE_STATUS.md](LICENSE_STATUS.md). The extracted SeeGround function retains
its Apache-2.0 license and attribution in `third_party/`. External datasets,
dependencies and model repositories retain their own terms.
