"""Create a fresh workspace; no model calls, inference, or downloads."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace',type=Path,required=True)
p.add_argument('--prepared-root',type=Path,required=True)
p.add_argument('--dataset-root',type=Path)
p.add_argument('--offline-gt',type=Path)
p.add_argument('--allow-api',action='store_true',help='Enable paid calls only in subsequent explicit run commands.')
a=p.parse_args();w=a.workspace.resolve()
if (w/'configs/run.json').exists():raise SystemExit('Workspace already initialized; refusing to reset configuration or budget.')
for name in ['configs','reports','results','episodes']: (w/name).mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/'configs/run.example.json').read_text())
cfg.update(workspace=str(w),prepared_root=str(a.prepared_root.resolve()),authorized=a.allow_api)
if a.dataset_root:cfg['dataset_root']=str(a.dataset_root.resolve())
if a.offline_gt:cfg['offline_gt_path']=str(a.offline_gt.resolve())
rows=json.loads((ROOT/'configs/selection_100.json').read_text())
for r in rows:r['prepared_dir']=str(a.prepared_root.resolve()/r['frame']['sample_token'])
for name,value in [('run.json',cfg),('selection.json',rows)]:
    (w/'configs'/name).write_text(json.dumps(value,indent=2)+'\n')
for name in ['prompts','docs']:shutil.copytree(ROOT/name,w/name,dirs_exist_ok=True)
print('Workspace created:',w)
print('Set PTA_RUN_DIR to this directory before invoking src/run_selected.py or src/evaluate_selected.py.')
print('Paid API calls enabled:',a.allow_api)
