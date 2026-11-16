"""Offline prerequisites for paid runs. This script never calls an API."""
import importlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common_run import W,read,save,require_compute
require_compute()
for name in ['numpy','scipy','torch','pytorch3d','PIL','matplotlib','pyquaternion','sklearn','requests']:
    importlib.import_module(name)
assert os.environ.get('PTA_BWRAP') or shutil.which('bwrap'),'bubblewrap is required'
cfg=read(W/'configs/run.json')
assert 0<cfg['budget_usd']<=200 and 0<cfg['max_code_iterations']<=5
assert cfg['model_id']=='gpt-6.1-sol' and cfg['reasoning_effort']=='high'
assert cfg['gt_inference'] is False
rows=read(W/'configs/selection.json')
assert len(rows)==100 and len({r['frame']['scene_name'] for r in rows})==100
for r in rows:
    assert (Path(r['prepared_dir'])/'observations.json').is_file()
    assert (Path(r['prepared_dir'])/'observations.npz').is_file()
receipt=dict(status='PASS',paid_calls=0,scope='Offline dependency and configuration checks; server/model compatibility is validated on each real response.')
save(W/'reports/runtime_checks.json',receipt)
print(json.dumps(receipt,indent=2))
