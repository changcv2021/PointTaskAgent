"""Frozen v4 inputs and explicit production tool adapter."""
import copy,datetime,hashlib,json,os,socket,sys
from pathlib import Path
ROOT=Path('${PROJECT_ROOT}')
W=ROOT/'evaluation/quartz_joint_v4';V1=ROOT/'evaluation/quartz_v1';V2=ROOT/'evaluation/quartz_flow_v2';V3=ROOT/'evaluation/quartz_nds_v3'
sys.path.insert(0,str(V2/'scripts'))
from v2_common import frames,baseline_detection,needs_strict_detection
from common import RAW,META,require_compute,file_sha,read_json,points
from geometry import name,score,filter_predictions,center_matches,detection_objective,global_points,in_box
sys.path.insert(0,str(ROOT/'runtime'))
from cross_task_tools import VERSION,REFINE_CENTER,ADD_BOX,LEGACY_REPLACE_BOX,TRACK_TO_DET,SEG_TO_DET,make_actions,apply_action

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    tmp.write_text(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def save_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def protocol():
    ph=file_sha(W/'protocol.json');assert ph==(W/'protocol.sha256').read_text().strip();cfg=read_json(W/'protocol.json')
    for p,h in cfg['source_protocols'].items():assert file_sha(Path(p))==h
    assert file_sha(ROOT/'runtime/cross_task_tools/__init__.py')==cfg['tool_sha256']
    return cfg,ph
def receipt(component,status,**kw):
    r=dict(component=component,status=status,job_id=os.environ.get('SLURM_JOB_ID'),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),**kw)
    save(W/'reports'/(component+'_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),r);print(json.dumps(r),flush=True)
def subset(smoke=False):
    fs=frames()
    if smoke:
        first=sorted({f['scene_name'] for f in fs})[0]
        fs=[f for f in fs if f['scene_name']==first]+[next(f for f in fs if needs_strict_detection(f) and f['scene_name']!=first)]
    return fs
def materialize(base,actions):
    result=list(base)
    for a in actions:result=apply_action(result,a)
    return result
def alternate(a,b):
    result=[]
    for i in range(max(len(a),len(b))):
        if i<len(a):result.append(a[i])
        if i<len(b):result.append(b[i])
    return result
