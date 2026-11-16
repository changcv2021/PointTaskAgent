"""Small adapters for the frozen offline NDS attribution experiment."""
import copy,datetime,hashlib,json,os,socket,sys
from pathlib import Path
ROOT=Path('${PROJECT_ROOT}')
W=ROOT/'evaluation/quartz_nds_v3';V1=ROOT/'evaluation/quartz_v1';V2=ROOT/'evaluation/quartz_flow_v2'
sys.path.insert(0,str(V2/'scripts'))
from v2_common import frames,baseline_detection,needs_strict_detection
from common import RAW,META,require_compute,file_sha,read_json
from geometry import name,score,filter_predictions,center_matches,detection_objective,apply_box_candidate,default_attribute

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    tmp.write_text(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8');tmp.replace(path)
def protocol():
    ph=file_sha(W/'protocol.json');assert ph==(W/'protocol.sha256').read_text().strip()
    cfg=read_json(W/'protocol.json')
    for p,h in cfg['source_protocols'].items():assert file_sha(Path(p))==h
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
def projected(base,c,variant,cfg):
    rule=cfg['variants'][variant];is_add=c['target'] is None
    if (is_add and rule['operations']=='replace') or (not is_add and rule['operations']=='append'):return None
    r=copy.deepcopy(c)
    if not is_add:
        old=base[c['target']]
        if 'preserve_fields' in rule:
            for k in rule['preserve_fields']:r['box'][k]=copy.deepcopy(old[k])
        elif isinstance(rule['replace_fields'],list):
            r['box']=copy.deepcopy(old)
            for k in rule['replace_fields']:r['box'][k]=copy.deepcopy(c['box'][k])
    assert r['box']['translation']==c['box']['translation']
    assert name(r['box'])==name(c['box']) and score(r['box'])==score(c['box'])
    return r
def materialize(base,patches):
    out=list(base)
    for c in patches:
        if c['target'] is None:out.append(c['box'])
        else:out[c['target']]=c['box']
    return out
def select(base,cs,frame,truth):
    objective=lambda boxes:detection_objective(truth['boxes'],filter_predictions(boxes,frame,truth['bike_racks']))
    current=list(base);best=objective(current);used=set();events=[];selected=[]
    for c in cs:
        target=c['target'];reason='no_gain';accepted=False
        if target is not None and target in used:reason='target_already_modified'
        else:
            trial=materialize(current,[c]);value=objective(trial)
            if value>best:
                current=trial;best=value;accepted=True;reason='accepted';selected.append(c['candidate_id'])
                if target is not None:used.add(target)
        events.append(dict(candidate_id=c['candidate_id'],accepted=accepted,reason=reason))
    return selected,events

