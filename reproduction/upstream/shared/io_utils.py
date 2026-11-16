"""Small provenance/I/O helpers; all new artifacts belong to this Slate run."""
import builtins
import contextlib
import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys

W = Path('${HISTORICAL_WORKSPACE}/ctgei_kimi_v6')
ROOT = Path('${PROJECT_ROOT}')
V1, V2, V4 = [ROOT/'evaluation'/x for x in ['quartz_v1','quartz_flow_v2','quartz_joint_v4']]
sys.path.insert(0,str(V2/'scripts'))
from v2_common import frames, baseline_detection, needs_strict_detection
from common import RAW, META, require_compute, points
from geometry import global_points, in_box, name, score, prior_box, RANGES
# Upstream imports prepend their own script directories. Restore this run's
# module precedence so its private Oracle is not confused with v1/oracle.py.
sys.path.insert(0,str(W/'src'))

def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    data=json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n'
    if path.suffix=='.gz':
        with gzip.open(tmp,'wt',encoding='utf-8') as stream:stream.write(data)
    else:tmp.write_text(data,encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_sha(array):return hashlib.sha256(array.tobytes()).hexdigest()

def save_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.partial')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)

def protocol():
    path=W/'protocol.json';p=read(path);h=sha(path)
    assert h==(W/'protocol.sha256').read_text().strip()
    for source,expected in p['source_sha256'].items():assert sha(source)==expected,source
    return p,h

def receipt(component,status,**kwargs):
    record=dict(component=component,status=status,job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
    save(W/'reports'/(component+'_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),record)
    print(json.dumps(record),flush=True)

@contextlib.contextmanager
def no_gt_io():
    """No annotation opens inside proposal/observation/replay code."""
    old_builtin,old_io=builtins.open,io.open
    def guard(fn):
        def wrapped(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)) and any(x in str(path) for x in ['private_gt','sample_annotation.json','/lidarseg/','oracle_private']):
                raise RuntimeError('Annotation access forbidden in observable execution: '+str(path))
            return fn(path,*args,**kwargs)
        return wrapped
    builtins.open,io.open=guard(old_builtin),guard(old_io)
    try:yield
    finally:builtins.open,io.open=old_builtin,old_io
