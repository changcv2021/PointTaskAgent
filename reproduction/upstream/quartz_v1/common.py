"""Shared, small I/O and reproducibility helpers for the frozen evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT=Path('${PROJECT_ROOT}')
WORK=ROOT/'evaluation/quartz_v1'
RAW=ROOT/'data/nuscenes'
META=dict(use_camera=False,use_lidar=True,use_radar=False,use_map=False,use_external=False)

def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0]!=os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run this workload on an allocated Slurm compute node')

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)

def atomic_npz(path,**values):
    import numpy as np
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as stream: np.savez_compressed(stream,**values)
    tmp.replace(path)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def frame_seed(token):
    return int(hashlib.sha256(('pointtaskagent-20261029-'+token).encode()).hexdigest()[:8],16)

def set_seed(token):
    import random
    import numpy as np
    import torch
    seed=frame_seed(token)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def protocol():
    p=WORK/'protocol.json'
    actual=file_sha(p)
    assert actual==(WORK/'protocol.sha256').read_text().strip(),'Protocol changed after freezing'
    return read_json(p),actual

def frames(scene=None,shard=0,shards=1):
    data=read_json(WORK/'public_frames.json')
    scenes=sorted({f['scene_name'] for f in data})
    chosen=set(scenes[shard::shards]) if scene is None else {scenes[0] if scene=='first' else scene}
    assert chosen and chosen<=set(scenes)
    return [f for f in data if f['scene_name'] in chosen]

def deadline():
    output=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'-o'],text=True)
    end=next(w.split('=',1)[1] for w in output.split() if w.startswith('EndTime='))
    return datetime.datetime.fromisoformat(end).timestamp()-180

def receipt(component,status,**extra):
    worker=os.environ.get('PTA_WORKER_ID')
    value=dict(component=component,status=status,job_id=os.environ.get('SLURM_JOB_ID'),worker_id=worker,host=socket.gethostname(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    suffix='_'+worker if worker else ''
    atomic_json(WORK/'reports'/f'{component}_{os.environ.get("SLURM_JOB_ID","local")}{suffix}.json',value)
    print(json.dumps(value),flush=True)

def points(frame):
    import numpy as np
    return np.fromfile(RAW/frame['lidar_path'],np.float32).reshape(-1,5)
