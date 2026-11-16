"""Version-two paths; baseline artifacts remain read-only."""
import sys,json,datetime,os,socket
from functools import lru_cache
from pathlib import Path
ROOT=Path('${PROJECT_ROOT}')
V1=ROOT/'evaluation/quartz_v1'
WORK=ROOT/'evaluation/quartz_flow_v2'
sys.path.insert(0,str(V1/'scripts'))
from common import RAW,META,read_json,atomic_json,atomic_npz,file_sha,require_compute,set_seed,deadline,points

@lru_cache(maxsize=1)
def manifest():return read_json(WORK/'public_windows.json')
def frames(scene=None,shard=0,shards=1):
    data=manifest()['frames'];scenes=sorted({f['scene_name'] for f in data})
    chosen=set(scenes[shard::shards]) if scene is None else {scenes[0] if scene=='first' else scene}
    return [f for f in data if f['scene_name'] in chosen]
def needs_strict_detection(frame):
    allowed=set(frame['window_tokens'])
    return any(s['sample_data_token'] not in allowed for s in frame['sweeps'])
def baseline_detection(frame):
    token=frame['sample_token']
    if needs_strict_detection(frame):
        path=WORK/'results/strict_baseline/lion'/(token+'.json')
        assert path.is_file(),'Missing strict baseline correction: '+token
        record=read_json(path)
        assert set(record['input_scan_tokens'])<=set(frame['window_tokens'])
        return record['results']
    return read_json(V1/'results/baseline/lion'/(token+'.json'))['results']
def protocol():
    actual=file_sha(WORK/'protocol.json')
    assert actual==(WORK/'protocol.sha256').read_text().strip()
    job=os.environ.get('SLURM_JOB_ID')
    if job:
        import tarfile
        restart=os.environ.get('SLURM_RESTART_COUNT','0')
        key=job+('_restart'+restart if restart!='0' else '')+('_'+os.environ['PTA_WORKER_ID'] if os.environ.get('PTA_WORKER_ID') else '')
        dest=WORK/'reports/run_inputs'/key
        stage=dest/'stages'/(Path(sys.argv[0]).stem+'.json')
        if not stage.exists():
            files=[p for folder in ['scripts','configs'] for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]+[WORK/'protocol.json',WORK/'protocol.sha256']
            atomic_json(stage,dict(protocol_sha256=actual,entrypoint=sys.argv,files_sha256={str(p.relative_to(WORK)):file_sha(p) for p in files},time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        if not (dest/'provenance.json').exists():
            dest.mkdir(parents=True,exist_ok=True)
            files=[p for folder in ['scripts','configs'] for p in (WORK/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]+[WORK/'protocol.json',WORK/'protocol.sha256']
            with tarfile.open(dest/'inputs.tar.gz','w:gz') as archive:
                for p in files:archive.add(p,arcname=str(p.relative_to(WORK)))
            atomic_json(dest/'provenance.json',dict(protocol_sha256=actual,files_sha256={str(p.relative_to(WORK)):file_sha(p) for p in files},time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return read_json(WORK/'protocol.json'),actual
def receipt(component,status,**extra):
    job=os.environ.get('SLURM_JOB_ID','local');worker=os.environ.get('PTA_WORKER_ID','')
    data=dict(component=component,status=status,job_id=job,worker_id=worker,restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0')),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),**extra)
    atomic_json(WORK/'reports'/f'{component}_{job}{"_"+worker if worker else ""}.json',data)
    print(json.dumps(data),flush=True)
