import hashlib,json,os
from pathlib import Path
PACKAGE_ROOT=Path(__file__).resolve().parents[1]
W=Path(os.environ.get('PTA_RUN_DIR',PACKAGE_ROOT/'workspace')).resolve()
def read(p):
    value=json.loads(Path(p).read_text(encoding='utf-8'))
    if Path(p).name=='run.json' and isinstance(value,dict) and os.environ.get('PTA_API_BASE_URL'):
        value['api_base_url']=os.environ['PTA_API_BASE_URL'].rstrip('/')
    return value
def clean(x):
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,float) and not __import__('math').isfinite(x):return None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(clean(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()

def require_compute():
    import socket
    host=socket.gethostname().split('.')[0]
    if host.startswith(('login','hpc-login')):
        raise RuntimeError('Run this workload inside a compute allocation, not a login node.')
    expected=os.environ.get('PTA_VERIFIED_COMPUTE_HOST')
    if os.environ.get('SLURM_JOB_ID') and expected and host!=expected:
        raise RuntimeError('Compute hostname does not match the verified allocation host.')
