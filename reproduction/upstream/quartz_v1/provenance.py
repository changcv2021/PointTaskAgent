"""Capture the exact small-file inputs at job start, excluding credentials/data."""
import os
import platform
import sys
import tarfile
from common import WORK,require_compute,atomic_json,file_sha,protocol

require_compute();_,phash=protocol();job=os.environ['SLURM_JOB_ID']
run_id=job+('_'+os.environ['PTA_WORKER_ID'] if os.environ.get('PTA_WORKER_ID') else '')
folder=WORK/'reports/run_inputs'/run_id;folder.mkdir(parents=True,exist_ok=True)
files=[WORK/'protocol.json',WORK/'protocol.sha256']
files += sorted(p for p in (WORK/'scripts').rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.sbatch'))
files += sorted(p for p in (WORK/'configs').iterdir() if p.is_file())
files += sorted((WORK/'reports').glob('*.lock.txt'))
files += sorted((WORK/'reports').glob('*_source_commit.txt'))
archive=folder/'inputs.tar.gz';temporary=archive.with_suffix('.partial')
hashes={}
with tarfile.open(temporary,'w:gz') as tar:
    for path in files:
        relative=str(path.relative_to(WORK));hashes[relative]=file_sha(path)
        tar.add(path,arcname=relative,recursive=False)
temporary.replace(archive)
atomic_json(folder/'provenance.json',dict(job_id=job,command=sys.argv[1:],python=sys.version,executable=sys.executable,platform=platform.platform(),modules=os.environ.get('LOADEDMODULES'),cuda_home=os.environ.get('CUDA_HOME'),cxx=os.environ.get('CXX'),ld_preload=os.environ.get('LD_PRELOAD'),protocol_sha256=phash,files=hashes,archive_sha256=file_sha(archive)))
print('Recorded run inputs:',folder,flush=True)
