import argparse,os,signal,socket,threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from collections import Counter
from pathlib import Path
from common_run import W,read,save,sha,PACKAGE_ROOT,require_compute
from agent import episode
from paid_api import Ledger,lock
STOP=threading.Event()
def run_one(row,cfg):
    token=row['frame']['sample_token'];e=row['episode'];out=W/'episodes'/token/e['episode_id']
    with lock(W/'results/episode_locks'/(token+'_'+e['episode_id']+'.lock')):
        if (out/'decision.json').exists():return read(out/'decision.json')
        if STOP.is_set():return None
        assert sha(Path(row['prepared_dir'])/'observations.json')==row['metadata_sha256']
        assert sha(Path(row['prepared_dir'])/'observations.npz')==row['arrays_sha256']
        d=episode(row['prepared_dir'],e,out,'azure',cfg)
        assert d['steps']<=5
        print('EPISODE',row['frame']['scene_name'],e['episode_id'],d['status'],d['candidate_id'],'steps',d['steps'],flush=True)
        return d
def main():
    require_compute()
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    cfg=read(W/'configs/run.json');rows=read(W/'configs/selection.json')
    rows=rows[:2] if a.pilot else rows[a.shard::a.shards]
    name='pilot' if a.pilot else 'shard_'+str(a.shard)
    with lock(W/'reports'/(name+'.lock')):
        result=[];start=time.monotonic()
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_one,r,cfg) for r in rows]
            try:
                for fut in as_completed(futures):
                    d=fut.result()
                    if d is not None:result.append(d)
                    state=dict(status='COMPLETE' if len(result)==len(rows) else 'RUNNING',completed=len(result),requested=len(rows),statuses=dict(Counter(d['status'] for d in result)),elapsed_s=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID','local'),budget=Ledger().summary())
                    save(W/'reports'/(name+'_progress.json'),state)
                    if sum(d['status']!='COMPLETE' for d in result)>=3:raise RuntimeError('Three abnormal episodes in shard; stop for diagnosis')
            except BaseException:
                STOP.set()
                for f in futures:f.cancel()
                raise
        if a.pilot and any(d['status']!='COMPLETE' for d in result):raise RuntimeError('Pilot episode failed')
if __name__=='__main__':main()
