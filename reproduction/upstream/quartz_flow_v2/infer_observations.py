"""Single-sweep detections and source-frame semantics without hidden history."""
import argparse,copy,time,os,json,fcntl
from v2_common import *
import infer as native_adapter

def main():
    p=argparse.ArgumentParser();p.add_argument('component',choices=['lion_sweeps','litept_sources','lion_strict_current']);p.add_argument('--scene');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    require_compute();_,ph=protocol();data=manifest();selected=frames(args.scene,args.shard,args.shards)
    if args.component=='lion_sweeps':tokens={t for f in selected for t in f['history_tokens']}
    else:tokens={f['causal_flow_tokens'][-2] for f in selected if f['causal_flow_available']}
    # Scheduling order only: finish complete scenes early for independent CPU
    # tracking. Every observation still has the same raw scan and token seed.
    ordered=sorted(tokens,key=lambda t:(data['scans'][t]['scene_name'],data['scans'][t]['timestamp_us'],t)) if args.component=='lion_sweeps' else sorted(tokens)
    scanlist=[copy.deepcopy(data['scans'][t]) for t in ordered]
    out=WORK/'results'/args.component
    if args.component=='lion_strict_current':
        scanlist=[copy.deepcopy(f) for f in selected if needs_strict_detection(f)]
        for f in scanlist:f['sweeps']=[s for s in f['sweeps'] if s['sample_data_token'] in f['window_tokens']]
        out=WORK/'results/strict_baseline/lion'
    key=lambda f:f['sample_token'] if args.component=='lion_strict_current' else f['lidar_token']
    out.mkdir(parents=True,exist_ok=True)
    existing={p.name for p in out.iterdir() if p.suffix=='.json'}
    pending=[(i,f,out/(key(f)+'.json')) for i,f in enumerate(scanlist) if key(f)+'.json' not in existing]
    print(json.dumps(dict(component=args.component,selected=len(scanlist),pending=len(pending))),flush=True)
    if not pending:receipt(args.component,'COMPLETE',selected=len(scanlist),new=0);return
    if args.scene is None and args.component!='lion_strict_current':assert (WORK/'reports/scene_acceptance.json').is_file(),'Full single-scene acceptance required'
    set_seed('v2-model')
    if args.component=='lion_sweeps':
        # The original NuScenesDataset helper always requests MAX_SWEEPS.
        # Supply a one-sweep config in this isolated adapter, keep strict weights.
        original=native_adapter.WORK;native_adapter.WORK=WORK
        run=native_adapter.detector(scanlist)
        native_adapter.WORK=original
    elif args.component=='lion_strict_current':
        # Established v1 LION adapter and checkpoint; only remove disallowed
        # input sweeps. Do not pad with duplicated points when fewer remain.
        import sys
        sys.path.insert(0,str(ROOT/'sources/quartz/LION'))
        from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
        original_get=NuScenesDataset.get_lidar_with_sweeps
        def clipped_get(ds,index,max_sweeps=1):return original_get(ds,index,min(max_sweeps,len(ds.infos[index]['sweeps'])+1))
        NuScenesDataset.get_lidar_with_sweeps=clipped_get
        run=native_adapter.detector(scanlist)
    else:run=native_adapter.segmentor(scanlist)
    import torch
    end=deadline();done=0;durations=[]
    # Multiple allocated jobs may help the same shard. Atomic file locks keep
    # exactly one writer per scan; deterministic seeds/input tensors unchanged.
    remaining=pending
    locks=out/'.locks';locks.mkdir(exist_ok=True)
    while remaining and time.time()+max(60,max(durations,default=0)*1.5)<=end:
        for i,f,dest in remaining:
            if time.time()+max(60,max(durations,default=0)*1.5)>end:break
            if dest.exists():continue
            with (locks/(key(f)+'.lock')).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if dest.exists():continue
                set_seed(key(f));start=time.time();result=run(i,f,dest);torch.cuda.synchronize()
                seconds=time.time()-start;durations.append(seconds)
                result.update(protocol_sha256=ph,lidar_token=f['lidar_token'],timestamp_us=f['timestamp_us'],input_scan_tokens=[f['lidar_token']],input_span_s=0.,seconds=seconds,job_id=os.environ['SLURM_JOB_ID'])
                if args.component=='lion_sweeps':result['max_sweeps']=1
                if args.component=='lion_strict_current':
                    result.update(input_scan_tokens=[f['lidar_token']]+[s['sample_data_token'] for s in f['sweeps']],input_span_s=f['window_span_s'],max_sweeps=1+len(f['sweeps']),sample_token=f['sample_token'])
                atomic_json(dest,result);done+=1
                print(json.dumps(dict(component=args.component,done=done,total=len(pending),token=f['lidar_token'],seconds=seconds)),flush=True)
        present={p.name for p in out.iterdir() if p.suffix=='.json'}
        remaining=[item for item in remaining if item[2].name not in present]
        if remaining and time.time()+60<end:time.sleep(1)
    receipt(args.component,'COMPLETE' if not remaining else 'PARTIAL',selected=len(scanlist),new=done,remaining=len(remaining))
    if remaining:raise SystemExit(75)
if __name__=='__main__':main()
