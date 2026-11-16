"""GT-free formal action pools; preserve old candidates and strict lineage."""
import argparse,builtins,io
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sklearn.cluster import DBSCAN
from joint_common import *
CFG,PH=protocol();MAPPING=read_json(V1/'configs/class_mappings.json')['detector_name_to_segmentation_index']
OPTS=read_json(V1/'protocol.json')['candidate_rules']['det_to_seg']

def det_seg_unchanged_rule(f,boxes):
    # Exact v1 Det->Seg algorithm with the strict detector passed as input.
    tok=f['sample_token']
    with np.load(V1/'results/baseline/litept'/(tok+'.npz')) as d:prob=d['probabilities'];labels=d['labels']
    xyz=global_points(points(f)[:,:3],f);cs=[];arrays={};reject=Counter();attempts=0
    order=sorted(range(len(boxes)),key=lambda i:(-score(boxes[i]),i))
    eligible=[i for i in order if score(boxes[i])>=OPTS['min_detection_score'] and name(boxes[i]) in MAPPING][:OPTS['max_boxes_per_frame']]
    for di in eligible:
        attempts+=1;box=boxes[di];label=MAPPING[name(box)]
        idx=np.flatnonzero(in_box(xyz,box)&(prob[:,label]>=OPTS['min_class_probability'])&(prob.max(1)<OPTS['max_original_confidence']))
        if len(idx)<5:reject['too_few_eligible_points']+=1;continue
        groups=DBSCAN(eps=.75,min_samples=5,n_jobs=1).fit_predict(xyz[idx]);valid=[]
        for g in sorted(set(groups)-{-1}):
            members=idx[groups==g]
            if np.any(labels[members]==label):valid.extend(members.tolist())
        idx=np.array(sorted(valid),dtype=np.int64)
        if not len(idx):reject['no_supported_component']+=1;continue
        for boost in OPTS['log_probability_boosts']:
            logits=np.log(np.maximum(prob[idx],1e-12));logits[:,label]+=boost
            updated=logits.argmax(1).astype(np.uint8);change=updated!=labels[idx]
            if not change.any():reject['no_label_change']+=1;continue
            ci=len(cs);arrays[f'indices_{ci}']=idx[change].astype(np.int32);arrays[f'labels_{ci}']=updated[change]
            cs.append(dict(candidate_id=ci,region_id=di,boost=boost,points=int(change.sum()),detection_score=score(box)))
    return dict(candidates=cs,attempts=attempts,rejections=dict(reject)),arrays

def generate(f):
    tok=f['sample_token'];out=W/'results/pools'/(tok+'.json')
    if out.exists():assert read_json(out)['protocol_sha256']==PH;return tok
    base=baseline_detection(f);v3path=V3/'results/pools'/(tok+'.json');old=read_json(v3path)
    assert old['baseline_sha256']==digest(base) and old['protocol_sha256']==file_sha(V3/'protocol.json')
    state_path=V2/'results/window_tracking'/(tok+'.json');state=read_json(state_path)
    assert state['input_scan_tokens']==f['history_tokens'] and state['max_input_span_s']<=.5
    tp=V2/'results/candidates/temporal_to_det'/(tok+'.json');temporal=read_json(tp)
    assert temporal['protocol_sha256']==file_sha(V2/'protocol.json')
    sources={'track':old['pools']['track']['full'],'seg':old['pools']['seg']['full'],'temporal':temporal['candidates']}
    det={};legacy={}
    for source,cs in sources.items():
        tool={'track':'TRACK_TO_DET','seg':'SEG_TO_DET','temporal':'TEMPORAL_TO_DET'}[source]
        det[source]=make_actions(tool,base,cs,state if source!='seg' else None)
        legacy[source]=make_actions(tool,base,cs,state if source!='seg' else None,legacy=True)
        if source in ['track','seg']:
            projected=old['pools'][source]['replace_center'];by={c['candidate_id']:c for c in projected}
            for a in det[source]:
                if a['operation']==REFINE_CENTER:assert a['box']==by[a['candidate_id']]['box']
    dp=V1/'results/candidates/det_to_seg'/(tok+'.json');dr=read_json(dp)
    # Smoke validates the unchanged rule on each unaffected frame; full runs only
    # regenerate the 80 frames whose detector inputs required strict rebasing.
    if needs_strict_detection(f) or SMOKE:
        new,arrays=det_seg_unchanged_rule(f,base)
        if not needs_strict_detection(f):
            assert new['candidates']==dr['candidates'] and new['rejections']==dr['rejections']
            with np.load(dp.with_suffix('.npz')) as oldarrays:
                assert set(oldarrays.files)==set(arrays)
                for k in arrays:np.testing.assert_array_equal(arrays[k],oldarrays[k])
        else:
            dp=W/'results/detseg_strict_candidates'/(tok+'.json');save(dp,dict(protocol_sha256=PH,**new));save_npz(dp.with_suffix('.npz'),**arrays);dr=read_json(dp)
    sp=V2/'results/candidates/track_window_to_seg'/(tok+'.json');sr=read_json(sp)
    assert sr['protocol_sha256']==file_sha(V2/'protocol.json')
    seg={}
    for source,path,record in [('det',dp,dr),('track',sp,sr)]:
        seg[source]=dict(candidates=record['candidates'],metadata_path=str(path),metadata_sha256=file_sha(path),arrays_path=str(path.with_suffix('.npz')),arrays_sha256=file_sha(path.with_suffix('.npz')))
    save(out,dict(protocol_sha256=PH,sample_token=tok,scene=f['scene_name'],baseline_sha256=digest(base),strict_rebased=needs_strict_detection(f),history_span_s=state['max_input_span_s'],history_sweeps=len(state['input_scan_tokens']),source_sha256={str(v3path):file_sha(v3path),str(tp):file_sha(tp),str(state_path):file_sha(state_path)},det=det,legacy=legacy,seg=seg))
    return tok

SMOKE=False
def main():
    global SMOKE
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args();SMOKE=a.smoke;require_compute();fs=subset(a.smoke)
    def guard(fn):
        def wrapped(path,*args,**kw):
            if isinstance(path,(str,bytes,os.PathLike)) and any(s in str(path) for s in ['private_gt','sample_annotation.json','lidarseg/']):raise RuntimeError('GT forbidden in formal candidate generation')
            return fn(path,*args,**kw)
        return wrapped
    builtins.open=guard(builtins.open);io.open=guard(io.open)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,_ in enumerate(pool.map(generate,fs,chunksize=1)):
            if (i+1)%100==0:print('formal pools',i+1,len(fs),flush=True)
    receipt('pools_smoke' if a.smoke else 'pools','COMPLETE',frames=len(fs),gt_read_denial=True)
if __name__=='__main__':main()
