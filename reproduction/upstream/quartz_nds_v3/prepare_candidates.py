"""Freeze attribute projection pools without loading any GT."""
import argparse,copy,builtins,io,time
from concurrent.futures import ProcessPoolExecutor
from nds_common import *
from candidates import target_box

CFG,PH=protocol()
def generate(f):
    tok=f['sample_token'];out=W/'results/pools'/(tok+'.json')
    if out.exists():assert read_json(out)['protocol_sha256']==PH;return tok
    base=baseline_detection(f);tp=V2/'results/candidates/track_window_to_det'/(tok+'.json');sp=V1/'results/candidates/seg_to_det'/(tok+'.json')
    track=read_json(tp);seg=read_json(sp)
    assert track['protocol_sha256']==file_sha(V2/'protocol.json') and seg['protocol_sha256']==file_sha(V1/'protocol.json')
    assert track.get('max_history_span_s',0)<=.5
    state=read_json(V2/'results/window_tracking'/(tok+'.json'))
    assert state['reset_from_empty'] and not state['old_tracker_states_read'] and state['input_scan_tokens']==f['history_tokens']
    assert len(f['history_tokens'])<=10 and state['max_input_span_s']<=.5
    conf={x['cluster_id']:x['confidence'] for x in seg['clusters']};rebased=[]
    for c in seg['candidates']:
        x=copy.deepcopy(c);box=x['box'];box['velocity']=[0.,0.];box['attribute_name']=default_attribute(name(box),box['velocity']);box['detection_score']=.5*conf[c['cluster_id']]
        x['target']=target_box(box,base)
        if x['target'] is not None:
            old=base[x['target']]
            for k in ['detection_score','velocity','attribute_name']:box[k]=copy.deepcopy(old[k])
        rebased.append(x)
    if not needs_strict_detection(f):assert rebased==seg['candidates'],'Semantic rebase changed an unaffected frame'
    pools={}
    for direction,cs in [('track',track['candidates']),('seg',rebased)]:
        pools[direction]={mode:[x for c in cs if (x:=projected(base,c,mode,CFG)) is not None] for mode in CFG['variants']}
    save(out,dict(protocol_sha256=PH,sample_token=tok,scene=f['scene_name'],baseline_sha256=digest(base),strict_correction=needs_strict_detection(f),max_history_span_s=state['max_input_span_s'],source_sha256={str(tp):file_sha(tp),str(sp):file_sha(sp)},pools=pools))
    return tok
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args();require_compute();fs=subset(a.smoke)
    # Deny privileged paths even if a future accidental import tries to read them.
    def guard(fn):
        def wrapped(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                text=str(path)
                if any(s in text for s in ['private_gt','gt_audit','sample_annotation.json','lidarseg/']):raise RuntimeError('GT access forbidden during candidate generation: '+text)
            return fn(path,*args,**kwargs)
        return wrapped
    builtins.open=guard(builtins.open);io.open=guard(io.open)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,_ in enumerate(pool.map(generate,fs,chunksize=1)):
            if (i+1)%100==0:print('frozen candidates',i+1,len(fs),flush=True)
    receipt('pools_smoke' if a.smoke else 'pools','COMPLETE',frames=len(fs),gt_read_denial=True,protocol_sha256=PH)
if __name__=='__main__':main()
