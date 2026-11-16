"""Export observable scan windows. Never load annotations or lidarseg labels."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pyquaternion import Quaternion
from v2_common import *

require_compute()
baseline=read_json(V1/'public_frames.json')
assert len(baseline)==6019 and len({f['scene_name'] for f in baseline})==150
tables={name:{r['token']:r for r in read_json(RAW/'v1.0-trainval'/(name+'.json'))} for name in ['sample_data','calibrated_sensor','ego_pose']}
scans={};out=[];audit=Counter();spans=[]
def scan(token,scene):
    if token in scans:return scans[token]
    sd=tables['sample_data'][token];cs=tables['calibrated_sensor'][sd['calibrated_sensor_token']];ego=tables['ego_pose'][sd['ego_pose_token']]
    qe=Quaternion(ego['rotation']);qc=Quaternion(cs['rotation']);q=qe*qc
    mat=np.eye(4);mat[:3,:3]=q.rotation_matrix;mat[:3,3]=np.asarray(qe.rotate(cs['translation']))+np.asarray(ego['translation'])
    ge=np.eye(4);ge[:3,:3]=qe.rotation_matrix;ge[:3,3]=ego['translation']
    row=dict(sample_token=token,lidar_token=token,lidar_path=sd['filename'],timestamp_us=sd['timestamp'],scene_name=scene,lidar2global=mat.tolist(),global2lidar=np.linalg.inv(mat).tolist(),lidar2global_quaternion=q.elements.tolist(),global2ego=np.linalg.inv(ge).tolist(),ego_translation=ego['translation'],sweeps=[])
    scans[token]=row
    return row

for frame_index,f in enumerate(baseline):
    current=scan(f['lidar_token'],f['scene_name']);t=f['timestamp_us']
    assert current['timestamp_us']==t
    # Use exactly the same unique historical scan identities as the frozen
    # LION input, with the additional strict 0.5-second causal bound.
    history=[]
    for s in f['sweeps']:
        tok=s['sample_data_token'];sd=tables['sample_data'][tok];age=t-sd['timestamp']
        if age<0:raise AssertionError('Future scan in baseline')
        if age>500000:audit['baseline_sweeps_outside_0p5']+=1;continue
        if age==0 or tok in history:continue
        scan(tok,f['scene_name']);history.append(tok)
    history.sort(key=lambda tok:scans[tok]['timestamp_us'])
    assert len(history)<=10
    window=history+[f['lidar_token']]
    span=(t-scans[window[0]]['timestamp_us'])/1e6;spans.append(span)
    assert 0<=span<=.5
    # Approx. 10 Hz using every second available 20 Hz sweep. Keep source,
    # target and all context inside the actual baseline scan window.
    reverse=window[::-1]
    causal=[reverse[min(2*i,len(reverse)-1)] for i in range(5)]
    causal=list(reversed(causal))
    valid=len(reverse)>=3
    if not valid:audit['causal_flow_no_past_pair']+=1
    elif len(set(causal))<5:audit['causal_flow_padded_boundary']+=1
    row=dict(f);row.update(window_tokens=window,history_tokens=history,window_span_s=span,causal_flow_tokens=causal,causal_flow_available=valid)
    # Native keyframe->next 10 Hz scan for independent model-quality scoring.
    # Future-target outputs must never be loaded by collaboration tools.
    sd=tables['sample_data'][f['lidar_token']];next_tok=sd['next']
    if next_tok and tables['sample_data'][next_tok]['next']:next_tok=tables['sample_data'][next_tok]['next']
    if next_tok:
        nxt=scan(next_tok,f['scene_name'])
        assert nxt['timestamp_us']>t
        row['native_flow_tokens']=[reverse[min(6,len(reverse)-1)],reverse[min(4,len(reverse)-1)],reverse[min(2,len(reverse)-1)],f['lidar_token'],next_tok]
    else:row['native_flow_tokens']=[];audit['native_flow_no_future_pair']+=1
    out.append(row)
    if (frame_index+1)%500==0:print(f'windows {frame_index+1}/6019 scans={len(scans)}',flush=True)
paths=[RAW/s['lidar_path'] for s in scans.values()]
with ThreadPoolExecutor(max_workers=8) as pool:
    exists=list(pool.map(lambda p:p.is_file(),paths))
missing=[str(p) for p,ok in zip(paths,exists) if not ok]
assert not missing,missing[:10]
atomic_json(WORK/'public_windows.json',{'frames':out,'scans':scans,'annotations_included':False})
receipt('windows','PASS',frames=len(out),scenes=150,unique_scans=len(scans),strict_window_max_s=max(spans),strict_window_min_s=min(spans),audit=dict(audit),baseline_protocol_sha256=(V1/'protocol.sha256').read_text().strip())
