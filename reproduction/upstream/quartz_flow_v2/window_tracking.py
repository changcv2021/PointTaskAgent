"""Rebuild MCTrack from single-scan observations separately for every window."""
import copy,sys,time,argparse
import numpy as np
import yaml
from pyquaternion import Quaternion
from v2_common import *
sys.path.insert(0,str(ROOT/'sources/quartz/MCTrack'))
from tracker.base_tracker import Base3DTracker
from dataset.baseversion_dataset import BaseVersionTrackingDataset

def inputs(scan,detections,index,cfg,dt):
    boxes=[]
    for d in detections:
        q=Quaternion(d['rotation'])
        boxes.append(dict(category=d['detection_name'],detection_score=d['detection_score'],global_xyz=d['translation'],global_orientation=d['rotation'],global_yaw=float(q.yaw_pitch_roll[0]),lwh=[d['size'][1],d['size'][0],d['size'][2]],global_velocity=d['velocity'],global_acceleration=[0.,0.],bbox_image={}))
    row=dict(frame_id=index,timestamp=scan['timestamp_us'],cur_sample_token=scan['lidar_token'],transform_matrix={'global2lidar':scan['global2lidar'],'global2ego':scan['global2ego'],'cameras_transform_matrix':{}},bboxes=boxes)
    data=BaseVersionTrackingDataset(scan['scene_name'],[row],cfg)[0]
    # Native BBox assumes a 0.5 s interval for backward matching; use the
    # measured sweep interval consistently with KF prediction here.
    for b in data.bboxes:
        b.global_xyz_last=(np.asarray(b.global_xyz[:2])-np.asarray(b.global_velocity)*dt).tolist()+[b.global_xyz[2]]
        b.global_xyz_lwh_yaw_last=b.global_xyz_last+b.lwh+[b.global_yaw]
    return data

def interval(tracker,dt):
    assert 0<dt<=.5
    tracker.cfg['FRAME_RATE']=1/dt
    for tr in tracker.all_trajs.values():
        tr.frame_rate=1/dt
        for key in ['cv_filter_pose','ca_filter_pose','ctra_filter_pose','kalman_filter_yaw','kalman_filter_size']:
            getattr(tr,key).dt=dt

def rebuild(frame,scans,cfg,ph):
    history=frame['history_tokens'];t=frame['timestamp_us']
    assert len(history)<=10 and all(0<t-scans[tok]['timestamp_us']<=500000 for tok in history)
    cfg=copy.deepcopy(cfg);cfg['FRAME_RATE']=20
    tracker=Base3DTracker(cfg);filtered=[];source_hashes={};previous=None
    for i,tok in enumerate(history):
        scan=scans[tok];path=WORK/'results/lion_sweeps'/(tok+'.json');d=read_json(path)
        assert d['protocol_sha256']==ph and d['input_scan_tokens']==[tok] and d['input_span_s']==0
        source_hashes[tok]=file_sha(path)
        dt=.05 if previous is None else (scan['timestamp_us']-previous)/1e6
        interval(tracker,dt);data=inputs(scan,d['results'],i,cfg,dt)
        filtered.append((scan,data.bboxes))
        tracker.track_single_frame(data);previous=scan['timestamp_us']
    if history:
        interval(tracker,(t-previous)/1e6)
        tracker.predict_before_associate()
    out=[]
    for tid,tr in sorted(tracker.all_trajs.items()):
        observed=next((b for b in reversed(tr.bboxes) if not b.is_fake and b.frame_id==tr.last_updated_frame),None)
        if observed is None:continue
        box=tr.bboxes[-1]
        out.append(dict(track_id=tid,category=box.category,box=list(map(float,box.global_xyz_lwh_yaw_predict)),score=float(observed.det_score),velocity=list(map(float,tr.kalman_filter_pose.x[2:4])),last_observation_timestamp=scans[history[tr.last_updated_frame]]['timestamp_us'],first_observation_timestamp=scans[history[tr.first_updated_frame]]['timestamp_us'],status=int(tr.status_flag)))
    # Fixed temporal control: propagate each filtered single-scan box with its
    # predicted velocity, then deterministic same-class 2 m suppression.
    pool=[]
    for scan,boxes in filtered:
        age=(t-scan['timestamp_us'])/1e6
        for b in boxes:
            center=np.asarray(b.global_xyz)+np.r_[b.global_velocity,0.]*age
            pool.append(dict(track_id=len(pool),category=b.category,box=center.tolist()+b.lwh+[b.global_yaw],score=float(b.det_score),velocity=b.global_velocity,last_observation_timestamp=scan['timestamp_us'],first_observation_timestamp=scan['timestamp_us'],status=1))
    control=[]
    for h in sorted(pool,key=lambda h:(-h['score']*np.exp(-(t-h['last_observation_timestamp'])/1e6),-h['last_observation_timestamp'],h['track_id'])):
        if any(h['category']==k['category'] and np.linalg.norm(np.asarray(h['box'][:2])-k['box'][:2])<2 for k in control):continue
        control.append(h)
    return dict(protocol_sha256=ph,sample_token=frame['sample_token'],history=out,fixed_temporal_history=control,input_scan_tokens=history,input_detection_sha256=source_hashes,reset_from_empty=True,old_tracker_states_read=False,max_input_span_s=frame['window_span_s'],input_sweeps=len(history),measurement_mode='strict pretrained LION, one scan per observation; native MCTrack thresholds; timestamp-correct prediction and backward match')

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    require_compute();_,ph=protocol();data=manifest();selected=frames(a.scene,a.shard,a.shards)
    cfg=yaml.safe_load((V1/'configs/mctrack_nuscenes.yaml').read_text());end=deadline();done=0;dur=[]
    for f in selected:
        path=WORK/'results/window_tracking'/(f['sample_token']+'.json')
        if path.exists():assert read_json(path)['protocol_sha256']==ph;done+=1;continue
        if time.time()+max(60,max(dur,default=0)*1.5)>end:break
        start=time.time();atomic_json(path,rebuild(f,data['scans'],cfg,ph));dur.append(time.time()-start);done+=1
        print(f'window_tracking {done}/{len(selected)} {dur[-1]:.3f}s',flush=True)
    receipt('window_tracking','COMPLETE' if done==len(selected) else 'PARTIAL',frames=done,selected=len(selected))
    if done<len(selected):raise SystemExit(75)
if __name__=='__main__':main()
