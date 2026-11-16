"""GT-free candidate generation. Inputs are public metadata, points and predictions."""
import argparse
from collections import Counter
import copy
import time
import numpy as np
from sklearn.cluster import DBSCAN
from pyquaternion import Quaternion
from common import WORK,frames,points,read_json,atomic_json,atomic_npz,protocol,require_compute,receipt,deadline
from geometry import global_points,in_box,name,score,prior_box,default_attribute,RANGES

def semantic_clusters(xyz,probability,settings,mapping):
    labels=probability.argmax(1);clusters=[]
    for cls,label in mapping.items():
        idx=np.flatnonzero((labels==label)&(probability[:,label]>=settings['min_confidence']))
        if len(idx)<settings['min_points']:continue
        assignment=DBSCAN(eps=settings['eps_m'],min_samples=settings['min_points'],n_jobs=1).fit_predict(xyz[idx])
        for cid in sorted(set(assignment)-{-1}):
            members=idx[assignment==cid]
            clusters.append(dict(category=cls,label=label,indices=members,confidence=float(probability[members,label].mean()),center=xyz[members].mean(0)))
    clusters.sort(key=lambda c:(-len(c['indices']),*c['center'],c['category']))
    return clusters[:settings['max_clusters_per_frame']]

def target_box(candidate,detections,gate=2.):
    matches=[(float(np.linalg.norm(np.array(d['translation'])[:2]-np.array(candidate['translation'])[:2])),i) for i,d in enumerate(detections) if name(d)==name(candidate)]
    if not matches:return None
    distance,index=min(matches)
    return index if distance<gate else None

def fit_semantic_box(frame,xyz,cluster,priors):
    pts=xyz[cluster['indices']]
    # DBSCAN min_samples is a density condition, not a lower bound on the
    # final membership after shared border points join an earlier cluster.
    # A singleton has no sample covariance and cannot define this PCA tool.
    if len(pts)<2:return None
    center=pts.mean(0)
    covariance=np.cov(pts[:,:2]-center[:2],rowvar=False)
    _,vectors=np.linalg.eigh(covariance);axis=vectors[:,-1]
    yaw=float(np.arctan2(axis[1],axis[0]));rotation=Quaternion(axis=[0,0,1],radians=yaw)
    local=(pts-center)@rotation.rotation_matrix
    lo,hi=np.quantile(local,[.02,.98],axis=0);extent=hi-lo+.2
    prior=np.array(priors[cluster['category']])
    if (extent>2*prior).any():return None
    extent=np.clip(extent,.5*prior,2*prior)
    center=center+rotation.rotation_matrix@((lo+hi)/2)
    return dict(sample_token=frame['sample_token'],translation=center.tolist(),size=extent[[1,0,2]].tolist(),rotation=rotation.elements.tolist(),velocity=[0.,0.],detection_name=cluster['category'],detection_score=.5*cluster['confidence'],attribute_name=default_attribute(cluster['category'],[0.,0.]))

def generate_frame(frame,settings,mapping,phash):
    token=frame['sample_token'];base=WORK/'results/baseline'
    det_record=read_json(base/'lion'/(token+'.json'));seg_record=read_json(base/'litept'/(token+'.json'))
    assert det_record['protocol_sha256']==seg_record['protocol_sha256']==phash
    detections=det_record['results']
    with np.load(base/'litept'/(token+'.npz')) as data:prob=data['probabilities'];labels=data['labels']
    xyz=global_points(points(frame)[:,:3],frame)
    assert len(xyz)==len(prob) and np.array_equal(labels,prob.argmax(1))
    clusters=semantic_clusters(xyz,prob,settings['semantic_clusters'],mapping)
    assignments=np.full(len(xyz),-1,dtype=np.int32)
    descriptions=[];box_candidates=[];reject=Counter()
    for ci,cluster in enumerate(clusters):
        assignments[cluster['indices']]=ci
        descriptions.append(dict(cluster_id=ci,category=cluster['category'],points=len(cluster['indices']),confidence=cluster['confidence']))
        if len(cluster['indices'])<2:reject['insufficient_pca_points']+=1;continue
        proposal=fit_semantic_box(frame,xyz,cluster,settings['class_lwh_m'])
        if proposal is None:reject['oversized_cluster']+=1;continue
        if np.linalg.norm(np.array(proposal['translation'])[:2]-np.array(frame['ego_translation'])[:2])>=RANGES[name(proposal)]:reject['outside_range']+=1;continue
        target=target_box(proposal,detections,settings['seg_to_det']['association_gate_m'])
        if target is not None:
            proposal['detection_score']=detections[target]['detection_score'];proposal['velocity']=detections[target]['velocity'];proposal['attribute_name']=detections[target]['attribute_name']
        box_candidates.append(dict(candidate_id=len(box_candidates),cluster_id=ci,target=target,box=proposal))
    out=WORK/'results/candidates/seg_to_det'
    atomic_npz(out/(token+'.npz'),cluster_assignments=assignments)
    atomic_json(out/(token+'.json'),dict(protocol_sha256=phash,candidates=box_candidates,clusters=descriptions,attempts=len(clusters),rejections=dict(reject)))

    opts=settings['det_to_seg'];seg_candidates=[];arrays={};reject=Counter();attempts=0
    order=sorted(range(len(detections)),key=lambda i:(-score(detections[i]),i))
    eligible=[i for i in order if score(detections[i])>=opts['min_detection_score'] and name(detections[i]) in mapping][:opts['max_boxes_per_frame']]
    for di in eligible:
        attempts+=1;det=detections[di];label=mapping[name(det)]
        idx=np.flatnonzero(in_box(xyz,det)&(prob[:,label]>=opts['min_class_probability'])&(prob.max(1)<opts['max_original_confidence']))
        if len(idx)<5:reject['too_few_eligible_points']+=1;continue
        group=DBSCAN(eps=.75,min_samples=5,n_jobs=1).fit_predict(xyz[idx]);valid=[]
        for g in sorted(set(group)-{-1}):
            members=idx[group==g]
            if np.any(labels[members]==label):valid.extend(members.tolist())
        idx=np.array(sorted(valid),dtype=np.int64)
        if not len(idx):reject['no_supported_component']+=1;continue
        for boost in opts['log_probability_boosts']:
            logits=np.log(np.maximum(prob[idx],1e-12));logits[:,label]+=boost
            updated=logits.argmax(1).astype(np.uint8);change=updated!=labels[idx]
            if not change.any():reject['no_label_change']+=1;continue
            ci=len(seg_candidates);arrays[f'indices_{ci}']=idx[change].astype(np.int32);arrays[f'labels_{ci}']=updated[change]
            seg_candidates.append(dict(candidate_id=ci,region_id=di,boost=boost,points=int(change.sum()),detection_score=score(det)))
    out=WORK/'results/candidates/det_to_seg'
    atomic_npz(out/(token+'.npz'),**arrays)
    atomic_json(out/(token+'.json'),dict(protocol_sha256=phash,candidates=seg_candidates,attempts=attempts,rejections=dict(reject)))

    state=read_json(base/'mctrack'/frame['scene_name']/(token+'.json'));assert state['protocol_sha256']==phash
    opts=settings['track_to_det'];track_candidates=[];reject=Counter();attempts=0;max_span=0.
    for hist in sorted(state['history'],key=lambda h:(-h['score'],h['track_id'])):
        if len(track_candidates)>=opts['max_candidates_per_frame']:break
        attempts+=1;age=(frame['timestamp_us']-hist['last_observation_timestamp'])/1e6
        assert age>0
        span=(frame['timestamp_us']-hist['first_observation_timestamp'])/1e6;max_span=max(max_span,span)
        if age>opts['max_observation_age_s']:reject['stale_history']+=1;continue
        proposal=prior_box(hist,token);idx=np.flatnonzero(in_box(xyz,proposal,opts['box_expansion_m']))
        if len(idx)<opts['min_lidar_points']:reject['insufficient_current_geometry']+=1;continue
        proposal['detection_score']=float(np.clip(hist['score']*np.exp(-age),0,1))
        rotation=Quaternion(proposal['rotation']).rotation_matrix
        local=(xyz[idx]-proposal['translation'])@rotation
        delta=(np.quantile(local,.02,axis=0)+np.quantile(local,.98,axis=0))/2
        shift=rotation@delta
        xy_norm=np.linalg.norm(shift[:2])
        if xy_norm>opts['max_shift_xy_m']:shift[:2]*=opts['max_shift_xy_m']/xy_norm
        shift[2]=np.clip(shift[2],-opts['max_shift_z_m'],opts['max_shift_z_m'])
        for variant in [0,1]:
            if len(track_candidates)>=opts['max_candidates_per_frame']:break
            candidate=copy.deepcopy(proposal)
            if variant:candidate['translation']=(np.array(candidate['translation'])+shift).tolist()
            target=target_box(candidate,detections)
            if target is not None:
                candidate['detection_score']=detections[target]['detection_score']
            track_candidates.append(dict(candidate_id=len(track_candidates),track_id=hist['track_id'],target=target,box=candidate,geometry_adjusted=bool(variant),history_span_s=span,observation_age_s=age))
    out=WORK/'results/candidates/track_to_det'
    atomic_json(out/(token+'.json'),dict(protocol_sha256=phash,candidates=track_candidates,attempts=attempts,rejections=dict(reject),max_history_span_s=max_span))

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
    require_compute();config,phash=protocol();mapping=read_json(WORK/'configs/class_mappings.json')['detector_name_to_segmentation_index'];selected=frames(args.scene,args.shard,args.shards);end=deadline();done=0;durations=[]
    for frame in selected:
        outputs=[WORK/'results/candidates'/direction/(frame['sample_token']+'.json') for direction in ['seg_to_det','det_to_seg','track_to_det']]
        if all(p.exists() for p in outputs):
            assert all(read_json(p)['protocol_sha256']==phash for p in outputs);done+=1;continue
        if time.time()+max(60.,max(durations,default=0.)*1.5)>end:break
        start=time.time();generate_frame(frame,config['candidate_rules'],mapping,phash);durations.append(time.time()-start);done+=1
        print(f"candidates {done}/{len(selected)} {durations[-1]:.3f}s",flush=True)
    receipt('candidates','COMPLETE' if done==len(selected) else 'PARTIAL',frames=done,selected=len(selected))
    if done<len(selected):raise SystemExit(75)

if __name__=='__main__':main()
