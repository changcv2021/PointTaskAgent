"""Frozen GT-free motion, correspondence and window-history tools."""
import argparse,copy,time
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from pyquaternion import Quaternion
from v2_common import *
from geometry import global_points,in_box,name,score,prior_box,default_attribute,RANGES,apply_box_candidate
from candidates import fit_semantic_box,target_box

def temporal_boxes(frame,xyz,history,detections,settings):
    result=[];rejections=Counter();attempts=0
    for h in sorted(history,key=lambda h:(-h['score'],h['track_id'])):
        if len(result)>=settings['max_candidates']:break
        attempts+=1;age=(frame['timestamp_us']-h['last_observation_timestamp'])/1e6
        assert 0<age<=.5
        proposal=prior_box(h,frame['sample_token'])
        if np.linalg.norm(np.asarray(proposal['translation'])[:2]-frame['ego_translation'][:2])>=RANGES[name(proposal)]:rejections['outside_range']+=1;continue
        idx=np.flatnonzero(in_box(xyz,proposal,.3))
        if len(idx)<5:rejections['insufficient_current_geometry']+=1;continue
        proposal['detection_score']=float(np.clip(h['score']*np.exp(-age),0,1))
        rot=Quaternion(proposal['rotation']).rotation_matrix
        local=(xyz[idx]-proposal['translation'])@rot
        shift=rot@((np.quantile(local,.02,axis=0)+np.quantile(local,.98,axis=0))/2)
        length=np.linalg.norm(shift[:2])
        if length>.5:shift[:2]*=.5/length
        shift[2]=np.clip(shift[2],-.3,.3)
        for adjust in [False,True]:
            if len(result)>=settings['max_candidates']:break
            box=copy.deepcopy(proposal)
            if adjust:box['translation']=(np.asarray(box['translation'])+shift).tolist()
            target=target_box(box,detections)
            raw_score=box['detection_score']
            if target is not None:box['detection_score']=score(detections[target])
            result.append(dict(candidate_id=len(result),region_id=h['track_id'],target=target,box=box,geometry_adjusted=adjust,evidence_score=raw_score,history_span_s=(frame['timestamp_us']-h['first_observation_timestamp'])/1e6))
    return result,attempts,dict(rejections)

def box_seg_candidates(xyz,prob,boxes,mapping):
    cs=[];arrays={};rejected=Counter();attempts=0;labels=prob.argmax(1)
    for di,box in enumerate(sorted(boxes,key=lambda b:-score(b))[:64]):
        if score(box)<.25:continue
        attempts+=1;label=mapping[name(box)]
        idx=np.flatnonzero(in_box(xyz,box)&(prob[:,label]>=.1)&(prob.max(1)<.9))
        if len(idx)<5:rejected['too_few_eligible_points']+=1;continue
        cl=DBSCAN(eps=.75,min_samples=5,n_jobs=1).fit_predict(xyz[idx]);keep=[]
        for k in sorted(set(cl)-{-1}):
            ix=idx[cl==k]
            if np.any(labels[ix]==label):keep.extend(ix.tolist())
        idx=np.asarray(sorted(keep),dtype=int)
        if not len(idx):rejected['no_supported_component']+=1;continue
        for boost in [.5,1.]:
            logits=np.log(np.maximum(prob[idx],1e-12));logits[:,label]+=boost
            updated=logits.argmax(1).astype(np.uint8);change=updated!=labels[idx]
            if not change.any():rejected['no_label_change']+=1;continue
            ci=len(cs);arrays[f'indices_{ci}']=idx[change].astype(np.int32);arrays[f'labels_{ci}']=updated[change]
            cs.append(dict(candidate_id=ci,region_id=di,boost=boost,points=int(change.sum())))
    return cs,arrays,attempts,dict(rejected)

def correspondence_seg(xyz,prob,source_prob,source_indices,valid,settings,mapping):
    original=prob.argmax(1);source_label=source_prob.argmax(1);cs=[];arrays={};reject=Counter();attempts=0
    # Semantic names come from actual previous-frame predictions; no GT class.
    for category in settings['dynamic_classes']:
        label=mapping[category]
        eligible=valid&(source_label[source_indices]==label)&(source_prob[source_indices,label]>=.6)&(prob[:,label]>=.1)&(prob.max(1)<.9)
        idx=np.flatnonzero(eligible)
        if len(idx)<5:reject['too_few_eligible_points']+=1;continue
        groups=DBSCAN(eps=.75,min_samples=5,n_jobs=1).fit_predict(xyz[idx])
        members=[idx[groups==g] for g in sorted(set(groups)-{-1})]
        members.sort(key=lambda v:(-len(v),int(v[0])))
        for ix in members:
            if attempts>=64:break
            region=attempts;attempts+=1
            for boost in [.5,1.]:
                logits=np.log(np.maximum(prob[ix],1e-12));logits[:,label]+=boost
                updated=logits.argmax(1).astype(np.uint8);change=updated!=original[ix]
                if not change.any():reject['no_label_change']+=1;continue
                ci=len(cs);arrays[f'indices_{ci}']=ix[change].astype(np.int32);arrays[f'labels_{ci}']=updated[change]
                cs.append(dict(candidate_id=ci,region_id=region,category=category,boost=boost,points=int(change.sum())))
    return cs,arrays,attempts,dict(reject)

def generate(frame,settings,mapping,scans,ph):
    tok=frame['sample_token'];base=V1/'results/baseline'
    detections=baseline_detection(frame)
    with np.load(base/'litept'/(tok+'.npz')) as d:prob=d['probabilities'];labels=d['labels']
    xyz=global_points(points(frame)[:,:3],frame)
    def save(direction,cs,attempts,reject,arrays=None,**extra):
        p=WORK/'results/candidates'/direction/(tok+'.json')
        if arrays is not None:atomic_npz(p.with_suffix('.npz'),**arrays)
        atomic_json(p,dict(protocol_sha256=ph,sample_token=tok,candidates=cs,attempts=attempts,rejections=reject,**extra))
    state=read_json(WORK/'results/window_tracking'/(tok+'.json'));assert state['protocol_sha256']==ph and state['reset_from_empty']
    for direction,key in [('track_window_to_det','history'),('temporal_to_det','fixed_temporal_history')]:
        cs,attempts,reject=temporal_boxes(frame,xyz,state[key],detections,settings['temporal'])
        save(direction,cs,attempts,reject,max_history_span_s=state['max_input_span_s'])
        if direction=='temporal_to_det':
            # A deployable, GT-free fixed control in addition to its Oracle.
            fixed=copy.deepcopy(detections);used=set();events=[]
            for c in cs:
                target=c['target'];accepted=False
                if not c['geometry_adjusted'] and c['evidence_score']>=.25:
                    if target is None:
                        if target_box(c['box'],fixed) is None:fixed=apply_box_candidate(fixed,c);accepted=True
                    elif target not in used and c['evidence_score']>score(detections[target]):
                        box=copy.deepcopy(c['box']);box['translation']=((np.asarray(box['translation'])+detections[target]['translation'])/2).tolist()
                        fixed=apply_box_candidate(fixed,{**c,'box':box});used.add(target);accepted=True
                events.append(dict(candidate_id=c['candidate_id'],accepted=accepted))
            atomic_json(WORK/'results/e2/temporal_fixed_to_det'/(tok+'.json'),dict(protocol_sha256=ph,results=fixed,events=events,uses_gt=False))
    trackboxes=[prior_box(h,tok) for h in state['history']]
    cs,arrays,attempts,reject=box_seg_candidates(xyz,prob,trackboxes,mapping)
    save('track_window_to_seg',cs,attempts,reject,arrays)
    flowpath=WORK/'results/flow/causal'/(tok+'.json');flow=read_json(flowpath)
    assert flow['protocol_sha256']==ph and flow['mode']=='causal'
    assignments=np.full(len(xyz),-1,np.int32);support_labels=np.full(len(xyz),-1,np.int16);clusters=[]
    if flow['status']!='COMPLETE':
        for direction in ['flow_to_det','flow_to_seg','ego_temporal_to_seg']:save(direction,[],0,{'no_causal_pair':1},{})
        atomic_npz(WORK/'results/flow_evidence'/(tok+'.npz'),motion_cluster=assignments,correspondence_support_label=support_labels)
        return
    assert flow['causal'] and flow['input_span_s']<=.5 and max(flow['input_timestamps_us'])<=frame['timestamp_us']
    with np.load(flowpath.with_suffix('.npz')) as d:
        velocity=d['current_velocity_global'];valid=d['current_correspondence_valid'];source_indices=d['current_source_indices'];ground=d['current_ground_mask']
    dynamic=valid&(np.linalg.norm(velocity,axis=1)>=settings['flow']['dynamic_speed_mps'])
    source=flow['source_token'];srcpath=WORK/'results/litept_sources'/(source+'.npz')
    with np.load(srcpath) as d:source_prob=d['probabilities']
    src_labels=source_prob.argmax(1);support_labels[dynamic]=src_labels[source_indices[dynamic]]
    # Cluster motion first, then assign category using current/source semantics.
    idx=np.flatnonzero(dynamic)
    if len(idx)>=5:
        features=np.c_[xyz[idx]/.75,velocity[idx]/2.]
        group=DBSCAN(eps=1.,min_samples=5,n_jobs=1).fit_predict(features)
        for g in sorted(set(group)-{-1}):
            ix=idx[group==g]
            if len(ix)<5:continue
            semantic=(prob[ix]+source_prob[source_indices[ix]])*.5
            mean=semantic.mean(0);label=int(mean.argmax());reverse={v:k for k,v in mapping.items()}
            if label not in reverse or reverse[label] not in settings['flow']['dynamic_classes'] or mean[label]<.5:continue
            med=np.median(velocity[ix],axis=0)
            if np.median(np.linalg.norm(velocity[ix]-med,axis=1))>1.5:continue
            clusters.append(dict(indices=ix,category=reverse[label],label=label,confidence=float(mean[label]),center=xyz[ix].mean(0),velocity=med))
    clusters.sort(key=lambda c:(-len(c['indices']),*c['center'],c['category']));clusters=clusters[:64]
    cs=[];reject=Counter();descriptions=[]
    for ci,cluster in enumerate(clusters):
        ix=cluster['indices'];assignments[ix]=ci
        descriptions.append(dict(cluster_id=ci,category=cluster['category'],points=len(ix),velocity=cluster['velocity'].tolist(),confidence=cluster['confidence']))
        box=fit_semantic_box(frame,xyz,cluster,settings['class_lwh_m'])
        if box is None:reject['invalid_geometry']+=1;continue
        box['velocity']=cluster['velocity'][:2].tolist();box['attribute_name']=default_attribute(name(box),box['velocity'])
        if np.linalg.norm(np.asarray(box['translation'])[:2]-frame['ego_translation'][:2])>=RANGES[name(box)]:reject['outside_range']+=1;continue
        target=target_box(box,detections)
        if target is not None:
            box['detection_score']=score(detections[target])
            bounded=copy.deepcopy(detections[target]);shift=np.asarray(box['translation'])-bounded['translation'];norm=np.linalg.norm(shift[:2])
            if norm>.5:shift[:2]*=.5/norm
            shift[2]=np.clip(shift[2],-.3,.3);bounded['translation']=(np.asarray(bounded['translation'])+shift).tolist();bounded['velocity']=box['velocity'];bounded['attribute_name']=box['attribute_name']
            cs.append(dict(candidate_id=len(cs),cluster_id=ci,target=target,box=bounded,variant='bounded_center_velocity'))
        if len(cs)<64:cs.append(dict(candidate_id=len(cs),cluster_id=ci,target=target,box=box,variant='motion_cluster_pca'))
        if len(cs)>=64:break
    save('flow_to_det',cs,len(clusters),dict(reject),clusters=descriptions)
    cs,arrays,attempts,reject=correspondence_seg(xyz,prob,source_prob,source_indices,dynamic,settings['flow'],mapping)
    save('flow_to_seg',cs,attempts,reject,arrays)
    # Same source semantics, but ego alignment only: isolates the usefulness of
    # learned motion correspondence from an extra frame of semantic evidence.
    sxyz=global_points(points(scans[source])[:,:3],scans[source]);dist,nn=cKDTree(sxyz).query(xyz,k=1,workers=1)
    back=cKDTree(xyz).query(sxyz,k=1,workers=1)[1]
    static_valid=(dist<=settings['flow']['max_correspondence_distance_m'])&(back[nn]==np.arange(len(xyz)))&~ground
    cs,arrays,attempts,reject=correspondence_seg(xyz,prob,source_prob,nn,static_valid,settings['flow'],mapping)
    save('ego_temporal_to_seg',cs,attempts,reject,arrays)
    atomic_npz(WORK/'results/flow_evidence'/(tok+'.npz'),motion_cluster=assignments,correspondence_support_label=support_labels)

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    require_compute();settings,ph=protocol();data=manifest();selected=frames(a.scene,a.shard,a.shards);end=deadline();done=0;dur=[]
    mapping=read_json(V1/'configs/class_mappings.json')['detector_name_to_segmentation_index']
    for f in selected:
        marker=WORK/'results/candidates/complete'/(f['sample_token']+'.json')
        if marker.exists():assert read_json(marker)['protocol_sha256']==ph;done+=1;continue
        if time.time()+max(60,max(dur,default=0)*1.5)>end:break
        start=time.time();generate(f,settings,mapping,data['scans'],ph);atomic_json(marker,dict(protocol_sha256=ph));dur.append(time.time()-start);done+=1
        print(f'candidates {done}/{len(selected)} {dur[-1]:.3f}s',flush=True)
    receipt('candidates','COMPLETE' if done==len(selected) else 'PARTIAL',frames=done,selected=len(selected))
    if done<len(selected):raise SystemExit(75)
if __name__=='__main__':main()
