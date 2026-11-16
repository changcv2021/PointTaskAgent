"""GT-free primitives extracted from the frozen candidate generators.
These primitives do not rebuild the complete preparation lineage by themselves.
Only existing-box center refinements are admitted by the coding input adapter.
"""
import copy
from collections import Counter
import numpy as np
from sklearn.cluster import DBSCAN
from pyquaternion import Quaternion
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

def track_to_det_proposals(frame,xyz,history,detections,settings):
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


def track_to_seg_patches(xyz,prob,boxes,mapping):
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

def det_to_seg_patches(xyz,prob,labels,boxes,mapping,settings):
    # Exact v1 Det->Seg algorithm with the strict detector passed as input.
    cs=[];arrays={};reject=Counter();attempts=0
    order=sorted(range(len(boxes)),key=lambda i:(-score(boxes[i]),i))
    eligible=[i for i in order if score(boxes[i])>=settings['min_detection_score'] and name(boxes[i]) in mapping][:settings['max_boxes_per_frame']]
    for di in eligible:
        attempts+=1;box=boxes[di];label=mapping[name(box)]
        idx=np.flatnonzero(in_box(xyz,box)&(prob[:,label]>=settings['min_class_probability'])&(prob.max(1)<settings['max_original_confidence']))
        if len(idx)<5:reject['too_few_eligible_points']+=1;continue
        groups=DBSCAN(eps=.75,min_samples=5,n_jobs=1).fit_predict(xyz[idx]);valid=[]
        for g in sorted(set(groups)-{-1}):
            members=idx[groups==g]
            if np.any(labels[members]==label):valid.extend(members.tolist())
        idx=np.array(sorted(valid),dtype=np.int64)
        if not len(idx):reject['no_supported_component']+=1;continue
        for boost in settings['log_probability_boosts']:
            logits=np.log(np.maximum(prob[idx],1e-12));logits[:,label]+=boost
            updated=logits.argmax(1).astype(np.uint8);change=updated!=labels[idx]
            if not change.any():reject['no_label_change']+=1;continue
            ci=len(cs);arrays[f'indices_{ci}']=idx[change].astype(np.int32);arrays[f'labels_{ci}']=updated[change]
            cs.append(dict(candidate_id=ci,region_id=di,boost=boost,points=int(change.sum()),detection_score=score(box)))
    return dict(candidates=cs,attempts=attempts,rejections=dict(reject)),arrays

