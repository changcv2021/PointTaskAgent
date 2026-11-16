"""Pure geometry/assignment; callers explicitly supply all inputs."""
import copy
import numpy as np
from pyquaternion import Quaternion

RANGES={'car':50.,'truck':50.,'bus':50.,'trailer':50.,'construction_vehicle':50.,'pedestrian':40.,'motorcycle':40.,'bicycle':40.,'traffic_cone':30.,'barrier':30.}

def name(box):return box.get('detection_name',box.get('tracking_name',box.get('category')))
def score(box):return box.get('detection_score',box.get('tracking_score',box.get('score',1.)))

def global_points(xyz,frame):
    transform=np.array(frame['lidar2global'])
    return xyz@transform[:3,:3].T+transform[:3,3]

def in_box(xyz,box,expand=0.):
    local=(xyz-np.array(box['translation']))@Quaternion(box['rotation']).rotation_matrix
    half=np.array(box['size'])[[1,0,2]]/2+expand
    return (np.abs(local)<=half+1e-8).all(1)

def prior_box(history,token):
    b=history['box']
    return dict(sample_token=token,translation=b[:3],size=[b[4],b[3],b[5]],rotation=Quaternion(axis=[0,0,1],radians=b[6]).elements.tolist(),velocity=history['velocity'],detection_name=history['category'],detection_score=max(0.,history['score']),attribute_name=default_attribute(history['category'],history['velocity']))

def default_attribute(cls,velocity):
    moving=np.linalg.norm(velocity)>0.2
    if cls in ['barrier','traffic_cone']:return ''
    if cls in ['bicycle','motorcycle']:return 'cycle.with_rider' if moving else 'cycle.without_rider'
    if cls=='pedestrian':return 'pedestrian.moving' if moving else 'pedestrian.standing'
    return 'vehicle.moving' if moving else ('vehicle.stopped' if cls=='bus' else 'vehicle.parked')

def filter_predictions(boxes,frame,bike_racks=()):
    result=[]
    for box in boxes:
        cls=name(box)
        if cls not in RANGES:continue
        if np.linalg.norm(np.array(box['translation'])[:2]-np.array(frame['ego_translation'])[:2])>=RANGES[cls]:continue
        if cls in ['bicycle','motorcycle'] and any(in_box(np.array([box['translation']]),rack)[0] for rack in bike_racks):continue
        result.append(box)
    return result

def center_matches(gt,pred,threshold=2.):
    """Same score-ordered greedy center matching as nuScenes detection accumulation."""
    if not gt or not pred:return []
    gpos=np.array([g['translation'][:2] for g in gt],dtype=float)
    ppos=np.array([p['translation'][:2] for p in pred],dtype=float)
    distances=np.linalg.norm(ppos[:,None,:]-gpos[None,:,:],axis=-1)
    same=np.array([name(p) for p in pred])[:,None]==np.array([name(g) for g in gt])[None,:]
    distances[~same]=np.inf
    result=[]
    for pi in sorted(range(len(pred)),key=lambda i:(score(pred[i]),i),reverse=True):
        gi=int(np.argmin(distances[pi]));distance=float(distances[pi,gi])
        if distance<threshold:result.append((gi,pi,distance));distances[:,gi]=np.inf
    return result

def apply_box_candidate(baseline,candidate):
    result=copy.deepcopy(baseline);target=candidate['target']
    if target is None:result.append(copy.deepcopy(candidate['box']))
    else:result[target]=copy.deepcopy(candidate['box'])
    return result

def detection_objective(gt,pred):
    net=0;dist=0.
    for threshold in [0.5,1.,2.,4.]:
        matches=center_matches(gt,pred,threshold)
        net+=2*len(matches)-len(pred)
        dist+=sum(m[2] for m in matches)
    return net,-dist
