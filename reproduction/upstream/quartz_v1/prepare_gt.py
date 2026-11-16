"""Privileged offline GT export; candidate generation must never import/read this."""
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_gt,add_center_dist,filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name
from common import WORK,RAW,frames,require_compute,atomic_json,receipt

require_compute()
nusc=NuScenes(version='v1.0-trainval',dataroot=str(RAW),verbose=False)
gt=load_gt(nusc,'val',DetectionBox,verbose=True)
gt=filter_eval_boxes(nusc,add_center_dist(nusc,gt),config_factory('detection_cvpr_2019').class_range,verbose=True)
result={}
for f in frames():
    token=f['sample_token'];sample=nusc.get('sample',token);annotations={};racks=[]
    for ann_token in sample['anns']:
        ann=nusc.get('sample_annotation',ann_token)
        if ann['category_name']=='static_object.bicycle_rack':racks.append({k:ann[k] for k in ['translation','size','rotation']})
        name=category_to_detection_name(ann['category_name'])
        if name is None:continue
        key=(name,tuple(ann['translation']))
        assert key not in annotations,'Ambiguous GT annotation location'
        annotations[key]=ann
    boxes=[]
    for box in gt[token]:
        ann=annotations[(box.detection_name,tuple(box.translation))]
        value=box.serialize()
        value.update(ann_token=ann['token'],instance_token=ann['instance_token'],num_lidar_pts=ann['num_lidar_pts'],distance=float(box.ego_dist))
        velocity=np.nan_to_num(np.array(value['velocity']),nan=0.).tolist()
        value['velocity']=velocity
        boxes.append(value)
    result[token]=dict(boxes=boxes,lidarseg_path=nusc.get('lidarseg',f['lidar_token'])['filename'],bike_racks=racks)
atomic_json(WORK/'private_gt.json',result)
receipt('offline_gt','COMPLETE',frames=len(result),annotations=sum(len(x['boxes']) for x in result.values()))
