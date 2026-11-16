"""Export only observable metadata; no annotations or semantic labels in this file."""
import pickle
import sys
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from common import ROOT,WORK,RAW,require_compute,read_json,atomic_json,receipt,protocol,file_sha

require_compute()
config,phash=protocol()
for component,path in [('detection','LION-Mamba/nus_mamba.pth'),('segmentation','LitePT-S/model_best.pth')]:
    assert file_sha(ROOT/'checkpoints'/path)==config['models'][component]['checkpoint_sha256']
sys.path.insert(0,str(ROOT/'sources/quartz/MCTrack'))
from preprocess.convert_nuscenes import load_camera_info,transform_matrix
assert read_json(ROOT/'reports/full_data_audit.json')['status']=='PASS'
assert read_json(ROOT/'reports/val_infos.json')['status']=='PREPARED'
nusc=NuScenes(version='v1.0-trainval',dataroot=str(RAW),verbose=False)
alignment=read_json(ROOT/'data/val_alignment.json')
with (RAW/'nuscenes_infos_10sweeps_val.pkl').open('rb') as f: infos=pickle.load(f)
infos={i['token']:i for i in infos}
out=[]
scene_counts={}
for row in alignment:
    sd=nusc.get('sample_data',row['lidar_token'])
    cs=nusc.get('calibrated_sensor',sd['calibrated_sensor_token'])
    ego=nusc.get('ego_pose',sd['ego_pose_token'])
    q=Quaternion(ego['rotation'])*Quaternion(cs['rotation'])
    translation=Quaternion(ego['rotation']).rotate(cs['translation'])+np.array(ego['translation'])
    matrix=np.eye(4);matrix[:3,:3]=q.rotation_matrix;matrix[:3,3]=translation
    np.testing.assert_allclose(np.linalg.inv(matrix)@matrix,np.eye(4),atol=1e-8)
    frame={k:row[k] for k in ['scene_name','scene_token','sample_token','lidar_token','timestamp_us','lidar_path']}
    frame.update(lidar2global=matrix.tolist(),global2lidar=np.linalg.inv(matrix).tolist(),lidar2global_quaternion=q.elements.tolist(),ego_translation=ego['translation'],frame_index=scene_counts.get(row['scene_token'],0))
    frame['global2ego']=transform_matrix(ego['translation'],ego['rotation'],inverse=True).tolist()
    frame['cameras_transform_matrix']=load_camera_info(nusc,nusc.get('sample',row['sample_token']))
    frame['sweeps']=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in sweep.items() if k in ['lidar_path','sample_data_token','transform_matrix','time_lag']} for sweep in infos[row['sample_token']]['sweeps']]
    scene_counts[row['scene_token']]=frame['frame_index']+1
    out.append(frame)
atomic_json(WORK/'public_frames.json',out)
receipt('public_manifest','PASS',scenes=len(scene_counts),frames=len(out),gt_fields_exported=False)
