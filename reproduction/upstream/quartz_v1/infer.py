"""Resumable native LION/LitePT inference, with no GT in model inputs."""
import argparse
import json
import logging
import os
import sys
import time
import numpy as np
from common import ROOT,WORK,frames,protocol,require_compute,set_seed,atomic_json,atomic_npz,read_json,receipt,deadline,points

def detector(selected):
    import torch
    from pyquaternion import Quaternion
    sys.path.insert(0,str(ROOT/'sources/quartz/LION'))
    os.chdir(ROOT/'sources/quartz/LION/tools')
    from pcdet.config import cfg,cfg_from_yaml_file
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.datasets.nuscenes.nuscenes_utils import cls_attr_dist
    from pcdet.models import build_network,load_data_to_gpu
    cfg_from_yaml_file(str(WORK/'configs/lion_nuscenes.yaml'),cfg)
    cfg.DATA_CONFIG.SET_NAN_VELOCITY_TO_ZEROS=False
    ds=NuScenesDataset(cfg.DATA_CONFIG,cfg.CLASS_NAMES,training=False,logger=logging.getLogger('lion'))
    ds.infos=[dict(token=f['sample_token'],lidar_path=f['lidar_path'],sweeps=[{**s,'transform_matrix':None if s['transform_matrix'] is None else np.array(s['transform_matrix'])} for s in f['sweeps']]) for f in selected]
    model=build_network(cfg.MODEL,len(cfg.CLASS_NAMES),ds).cuda().eval()
    state=torch.load(ROOT/'checkpoints/LION-Mamba/nus_mamba.pth',map_location='cpu')['model_state']
    model.load_state_dict(state,strict=True)
    def run(index,frame,dest):
        batch=ds.collate_batch([ds[index]])
        assert 'gt_boxes' not in batch
        load_data_to_gpu(batch)
        with torch.inference_mode(): pred,_=model(batch)
        p=pred[0]
        boxes=p['pred_boxes'].cpu().numpy(); scores=p['pred_scores'].cpu().numpy(); labels=p['pred_labels'].cpu().numpy()
        assert boxes.shape==(len(scores),9) and np.isfinite(boxes).all() and np.isfinite(scores).all()
        transform=np.array(frame['lidar2global']); qglobal=Quaternion(frame['lidar2global_quaternion'])
        result=[]
        for box,score,label in zip(boxes,scores,labels):
            name=cfg.CLASS_NAMES[int(label)-1]
            center=transform[:3,:3]@box[:3]+transform[:3,3]
            vel=transform[:3,:3]@np.r_[box[7:9],0.]
            rotation=qglobal*Quaternion(axis=[0,0,1],radians=float(box[6]))
            attr=None
            if np.linalg.norm(vel[:2])>0.2:
                if name in ['car','construction_vehicle','bus','truck','trailer']: attr='vehicle.moving'
                elif name in ['bicycle','motorcycle']: attr='cycle.with_rider'
            elif name=='pedestrian': attr='pedestrian.standing'
            elif name=='bus': attr='vehicle.stopped'
            if attr is None: attr=max(cls_attr_dist[name].items(),key=lambda x:x[1])[0]
            result.append(dict(sample_token=frame['sample_token'],translation=center.tolist(),size=box[[4,3,5]].tolist(),rotation=rotation.elements.tolist(),velocity=vel[:2].tolist(),detection_name=name,detection_score=float(score),attribute_name=attr))
        atomic_npz(dest.with_suffix('.npz'),boxes=boxes,scores=scores,labels=labels)
        return dict(results=result,boxes=len(result),max_sweeps=10)
    return run

def segmentor(selected):
    import torch
    from copy import deepcopy
    sys.path.insert(0,str(ROOT/'sources/quartz/LitePT'))
    os.chdir(ROOT/'sources/quartz/LitePT')
    from utils.config import Config
    from models import build_model
    from datasets.nuscenes import NuScenesDataset
    from datasets.utils import collate_fn
    cfg=Config.fromfile(str(WORK/'configs/litept_nuscenes.py'))
    model=build_model(cfg.model).cuda().eval()
    state=torch.load(ROOT/'checkpoints/LitePT-S/model_best.pth',map_location='cpu',weights_only=False)['state_dict']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    class ObservableDataset(NuScenesDataset):
        def get_data_list(self): return selected
        def get_data_name(self,idx): return selected[idx]['lidar_token']
        def get_data(self,idx):
            p=points(selected[idx])
            return dict(coord=p[:,:3],strength=p[:,3:4]/255,segment=np.full(len(p),-1,dtype=np.int64),name=self.get_data_name(idx))
    settings=deepcopy(dict(cfg.data.test)); settings.pop('type')
    ds=ObservableDataset(**settings)
    def run(index,frame,dest):
        data=ds[index]
        n=len(data['segment'])
        pred=torch.zeros((n,16),dtype=torch.float32,device='cuda')
        votes=torch.zeros(n,dtype=torch.float32,device='cuda')
        for fragment in data['fragment_list']:
            batch=collate_fn([fragment])
            batch={k:v.cuda(non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            assert 'segment' not in batch
            with torch.inference_mode(): prob=model(batch)['seg_logits'].softmax(-1)
            # Identical summation order to the native tester. Each fragment has unique point indices.
            ix=batch['index']
            assert len(ix)==len(torch.unique(ix))
            pred[ix]+=prob; votes[ix]+=1
        assert (votes>0).all() and torch.isfinite(pred).all()
        native_labels=pred.argmax(1)
        pred=pred/votes[:,None]
        assert torch.equal(native_labels,pred.argmax(1))
        inv=torch.as_tensor(data['inverse'],dtype=torch.long,device='cuda') if 'inverse' in data else torch.arange(n,device='cuda')
        probability=pred[inv].cpu().numpy()
        labels=native_labels[inv].cpu().numpy().astype(np.uint8)
        assert len(labels)==len(points(frame))
        assert np.array_equal(labels,probability.argmax(1))
        np.testing.assert_allclose(probability.sum(1),1,atol=2e-5)
        atomic_npz(dest.with_suffix('.npz'),probabilities=probability,labels=labels,votes=votes[inv].cpu().numpy().astype(np.uint16))
        return dict(points=len(labels),tta_augmentations=len(ds.aug_transform),fragments=len(data['fragment_list']),native_labels_equal=True,probability_dtype='float32')
    return run

def main():
    p=argparse.ArgumentParser();p.add_argument('component',choices=['lion','litept']);p.add_argument('--scene');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    args=p.parse_args(); require_compute()
    run_id=os.environ['SLURM_JOB_ID']+('_'+os.environ['PTA_WORKER_ID'] if os.environ.get('PTA_WORKER_ID') else '')
    if not (WORK/'reports/run_inputs'/run_id/'provenance.json').exists():
        import provenance  # Records inputs for already-submitted Slurm wrappers as well.
    import torch
    assert torch.cuda.is_available()
    _,phash=protocol()
    if args.scene is None:
        assert any(read_json(path).get('status')=='PASS' and read_json(path).get('protocol_sha256')==phash for path in (WORK/'reports').glob('end_to_end_scene_*.json')), 'Complete the real-scene functional acceptance before scaling inference'
    selected=frames(args.scene,args.shard,args.shards)
    out=WORK/'results/baseline'/args.component;out.mkdir(parents=True,exist_ok=True)
    end=deadline();pending=[]
    for i,frame in enumerate(selected):
        dest=out/(frame['sample_token']+'.json')
        if dest.exists():
            previous=read_json(dest)
            assert previous['protocol_sha256']==phash and dest.with_suffix('.npz').is_file()
        else: pending.append((i,frame,dest))
    if not pending:
        receipt(args.component,'COMPLETE',selected=len(selected),processed=0,protocol_sha256=phash); return
    set_seed('model-startup')
    run={'lion':detector,'litept':segmentor}[args.component](selected)
    completed=0;durations=[]
    for i,frame,dest in pending:
        if time.time()+max(30.,max(durations,default=0.)*1.5)>end: break
        start=time.time();set_seed(frame['sample_token'])
        result=run(i,frame,dest)
        torch.cuda.synchronize()
        seconds=time.time()-start;durations.append(seconds)
        result.update(sample_token=frame['sample_token'],lidar_token=frame['lidar_token'],scene_token=frame['scene_token'],protocol_sha256=phash,seconds=seconds,job_id=os.environ['SLURM_JOB_ID'],peak_gpu_bytes=torch.cuda.max_memory_allocated())
        atomic_json(dest,result);completed+=1
        print(json.dumps(dict(component=args.component,frame=i,token=frame['sample_token'],seconds=seconds,done=completed,total=len(pending))),flush=True)
    receipt(args.component,'COMPLETE' if completed==len(pending) else 'PARTIAL',selected=len(selected),processed=completed,remaining=len(pending)-completed,protocol_sha256=phash,seconds_sum=sum(durations))
    if completed<len(pending):raise SystemExit(75)

if __name__=='__main__': main()
