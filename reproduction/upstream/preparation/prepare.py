"""Trusted GT-free adapter of frozen v4 pools. No candidate regeneration."""
import sys,copy,argparse,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,'${HISTORICAL_WORKSPACE}/ctgei_kimi_v6/src')
from state import Episode
from io_utils import frames,V1,V2,V4,points,prior_box,score,no_gt_io,require_compute
sys.path.insert(0,str(Path(__file__).parent))
from common_run import W,read,save,sha,digest

def prepare(f):
    token=f['sample_token'];out=W/'prepared'/token
    if (out/'ready.json').exists():return read(out/'ready.json')
    with no_gt_io():
        ep=Episode(f);pool_path=V4/'results/pools'/(token+'.json');pool=read(pool_path)
        assert pool['baseline_sha256']==digest(ep.detections[0])
        out.mkdir(parents=True,exist_ok=True)
        scan=points(f);arrays=dict(xyz=ep.xyz,intensity=scan[:,3],labels=ep.labels[0],probabilities=ep.probabilities)
        all_actions={};det_groups={};seg_groups={}
        rawtrack=read(V2/'results/candidates/track_window_to_det'/(token+'.json'))
        track_origin={a['candidate_id']:a for a in rawtrack['candidates']}
        for source,tool in [('track','TRACK_TO_DET'),('seg','SEG_TO_DET')]:
            for original in pool['det'][source]:
                if original['operation']!='REFINE_CENTER':continue
                a=copy.deepcopy(original);cid=a['uid'];target=a['target'];base=ep.detections[0][target]
                assert {k:v for k,v in a['box'].items() if k!='translation'}=={k:v for k,v in base.items() if k!='translation'}
                a['offset_xyz_m']=(np.array(a['box']['translation'])-base['translation']).tolist()
                if source=='track':
                    raw=track_origin[a['candidate_id']];assert raw['target']==target
                    a['source_track_id']=raw['region_id'];a['association_status']='generator_proposal_not_confirmed_identity'
                all_actions[cid]=a;det_groups.setdefault(target,[]).append(cid)
        trackboxes=sorted([prior_box(h,token) for h in ep.temporal['history']],key=lambda b:-score(b))[:64]
        for source,tool in [('det','DET_TO_SEG'),('track','TRACK_TO_SEG')]:
            src=pool['seg'][source];assert sha(src['arrays_path'])==src['arrays_sha256'];groups={}
            with np.load(src['arrays_path']) as data:
                for c in src['candidates']:
                    ci=c['candidate_id'];cid=tool+':'+str(ci);rid=tool+':'+str(c['region_id']);key='patch_'+tool+'_'+str(ci)
                    indices=data['indices_'+str(ci)].copy();labels=data['labels_'+str(ci)].copy()
                    assert len(indices)==len(labels) and len(np.unique(indices))==len(indices)
                    assert ((indices>=0)&(indices<len(scan))).all() and ((labels>=0)&(labels<16)).all()
                    assert (labels!=ep.labels[0][indices]).all()
                    arrays[key+'_indices']=indices;arrays[key+'_labels']=labels
                    source_box=(ep.detections[0] if source=='det' else trackboxes)[c['region_id']]
                    a=dict(c,uid=cid,tool=tool,operation='PATCH_LABELS',array_key=key,region_uid=rid,source_box=source_box)
                    all_actions[cid]=a;groups.setdefault(rid,[]).append(cid)
            seg_groups[source]=list(groups.items())
        episodes=[]
        for target in sorted(det_groups):
            episodes.append(dict(episode_id='D-'+str(target),task='detection',target=target,candidate_ids=det_groups[target],reference_box=ep.detections[0][target]))
        # EXACT v4 joint ordering: alternate original det / track region groups.
        for pair in itertools.zip_longest(seg_groups['det'],seg_groups['track']):
            for item in pair:
                if item is None:continue
                rid,cids=item;episodes.append(dict(episode_id=rid.replace(':','-'),task='segmentation',region_id=rid,candidate_ids=cids,reference_box=all_actions[cids[0]]['source_box']))
        np.savez(out/'observations.npz',**arrays)
        mapping=read(V1/'configs/class_mappings.json')
        obs=dict(sample_token=token,scene=f['scene_name'],timestamp_us=f['timestamp_us'],lidar_token=f['lidar_token'],lidar2global=f['lidar2global'],detections=ep.detections[0],temporal=ep.temporal,candidates=all_actions,episodes=episodes,class_names=mapping['litept16_classes'],detection_class_mapping=mapping['detector_name_to_segmentation_index'])
        save(out/'observations.json',obs)
        record=dict(sample_token=token,scene=f['scene_name'],points=len(scan),detection_episodes=len(det_groups),segmentation_episodes=len(episodes)-len(det_groups),all_candidates=len(all_actions),deterministic_keep_boxes=len(ep.detections[0])-len(det_groups),pool_sha256=sha(pool_path),arrays_sha256=sha(out/'observations.npz'),metadata_sha256=sha(out/'observations.json'),history_span_s=ep.temporal['max_input_span_s'],history_sweeps=len(ep.temporal['input_scan_tokens']),gt_used=False)
        save(out/'ready.json',record);return record

def main():
    require_compute();p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    fs=frames();assert len(fs)==6019 and len({f['scene_name'] for f in fs})==150
    save(W/'configs/frames.json',[{k:f[k] for k in ['sample_token','scene_name','lidar_token','timestamp_us']} for f in fs])
    if a.limit:fs=fs[:a.limit]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=[]
        for row in pool.map(prepare,fs,chunksize=1):
            rows.append(row)
            if len(rows)%100==0:print('prepared',len(rows),len(fs),flush=True)
    save(W/'reports/preparation.json',dict(frames=len(rows),scenes=len({r['scene'] for r in rows}),detection_episodes=sum(r['detection_episodes'] for r in rows),segmentation_episodes=sum(r['segmentation_episodes'] for r in rows),candidates=sum(r['all_candidates'] for r in rows),gt_used=False,rows=rows))
if __name__=='__main__':main()
