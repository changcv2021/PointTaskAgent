"""GT is loaded ONLY after inference; scores every frozen candidate offline."""
import copy,csv,importlib.util,os,socket
from pathlib import Path
from collections import Counter
import numpy as np
from common_run import W,read,save,sha,PACKAGE_ROOT,require_compute
from paid_api import Ledger
def seg_metrics(cm):
    hit=cm.diagonal();union=cm.sum(0)+cm.sum(1)-hit;count=cm.sum(1)
    iou=np.divide(hit,union,out=np.full(16,np.nan),where=union>0)
    acc=np.divide(hit,count,out=np.full(16,np.nan),where=count>0)
    return dict(mIoU_percent=100*float(np.nanmean(iou)),mAcc_percent=100*float(np.nanmean(acc)),iou_per_class=iou.tolist(),accuracy_per_class=acc.tolist(),confusion_matrix=cm.tolist())
def cm_for(gt,labels):
    mask=gt>=0
    return np.bincount(16*gt[mask]+labels[mask],minlength=256).reshape(16,16)
def comparison(a,b):
    if isinstance(a,(list,tuple)):
        if a[0]!=b[0]:return 1 if a[0]>b[0] else -1
        return 0 if abs(a[1]-b[1])<1e-9 else (1 if a[1]>b[1] else -1)
    return (a>b)-(a<b)
def main():
    require_compute()
    rows=read(W/'configs/selection.json');assert len(rows)==100
    for row in rows:assert (W/'episodes'/row['frame']['sample_token']/row['episode']['episode_id']/'decision.json').exists(),'Do not present unfinished cases as KEEP'
    out=W/'reports/evaluation';out.mkdir(parents=True,exist_ok=True)
    import geometry as geom
    cfg=read(W/'configs/run.json');raw=Path(cfg['dataset_root'])
    truth=read(cfg['offline_gt_path']);mapping=read(PACKAGE_ROOT/'configs/class_mappings.json')
    lut=np.array([mapping['raw32_to_litept16'][str(i)] for i in range(32)])
    from nuscenes.nuscenes import NuScenes
    nusc=NuScenes(version='v1.0-trainval',dataroot=str(raw),verbose=False)
    records=[];pred={'baseline':{},'gpt61sol':{}};cms={k:np.zeros((16,16),np.int64) for k in pred}
    for ix,row in enumerate(rows):
        frame=row['frame'];token=frame['sample_token'];e=row['episode'];folder=Path(row['prepared_dir'])
        epdir=W/'episodes'/token/e['episode_id'];d=read(epdir/'decision.json');m=read(folder/'observations.json')
        assert sha(folder/'observations.json')==row['metadata_sha256'];assert d['steps']<=5 and d['gt_used'] is False
        cid=d['candidate_id'];assert cid in ['KEEP']+e['candidate_ids']
        rec=dict(index=ix,scene=frame['scene_name'],sample_token=token,task=e['task'],episode=e['episode_id'],status=d['status'],selected=cid,code_attempts=d['steps'],model_calls=len(d['usage']),summary=d['summary'],episode_dir=str(epdir),gt_used_only_offline=True)
        candidate_scores={};before=None
        if e['task']=='detection':
            ego=nusc.get('ego_pose',nusc.get('sample_data',frame['lidar_token'])['ego_pose_token'])['translation']
            gt=truth[token]
            def objective(boxes):return geom.detection_objective(gt['boxes'],geom.filter_predictions(boxes,{'ego_translation':ego},gt['bike_racks']))
            base=m['detections'];before=objective(base);candidate_scores['KEEP']=before
            for candidate_id in e['candidate_ids']:
                a=m['candidates'][candidate_id];trial=geom.apply_box_candidate(base,a);candidate_scores[candidate_id]=objective(trial)
            updated=base if cid=='KEEP' else geom.apply_box_candidate(base,m['candidates'][cid])
            pred['baseline'][token]=base;pred['gpt61sol'][token]=updated
            selected=candidate_scores[cid]
            rec.update(baseline_objective=before,selected_objective=selected,objective_definition='Lexicographic: sum over 0.5/1/2/4m of 2*TP-Npred, then negative summed matched XY distance. Not AP/NDS.',
                       delta_primary=selected[0]-before[0],matched_xy_error_reduction_m=selected[1]-before[1])
        else:
            with np.load(folder/'observations.npz') as arr:
                base=arr['labels'].copy();gt=lut[np.fromfile(raw/truth[token]['lidarseg_path'],np.uint8)]
                valid=gt>=0;before=0;candidate_scores['KEEP']=0;effects={};updated=base.copy()
                for candidate_id in e['candidate_ids']:
                    a=m['candidates'][candidate_id];idx=arr[a['array_key']+'_indices'];labels=arr[a['array_key']+'_labels'];v=valid[idx]
                    repaired=int(((base[idx]!=gt[idx])&(labels==gt[idx])&v).sum());harmed=int(((base[idx]==gt[idx])&(labels!=gt[idx])&v).sum())
                    candidate_scores[candidate_id]=repaired-harmed;effects[candidate_id]=dict(repaired_points=repaired,harmed_points=harmed,net_correct_points=repaired-harmed,changed_points=len(idx))
                    if cid==candidate_id:updated[idx]=labels
            effect=effects.get(cid,dict(repaired_points=0,harmed_points=0,net_correct_points=0,changed_points=0))
            bcm=cm_for(gt,base);ucm=cm_for(gt,updated);cms['baseline']+=bcm;cms['gpt61sol']+=ucm
            bm=seg_metrics(bcm);um=seg_metrics(ucm);selected=candidate_scores[cid]
            rec.update(effect,baseline_mIoU_percent=bm['mIoU_percent'],selected_mIoU_percent=um['mIoU_percent'],delta_mIoU_pp=um['mIoU_percent']-bm['mIoU_percent'],
                       baseline_mAcc_percent=bm['mAcc_percent'],selected_mAcc_percent=um['mAcc_percent'],delta_mAcc_pp=um['mAcc_percent']-bm['mAcc_percent'],all_candidate_effects=effects)
        best=max(candidate_scores.values());qualified=[k for k,v in candidate_scores.items() if comparison(v,best)==0] if comparison(best,before)>0 else ['KEEP']
        sign=comparison(selected,before)
        rec.update(effect={-1:'harmful',0:'neutral',1:'beneficial'}[sign] if cid!='KEEP' else ('KEEP' if d['status']=='COMPLETE' else 'ABNORMAL_FALLBACK'),
                   oracle_qualified=d['status']=='COMPLETE' and cid in qualified,qualified_actions=qualified,candidate_objectives=candidate_scores)
        trace=read(epdir/'diagnostic_trace.json') if (epdir/'diagnostic_trace.json').exists() else {'steps':[]}
        rec['code_or_format_errors']=sum(s.get('execution_results',{}).get('status') in ['ERROR','FORMAT_ERROR'] for s in trace['steps'])
        rec['successful_code_steps']=sum(s.get('execution_results',{}).get('status')=='OK' for s in trace['steps'])
        rec['diagnostic_questions']=[s.get('diagnostic_question') for s in trace['steps']]
        rec['model_requested_views']=d['model_requested_views']
        rec['finalization']=d.get('finalization','PYTHON_RETURN_DECISION')
        records.append(rec)
        if (ix+1)%10==0:print('OFFLINE SCORED',ix+1,100,flush=True)
    save(out/'cases.json',records)
    fields=['index','scene','sample_token','task','episode','status','selected','effect','oracle_qualified','code_attempts','successful_code_steps','model_calls','code_or_format_errors','model_requested_views','finalization','delta_primary','matched_xy_error_reduction_m','repaired_points','harmed_points','net_correct_points','delta_mIoU_pp','delta_mAcc_pp','episode_dir']
    with (out/'cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    top={}
    for task in ['detection','segmentation']:
        subset=[r for r in records if r['task']==task and r['effect']=='beneficial']
        key=(lambda r:(r['delta_primary'],r['matched_xy_error_reduction_m'])) if task=='detection' else (lambda r:(r['net_correct_points'],r['delta_mIoU_pp']))
        top[task]=sorted(subset,key=key,reverse=True)
    save(out/'improved_cases.json',top)
    # Official detection metric on exactly the 50 detection-example frames.
    from official_metrics import detection_metric
    metrics={}
    for branch in ['baseline','gpt61sol']:
        det,details=detection_metric(nusc,pred[branch],list(pred[branch]));save(out/(branch+'_detection_details.json'),details)
        metrics[branch]=dict(mAP_percent=100*det['mean_ap'],NDS_percent=100*det['nd_score'],segmentation=seg_metrics(cms[branch]),detection=det)
    counts={task:dict(Counter(r['effect'] for r in records if r['task']==task)) for task in ['detection','segmentation']}
    summary=dict(status='COMPLETE',model='gpt-6.1-sol',reasoning_effort='high',examples=100,scenes=len({r['scene'] for r in records}),counts=counts,
        oracle_qualified=sum(r['oracle_qualified'] for r in records),max_code_attempts=max(r['code_attempts'] for r in records),
        metrics=metrics,metric_coverage='Detection: 50 detection-example frames. Segmentation: 50 segmentation-example frames. Different, disjoint scene sets; subset results, not full nuScenes validation.',
        budget=Ledger().summary(),selection_sha256=sha(W/'configs/selection.json'),evaluator_sha256=sha(Path(__file__)),gt_inference=False,gt_evaluation_only=True)
    save(out/'summary.json',summary)
    print('EVALUATION COMPLETE',counts,Ledger().summary(),flush=True)
if __name__=='__main__':main()
