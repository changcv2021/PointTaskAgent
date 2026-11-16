"""100 examples / 100 scenes; no GT reads or outcome-conditioned selection."""
import os,hashlib,collections
from pathlib import Path
from common_run import W,read,save,sha
def ranked(x):return hashlib.sha256(('20261115:'+str(x)).encode()).hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID')
    cfg=read(W/'configs/run.json');source=Path(cfg['prepared_root'])
    rows=read(cfg['preparation_index'])['rows'];frames={f['sample_token']:f for f in read(cfg['frame_index'])}
    byscene=collections.defaultdict(list)
    for r in rows:byscene[r['scene']].append(r)
    for group in byscene.values():group.sort(key=lambda r:frames[r['sample_token']]['timestamp_us'])
    segscenes=sorted([s for s,g in byscene.items() if any(r['segmentation_episodes']>0 for r in g)],key=ranked)[:50]
    detscenes=sorted([s for s,g in byscene.items() if s not in segscenes and any(r['detection_episodes']>0 for r in g)],key=ranked)[:50]
    assert len(segscenes)==len(detscenes)==50
    selected=[];category_counts=collections.Counter()
    for task,scenes in [('detection',detscenes),('segmentation',segscenes)]:
        for scene in scenes:
            g=byscene[scene];eligible=[(i,r) for i,r in enumerate(g) if r[task+'_episodes']>0]
            _,r=min(eligible,key=lambda ir:(abs(ir[0]-len(g)//2),ranked(ir[1]['sample_token'])))
            token=r['sample_token'];folder=source/token;m=read(folder/'observations.json')
            choices=[e for e in m['episodes'] if e['task']==task]
            if task=='detection':
                visible=[e for e in choices if e['reference_box'].get('detection_score',0)>=0.1]
                if visible:choices=visible
            def order(e):
                cls=e['reference_box']['detection_name'];tools={cid.split(':')[0] for cid in e['candidate_ids']}
                return (-len(tools),category_counts[(task,cls)],ranked(token+e['episode_id']))
            e=min(choices,key=order);category_counts[(task,e['reference_box']['detection_name'])]+=1
            selected.append(dict(frame=frames[token],episode=e,prepared_dir=str(folder),metadata_sha256=sha(folder/'observations.json'),arrays_sha256=read(folder/'ready.json')['arrays_sha256']))
    # Alternate tasks to distribute task mix over each worker shard.
    det=[x for x in selected if x['episode']['task']=='detection'];seg=[x for x in selected if x['episode']['task']=='segmentation']
    selected=[x for pair in zip(det,seg) for x in pair]
    assert len(selected)==100 and len({x['frame']['scene_name'] for x in selected})==100
    save(W/'configs/selection.json',selected)
    save(W/'reports/selection.json',dict(status='PASS',episodes=100,scenes=100,task_counts={'detection':50,'segmentation':50},gt_used=False,
         method='Seeded scene ordering; one eligible midpoint frame/scene; prefer multi-tool detection choices and diversify predicted classes; no GT/outcome filtering.',
         class_counts={str(k):v for k,v in category_counts.items()},selection_sha256=sha(W/'configs/selection.json')))
    print('Selected 100 episodes, 100 distinct scenes, 50 detection + 50 segmentation',flush=True)
if __name__=='__main__':main()
