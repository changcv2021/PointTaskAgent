"""Immutable source versions and atomic synchronous D/S round commits."""
import copy
import numpy as np
from io_utils import *

class Episode:
    def __init__(self,frame):
        self.frame=frame;self.token=frame['sample_token'];self.round=1
        self.detections=[baseline_detection(frame)]
        with np.load(V1/'results/baseline/litept'/(self.token+'.npz')) as a:
            self.probabilities=a['probabilities'];self.labels=[a['labels']]
        self.probabilities.flags.writeable=False;self.labels[0].flags.writeable=False
        self.xyz=global_points(points(frame)[:,:3],frame)
        self.temporal=read(V2/'results/window_tracking'/(self.token+'.json'))
        t=self.temporal
        assert t['reset_from_empty'] and not t['old_tracker_states_read']
        assert 0<=t['max_input_span_s']<=.5 and len(t['input_scan_tokens'])<=10
        assert t['input_scan_tokens']==frame['history_tokens']
        for h in t['history']:
            assert 0<frame['timestamp_us']-h['last_observation_timestamp']<=500000
            assert 0<frame['timestamp_us']-h['first_observation_timestamp']<=500000
        self.initial_noncenter=[{k:v for k,v in b.items() if k!='translation'} for b in self.detections[0]]
        self.temporal_sha=digest(t);self.probability_sha=array_sha(self.probabilities)
        self.commits=[];self.applied_add_signatures=set()

    def snapshot(self):
        return dict(sample_token=self.token,round=self.round,
                    D=[digest(d) for d in self.detections],S=[array_sha(s) for s in self.labels],T0=self.temporal_sha,
                    original_probabilities_sha256=self.probability_sha)

    def source(self,task,version):
        if not isinstance(version,int) or not 0<=version<self.round:raise ValueError('Source must be a committed version 0..r-1')
        return self.detections[version] if task=='detection' else self.labels[version]

    def commit(self,detections,labels,source_snapshot,actions):
        assert source_snapshot==self.snapshot(),'Round-start sources changed during decision execution'
        assert self.round<=3 and labels.shape==self.labels[0].shape
        for i,original in enumerate(self.initial_noncenter):
            assert {k:v for k,v in detections[i].items() if k!='translation'}==original
        assert digest(self.temporal)==self.temporal_sha and array_sha(self.probabilities)==self.probability_sha
        assert len(detections)<=500 and ((labels>=0)&(labels<16)).all()
        changed=digest(detections)!=digest(self.detections[-1]) or not np.array_equal(labels,self.labels[-1])
        self.detections.append(copy.deepcopy(detections));labels=labels.copy();labels.flags.writeable=False;self.labels.append(labels)
        self.commits.append(dict(round=self.round,source_snapshot=source_snapshot,actions=actions,
                                 detection_sha256=digest(detections),labels_sha256=array_sha(labels),changed=changed,formal_commits=1))
        self.round+=1
        return changed

    def carry_stopped(self):
        """Early stopping retains the final state in every subsequent round metric."""
        while self.round<=3:self.commit(self.detections[-1],self.labels[-1],self.snapshot(),[])

def apply_detection(current,action):
    result=copy.deepcopy(current)
    if action['operation']=='REFINE_CENTER':
        target=action['target'];assert target is not None and 0<=target<len(result)
        result[target]['translation']=list(action['box']['translation'])
    elif action['operation']=='ADD_BOX':
        assert action['target'] is None and len(result)<500
        result.append(copy.deepcopy(action['box']))
    else:raise ValueError('Only explicit REFINE_CENTER or ADD_BOX is allowed')
    return result

def apply_segmentation(current,action,claimed):
    indices=np.asarray(action['indices'],np.int64);labels=np.asarray(action['labels'],np.uint8)
    assert len(indices)==len(labels)==len(set(indices.tolist()))
    assert ((indices>=0)&(indices<len(current))).all() and (labels<16).all()
    allowed=~claimed[indices] # Determined only by prior accepted actions, never GT.
    result=current.copy();result[indices[allowed]]=labels[allowed]
    new_claimed=claimed.copy();new_claimed[indices[allowed]]=True
    return result,new_claimed
