"""Observation-only kernel interface. Global metres, row-index point IDs."""
import copy,json,hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as cc

class LabelNames(dict):
    def __getitem__(self,key):return super().__getitem__(str(key))
    def get(self,key,default=None):return super().get(str(key),default)
    def __contains__(self,key):return super().__contains__(str(key))

@dataclass
class PointSet:
    point_ids:np.ndarray
    xyz:np.ndarray
    intensity:np.ndarray
    semantic_labels:np.ndarray
    semantic_probs:np.ndarray
    def __len__(self):return len(self.point_ids)
    def __getitem__(self,selection):
        return PointSet(*(np.atleast_1d(getattr(self,k)[selection]) if k!='xyz' and k!='semantic_probs' else np.atleast_2d(getattr(self,k)[selection]) for k in self.__dataclass_fields__))
    def select(self,selection):return self[selection]

@dataclass
class Box:
    id:str
    center:np.ndarray
    size_lwh:np.ndarray
    rotation:np.ndarray
    class_name:str
    score:float
    raw:dict
    @property
    def size(self):return self.size_lwh
    @classmethod
    def from_raw(cls,b,bid):
        q=b['rotation'];w,l,h=b['size']
        return cls(bid,np.array(b['translation']),np.array([l,w,h]),Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(),b.get('detection_name',b.get('tracking_name')),float(b.get('detection_score',b.get('tracking_score',0))),copy.deepcopy(b))

@dataclass
class SemanticPatch:
    id:str
    point_ids:np.ndarray
    new_labels:np.ndarray
    region_point_ids:np.ndarray
    source_box:Box

class Scene:
    def __init__(self,folder,episode_id):
        folder=Path(folder);self._meta=json.loads((folder/'observations.json').read_text())
        with np.load(folder/'observations.npz') as a:self._a={k:a[k] for k in a.files}
        self._a={k:np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape) for k,a in self._a.items()}
        self._episode=next(e for e in self._meta['episodes'] if e['episode_id']==episode_id)
        self.task=self._episode['task'];self.candidate_ids=['KEEP']+self._episode['candidate_ids'];self.class_names=LabelNames(self._meta['class_names'])
        self._reference=Box.from_raw(self._episode['reference_box'],'D0' if self.task=='detection' else 'SOURCE')
        self.sensor_origin=np.array(self._meta['lidar2global'])[:3,3]
        self._access_events=[]
    def _note(self,sources,api,**details):
        self._access_events.append(dict(source_ids=list(sources),api=api,retrieval_only=True,**details))
    def evidence_catalog(self):
        return {
            'LIDAR_CURRENT':dict(description='Actual current-scan XYZ and intensity; no completed surfaces or historical points',upstream=['CURRENT_SCAN']),
            'SENSOR_POSE':dict(description='Actual current LiDAR origin and global transform',upstream=['CALIBRATION_POSE']),
            'D0':dict(description='Frozen current LION-Mamba detections, never D1',upstream=['LION_WEIGHTS','CURRENT_SCAN','ALLOWED_HISTORY']),
            'S0':dict(description='Frozen LitePT-S semantic labels/probabilities, not instances or GT',upstream=['LITEPT_WEIGHTS','CURRENT_SCAN']),
            'T0':dict(description='Strict-window MCTrack priors from historical LION measurements; identities unconfirmed',upstream=['LION_WEIGHTS','ALLOWED_HISTORY']),
            'CANDIDATES':dict(description='Existing executable proposals, not independent evidence of correctness',upstream=['D0','S0','T0','CURRENT_SCAN']),
        }
    def candidate_provenance(self,cid):
        a=self.candidate(cid)
        tool=a.get('tool','KEEP')
        dependencies={'SEG_TO_DET':['D0','S0','LIDAR_CURRENT'],
                      'TRACK_TO_DET':['D0','T0','LIDAR_CURRENT'],
                      'DET_TO_SEG':['D0','S0','LIDAR_CURRENT'],
                      'TRACK_TO_SEG':['T0','S0','LIDAR_CURRENT'],
                      'KEEP':['D0' if self.task=='detection' else 'S0']}[tool]
        basis={'SEG_TO_DET':'Semantic confidence/cluster geometry and current-box association; center retained from original proposal',
               'TRACK_TO_DET':'Historical prior and current-LiDAR support/possible recentering, associated to current detection',
               'DET_TO_SEG':'Current detector box, S0 probabilities, geometric neighborhood and original probability boost',
               'TRACK_TO_SEG':'Historical source box, S0 probabilities, geometric neighborhood and original probability boost',
               'KEEP':'Original frozen prediction'}[tool]
        return dict(candidate_id=cid,tool=tool,direct_generator_sources=dependencies,generation_basis=basis,
                    source_track_id=a.get('source_track_id'),source_region_id=a.get('region_uid',a.get('region_id')),
                    exact_generator_point_membership='Not cached; do not claim validation points were excluded from generation',
                    association_status=a.get('association_status','proposal_only'),snapshot='D0/S0/T0')
    def check_evidence_sources(self,candidate_ids,source_ids):
        catalog=self.evidence_catalog();source_ids=list(dict.fromkeys(source_ids))
        unknown=[s for s in source_ids if s not in catalog]
        if unknown:raise ValueError('Unknown evidence sources: '+str(unknown)+'; use scene.evidence_catalog()')
        rows=[]
        for cid in candidate_ids:
            provenance=self.candidate_provenance(cid);deps=provenance['direct_generator_sources']
            rows.append(dict(candidate_id=cid,generator_sources=deps,reused_generator_sources=[s for s in source_ids if s in deps],
                             additional_source_types=[s for s in source_ids if s not in deps and s!='CANDIDATES'],
                             shared_detector_lineage='D0' in source_ids and 'T0' in deps or 'T0' in source_ids and 'D0' in deps,
                             source_check='Dependency description only; no reliability score, independence proof or candidate ranking'))
        return rows
    @property
    def reference_boxes(self):return [self.box('KEEP')]
    @property
    def detection_boxes(self):
        self._note(['D0'],'detection_boxes');return [Box.from_raw(b,'DET:'+str(i)) for i,b in enumerate(self._meta['detections'])]
    def points(self,bounds=None,point_ids=None):
        ids=np.arange(len(self._a['xyz'])) if point_ids is None else np.asarray(point_ids,dtype=np.int64)
        if ids.ndim!=1 or ((ids<0)|(ids>=len(self._a['xyz']))).any():raise ValueError('point_ids must be a 1-D array of current-scan row IDs in [0,N)')
        if bounds is not None:
            if isinstance(bounds,Box):mask=Geom.points_in_box(self._a['xyz'][ids],bounds)
            else:
                lo,hi=np.asarray(bounds);mask=((self._a['xyz'][ids]>=lo)&(self._a['xyz'][ids]<=hi)).all(1)
            ids=ids[mask]
        self._note(['LIDAR_CURRENT','S0'],'points',count=len(ids),point_ids_sha256=hashlib.sha256(ids.astype(np.int64).tobytes()).hexdigest())
        return PointSet(ids,self._a['xyz'][ids],self._a['intensity'][ids],self._a['labels'][ids],self._a['probabilities'][ids])
    def candidate(self,cid):
        if cid=='KEEP':return dict(uid='KEEP',candidate_id='KEEP',operation='KEEP')
        if cid not in self.candidate_ids:raise KeyError('Use a full candidate ID from scene.candidate_ids: '+str(self.candidate_ids))
        a=copy.deepcopy(self._meta['candidates'][cid]);a['native_candidate_id']=a['candidate_id'];a['candidate_id']=cid;return a
    def candidate_table(self):
        result=[]
        for cid in self.candidate_ids:
            a=self.candidate(cid);result.append({k:v for k,v in a.items() if k not in ['box','source_box','array_key']})
        return result
    def box(self,cid='KEEP'):
        if not isinstance(cid,str):raise ValueError('box expects a full string ID from scene.candidate_ids, e.g. KEEP or SEG_TO_DET:4')
        if cid in ['KEEP','D0','SOURCE']:
            source='D0' if self.task=='detection' or self._episode['episode_id'].startswith('DET_TO_SEG') else 'T0'
            self._note([source],'box',candidate_id=cid);return Box.from_raw(self._episode['reference_box'],self._reference.id)
        if cid.startswith('DET:'):
            self._note(['D0'],'box',candidate_id=cid);return Box.from_raw(self._meta['detections'][int(cid.split(':')[1])],cid)
        a=self.candidate(cid)
        self._note(['CANDIDATES'],'box',candidate_id=cid)
        return Box.from_raw(a.get('box',a.get('source_box')),cid)
    def patch(self,cid):
        a=self.candidate(cid)
        if a['operation']!='PATCH_LABELS':raise ValueError('Not a semantic patch')
        self._note(['CANDIDATES'],'patch',candidate_id=cid)
        key=a['array_key'];ids=self._a[key+'_indices'];new=self._a[key+'_labels']
        region=np.unique(np.concatenate([self._a[self.candidate(c)['array_key']+'_indices'] for c in self.candidate_ids[1:]]))
        return SemanticPatch(cid,ids.copy(),new.copy(),region,Box.from_raw(a['source_box'],'SOURCE'))
    def semantic(self,point_ids):
        ids=np.asarray(point_ids,dtype=np.int64);self._note(['S0'],'semantic',count=len(ids));return dict(labels=self._a['labels'][ids].copy(),probabilities=self._a['probabilities'][ids].copy())
    def semantic_indices_for_detection_class(self,name):return [self._meta['detection_class_mapping'][name]]
    def labels_for(self,point_ids,candidate_id='KEEP'):
        self._note(['S0'] if candidate_id=='KEEP' else ['S0','CANDIDATES'],'labels_for',candidate_id=candidate_id)
        ids=np.asarray(point_ids,dtype=np.int64);labels=self._a['labels'][ids].copy()
        if candidate_id!='KEEP':
            patch=self.patch(candidate_id);idx=np.full(len(self._a['labels']),-1,dtype=np.int64);idx[patch.point_ids]=np.arange(len(patch.point_ids));where=idx[ids];mask=where>=0;labels[mask]=patch.new_labels[where[mask]]
        return labels
    def track_history(self):
        self._note(['T0'],'track_history')
        records=copy.deepcopy(self._meta['temporal']['history'])
        for r in records:
            r['association_status']='unconfirmed_temporal_prior';r['age_s']=(self._meta['timestamp_us']-r['last_observation_timestamp'])/1e6
        return records
    def initial_points(self):
        b=self._reference;padding=float(np.clip(.24*max(b.size_lwh[:2]),.6,2.))
        local=Geom.to_box_coordinates(self._a['xyz'],b);ids=np.flatnonzero((np.abs(local)<=b.size_lwh/2+padding).all(1))
        if self.task=='segmentation':
            patchids=np.unique(np.concatenate([self.patch(c).point_ids for c in self.candidate_ids[1:]]));ids=np.union1d(ids,patchids)
        return self.points(point_ids=ids)
    def summary(self):
        p=self.initial_points();b=self._reference
        return dict(task=self.task,episode_id=self._episode['episode_id'],sample_token=self._meta['sample_token'],scene=self._meta['scene'],coordinate_frame='nuScenes global XYZ metres, Z up; point IDs current scan row indices',sensor_origin=self.sensor_origin.tolist(),initial_box=b.raw,local_points=len(p),inside_points=int(Geom.points_in_box(p.xyz,b).sum()),candidate_ids=self.candidate_ids,candidate_table=self.candidate_table(),candidate_provenance=[self.candidate_provenance(c) for c in self.candidate_ids],evidence_catalog=self.evidence_catalog(),history_span_s=self._meta['temporal']['max_input_span_s'],history_sweeps=len(self._meta['temporal']['input_scan_tokens']),tracking_association='unconfirmed unless candidate provenance explicitly names a source track; source identity is still a proposal',class_names=self.class_names,probabilities='original LitePT-S TTA probabilities, not regenerated')

class Geom:
    @staticmethod
    def _xyz(x):return x.xyz if isinstance(x,PointSet) else np.asarray(x)
    @staticmethod
    def to_box_coordinates(points,box):return (Geom._xyz(points)-box.center)@box.rotation
    @staticmethod
    def points_in_box(points,box,padding=0.):return (np.abs(Geom.to_box_coordinates(points,box))<=box.size_lwh/2+padding).all(-1)
    @staticmethod
    def signed_box_distance(points,box):
        q=np.abs(Geom.to_box_coordinates(points,box))-box.size_lwh/2
        return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(axis=-1),0)
    @staticmethod
    def outside_box_distance(points,box):return np.maximum(Geom.signed_box_distance(points,box),0)
    @staticmethod
    def knn(reference,queries=None,k=8):
        r=Geom._xyz(reference);q=r if queries is None else Geom._xyz(queries);return cKDTree(r).query(q,k=min(k,len(r)))
    @staticmethod
    def radius_edges(points,radius):return np.array(sorted(cKDTree(Geom._xyz(points)).query_pairs(radius)),dtype=np.int64).reshape(-1,2)
    @staticmethod
    def connected_components(n_points,edges):
        edges=np.asarray(edges,dtype=np.int64).reshape(-1,2);m=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(n_points,n_points));return cc(m,directed=False)
    @staticmethod
    def transform(points,matrix):
        xyz=Geom._xyz(points);m=np.asarray(matrix);return xyz@m[:3,:3].T+m[:3,3]
