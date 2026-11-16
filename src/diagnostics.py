"""Execute an agent-authored test symmetrically; no built-in metric or selection."""
import hashlib,inspect,traceback
import numpy as np
from common_run import clean

class Diagnostics:
    def __init__(self,scene,record):self.scene=scene;self.record=record;self.runs=[]
    def compare(self,question,test,candidate_ids=None,point_ids=None,evidence_sources=(),assumptions=None):
        ids=list(self.scene.candidate_ids if candidate_ids is None else candidate_ids)
        if not ids or len(set(ids))!=len(ids) or any(c not in self.scene.candidate_ids for c in ids):
            raise ValueError('Use distinct existing executable candidate IDs')
        if not callable(test):raise TypeError('test must be your Python callable: test(candidate_id, points)')
        points=self.scene.initial_points() if point_ids is None else self.scene.points(point_ids=point_ids)
        if len(np.unique(points.point_ids))!=len(points):raise ValueError('Use unique original point IDs')
        sources=list(evidence_sources);source_checks=self.scene.check_evidence_sources(ids,sources)
        try:source=inspect.getsource(test)
        except (OSError,TypeError):source=None
        program_hash=hashlib.sha256((source or repr(test.__code__.co_code)).encode()).hexdigest()
        point_hash=hashlib.sha256(points.point_ids.astype(np.int64).tobytes()).hexdigest()
        rows=[]
        for cid in ids:
            # Fresh views of the identical original support prevent one test's
            # mutations of its PointSet from changing subsequent test inputs.
            p=self.scene.points(point_ids=points.point_ids)
            for name in p.__dataclass_fields__:getattr(p,name).flags.writeable=False
            start=len(self.scene._access_events)
            try:result=clean(test(cid,p));status='OK';error=None
            except Exception:result=None;status='ERROR';error=traceback.format_exc()
            rows.append(dict(candidate_id=cid,status=status,result=result,error=error,
                             program_sha256=program_hash,point_ids_sha256=point_hash,
                             source_reads=self.scene._access_events[start:]))
        record=dict(diagnostic_question=str(question),program_source=source,program_sha256=program_hash,
                    evaluated_candidates=ids,shared_point_ids=points.point_ids.tolist(),point_ids_sha256=point_hash,
                    execution_results=rows,declared_evidence_sources=sources,source_checks=source_checks,
                    assumptions=clean(assumptions),same_callable_and_input_points=True,
                    caveat='Agent function may depend on global state or candidate-specific branches; identical invocation is not a proof of unbiased logic. Review the saved code. No independence claim or ranking is produced.')
        eid=self.record('candidate_symmetric_test',record,point_ids=points.point_ids,evidence_sources=sources,candidate_ids=ids)
        record['evidence_id']=eid;self.runs.append(record)
        return dict(evidence_id=eid,results={row['candidate_id']:row for row in rows},source_checks=source_checks)
