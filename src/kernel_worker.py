"""Persistent execution inside a filesystem/network/PID-isolated bwrap process."""
import sys,os,json,io,traceback,contextlib,signal,time,ast,faulthandler,linecache
faulthandler.dump_traceback_later(120,repeat=True)
from pathlib import Path
import numpy as np
import scipy
from scene_api import Scene,Geom
from visualization import Viz
from common_run import clean
from diagnostics import Diagnostics

scene=Scene('/input',sys.argv[1]);viz=Viz(scene,'/output/views');reports={};events=[];decision=None
def report(name,values,point_ids=None,evidence_sources=(),candidate_ids=()):
    checks=scene.check_evidence_sources(list(candidate_ids),list(evidence_sources))
    eid='E'+str(len(reports)+1);record=dict(evidence_id=eid,name=name,values=clean(values),declared_evidence_sources=list(evidence_sources),evaluated_candidates=list(candidate_ids),source_checks=checks)
    if point_ids is not None:
        pts=scene.points(point_ids=point_ids);record['point_ids']=pts.point_ids.tolist()
    reports[eid]=record;events.append(dict(type='report',**record));return eid
def show(view,caption=''):
    if isinstance(view,str):view=viz.load_view(view)
    if view['view_id'] not in viz.views:raise ValueError('show requires an actually rendered view')
    events.append(dict(type='image',view_id=view['view_id'],caption=caption));return view['view_id']
class Finished(BaseException):pass
def ReturnDecision(candidate_id,evidence_ids,summary):
    global decision
    if any(e['type']=='image' for e in events):raise ValueError('Inspect the newly requested images in the next observation before submitting a decision')
    new_measurements={e['evidence_id'] for e in events if e['type']=='report'}
    if new_measurements.intersection(evidence_ids) or any(e.get('name')=='candidate_symmetric_test' for e in events):
        raise ValueError('Interpret newly executed measurements in the next observation before citing them in a final decision')
    if candidate_id not in scene.candidate_ids:raise ValueError('Choose one existing candidate ID or KEEP')
    if any(e not in reports and e not in viz.views for e in evidence_ids):raise ValueError('Unknown evidence ID')
    decision=dict(candidate_id=candidate_id,evidence_ids=list(evidence_ids),summary=str(summary));raise Finished()
def timeout(signum,frame):raise TimeoutError('Python cell exceeded configured runtime')
signal.signal(signal.SIGALRM,timeout)
diagnose=Diagnostics(scene,report)
scope=dict(np=np,scipy=scipy,scene=scene,geom=Geom,viz=viz,diagnose=diagnose,report=report,show=show,ReturnDecision=ReturnDecision)
def emit(data):sys.__stdout__.write(json.dumps(clean(data),allow_nan=False)+'\n');sys.__stdout__.flush()
try:
    points=scene.initial_points();camera=viz.preset_camera(points=points)
    view=viz.render(points,camera,box_ids=['SOURCE' if scene.task=='segmentation' else 'KEEP'],color_mode='semantic' if scene.task=='segmentation' else 'intensity',title='Initial '+scene.task+' observation')
    emit(dict(status='READY',summary=scene.summary(),initial_view=view))
    faulthandler.cancel_dump_traceback_later()
except BaseException:emit(dict(status='INIT_ERROR',error=traceback.format_exc()));raise SystemExit(2)
cell_number=0
for line in sys.stdin:
    request=json.loads(line);code=request['code'];events.clear();decision=None;buf=io.StringIO();status='OK';error=None;started=time.monotonic()
    cell_number+=1;scene._access_events.clear();view_ids_before=set(viz.views);diagnostic_start=len(diagnose.runs)
    try:
        tree=ast.parse(code,mode='exec');signal.alarm(int(request.get('timeout_s',90)))
        filename='/output/cell_'+str(cell_number)+'.py'
        linecache.cache[filename]=(len(code),None,code.splitlines(True),filename)
        with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):exec(compile(tree,filename,'exec'),scope,scope)
    except Finished:status='DECISION'
    except BaseException:status='ERROR';error=traceback.format_exc()
    finally:signal.alarm(0)
    variables={k:dict(type=type(v).__name__,shape=list(v.shape) if hasattr(v,'shape') else None) for k,v in scope.items() if not k.startswith('_') and k not in ['np','scipy','scene','geom','viz','diagnose','report','show','ReturnDecision']}
    emit(dict(status=status,stdout=buf.getvalue(),error=error,events=events,views={e['view_id']:viz.views[e['view_id']] for e in events if e['type']=='image'},generated_images=[v for k,v in viz.views.items() if k not in view_ids_before],source_reads=scene._access_events,diagnostics=diagnose.runs[diagnostic_start:],variables=variables,decision=decision,elapsed_s=time.monotonic()-started))
    if decision:break
