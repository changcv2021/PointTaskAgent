"""Finalize the pilot's four completed code attempts, without redoing them."""
import sys,base64,json,os,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common_run import W,read,save,sha
from agent import Client
from diagnostic_trace import save_trace
assert os.environ.get('SLURM_JOB_ID')
row=read(W/'configs/selection.json')[1];e=row['episode'];token=row['frame']['sample_token'];out=W/'episodes'/token/e['episode_id']
if (out/'decision.json').exists():raise SystemExit(0)
interruption=read(out/'paid_interruption.json')
assert interruption['reason']=='Input bound exceeded; no silent truncation'
feedback=[read(out/f'feedback_{i}.json') for i in range(1,5)]
assert all(f['status']=='OK' for f in feedback) and not (out/'cell_5.py').exists()
messages=read(out/'step_5_request.json');byhash={sha(p):p for p in (out/'views').glob('*.png')}
for m in messages:
    if isinstance(m['content'],list):
        for i,p in enumerate(m['content']):
            if p['type']=='image_bytes':
                data=byhash[p['sha256']].read_bytes();assert len(data)==p['bytes']
                m['content'][i]=dict(type='image_url',image_url={'url':'data:image/png;base64,'+base64.b64encode(data).decode()})
messages.append(dict(role='user',content='The four executed diagnostic attempts and their feedback above are the completed investigation. Stop writing code now. Return only JSON with candidate_id, evidence_ids (list), and summary (string), selecting one of '+json.dumps(['KEEP']+e['candidate_ids'])+'. Base the choice on the actual executed evidence; acknowledge uncertainty. Do not generate another code cell.'))
text,usage=Client('azure',read(W/'configs/run.json')).call(messages,out,'final_decision',300)
d=json.loads(text);assert d['candidate_id'] in ['KEEP']+e['candidate_ids']
assert isinstance(d['summary'],str) and isinstance(d['evidence_ids'],list)
valid={'V1'}
for fb in feedback:
    valid.update(fb.get('views',{}))
    valid.update(ev['evidence_id'] for ev in fb.get('events',[]) if ev.get('type')=='report')
assert set(d['evidence_ids']).issubset(valid)
trace=read(out/'diagnostic_trace.json')['steps']
trace[-1]['agent_interpretation'].update(after_execution=d['summary'],interpreted_in_step='final_json_after_context_guard')
save_trace(out,trace)
save(out/'terminal_decision.json',dict(answer=d,executed_code=False,code_attempts=4,gt_used=False,reason='Early JSON-only finalization after local input-size guard; no code rerun or new code attempt'))
save(out/'decision.json',dict(status='COMPLETE',**d,task=e['task'],episode_id=e['episode_id'],sample_token=token,steps=4,max_code_iterations=5,
    elapsed_s=None,usage=interruption['completed_usage']+[usage],model='gpt-6.1-sol',gt_used=False,diagnostic_trace='diagnostic_trace.json',diagnostic_steps=4,
    model_requested_views=sum(len(fb.get('views',{})) for fb in feedback),finalization='JSON_ONLY_AFTER_4_CODE_ATTEMPTS_CONTEXT_GUARD'))
print('Pilot segmentation finalized from existing four code attempts:',d['candidate_id'],flush=True)
