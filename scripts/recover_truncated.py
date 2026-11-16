"""Recover only pre-execution output truncations; preserve global five-attempt cap."""
import sys,os,copy,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common_run import W,read,save
import agent
from diagnostic_trace import build_record,save_trace
assert os.environ.get('SLURM_JOB_ID')
rows=read(W/'configs/selection.json');recovered=[]
for row in rows:
    e=row['episode'];out=W/'episodes'/row['frame']['sample_token']/e['episode_id'];p=out/'decision.json'
    if not p.exists():continue
    old=read(p)
    if old['status']!='ABNORMAL_FALLBACK' or not old.get('reason','').startswith('MODEL_OUTPUT_TRUNCATED'):continue
    # Do not reset an already executed program or retry a previously attempted continuation.
    if list(out.glob('feedback_*.json')) or old['steps']>1:continue
    dest=out/'continuation_after_truncation'
    if dest.exists():continue
    used_attempts=old['steps'];cfg=copy.deepcopy(read(W/'configs/run.json'))
    cfg.update(max_code_iterations=5-used_attempts,max_output_tokens=16384,planner_max_output_tokens=8192,final_max_output_tokens=8192)
    save(out/'pre_recovery_decision.json',old)
    original_usage=[]
    for f in [out/'planner_response.json']+sorted(out.glob('step_*_response.json')):
        if f.exists():original_usage.append(read(f)['response']['usage'])
    OriginalClient=agent.Client
    plan=read(out/'initial_plan.json')['plan'] if (out/'initial_plan.json').exists() else None
    class RecoveryClient(OriginalClient):
        def call(self,messages,outpath,label,timeout):
            if label=='planner' and plan is not None:
                return plan,{'reused_prior_planner':True,'input_tokens':0,'output_tokens':0}
            return super().call(messages,outpath,label,timeout)
    agent.Client=RecoveryClient
    try:new=agent.episode(row['prepared_dir'],e,dest,'azure',cfg)
    finally:agent.Client=OriginalClient
    assert used_attempts+new['steps']<=5
    logical=[]
    if used_attempts:
        logical.append(build_record(1,'',None,dict(status='FORMAT_ERROR',error='Original first code-generation request exhausted max_output_tokens before any code executed; attempt and cost retained.'),['V1']))
    for s in read(dest/'diagnostic_trace.json')['steps']:
        s=copy.deepcopy(s);s['continuation_step']=s['step'];s['step']+=used_attempts;s['artifact_dir']=str(dest);logical.append(s)
    save_trace(out,logical)
    result=dict(new,steps=used_attempts+new['steps'],max_code_iterations=5,diagnostic_steps=len(logical),
        usage=original_usage+[u for u in new['usage'] if not u.get('reused_prior_planner')],
        continuation_dir=str(dest),recovered_from='PRE_EXECUTION_OUTPUT_TRUNCATION',original_code_attempts=used_attempts,
        continuation_output_token_limit=16384,original_output_token_limit=read(W/'configs/run.json')['max_output_tokens'])
    save(p,result);recovered.append(dict(scene=row['frame']['scene_name'],episode=e['episode_id'],status=result['status'],total_code_attempts=result['steps']))
    print('RECOVERY',recovered[-1],flush=True)
save(W/'reports'/('truncation_recovery_'+os.environ['SLURM_JOB_ID']+'.json'),dict(recovered=recovered,selection_uses_gt=False,budget_reset=False))
