import base64,json,re,time,traceback,hashlib
from pathlib import Path
import requests
from functools import lru_cache
from common_run import W,read,save,sha,PACKAGE_ROOT
from kernel import Kernel
from paid_api import PaidClient,PaidStop,output_text
from diagnostic_trace import sections,build_record,save_trace,feedback_for_model,image_ids

@lru_cache(maxsize=1)
def source_manifest():return {str(p.relative_to(PACKAGE_ROOT)):sha(p) for d in ['src','prompts'] for p in (PACKAGE_ROOT/d).glob('*') if p.is_file()}

def image_part(view,out):
    # Never follow arbitrary model-provided paths outside this episode output.
    path=Path(out)/'views'/(view['view_id']+'.png');assert path.is_file() and path.resolve().is_relative_to(Path(out).resolve())
    assert sha(path)==view['sha256'];data=path.read_bytes();assert data.startswith(b'\x89PNG\r\n\x1a\n')
    return dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(data).decode()))

def archive_messages(messages,out,label):
    """Small manifest binds every image-bearing request to actual PNG bytes."""
    rows=[]
    for m in messages:
        content=m['content']
        if isinstance(content,list):
            parts=[]
            for p in content:
                if p['type']=='image_url':
                    data=base64.b64decode(p['image_url']['url'].split(',',1)[1]);parts.append(dict(type='image_bytes',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
                else:parts.append(p)
            rows.append(dict(role=m['role'],content=parts))
        else:rows.append(m)
    save(Path(out)/(label+'_request.json'),rows)

class Client:
    def __init__(self,url,config):
        self.cfg=config;self.api=PaidClient(config)
        assert read(W/'reports/runtime_checks.json')['status']=='PASS'
    def call(self,messages,out,label,timeout):
        archive_messages(messages,out,label)
        start=time.monotonic()
        maximum=self.cfg['planner_max_output_tokens'] if label=='planner' else self.cfg['final_max_output_tokens'] if label=='final_decision' else self.cfg['max_output_tokens']
        raw,key,cached=self.api.ask(messages,max_tokens=maximum,timeout=min(timeout,self.cfg['api_request_timeout_s']),tag=str(Path(out).relative_to(W/'episodes'))+'/'+label)
        save(Path(out)/(label+'_response.json'),dict(elapsed_s=time.monotonic()-start,request_key=key,cached=cached,response=raw))
        if raw.get('status')!='completed':raise RuntimeError('MODEL_OUTPUT_TRUNCATED:'+str(raw.get('incomplete_details')))
        self.last_reasoning=''
        return output_text(raw),raw['usage']

def compact_images(messages):
    """Keep recent real images; older ones stay addressable via load_view/show."""
    count=sum(isinstance(m['content'],list) and any(p['type']=='image_url' for p in m['content']) for m in messages)
    if count<6:return
    seen=0
    for m in reversed(messages):
        if not isinstance(m['content'],list):continue
        if any(p['type']=='image_url' for p in m['content']):
            seen+=1
            if seen>3:m['content']=[p for p in m['content'] if p['type']!='image_url']+[dict(type='text',text='Earlier image bytes archived to conserve context; view IDs remain available through viz.load_view and show.')]

def episode(folder,e,out,url,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'decision.json').exists():return read(out/'decision.json')
    previous=[p for p in out.iterdir() if p.name!='attempts']
    if previous:
        archived=out/'attempts'/str(time.time_ns());archived.mkdir(parents=True)
        for p in previous:p.rename(archived/p.name)
        save(archived/'interruption.json',dict(status='INTERRUPTED_BEFORE_DECISION',included_in_metrics=False))
    save(out/'execution_manifest.json',dict(config=config,source_files=source_manifest(),model_revision=config['model_revision'],gt_used=False))
    start=time.monotonic();kernel=None;steps=0;errors=0;usage=[];result=None;diagnostic_records=[]
    try:
        kernel=Kernel(folder,e['episode_id'],out);initial=kernel.initial;save(out/'initial.json',initial)
        client=Client(url,config);summary=json.dumps(initial['summary'],ensure_ascii=False)
        planner=[dict(role='system',content=(W/'prompts/planner.txt').read_text()),dict(role='user',content=[dict(type='text',text=summary+'\nActual initial point-cloud image:'),image_part(initial['initial_view'],out)])]
        plan,u=client.call(planner,out,'planner',config['episode_timeout_s']);usage.append(u)
        save(out/'initial_plan.json',dict(plan=plan,initial_image=initial['initial_view']['view_id'],image_sha256=initial['initial_view']['sha256'],candidate_provenance=initial['summary']['candidate_provenance'],evidence_catalog=initial['summary']['evidence_catalog'],model_interpretation_not_measurement=True,gt_used=False))
        system=(W/'prompts/agent.txt').read_text()+'\nReturn exactly one fenced ```python cell containing executable Python.\n'+(W/'prompts'/f"{e['task']}.txt").read_text()+'\n'+(W/'docs/kernel_api.txt').read_text()
        messages=[dict(role='system',content=system),dict(role='user',content=[dict(type='text',text=summary+'\nIndependent initial plan:\n'+plan+'\nInitial image '+initial['initial_view']['view_id']+':'),image_part(initial['initial_view'],out)])]
        transcript=[]
        while steps<config['max_code_iterations'] and time.monotonic()-start<config['episode_timeout_s']:
            steps+=1;remaining=config['episode_timeout_s']-(time.monotonic()-start)
            if remaining<5:break
            compact_images(messages)
            messages.append(dict(role='user',content=f'Code attempt {steps}/{config["max_code_iterations"]}. Errors consume an attempt. At the cap you must stop writing diagnostic code; a JSON-only final selection will follow if needed.'))
            observed_images=image_ids(messages,out)
            text,u=client.call(messages,out,'step_'+str(steps),remaining);usage.append(u)
            interpretation=sections(text).get('agent_interpretation')
            if diagnostic_records:
                diagnostic_records[-1]['agent_interpretation'].update(after_execution=interpretation,interpreted_in_step=steps,interpretation_observed_image_ids=observed_images)
                save_trace(out,diagnostic_records)
            assistant=dict(role='assistant',content=text)
            if config['preserve_thinking'] and client.last_reasoning:assistant['reasoning_content']=client.last_reasoning
            messages.append(assistant)
            blocks=re.findall(r'```(?:python|py)\s*\n(.*?)```',text,re.S);code=None;kernel_error=None
            if len(blocks)!=1:
                errors+=1;feedback=dict(status='FORMAT_ERROR',error='Return exactly ONE complete fenced python cell in final content. Your reasoning text is not executed.')
            else:
                code=blocks[0];(out/('cell_'+str(steps)+'.py')).write_text(code,encoding='utf-8')
                diagnostic_records.append(build_record(steps,text,code,dict(status='EXECUTION_STARTED'),observed_images));save_trace(out,diagnostic_records)
                try:feedback=kernel.execute(code,config['cell_timeout_s'])
                except Exception as exc:
                    kernel_error=exc;feedback=dict(status='ERROR',error=traceback.format_exc(),infrastructure_failure=True)
                errors=errors+1 if feedback['status']=='ERROR' else 0
            save(out/('feedback_'+str(steps)+'.json'),feedback);transcript.append(feedback)
            record=build_record(steps,text,code,feedback,observed_images)
            if diagnostic_records and diagnostic_records[-1]['step']==steps:diagnostic_records[-1]=record
            else:diagnostic_records.append(record)
            save_trace(out,diagnostic_records)
            if kernel_error is not None:raise kernel_error
            if feedback.get('decision'):
                cid=feedback['decision']['candidate_id'];assert cid in ['KEEP']+e['candidate_ids']
                result=dict(status='COMPLETE',**feedback['decision']);break
            if errors>=config['max_consecutive_errors']:raise RuntimeError('CONSECUTIVE_EXECUTION_ERRORS')
            feedback_text=json.dumps(feedback_for_model(feedback),ensure_ascii=False)
            if len(feedback_text)>24000:feedback_text=feedback_text[:24000]+'\n[Long stdout/report archived; compute a compact summary in the next cell. All scene candidates remain accessible.]'
            parts=[dict(type='text',text='Actual Python execution feedback:\n'+feedback_text)]
            for view in feedback.get('views',{}).values():
                facts=dict(camera=view['camera'],points=view['points'],boxes=view['boxes'],color_mode=view['color_mode'],highlighted_points=len(view.get('highlight_ids',[])),point_ids_file=view.get('point_ids_file'),tracking_status=view['tracks'])
                parts.extend([dict(type='text',text='Actual renderer metadata for '+view['view_id']+' (authoritative; any agent-written caption is unverified): '+json.dumps(facts)),image_part(view,out)])
            messages.append(dict(role='user',content=parts))
        if result is None and steps==config['max_code_iterations']:
            allowed=['KEEP']+e['candidate_ids']
            messages.append(dict(role='user',content='The five-code-attempt cap has been reached. Do not write or execute any more code. Read the last feedback and images, then return only JSON with candidate_id, evidence_ids (list), and summary (string). Choose one of '+json.dumps(allowed)+'. Explain actual measured evidence and limitations; no GT is available.'))
            text,u=client.call(messages,out,'final_decision',config['api_request_timeout_s']);usage.append(u)
            answer=json.loads(text)
            assert answer.get('candidate_id') in allowed and isinstance(answer.get('summary'),str)
            assert isinstance(answer.get('evidence_ids'),list)
            valid_evidence=set(image_ids(messages,out))|{initial['initial_view']['view_id']}
            for fb in transcript:
                for ev in fb.get('events',[]):
                    if ev.get('type')=='report':valid_evidence.add(ev['evidence_id'])
                valid_evidence.update(fb.get('views',{}))
            assert set(answer['evidence_ids']).issubset(valid_evidence),'Unknown final evidence IDs'
            result=dict(status='COMPLETE',candidate_id=answer['candidate_id'],evidence_ids=answer['evidence_ids'],summary=answer['summary'],finalization='JSON_ONLY_AFTER_CODE_CAP')
            save(out/'terminal_decision.json',dict(answer=answer,executed_code=False,code_attempts=steps,gt_used=False))
            if diagnostic_records:
                diagnostic_records[-1]['agent_interpretation'].update(after_execution=answer['summary'],interpreted_in_step='final_json',interpretation_observed_image_ids=image_ids(messages,out))
        if result is None:raise TimeoutError('EPISODE_RUNTIME_LIMIT')
    except PaidStop as exc:
        save(out/'paid_interruption.json',dict(status='PAID_RUN_INTERRUPTED',reason=str(exc),completed_usage=usage,gt_used=False,not_a_keep_decision=True))
        raise
    except Exception as exc:
        result=dict(status='ABNORMAL_FALLBACK',candidate_id='KEEP',reason=str(exc),traceback=traceback.format_exc(),evidence_ids=[],summary='Runtime failure; not an Agent KEEP decision')
    finally:
        if kernel:kernel.close()
    save_trace(out,diagnostic_records)
    result.update(task=e['task'],episode_id=e['episode_id'],sample_token=Path(folder).name,steps=steps,max_code_iterations=config['max_code_iterations'],elapsed_s=time.monotonic()-start,usage=usage,model=config['model_id'],gt_used=False,diagnostic_trace='diagnostic_trace.json',diagnostic_steps=len(diagnostic_records),model_requested_views=sum(len(f.get('views',{})) for f in locals().get('transcript',[])))
    save(out/'decision.json',result);return result
