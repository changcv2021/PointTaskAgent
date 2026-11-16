"""Auditable code/feedback/interpretation records, independent of GT or scores."""
import json,re,hashlib
from common_run import save

def sections(text):
    fields={};current=None
    for line in text.split('```',1)[0].splitlines():
        plain=line.strip().lstrip('#').strip().replace('**','')
        match=re.match(r'^(Diagnostic Question|Evaluated Candidates|Evidence Sources|Agent Interpretation|Purpose|Code)\s*:\s*(.*)$',plain,re.I)
        if match:
            current=match.group(1).lower().replace(' ','_');fields[current]=match.group(2)
        elif current:fields[current]+='\n'+line
    return {k:v.strip() for k,v in fields.items()}

def image_ids(messages,out):
    from pathlib import Path
    import base64
    hashes={}
    for p in (Path(out)/'views').glob('V*.json'):
        v=json.loads(p.read_text());hashes[v['sha256']]=v['view_id']
    ids=[]
    for m in messages:
        if not isinstance(m.get('content'),list):continue
        for part in m['content']:
            if part.get('type')=='image_url':
                h=hashlib.sha256(base64.b64decode(part['image_url']['url'].split(',',1)[1])).hexdigest()
                ids.append(hashes.get(h,'unmapped:'+h))
    return list(dict.fromkeys(ids))

def build_record(step,text,code,feedback,shown_before):
    s=sections(text);tests=feedback.get('diagnostics',[])
    invoked=list(dict.fromkeys(c for t in tests for c in t['evaluated_candidates']))
    accessed=list(dict.fromkeys(r['candidate_id'] for r in feedback.get('source_reads',[]) if 'candidate_id' in r))
    return dict(step=step,diagnostic_question=s.get('diagnostic_question',s.get('purpose')),
                executed_test_questions=[t['diagnostic_question'] for t in tests],
                generated_code=code,code_sha256=hashlib.sha256((code or '').encode()).hexdigest(),
                evaluated_candidates=dict(same_program_invocations=invoked,declared=s.get('evaluated_candidates'),metadata_access_only=accessed),
                execution_results={k:v for k,v in feedback.items() if k not in ['views','generated_images','source_reads']},
                evidence_sources=dict(agent_declared=s.get('evidence_sources'),test_declared=[t['declared_evidence_sources'] for t in tests],api_retrievals=feedback.get('source_reads',[]),dependency_checks=[t['source_checks'] for t in tests],retrieval_caveat='API retrieval is not proof that every returned field was used; direct private-array reads are not instrumented.'),
                generated_images=feedback.get('generated_images',[]),images_queued_for_next_model=list(feedback.get('views',{})),
                images_present_in_model_request=shown_before,
                agent_interpretation=dict(before_execution=s.get('agent_interpretation'),after_execution=None,interpreted_in_step=None),
                missing_narrative_fields=[k for k in ['diagnostic_question','evaluated_candidates','evidence_sources','agent_interpretation'] if not s.get(k)],
                final_decision=feedback.get('decision'))

def save_trace(out,records):
    for row in records:save(out/'diagnostics'/('step_'+str(row['step']).zfill(3)+'.json'),row)
    save(out/'diagnostic_trace.json',dict(schema='disagreement_program_verification_v1',steps=records,gt_used=False,
                                         interpretations_are_model_statements=True,missing_fields_not_invented=True))

def feedback_for_model(feedback):
    result={k:v for k,v in feedback.items() if k not in ['views','generated_images','source_reads','diagnostics','events']}
    result['events']=[]
    for event in feedback.get('events',[]):
        if event.get('name')=='candidate_symmetric_test':
            result['events'].append(dict(type='report',name=event['name'],evidence_id=event['evidence_id'],details_in='diagnostics'))
        else:result['events'].append(event)
    result['diagnostics']=[]
    for t in feedback.get('diagnostics',[]):
        result['diagnostics'].append({k:v for k,v in t.items() if k not in ['shared_point_ids','program_source']})
        result['diagnostics'][-1]['shared_point_count']=len(t['shared_point_ids'])
    retrievals={}
    for r in feedback.get('source_reads',[]):
        key=(r['api'],tuple(r['source_ids']))
        retrievals[key]=retrievals.get(key,0)+1
    result['source_retrieval_summary']=[dict(api=k[0],source_ids=k[1],calls=v,retrieval_not_independence_proof=True) for k,v in retrievals.items()]
    result['generated_images']=[dict(view_id=v['view_id'],points=v['points'],highlight_count=len(v.get('highlight_ids',[])),shown=v['view_id'] in feedback.get('views',{})) for v in feedback.get('generated_images',[])]
    return result
