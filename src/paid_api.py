"""Azure Responses API. All workers share a durable conservative USD ledger."""
import os,base64,contextlib,copy,datetime,fcntl,json,re,struct,time,urllib.request,urllib.error
from pathlib import Path
from common_run import W,read,save,digest

class PaidStop(RuntimeError): pass
class BudgetStop(PaidStop): pass
class APIStop(PaidStop): pass
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

class Ledger:
    def __init__(self,root=W,config=None):
        self.root=Path(root);self.cfg=config or read(self.root/'configs/run.json')
        assert self.cfg['authorized'] and 0<self.cfg['budget_usd']<=200
        assert self.cfg['budget_input_nano_usd_per_token']>=6000
        assert self.cfg['budget_output_nano_usd_per_token']>=20000
        self.path=self.root/'reports/budget.json';self.limit=int(self.cfg['budget_usd']*1e9)
    def state(self):
        s=read(self.path) if self.path.exists() else dict(limit_nano_usd=self.limit,requests={})
        assert s['limit_nano_usd']==self.limit
        return s
    def ensure_active(self):
        for name in ['STOP_REQUESTED','BUDGET_STOP','API_STOP']:
            if (self.root/name).exists():raise PaidStop(name+' is active')
    def summary(self):
        with lock(self.path.with_suffix('.lock')):
            s=self.state();used=sum(v['amount_nano_usd'] for v in s['requests'].values())
            settled=sum(v['amount_nano_usd'] for v in s['requests'].values() if v['status']=='ACCOUNTED')
            return dict(limit_usd=self.limit/1e9,accounted_conservative_usd=settled/1e9,reserved_or_unresolved_usd=(used-settled)/1e9,total_committed_usd=used/1e9,requests=len(s['requests']))
    def reserve(self,key,inp,out,metadata):
        amount=inp*self.cfg['budget_input_nano_usd_per_token']+out*self.cfg['budget_output_nano_usd_per_token']
        with lock(self.path.with_suffix('.lock')):
            self.ensure_active();s=self.state()
            if key in s['requests']:raise APIStop('Unresolved earlier request; no duplicate charge')
            used=sum(v['amount_nano_usd'] for v in s['requests'].values())
            if used+amount>self.limit:raise BudgetStop('Next reservation would exceed shared 200 USD ceiling')
            s['requests'][key]=dict(status='RESERVED',amount_nano_usd=amount,input_upper=inp,output_upper=out,time_utc=now(),metadata=metadata)
            save(self.path,s)
    def settle(self,key,usage,metadata):
        assert isinstance(usage,dict)
        inp=usage.get('input_tokens');out=usage.get('output_tokens')
        assert type(inp) is int and type(out) is int
        with lock(self.path.with_suffix('.lock')):
            s=self.state();v=s['requests'][key]
            assert 0<=inp<=v['input_upper'] and 0<=out<=v['output_upper']
            v.update(status='ACCOUNTED',amount_nano_usd=inp*self.cfg['budget_input_nano_usd_per_token']+out*self.cfg['budget_output_nano_usd_per_token'],usage=usage,response_metadata=metadata,settled_utc=now())
            assert sum(x['amount_nano_usd'] for x in s['requests'].values())<=self.limit
            save(self.path,s)
    def error(self,key,detail):
        with lock(self.path.with_suffix('.lock')):
            s=self.state();v=s['requests'][key]
            if v['status']!='ACCOUNTED':v['status']='UNRESOLVED_ERROR_RESERVED'
            v['error']=detail;save(self.path,s)

def credential(env_name='PTA_API_KEY'):
    value=os.environ.get(env_name,'')
    if not value:raise APIStop('Set the API key environment variable '+env_name)
    return value

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a,**k):return None

def payload_for(messages,cfg,maximum):
    items=[]
    for m in messages:
        content=m['content']
        if isinstance(content,list):
            content=[dict(type='input_image',image_url=p['image_url']['url'],detail='high') if p['type']=='image_url' else dict(type='input_text',text=p['text']) for p in content]
        items.append(dict(role=m['role'],content=content))
    return dict(model=cfg['model_id'],input=items,reasoning={'effort':'high'},max_output_tokens=maximum,store=False,service_tier='default')

def input_bound(payload,cfg):
    small=copy.deepcopy(payload);images=0
    for m in small['input']:
        if not isinstance(m['content'],list):continue
        for p in m['content']:
            if p['type']!='input_image':continue
            assert p['detail']=='high' and p['image_url'].startswith('data:image/png;base64,')
            raw=base64.b64decode(p['image_url'].split(',',1)[1],validate=True)
            assert raw[:8]==b'\x89PNG\r\n\x1a\n'
            w,h=struct.unpack('>II',raw[16:24]);assert 0<w<=4096 and 0<h<=4096
            p['image_url']='PNG';images+=1
    # Text UTF-8 bytes bound byte-BPE tokens; 8192/image exceeds documented high-detail cap.
    n=len(json.dumps(small,ensure_ascii=False,separators=(',',':')).encode())+8192*(images+1)
    if n>cfg['max_request_input_bound']:raise APIStop('Input bound exceeded; no silent truncation')
    return n,images

def output_text(raw):
    return ''.join(c.get('text','') for o in raw.get('output',[]) if o.get('type')=='message' for c in o.get('content',[]) if c.get('type')=='output_text')

class PaidClient:
    def __init__(self,config=None):
        self.cfg=config or read(W/'configs/run.json');self.ledger=Ledger(config=self.cfg)
        assert self.cfg['model_id']=='gpt-6.1-sol' and self.cfg['reasoning_effort']=='high'
        assert self.cfg['api_base_url'].startswith('https://'),'Set PTA_API_BASE_URL to the Responses API base URL'
        self._key=credential(self.cfg.get('api_key_env','PTA_API_KEY'));self.opener=urllib.request.build_opener(NoRedirect())
    def ask(self,messages,max_tokens=None,timeout=None,tag='experiment'):
        maximum=max_tokens or self.cfg['max_output_tokens'];payload=payload_for(messages,self.cfg,maximum)
        if tag.endswith('/final_decision'):
            payload['text']={'format':{'type':'json_object'}}
            payload['input'][0]['content']+='\nFINALIZATION MODE: diagnostic code execution has ended. This final response MUST be only a JSON object with candidate_id, evidence_ids and summary. This overrides the earlier Python-cell response format. No more code.'
        fingerprint=digest(payload);key=digest(dict(payload_sha256=fingerprint,tag=tag))
        path=W/'results/api'/(key+'.json')
        with lock(W/'results/api_locks'/(key+'.lock')):
            if path.exists():return read(path),key,True
            inp,images=input_bound(payload,self.cfg)
            self.ledger.reserve(key,inp,maximum,dict(tag=tag,model=payload['model'],reasoning_effort='high',images=images,payload_sha256=fingerprint))
            data=json.dumps(payload,separators=(',',':'),ensure_ascii=False).encode()
            req=urllib.request.Request(self.cfg['api_base_url']+'/responses',data,{'Content-Type':'application/json','Authorization':'Bearer '+self._key})
            try:
                with self.opener.open(req,timeout=timeout or self.cfg['api_request_timeout_s']) as r:
                    body=r.read().decode().replace(self._key,'[REDACTED]')
                raw=json.loads(body)
                rawpath=W/'results/api_raw'/(key+'.json');save(rawpath,raw)
                self.ledger.settle(key,raw.get('usage'),dict(model=raw.get('model'),id=raw.get('id'),status=raw.get('status')))
                assert raw.get('model') in ['gpt-6.1-sol','gpt-6.1-sol-2026-09-29'],'Unexpected returned model'
                assert raw.get('service_tier','default') in ['default',None],'Unexpected billing tier'
                assert raw.get('reasoning',{}).get('effort')=='high','HIGH not confirmed by response'
                save(path,raw)
            except Exception as e:
                detail=dict(type=type(e).__name__,message=str(e).replace(self._key,'[REDACTED]')[:1000])
                if isinstance(e,urllib.error.HTTPError):detail.update(http_status=e.code,body=e.read(2000).decode(errors='replace').replace(self._key,'[REDACTED]'))
                self.ledger.error(key,detail);save(W/'API_STOP',dict(time_utc=now(),request_key=key,error=detail))
                raise APIStop('API failure; reservation retained; inspect API_STOP before retry') from None
        return raw,key,False
