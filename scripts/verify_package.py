"""Static package checks. Does not import experiment entry points or call APIs."""
import ast,json,os,re,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
azure_resource_host = re.compile(
    r"\b[a-z0-9][a-z0-9-]*\.(?:services\.ai|openai|cognitiveservices)\.azure\.com\b",
    re.IGNORECASE,
)
# A reviewer can run the checks as-is. A publisher may also supply a bare resource name.
private_resource_name = os.environ.get("AZURE_RESOURCE_NAME", "").strip().casefold()
errors=[];count=0
for p in ROOT.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if p.is_symlink():errors.append(str(rel)+': symlink')
    if p.name in ['api.md','.env'] or p.suffix in ['.pth','.pt','.ckpt','.bin','.npz','.npy','.pyc']:
        errors.append(str(rel)+': excluded payload type')
    if p.suffix=='.py':
        count+=1
        try:ast.parse(p.read_text())
        except SyntaxError as e:errors.append(str(rel)+': '+str(e))
    if p.suffix in ['.py','.json','.txt','.md','.toml','.yaml','.yml','.sh']:
        t=p.read_text()
        endpoint_text = re.sub(r"""['"]\s*\+\s*['"]""", "", t) if p.suffix == ".py" else t
        if azure_resource_host.search(endpoint_text) or (private_resource_name and private_resource_name in endpoint_text.casefold()):
            errors.append(str(rel)+': private API endpoint')
        if re.search(r'\bsk-[A-Za-z0-9_-]{20,}',t):errors.append(str(rel)+': possible API key')
        if rel.parts[0] in ['src','scripts','configs','tools'] and any('/'+'N/'+s+'/' in t for s in ['project','slate']):
            errors.append(str(rel)+': machine-specific path in active code/config')
cfg=json.loads((ROOT/'configs/run.example.json').read_text())
assert cfg['authorized'] is False and 'credential_file' not in cfg
rows=json.loads((ROOT/'configs/selection_100.json').read_text())
assert len(rows)==100 and len({r['frame']['scene_name'] for r in rows})==100
for r in rows:assert not Path(r['prepared_dir']).is_absolute()
report={'status':'PASS' if not errors else 'FAIL','python_files_parsed':count,'errors':errors,
        'examples':len(rows),'paid_calls':0,'scope':'Static checks only; see separate runtime validation receipt.'}
print(json.dumps(report,indent=2))
if errors:sys.exit(1)
