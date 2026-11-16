"""Public GT-free cross-task detection actions, version 4.0.0.

REFINE_CENTER is the default for an associated detection. ADD_BOX is explicit.
Old complete replacements are exposed only as LEGACY_REPLACE_BOX ablations.
This module never reads annotations and makes no Oracle selection decisions.
"""
import copy
import math

VERSION = '4.0.0'
REFINE_CENTER = 'REFINE_CENTER'
ADD_BOX = 'ADD_BOX'
LEGACY_REPLACE_BOX = 'LEGACY_REPLACE_BOX'
PRESERVED_FIELDS = ('size','rotation','velocity','detection_name','detection_score','attribute_name')

def validate_history(history):
    if not history.get('reset_from_empty') or history.get('old_tracker_states_read'):
        raise ValueError('Tracking must be rebuilt from an empty strict-window state')
    if not 0 <= history['max_input_span_s'] <= .5 or len(history['input_scan_tokens']) > 10:
        raise ValueError('Tracking history exceeds 0.5 seconds or 10 historical sweeps')

def make_actions(tool, baseline, candidates, history=None, legacy=False):
    if tool not in ('TRACK_TO_DET','SEG_TO_DET','TEMPORAL_TO_DET'):
        raise ValueError('Unsupported detection tool: '+tool)
    if tool in ('TRACK_TO_DET','TEMPORAL_TO_DET'):
        if history is None:raise ValueError('Strict temporal lineage is required')
        validate_history(history)
    actions=[]
    for c in candidates:
        target=c['target'];proposal=c['box']
        if len(proposal['translation'])!=3 or not all(math.isfinite(x) for x in proposal['translation']):
            raise ValueError('Candidate center must be a finite XYZ position')
        if target is None:operation=ADD_BOX;box=copy.deepcopy(proposal)
        else:
            if not isinstance(target,int) or not 0<=target<len(baseline):raise ValueError('Invalid original detection index')
            original=baseline[target]
            if original['detection_name']!=proposal['detection_name']:raise ValueError('Refinement cannot change class')
            operation=LEGACY_REPLACE_BOX if legacy else REFINE_CENTER
            box=copy.deepcopy(proposal if legacy else original)
            box['translation']=copy.deepcopy(proposal['translation'])
            if not legacy:
                assert all(box[k]==original[k] for k in PRESERVED_FIELDS)
                assert {k:v for k,v in box.items() if k!='translation'}=={k:v for k,v in original.items() if k!='translation'}
        actions.append(dict(candidate_id=c['candidate_id'],uid=tool+':'+str(c['candidate_id']),tool=tool,tool_version=VERSION,operation=operation,target=target,box=box))
    return actions

def TRACK_TO_DET(baseline,candidates,history,legacy=False):
    return make_actions('TRACK_TO_DET',baseline,candidates,history,legacy)

def SEG_TO_DET(baseline,candidates,legacy=False):
    return make_actions('SEG_TO_DET',baseline,candidates,legacy=legacy)

def apply_action(current,action):
    result=list(current);target=action['target']
    if action['operation']==ADD_BOX:
        if target is not None:raise ValueError('ADD_BOX cannot overwrite a detection')
        result.append(copy.deepcopy(action['box']))
    else:
        if target is None:raise ValueError('Refinement requires an existing detection')
        if action['operation']==REFINE_CENTER:
            box=copy.deepcopy(result[target]);box['translation']=copy.deepcopy(action['box']['translation']);result[target]=box
        elif action['operation']==LEGACY_REPLACE_BOX:result[target]=copy.deepcopy(action['box'])
        else:raise ValueError('Unknown operation')
    return result
