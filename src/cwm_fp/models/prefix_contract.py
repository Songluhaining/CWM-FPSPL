"""Future observer-supplied prefix contract. No slicing of post-hoc full graphs."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PrefixCursor:
    test_identity:tuple[str,str,str,str]
    completed_event:int

FORBIDDEN={'test_result','assertion_passed','posterior','whole_trace_statistics','quality','coverage','label','y_true'}

def validate_prefix(cursor:PrefixCursor,records):
    """Validate explicit availability; unknown availability cannot enter input.

    Records must come from a separately certified observer, not be inferred from
    complete-graph membership. Current real packages retain evidence for design,
    but do not yet provide such a producer or state-transition training dataset.
    """
    if type(cursor.completed_event) is not int or cursor.completed_event< -1 or len(cursor.test_identity)!=4:raise ValueError('Invalid prefix cursor')
    def check_fields(x,static=False):
        if isinstance(x,dict):
            if FORBIDDEN.intersection(x):raise ValueError('Future/terminal/posterior metadata in prefix')
            if static and {'executed','observed_successor','branch_taken','value_observed'}.intersection(x):raise ValueError('Post-execution flags on static structure')
            for v in x.values():check_fields(v,static)
        elif isinstance(x,list):
            for v in x:check_fields(v,static)
    for r in records:
        check_fields(r,r.get('evidence_kind')=='static')
        if tuple(r['test_identity'])!=cursor.test_identity:raise ValueError('Cross-test prefix')
        if r['evidence_kind']=='static':
            if 'executed' in r or r.get('value_observed'):raise ValueError('Post-execution state on static structure')
        elif r['evidence_kind']=='observed':
            at=r.get('available_after_event')
            if type(at) is not int or not 0<=at<=cursor.completed_event:raise ValueError('Unavailable/future evidence')
            times=[r[k] for k in ('event_index','use_event','call_return_event') if k in r]+r.get('depends_on_events',[])
            if any(type(i) is not int or not 0<=i<=at for i in times):raise ValueError('Invalid or future evidence dependency')
            if 'value' in r and type(r.get('observed')) is not bool:raise ValueError('Value requires explicit observed/missing mask')
            if r.get('observed') is False and r.get('value') is not None:raise ValueError('Invented missing state')
            if r.get('observed') is True and r.get('value') is None:raise ValueError('Observed state without actual value')
        else:raise ValueError('Inferred/unknown state is not an observation')
    return True
