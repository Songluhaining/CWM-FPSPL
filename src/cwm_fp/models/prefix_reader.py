"""Consume a frozen prefix index without opening any transition/product target.

Integrity hashes can cover complete shared files offline. They are never model
features; replay consumes observation records only through the requested clock.
"""
from pathlib import Path
import json
from cwm_fp.data.common import sha256
from cwm_fp.lddg.static import load_static
from .prefix_observer import ObservationReplay,PrefixBuilder

def load_prefix(root:Path,trace_index:int,event_index:int,*,expected_manifest_sha256:str):
    root=Path(root)
    manifest=root/'prefix_manifest.json'
    if sha256(manifest)!=expected_manifest_sha256:raise ValueError('Prefix manifest anchor mismatch')
    m=json.loads(manifest.read_text(encoding='utf8'))
    if m['schema']!='physical_prefix_dataset_v1':raise ValueError('Unknown prefix dataset')
    trace=next((r for r in m['traces'] if r['trace_index']==trace_index),None)
    if trace is None or trace['partition'] not in ('train','val'):raise ValueError('Unknown/disallowed source trace')
    def checked(name):
        ref=trace['input_files'][name];path=(root/ref['path']).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path)!=ref['sha256']:raise ValueError('Corrupt/escaping prefix input')
        return path
    index=checked('prefix_index')
    with index.open(encoding='utf8') as f:
        item=next((r for line in f if (r:=json.loads(line))['event_index']==event_index),None)
    if item is None:raise ValueError('No accepted prefix at requested event')
    static=load_static(checked('static_bytecode'));bundle=json.loads(checked('static').read_text(encoding='utf8'))
    framework=root/m['framework_static']['path']
    if sha256(framework)!=m['framework_static']['sha256']:raise ValueError('Framework static mismatch')
    observer=ObservationReplay(trace['identity'][3])
    # Shared stream begins immediately after the real root method marker, which
    # was independently verified against the original trace in phase05A.
    observer.active=True;observer.entry=True
    with checked('observations').open(encoding='utf8') as f:
        for line in f:
            r=json.loads(line)
            if r['clock']>item['clock']:break
            observer.push(r)
    result=PrefixBuilder(static,bundle,load_static(framework)).build(observer)
    if result.fingerprint()!=item['input_sha256']:raise ValueError('Reproduced model input differs')
    return result
