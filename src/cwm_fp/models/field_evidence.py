"""Evidence classification for field reads; never alters strict admission.

An object hash is deliberately not a reference identity or a typed null tag.
This module does not implement a new heap interpreter or repair write lineage.
"""
from __future__ import annotations
import re

INTEGER_RANGES={'I':(-(1<<31),(1<<31)-1),'B':(-128,127),'S':(-32768,32767)}

def read_value(raw, descriptor):
    """Decode only the frozen producer's exact primitive callback format."""
    unknown={'valid':False,'value':None,'dtype':None,'raw_callback':raw,'reason':None}
    if raw is None:return {**unknown,'reason':'no_value_callback'}
    if ':' not in raw:return {**unknown,'reason':'malformed_callback'}
    kind,text=raw.split(':',1)
    if descriptor.startswith(('L','[')):
        return {**unknown,'reason':'object_hash_not_reference_or_typed_null' if kind=='Obj' else 'reference_callback_not_supported'}
    if kind!=descriptor:return {**unknown,'reason':'callback_descriptor_mismatch'}
    if descriptor=='Z':
        if text not in ('true','false'):return {**unknown,'reason':'invalid_boolean_callback'}
        return {'valid':True,'dtype':'I','value':int(text=='true'),'raw_callback':raw,'reason':None}
    if descriptor not in INTEGER_RANGES:return {**unknown,'reason':'unsupported_primitive_type'}
    if not re.fullmatch(r'-?\d+',text):return {**unknown,'reason':'malformed_integer_callback'}
    value=int(text);lo,hi=INTEGER_RANGES[descriptor]
    if not lo<=value<=hi:return {**unknown,'reason':'integer_out_of_range'}
    return {'valid':True,'dtype':'I','value':value,'raw_callback':raw,'reason':None}

def classify_read(*,opcode,descriptor,raw_callback,pre_clock,callback_clock,completed_clock,
                  identity_valid,frame_stack_valid,receiver_valid,association_valid,
                  earlier_gaps,write_provenance_complete=False):
    """A-E classification with separate observation and provenance dimensions.

    `frame_stack_valid` is established from the pre-consumption interpreter;
    `association_valid` from the physical record and pinned callback emitter.
    Callers must not use a final test assertion as an association witness.
    """
    obs=read_value(raw_callback,descriptor)
    out={'category':None,'observed_value_valid':False,'upstream_write_provenance_complete':bool(write_provenance_complete),
         'current_heap_complete':False,'local_transition_valid':False,'full_product_admission':False,
         'callback_present':raw_callback is not None,'read_completed':completed_clock is not None,
         'value':None,'dtype':None,'available_at':None,'reason':None,'decoded_observation':obs}
    if opcode not in (178,180):return {**out,'category':'E','reason':'not_a_field_read'}
    if earlier_gaps:return {**out,'category':'D','reason':'earlier_unverified_semantics'}
    if not identity_valid or not frame_stack_valid or not receiver_valid or not association_valid:
        return {**out,'category':'C','reason':'instruction_frame_receiver_or_callback_association_unverified'}
    if completed_clock is None or completed_clock<=pre_clock:
        return {**out,'category':'C','reason':'read_completion_not_certified'}
    if raw_callback is not None and (callback_clock is None or callback_clock<=pre_clock or callback_clock!=completed_clock):
        return {**out,'category':'C','reason':'callback_time_not_certified'}
    if obs['reason']=='callback_descriptor_mismatch':return {**out,'category':'C','reason':obs['reason']}
    if not obs['valid']:return {**out,'category':'B','reason':obs['reason']}
    if write_provenance_complete:
        return {**out,'category':'E','observed_value_valid':True,'value':obs['value'],'dtype':obs['dtype'],
                'available_at':callback_clock,'reason':'not_a_missing_write_provenance_case'}
    return {**out,'category':'A','observed_value_valid':True,'value':obs['value'],'dtype':obs['dtype'],
            'available_at':callback_clock,'reason':'real_read_value_only_upstream_write_unknown'}

def visible_read(evidence,clock):
    """Read-occurrence observation only, never a persistent heap-slot value.

    This is an evidence visibility guard, not a local transition producer.
    A verified downstream stack/frame chain is still required independently.
    """
    known=evidence['category']=='A' and evidence['observed_value_valid'] and evidence['available_at']<clock
    return {'value':evidence['value'] if known else None,'observed':known,
            'upstream_write_provenance_complete':False,'current_heap_complete':False}
