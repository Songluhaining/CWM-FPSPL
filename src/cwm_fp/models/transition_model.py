"""Finite observational next-event/branch predictors; no JVM/state simulation."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torch import nn
from .graph_encoder import GraphEncoder, mean_pool

KINDS=('seq_gru','no_dep','rgcn')
DEPENDENCIES=(1,2,3,4)  # control, data, produces, consumes; no indirect bypass

@dataclass(frozen=True)
class StructureInput:
    x:np.ndarray
    edge_index:np.ndarray
    edge_type:np.ndarray
    anchor:int

    @property
    def event_nodes(self):return np.flatnonzero(self.x[:,19]).astype(np.int64)
    @property
    def static_count(self):return int(self.event_nodes[0])

def from_snapshot(snapshot):
    m=snapshot.model
    anchors=np.flatnonzero(m['current_event'][:,0])
    if len(anchors)!=1:raise ValueError('Ambiguous observed query anchor')
    x=m['x'].astype(np.uint8,copy=True);ei=m['edge_index'];et=m['edge_type']
    if x.shape[1]!=22 or not np.isin(m['x'],[0,1]).all() or x[:,20:].any():raise ValueError('Invalid structural feature whitelist')
    order=np.lexsort((ei[1],ei[0],et))
    inp=StructureInput(x,ei[:,order].astype(np.int32),et[order].astype(np.uint8),int(anchors[0]))
    validate_input(inp);return inp

def validate_input(inp):
    if type(inp) is not StructureInput:raise TypeError('Only whitelisted structural input, never a target dictionary')
    if inp.x.ndim!=2 or inp.x.shape[1]!=22 or not np.isin(inp.x,[0,1]).all():raise ValueError('Invalid node matrix')
    if not len(inp.event_nodes) or inp.anchor!=inp.event_nodes[-1]:raise ValueError('Anchor is not latest observed event')
    if inp.edge_index.shape!=(2,len(inp.edge_type)):raise ValueError('Invalid edge arrays')
    if inp.edge_index.size and (inp.edge_index.min()<0 or inp.edge_index.max()>=len(inp.x)):raise ValueError('Bad edge endpoint')
    if set(inp.edge_type.tolist())-set((1,2,3,4,6)):raise ValueError('Unexpected prefix relation')
    sn=inp.static_count
    if inp.x[:sn,17].any() or inp.x[:,20:].any():raise ValueError('Future static/config flags')
    cross=(inp.edge_index[0]<sn)!=(inp.edge_index[1]<sn)
    if cross.any():raise ValueError('Static sharing requires disconnected observed static component')
    return True

def view_edges(inp,kind):
    mask=~np.isin(inp.edge_type,DEPENDENCIES) if kind=='no_dep' else np.ones(len(inp.edge_type),bool)
    return inp.edge_index[:,mask],inp.edge_type[mask]

def signature(inp,kind):
    """Actual input bytes, ordered local nodes/canonical edge order, no source IDs.

    This is exact input equivalence, not a general graph-isomorphism algorithm.
    """
    h=hashlib.sha256()
    arrays=[inp.x[inp.event_nodes]] if kind=='seq_gru' else [inp.x,*view_edges(inp,kind),np.asarray([inp.anchor],np.int64)]
    for a in arrays:h.update(str((a.shape,a.dtype)).encode());h.update(a.tobytes())
    return h.hexdigest()

def tensors(inp,kind,device):
    ei,et=view_edges(inp,kind)
    return torch.as_tensor(inp.x,dtype=torch.float32,device=device),torch.as_tensor(ei,dtype=torch.long,device=device),torch.as_tensor(et,dtype=torch.long,device=device)

class TransitionModel(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in KINDS:raise ValueError('Unknown transition model')
        self.kind=kind
        if kind=='seq_gru':
            self.gru=nn.GRU(22,32,num_layers=1,batch_first=True,bidirectional=False);width=32
        else:
            self.encoder=GraphEncoder('rgcn',hidden=32,layers=2);width=99
        self.context=nn.Sequential(nn.Linear(width,32),nn.ReLU())
        self.head_a=nn.Linear(32,9);self.head_b=nn.Linear(32,1)

    def forward(self,inputs,*,share_static=True):
        if not inputs or any(type(i) is not StructureInput for i in inputs):raise TypeError('Only nonempty structural input batch')
        device=next(self.parameters()).device
        if self.kind=='seq_gru':
            seq=[torch.as_tensor(i.x[i.event_nodes],dtype=torch.float32,device=device) for i in inputs]
            packed=nn.utils.rnn.pack_sequence(seq,enforce_sorted=False)
            _,h=self.gru(packed);context=h[-1]
        else:context=self.graph_context(inputs,device,share_static)
        h=self.context(context)
        return self.head_a(h),self.head_b(h).squeeze(-1)

    def graph_context(self,inputs,device,share_static):
        # Identical disconnected static components may share computation, never
        # dynamic edges. Repeated context indexing preserves summed gradients.
        xs=[];edges=[];relations=[];anchor=[];dyn_nodes=[];dyn_owner=[];stat_nodes=[];stat_owner=[]
        stat_query=[];static_seen={};offset=0
        for n,inp in enumerate(inputs):
            x,ei,et=tensors(inp,self.kind,device);sn=inp.static_count
            sm=ei[0]<sn;sei=ei[:,sm];setype=et[sm]
            h=hashlib.sha256(inp.x[:sn].tobytes()+sei.cpu().numpy().tobytes()+setype.cpu().numpy().tobytes()).hexdigest() if share_static else str(n)
            if h not in static_seen:
                group=len(static_seen);static_seen[h]=group
                xs.append(x[:sn]);edges.append(sei+offset);relations.append(setype)
                stat_nodes.extend(range(offset,offset+sn));stat_owner.extend([group]*sn);offset+=sn
            stat_query.append(static_seen[h])
            xs.append(x[sn:]);edges.append(ei[:,~sm]-sn+offset);relations.append(et[~sm])
            anchor.append(offset+inp.anchor-sn)
            ids=inp.event_nodes-sn+offset;dyn_nodes.extend(ids.tolist());dyn_owner.extend([n]*len(ids));offset+=len(x)-sn
        ei=torch.cat(edges,1);et=torch.cat(relations);x=torch.cat(xs)
        h=torch.relu(self.encoder.input(x))
        for layer in self.encoder.layers:h=layer(h,ei,et)
        anchor_h=h[torch.tensor(anchor,device=device)]
        dynamic=mean_pool(h[torch.tensor(dyn_nodes,device=device)],torch.tensor(dyn_owner,device=device),len(inputs))
        if stat_nodes:
            static=mean_pool(h[torch.tensor(stat_nodes,device=device)],torch.tensor(stat_owner,device=device),len(static_seen))[torch.tensor(stat_query,device=device)]
        else:static=h.new_zeros((len(inputs),32))
        flags=h.new_tensor([[1.,float(len(i.event_nodes)>0),float(i.static_count>0)] for i in inputs])
        return torch.cat([anchor_h,dynamic,static,flags],1)

def parameter_report(model):
    total=sum(p.numel() for p in model.parameters())
    # NoDep keeps exactly the same parameter form. Only temporal coefficient row
    # participates; bases/self/input/head still participate normally.
    inactive=24 if model.kind=='no_dep' else 8 if model.kind=='rgcn' else 0
    return {'total':total,'structurally_active':total-inactive,'inactive_relation_coefficients':inactive,
        'note':'full prefix uses5 of7 relations; NoDep uses1; local gradients may legitimately be zero'}

def weighted_loss(logits,targets,weights,scale=1.):
    """Three global hierarchy-normalized parts; no re-normalizing sparse batches."""
    a,b=logits;ya=targets[:,0].long();yb=targets[:,1];valid_a=ya>=0;valid_b=yb>=0
    la=torch.nn.functional.cross_entropy(a[valid_a],ya[valid_a],reduction='none') if valid_a.any() else a.sum()*0
    lb=torch.nn.functional.binary_cross_entropy_with_logits(b[valid_b],yb[valid_b],reduction='none') if valid_b.any() else b.sum()*0
    parts=[]
    for column in (0,1):parts.append((la*weights[valid_a,column]).sum() if valid_a.any() else la)
    parts.append((lb*weights[valid_b,2]).sum() if valid_b.any() else lb)
    return sum(parts)*scale/3,torch.stack(parts).detach()
