"""NOT_VERIFIED: future explicitly authorized Linux/CUDA engineering acceptance.

Copy the repository, frozen05A prefix source and05B cache first. This entry does
not fit an experiment or reuse the CPU protocol as a GPU training protocol.
"""
import argparse,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import sha256,write_json
from cwm_fp.models.transition_data import verify_source,read_cache,read_targets,hierarchy_weights,access_guard
from cwm_fp.models.transition_model import TransitionModel,weighted_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',required=True,help='Explicit authorized device, e.g. cuda:0')
    a=p.parse_args()
    if platform.system()!='Linux' or not a.device.startswith('cuda:'):raise ValueError('Explicit Linux CUDA session required')
    if a.out.exists():raise ValueError('Use a new independent acceptance directory')
    a.out.mkdir(parents=True);access=access_guard(a.source,{'train'});verify_source(a.source)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent fallback')
    device=torch.device(a.device);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device);torch.set_num_threads(1)
    gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False,timeout=15)
    index,store=read_cache(a.cache_root,'train');targets=read_targets(a.cache_root,index);weights=hierarchy_weights(index['rows'],targets)
    ids=list(range(min(64,len(targets))));inputs=[store[index['rows'][i]['input_key']] for i in ids]
    y=torch.tensor([[targets[i]['A'],targets[i]['B']] for i in ids],dtype=torch.float32,device=device);w=torch.tensor(weights[ids],dtype=torch.float32,device=device)
    torch.manual_seed(20261101);model=TransitionModel('rgcn').to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);start=time.perf_counter()
    loss,_=weighted_loss(model(inputs),y,w);loss.backward();grad=float(model.encoder.input.weight.grad.norm());opt.step();torch.cuda.synchronize(device)
    if grad<=0:raise ValueError('Encoder gradient not established')
    write_json(a.out/'linux_acceptance.json',{'status':'ENGINEERING_ONLY','platform':platform.platform(),'python':sys.version,'torch':str(torch.__version__),
        'device':str(device),'device_name':torch.cuda.get_device_name(device),'nvidia_smi_exit':gpu.returncode,'nvidia_smi_stdout':gpu.stdout,
        'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
        'prefixes':len(ids),'optimizer_updates':1,'encoder_gradient_norm':grad,'formal_training':False,'requires_new_device_protocol_before_fitting':True,
        'cache_index_sha256':sha256(a.cache_root/'cache/index_train.json'),'source_target_access':access})

if __name__=='__main__':main()
