"""Bounded engineering acceptance, not a new model comparison or selection."""
import argparse,copy,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from cwm_fp.data.common import ROOT,sha256,write_json
from cwm_fp.models.transition_data import access_guard,verify_source,read_targets,hierarchy_weights,load
from cwm_fp.models.transition_train import dataset,trajectory_batches,target_arrays
from cwm_fp.models.transition_model import TransitionModel,KINDS,weighted_loss

SEED=20261101
TOLERANCES={'logits':{'atol':2e-5,'rtol':2e-5},'loss':{'atol':2e-6,'rtol':2e-5},
    'gradient':{'atol':3e-5,'rtol':3e-5},'updated_parameters':{'atol':1e-4,'rtol':1e-4},
    'old_checkpoint_predictions':{'atol':2e-5,'rtol':2e-5}}

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--cache-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--prior-updates',type=int,default=1);p.add_argument('--models',nargs='+',choices=KINDS,default=list(KINDS));a=p.parse_args()
    if platform.system()!='Linux' or not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Verified Linux CUDA required; no fallback')
    if a.out.exists():raise ValueError('Do not overwrite a prior attempt')
    a.out.mkdir(parents=True);start=time.monotonic();device=torch.device(a.device);torch.cuda.set_device(device)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    access_guard(a.source,{'train'});verify_source(a.source)
    def no_val(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and str(args[0]).replace('\\','/').endswith('targets_val.json'):raise PermissionError('No validation targets in training acceptance')
    sys.addaudithook(no_val)
    old=load(ROOT/'outputs/phase05B_protocol.json')
    if sha256(a.cache_root/'cache/index_train.json')!=old['cache_train_index_sha256']:raise ValueError('Training inputs changed')
    protocol={'label':'server_acceptance_only','seed':SEED,'epochs_per_model':1,'max_total_updates':300,'prior_updates':a.prior_updates,
        'microbatch':64,'trajectories_per_batch':4,'tolerances':TOLERANCES,'threads':2,'device':a.device,'old_protocol_sha256':sha256(ROOT/'outputs/phase05B_protocol.json'),
        'new_validation_checkpoint_selection':False,'code_sha256':sha256(Path(__file__)),'models':a.models,
        'restore_policy':'state tensors exact; logits use the original predeclared logits tolerance; byte-exact output remains a separately reported strict probe'}
    write_json(a.out/'protocol.json',protocol)
    updates=a.prior_updates;records=[];errors=[]
    result={'server_acceptance_only':True,'platform':platform.platform(),'python':sys.version,'python_executable':sys.executable,
        'cwd':os.getcwd(),'container_hostname':platform.node(),'pid':os.getpid(),'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
        'numpy':np.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':a.device,
        'gpu_name':torch.cuda.get_device_name(device),'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),
        'records':records,'failures':errors,'tolerances':TOLERANCES}
    def budget():
        if updates>300 or time.monotonic()-start>6000 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*1024**3:raise RuntimeError('Acceptance budget exceeded')
    def step(model,opt,inputs,y,w,ids,scale,update=True):
        nonlocal updates
        dev=next(model.parameters()).device;model.train();opt.zero_grad(set_to_none=True);total=0.
        sync(dev);tick=time.perf_counter()
        for j in range(0,len(ids),64):
            ix=ids[j:j+64];loss,_=weighted_loss(model([inputs[i] for i in ix]),y[ix],w[ix],scale);loss.backward();total+=float(loss.detach().cpu())
        if update:opt.step();updates+=1;budget()
        sync(dev);return total,time.perf_counter()-tick
    try:
        index,inputs,readseconds=dataset(a.cache_root,'train');targets=read_targets(a.cache_root,index);weights=hierarchy_weights(index['rows'],targets)
        if len({r['trace_index'] for r in index['rows']})!=283 or len({tuple(r['identity'][:3]) for r in index['rows']})!=17:raise ValueError('Training scope changed')
        batches=trajectory_batches(index['rows'],np.random.default_rng(SEED));ids=batches[0];scale=len(batches)
        cpu_y=target_arrays(targets,'cpu');cpu_w=torch.as_tensor(weights,dtype=torch.float32)
        sync(device);tick=time.perf_counter();gpu_y=cpu_y.to(device);gpu_w=cpu_w.to(device);sync(device)
        result.update(read_seconds=readseconds,supervision_transfer_seconds=time.perf_counter()-tick,train_prefixes=len(inputs),train_traces=283,train_products=17,fixed_batch_prefixes=len(ids))
        # Separate component probe. This is NOT misreported as an additive timing
        # decomposition of the unchanged old forward(), which also packs tensors.
        tick=time.perf_counter();prepared=[tuple(torch.as_tensor(z.copy()) for z in (inputs[i].x,inputs[i].edge_index,inputs[i].edge_type)) for i in ids]
        result['component_probe_cpu_prepare_seconds']=time.perf_counter()-tick
        sync(device);tick=time.perf_counter();transferred=[tuple(t.to(device) for t in row) for row in prepared];sync(device)
        result['component_probe_transfer_seconds']=time.perf_counter()-tick;del transferred,prepared
        for kind in a.models:
            budget();torch.cuda.reset_peak_memory_stats(device);torch.manual_seed(SEED);cpu=TransitionModel(kind);gpu=copy.deepcopy(cpu).to(device)
            before={k:v.detach().clone() for k,v in cpu.state_dict().items()}
            co=torch.optim.Adam(cpu.parameters(),lr=.001);go=torch.optim.Adam(gpu.parameters(),lr=.001)
            rec={'model':kind};records.append(rec)
            with torch.no_grad():
                cp=cpu([inputs[i] for i in ids]);gp=gpu([inputs[i] for i in ids])
                for x,z in zip(cp,gp):torch.testing.assert_close(x,z.cpu(),**TOLERANCES['logits'])
                rec['initial_max_output_difference']=max(float((x-z.cpu()).abs().max()) for x,z in zip(cp,gp))
            c_loss,c_time=step(cpu,co,inputs,cpu_y,cpu_w,ids,scale);g_loss,g_time=step(gpu,go,inputs,gpu_y,gpu_w,ids,scale)
            torch.testing.assert_close(torch.tensor(c_loss),torch.tensor(g_loss),**TOLERANCES['loss'])
            required=['gru.weight_ih_l0','gru.weight_hh_l0'] if kind=='seq_gru' else ['encoder.input.weight','encoder.layers.0.bases','encoder.layers.1.bases']
            required+=['head_a.weight','head_b.weight'];grad={};maxgrad=0.;maxparam=0.
            for (name,c),(name2,g) in zip(cpu.named_parameters(),gpu.named_parameters()):
                if name!=name2 or c.grad is None or g.grad is None:raise ValueError('Gradient graph differs')
                torch.testing.assert_close(c.grad,g.grad.cpu(),**TOLERANCES['gradient']);torch.testing.assert_close(c,g.cpu(),**TOLERANCES['updated_parameters'])
                grad[name]=float(g.grad.norm());maxgrad=max(maxgrad,float((c.grad-g.grad.cpu()).abs().max()));maxparam=max(maxparam,float((c-g.cpu()).abs().max()))
            if not all(grad[n]>0 for n in required):raise ValueError('Expected gradient missing')
            changed=[k for k,v in gpu.state_dict().items() if not torch.equal(before[k],v.cpu())]
            if not all(k in changed for k in required):raise ValueError('Expected parameter update missing')
            rec['changed_parameters']=changed
            rec.update(cpu_gpu_single_step=True,gradient_norms=grad,max_gradient_difference=maxgrad,max_updated_parameter_difference=maxparam,cpu_loss=c_loss,gpu_loss=g_loss)
            # Existing checkpoint, saved TRAIN predictions and optimizer recovery.
            checkpoint=a.cache_root/'runs'/f'{kind}_{SEED}'/'epoch20.pt';saved=torch.load(checkpoint,map_location=device,weights_only=True)
            recovered=TransitionModel(kind).to(device);recovered.load_state_dict(saved['model']);ro=torch.optim.Adam(recovered.parameters(),lr=.001);ro.load_state_dict(saved['optimizer']);recovered.eval()
            with torch.no_grad():pa,pb=recovered(inputs[:8]);pa=pa.softmax(1).cpu();pb=pb.sigmoid().cpu()
            with np.load(checkpoint.with_name('epoch20_train.npz'),allow_pickle=False) as z:
                torch.testing.assert_close(pa,torch.as_tensor(z['A'][:8]),**TOLERANCES['old_checkpoint_predictions']);torch.testing.assert_close(pb,torch.as_tensor(z['B'][:8]),**TOLERANCES['old_checkpoint_predictions'])
            oldsteps=[float(s['step']) for s in ro.state.values() if 'step' in s]
            step(recovered,ro,inputs,gpu_y,gpu_w,ids,scale)
            newsteps=[float(s['step']) for s in ro.state.values() if 'step' in s]
            if not all(b==a+1 for a,b in zip(oldsteps,newsteps)):raise ValueError('Optimizer not resumed')
            out=a.out/f'{kind}_recovered.pt';torch.save({'model':recovered.state_dict(),'optimizer':ro.state_dict(),'server_acceptance_only':True},out)
            restored=TransitionModel(kind).to(device);restored.load_state_dict(torch.load(out,map_location=device,weights_only=True)['model'])
            if not all(torch.equal(v,restored.state_dict()[k]) for k,v in recovered.state_dict().items()):raise ValueError('Restored state tensors changed')
            recovered.eval();restored.eval()
            with torch.no_grad():
                original=recovered(inputs[:8]);again=recovered(inputs[:8]);reload=restored(inputs[:8])
                for x,z in zip(original,reload):torch.testing.assert_close(x,z,**TOLERANCES['logits'])
            rec.update(old_checkpoint_loaded=True,old_train_predictions_match=True,optimizer_resumed=True,
                save_restore_state_exact=True,save_restore_exact=all(torch.equal(x,z) for x,z in zip(original,reload)),
                restore_max_output_difference=max(float((x-z).abs().max()) for x,z in zip(original,reload)),
                same_instance_repeat_max_difference=max(float((x-z).abs().max()) for x,z in zip(original,again)),
                restore_within_original_logit_tolerance=True,old_checkpoint_sha256=sha256(checkpoint))
            # Fixed-batch warmup, then synchronized repeated forward/backward timing
            # without optimizer updates. CPU and GPU retain their compared states.
            for model,opt,y,w in ((cpu,co,cpu_y,cpu_w),(gpu,go,gpu_y,gpu_w)):step(model,opt,inputs,y,w,ids,scale,False)
            ctimes=[step(cpu,co,inputs,cpu_y,cpu_w,ids,scale,False)[1] for _ in range(3)]
            gtimes=[step(gpu,go,inputs,gpu_y,gpu_w,ids,scale,False)[1] for _ in range(3)]
            rec.update(cpu_fixed_batch_seconds=ctimes,gpu_fixed_batch_seconds=gtimes,speedup_median=float(np.median(ctimes)/np.median(gtimes)),timing_includes_old_tensor_packing=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                step(gpu,go,inputs,gpu_y,gpu_w,ids,scale,False)
            deviceevents=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            rec['profile_device_kernel_seconds']=sum(e.time_range.elapsed_us() for e in deviceevents if not any(x in e.name.lower() for x in ('memcpy','memset')))/1e6 if deviceevents else None
            rec['profile_device_copy_seconds']=sum(e.time_range.elapsed_us() for e in deviceevents if 'memcpy' in e.name.lower())/1e6 if deviceevents else None
            rec['profile_note']='one additional unchanged fixed-batch forward/backward, no optimizer update; kernel totals exclude host launch gaps; unavailable profiler yields null'
            # One fresh, full epoch, same original order generator and supervision.
            torch.manual_seed(SEED);model=TransitionModel(kind).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);visited=[];tick=time.perf_counter();losses=[]
            for batch in batches:
                budget();loss,_=step(model,opt,inputs,gpu_y,gpu_w,batch,scale);losses.append(loss);visited.extend(batch)
            if sorted(visited)!=list(range(len(inputs))):raise ValueError('Omitted/repeated prefix in epoch')
            sync(device);rec.update(epoch_completed=True,epoch_updates=len(batches),epoch_prefixes=len(visited),epoch_seconds=time.perf_counter()-tick,epoch_training_objective=float(np.mean(losses)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':1,'seed':SEED,'server_acceptance_only':True},a.out/f'{kind}_epoch01.pt')
            write_json(a.out/'progress.json',result);print('ACCEPTED',kind,rec['epoch_seconds'],'updates',updates,flush=True)
        result['status']='PASSED'
    except Exception as exc:
        errors.append({'type':type(exc).__name__,'message':str(exc)});result['status']='FAILED';raise
    finally:
        result.update(optimizer_updates_including_prior=updates,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),validation_targets_read=False,test_trajectories_read=False,
            host_environment_installed=False,generalization_ranking=False)
        write_json(a.out/'server_acceptance.json',result)
if __name__=='__main__':main()
