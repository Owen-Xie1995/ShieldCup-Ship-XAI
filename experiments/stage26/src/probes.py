"""Frozen local PCG tile subsets and exact source/device acceptance; no official RNG draws."""
import json,shutil,time
from pathlib import Path
import numpy as np
import torch
import common as c
import occlusion_raw as raw
from legacy import Legacy

def prepare():
    folder=c.ROOT/'probes';folder.mkdir(exist_ok=True)
    source=c.WORKSPACE/'output_mac_stage25/probes/fixed_inputs.npz';dest=folder/'fixed_inputs.npz'
    if dest.exists() and c.sha(dest)!=c.sha(source):raise RuntimeError('Frozen inputs changed')
    if not dest.exists():shutil.copyfile(source,dest)
    ids={f'tile{tile}_ids128':np.random.Generator(np.random.PCG64(seed)).choice(raw.count(tile),128,replace=False).astype(np.int64)
         for tile,seed in ((2,20261004),(4,20261005),(8,20261006))}
    path=folder/'frozen_indices.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(ids) or any(z[k].dtype!=v.dtype or z[k].shape!=v.shape or c.arrsha(z[k])!=c.arrsha(v) for k,v in ids.items()):raise RuntimeError('Frozen indices changed')
    else:c.atomic_npz(path,**ids)
    record={'status':'FROZEN_BEFORE_NEW_FORWARD','input_source':str(source),'input_source_sha256':c.sha(source),'fixed_inputs_sha256':c.sha(dest),'indices_file_sha256':c.sha(path),
            'arrays':{k:{'sha256':c.arrsha(v),'shape':list(v.shape),'dtype':v.dtype.str} for k,v in ids.items()},
            'seeds':{'tile2_ids128':20261004,'tile4_ids128':20261005,'tile8_ids128':20261006},
            'acceptance':{'rtol':1e-3,'atol':1e-4,'relative_tile_drop_L2_max':.01,'denominator_floor':1e-8,'near_zero_sign_exclude':1e-6},
            'B_probe_cumulative_limit_seconds':120,'not_full_CPU_final_support_acceptance':True,'sampling_uses_local_Generator_only':True}
    c.atomic_json(folder/'frozen_sampling.json',record)
    (folder/'frozen_sampling.sha256').write_text(c.sha(folder/'frozen_sampling.json')+'  frozen_sampling.json\n')

def read():
    meta=json.loads((c.ROOT/'probes/frozen_sampling.json').read_text())
    if c.sha(c.ROOT/'probes/fixed_inputs.npz')!=meta['fixed_inputs_sha256'] or c.sha(c.ROOT/'probes/frozen_indices.npz')!=meta['indices_file_sha256']:raise RuntimeError('Frozen sampling files changed')
    with np.load(c.ROOT/'probes/fixed_inputs.npz',allow_pickle=False) as z:inputs={k:np.ascontiguousarray(z[k]) for k in z.files}
    with np.load(c.ROOT/'probes/frozen_indices.npz',allow_pickle=False) as z:indices={k:np.ascontiguousarray(z[k]) for k in z.files}
    return inputs,indices

def subset(model,x,y,tile,ids,device,batch,clock,label,prefix):
    clock.check();started=time.monotonic();folder=c.ROOT/'probes';path=folder/f'{prefix}_{label}_tile{tile}_{device}_batch{batch}.npz'
    if next(model.parameters()).device.type!=device or next(model.parameters()).dtype!=torch.float32 or model.training:raise RuntimeError('Probe model device/FP32/eval changed')
    b=float(x.min());masked=np.empty((len(ids),5),np.float32);done=0
    with torch.no_grad():
        logits=raw.forward(model,x,device)
        for first in range(0,len(ids),batch):
            clock.check();current=ids[first:first+batch];masked[first:first+len(current)]=raw.forward(model,raw.mask_batch(x,np.float32(b),current,tile),device);done=first+len(current)
            arrays={'input':x,'target':y,'logits':logits,'tile_ids':ids,'masked_logits':masked[:done].copy()}
            metadata={'status':'COMPLETE' if done==len(ids) else 'PARTIAL','label':label,'tile':tile,'device':device,'output_device':device,'precision':'float32','output_dtype':'float32','inference_batch':batch,'baseline':b,'target':int(y[0]),'predicted_class':int(logits.argmax(1)[0]),'masks_completed':done,'masks_planned':len(ids),'weights_sha256':c.WEIGHT_SHA,'elapsed_seconds':time.monotonic()-started,'hashes':{k:c.arrsha(v) for k,v in arrays.items()}}
            c.atomic_npz(path,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)));clock.update(method=f'subset_tile{tile}',input_kind=label,masks_completed=done,masks_planned=len(ids),device=device);clock.check()
    c.sync(device)
    metadata.update({'elapsed_seconds':time.monotonic()-started,'memory':{k:int(getattr(torch.mps,k)()) for k in ('current_allocated_memory','driver_allocated_memory','recommended_max_memory')} if device=='mps' else {}})
    c.atomic_npz(path,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)));c.atomic_json(path.with_suffix('.json'),metadata)
    return arrays,metadata,path

def compare(reference,current,target,full_masked=True):
    rlog=np.asarray(reference['logits']);clog=current['logits']
    if full_masked:rprobe=np.asarray(reference['masked_logits']);cprobe=current['masked_logits']
    else:rprobe=np.asarray(reference['masked_target_logits']);cprobe=current['masked_logits'][:,target]
    rd=rlog[0,target].astype(np.float64)-rprobe[:,target].astype(np.float64) if full_masked else np.float64(rlog[0,target])-rprobe.astype(np.float64)
    cd=clog[0,target].astype(np.float64)-current['masked_logits'][:,target].astype(np.float64)
    relative=float(np.linalg.norm(cd-rd)/max(float(np.linalg.norm(rd)),1e-8));eligible=np.maximum(np.abs(rd),np.abs(cd))>1e-6
    finite=all(np.isfinite(a).all() for a in (rlog,clog,rprobe,cprobe,rd,cd));classification=int(rlog.argmax(1)[0])==int(clog.argmax(1)[0])
    checks={'finite':finite,'original_five_logits_allclose':bool(np.allclose(rlog,clog,rtol=1e-3,atol=1e-4)),
            'probe_logits_allclose':bool(np.allclose(rprobe,cprobe,rtol=1e-3,atol=1e-4)),'predicted_class_same':classification,'tile_drop_relative_L2_le_1percent':relative<=.01}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'probe_comparison_scope':'all five masked logits' if full_masked else 'saved masked target logits only; other four legacy classes unavailable',
            'original5_max_abs_difference':float(np.max(np.abs(clog.astype(np.float64)-rlog))), 'masked_max_abs_difference':float(np.max(np.abs(cprobe.astype(np.float64)-rprobe))),
            'tile_drop_relative_L2':relative,'near_zero_excluded_count':int((~eligible).sum()),'eligible_sign_count':int(eligible.sum()),
            'sign_mismatch_count':int((np.sign(rd[eligible])!=np.sign(cd[eligible])).sum())}

def source_acceptance(model,inputs,indices,clock):
    legacy=Legacy();cases=[]
    for tile in (4,8):
        ids=indices[f'tile{tile}_ids128']
        for label in ('original','perturbed'):
            batch=32 if tile==8 and label=='perturbed' else 64
            found=legacy.find(inputs[label],inputs['target'],tile,batch)
            if found is None:raise RuntimeError(f'Missing validated legacy source:{tile},{label},batch{batch}')
            a,m,p=found;ref={'logits':a['logits']}
            if tile==4:ref['masked_logits']=a['masked_logits'][ids]
            else:ref['masked_target_logits']=a['masked_target_logits'][ids]
            new,record,path=subset(model,inputs[label],inputs['target'],tile,ids,'mps',batch,clock,label,'A_source')
            gate=compare(ref,new,int(inputs['target'][0]),full_masked=tile==4)
            gate.update({'tile':tile,'label':label,'legacy_inference_batch':batch,'new_inference_batch':batch,'legacy_path':str(p),'legacy_file_sha256':c.sha(p),'new_probe_path':str(path),'new_probe_file_sha256':c.sha(path),'source_identity':m,'model_parameter_device':str(next(model.parameters()).device)})
            c.atomic_json(c.ROOT/'probes'/f'A_source_{label}_tile{tile}_acceptance.json',gate);cases.append(gate)
    result={'status':'PASS' if all(x['status']=='PASS' for x in cases) else 'FAIL','cases':cases,'not_full_CPU_final_support_acceptance':True}
    c.atomic_json(c.ROOT/'probes/A_source_acceptance.json',result);return result

def device_acceptance2(model,inputs,indices,clock,batch=64):
    results={};ids=indices['tile2_ids128']
    with clock.probe_scope():
        for device in ('cpu','mps'):
            model.to(device);model.eval();c.sync(device);clock.check();results[device]={}
            for label in ('original','perturbed'):
                results[device][label]=subset(model,inputs[label],inputs['target'],2,ids,device,batch,clock,label,'B_device')
        cases={}
        for label in ('original','perturbed'):
            ref,rm,rp=results['cpu'][label];new,nm,npth=results['mps'][label];gate=compare(ref,new,int(inputs['target'][0]),True)
            gate.update({'cpu_path':str(rp),'cpu_sha256':c.sha(rp),'mps_path':str(npth),'mps_sha256':c.sha(npth),'batch':batch});cases[label]=gate
        clock.check()
    result={'status':'PASS' if all(g['status']=='PASS' for g in cases.values()) else 'FAIL','cases':cases,'elapsed_seconds':clock.last_probe_seconds,'cumulative_limit_seconds':120,
            'full_CPU_2x2_final_support_acceptance':'NOT_PERFORMED','batch':batch}
    c.atomic_json(c.ROOT/'probes/B_device_acceptance.json',result);return result
