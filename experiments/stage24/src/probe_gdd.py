"""Actual CPU/MPS backward probe on frozen Stage23 original and perturbation."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import common as c
import gdd_raw as gdd
import postprocess as post

def run(device,max_wall):
    began=time.monotonic();c.check_device(device);official=c.frozen_sources();sys.path.insert(0,str(official/'vendor'))
    from kaggle.model.model import load_model
    torch.set_num_threads(4);model=load_model(str(c.WEIGHTS),device=torch.device(device));model.eval()
    with np.load(c.WORKSPACE/'output_mac_stage23/device/probe_inputs.npz',allow_pickle=False) as z:
        inputs={k:np.ascontiguousarray(z[k]) for k in ('original','perturbed','target')}
    cases=[]
    for name in ('original','perturbed'):
        if time.monotonic()-began>=max_wall:raise TimeoutError('A probe budget reached')
        arrays,record=gdd.compute(model,inputs[name][None],inputs['target'],device=device)
        for method in post.METHODS:
            for key,value in post.process(arrays['raw'],method).items(): arrays[f'{method}_{key}']=value
        meta={'input':name,'device':device,'record':record,'raw_spec':gdd.specification(device),
              'probe_acceptance_sha256':c.sha(c.ROOT/'probe_acceptance.json'),'hashes':{k:c.arrsha(v) for k,v in arrays.items()}}
        path=c.ROOT/'probes'/f'{device}_{name}.npz'
        c.atomic_npz(path,**arrays,metadata=np.array(json.dumps(meta,sort_keys=True)))
        cases.append({'input':name,'device':device,'file':str(path),'file_sha256':c.sha(path),'record':record})
        c.atomic_json(c.ROOT/'progress.json',{'phase':'A_BACKWARD_PROBE','method':gdd.METHOD,'device':device,
                       'input_kind':name,'completed_inputs':len(cases),'planned_inputs':2,'elapsed_seconds':time.monotonic()-began,
                       'last_completed_at_unix':time.time(),'status':'running' if len(cases)<2 else 'COMPLETE'})
        print(json.dumps(cases[-1]),flush=True)
    result={'status':'COMPLETE','device':device,'cases':cases,'process_elapsed_seconds':time.monotonic()-began}
    c.atomic_json(c.ROOT/'probes'/f'{device}.run.json',result);return result

def compare():
    limits=json.loads((c.ROOT/'probe_acceptance.json').read_text());cases={}
    for name in ('original','perturbed'):
        a=np.load(c.ROOT/'probes'/f'cpu_{name}.npz',allow_pickle=False);b=np.load(c.ROOT/'probes'/f'mps_{name}.npz',allow_pickle=False)
        row={};close=np.isclose(a['logits'],b['logits'],rtol=limits['logit_rtol'],atol=limits['logit_atol'])
        row['logit_fail_count']=int((~close).sum());row['class_match']=int(a['logits'][0].argmax())==int(b['logits'][0].argmax())
        row['logit_max_abs_difference']=float(np.max(np.abs(a['logits']-b['logits'])))
        passed=bool(close.all() and row['class_match'])
        for field in ('gradient','raw'):
            rel=float(np.linalg.norm(a[field].astype(np.float64)-b[field].astype(np.float64))/max(np.linalg.norm(a[field].astype(np.float64)),1e-8))
            row[field+'_relative_l2']=rel;passed=passed and rel<=limits['gradient_and_raw_relative_l2_max']
        for method in post.METHODS:
            ac=a[method+'_final']!=0;bc=b[method+'_final']!=0
            iou=float(np.count_nonzero(ac&bc)/max(np.count_nonzero(ac|bc),1));row[method+'_support_iou']=iou
            passed=passed and iou>=limits['final_support_iou_min']
        for archive in (a,b):
            raw=(archive['input']-np.float32(archive['input'].min()))*archive['gradient']
            if not np.array_equal(raw,archive['raw']):raise RuntimeError('Saved g×difference reconstruction failed')
            for field in archive.files:
                if field!='metadata' and not np.isfinite(archive[field]).all():raise RuntimeError('Nonfinite probe')
        row['pass']=bool(passed);cases[name]=row;a.close();b.close()
    result={'status':'PASS' if all(r['pass'] for r in cases.values()) else 'FAIL','cases':cases,
            'acceptance_sha256':c.sha(c.ROOT/'probe_acceptance.json')}
    c.atomic_json(c.ROOT/'probes/comparison.json',result);print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps']);p.add_argument('--compare',action='store_true');p.add_argument('--max-wall-seconds',type=float,default=1200)
    args=p.parse_args();compare() if args.compare else run(args.device,args.max_wall_seconds)
