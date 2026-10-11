"""Independent saved subset/source/device comparisons; no classifier forward."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and arrsha(a)==arrsha(b)
def load(p):
    with np.load(p,allow_pickle=False) as z:
        a={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'};m=json.loads(str(z['metadata'].item())) if 'metadata' in z.files else None
    return a,m
def gate(rlog,rmasked,clog,cmasked,target,full):
    rprobe=rmasked if full else rmasked.reshape(-1);cprobe=cmasked if full else cmasked[:,target]
    rd=np.float64(rlog[0,target])-(rprobe[:,target].astype(np.float64) if full else rprobe.astype(np.float64))
    cd=np.float64(clog[0,target])-cmasked[:,target].astype(np.float64);denom=max(np.linalg.norm(rd),1e-8);relative=float(np.linalg.norm(cd-rd)/denom)
    eligible=np.maximum(np.abs(rd),np.abs(cd))>1e-6
    checks={'finite':all(np.isfinite(a).all() for a in [rlog,clog,rprobe,cprobe,rd,cd]),'original_five_logits_allclose':bool(np.allclose(rlog,clog,rtol=1e-3,atol=1e-4)),'probe_logits_allclose':bool(np.allclose(rprobe,cprobe,rtol=1e-3,atol=1e-4)),'predicted_class_same':int(rlog.argmax(1)[0])==int(clog.argmax(1)[0]),'tile_drop_relative_L2_le_1percent':relative<=.01}
    return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'original5_max_abs_difference':float(np.max(np.abs(clog.astype(np.float64)-rlog))),'masked_max_abs_difference':float(np.max(np.abs(cprobe.astype(np.float64)-rprobe))),'tile_drop_relative_L2':relative,'near_zero_excluded_count':int((~eligible).sum()),'eligible_sign_count':int(eligible.sum()),'sign_mismatch_count':int((np.sign(rd[eligible])!=np.sign(cd[eligible])).sum())}
def compare_record(computed,saved):
    for k,v in computed.items():
        if isinstance(v,float):assert abs(saved[k]-v)<=1e-15,(k,saved[k],v)
        else:assert saved[k]==v,(k,saved[k],v)
def masked_class_qa(reference,current,full):
    if not full:
        return {'status':'NOT_VERIFIABLE_TARGET_ONLY_LEGACY','reason':'Historical 8x8 masked outputs preserve only the supplied target column; four other classes cannot be reconstructed.','compared_count':0,'mismatch_count':None,'reference_classes':None,'current_classes':None}
    reference_classes=np.asarray(reference).argmax(axis=1);current_classes=np.asarray(current).argmax(axis=1)
    mismatch=int((reference_classes!=current_classes).sum())
    return {'status':'PASS' if mismatch==0 else 'FAIL','comparison':'Each corresponding masked prediction class agrees across source/device; classes may differ from the unmasked prediction.','compared_count':len(reference_classes),'mismatch_count':mismatch,'reference_classes':reference_classes.tolist(),'current_classes':current_classes.tolist()}
def probe(p,fixed,ids,tile,label,device,batch):
    a,m=load(p);assert set(a)=={'input','target','logits','tile_ids','masked_logits'} and set(m['hashes'])==set(a)
    assert exact(a['input'],fixed[label]) and exact(a['target'],fixed['target']) and exact(a['tile_ids'],ids[f'tile{tile}_ids128'])
    expected={'input':((1,3,224,224),np.float32),'target':((1,),np.int64),'logits':((1,5),np.float32),'tile_ids':((128,),np.int64),'masked_logits':((128,5),np.float32)}
    for k,v in a.items():
        shape,dtype=expected[k];assert v.shape==shape and v.dtype==dtype and np.isfinite(v).all() and arrsha(v)==m['hashes'][k]
    assert m['status']=='COMPLETE' and m['label']==label and m['tile']==tile and m['device']==m['output_device']==device and m['precision']==m['output_dtype']=='float32'
    assert m['inference_batch']==batch and m['baseline']==float(a['input'].min()) and m['target']==int(a['target'][0]) and m['predicted_class']==int(a['logits'].argmax(1)[0])
    assert m['masks_completed']==m['masks_planned']==128 and m['weights_sha256']=='eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473'
    assert json.loads(Path(p).with_suffix('.json').read_text())==m
    return a,m
def main():
    fixed,_=load(ROOT/'probes/fixed_inputs.npz');ids,_=load(ROOT/'probes/frozen_indices.npz');result={'status':'PASS','model_forwards':0,'A_source_cases':[],'B_device_cases':[],'full_CPU_final_support_acceptance':'NOT_PERFORMED'}
    assert sha(ROOT/'probes/fixed_inputs.npz')==sha(ROOT.parent/'output_mac_stage25/probes/fixed_inputs.npz')
    source_path=ROOT/'probes/A_source_acceptance.json'
    if source_path.exists():
        saved=json.loads(source_path.read_text());assert len(saved['cases'])==4
        for s in saved['cases']:
            tile=s['tile'];label=s['label'];batch=32 if tile==8 and label=='perturbed' else 64
            assert s['legacy_inference_batch']==s['new_inference_batch']==batch and s['model_parameter_device'].split(':')[0]=='mps'
            lp=Path(s['legacy_path']);npth=Path(s['new_probe_path']);assert sha(lp)==s['legacy_file_sha256'] and sha(npth)==s['new_probe_file_sha256']
            current,_=probe(npth,fixed,ids,tile,label,'mps',batch);legacy,m=load(lp);selected=ids[f'tile{tile}_ids128'];target=int(fixed['target'][0])
            if tile==4:
                assert exact(legacy['input'],fixed[label]) and exact(legacy['target'],fixed['target']);rlog=legacy['logits'];rmasked=legacy['masked_logits'][selected]
                assert m['identity']['spec']['inference_batch']==batch
            else:
                assert m['input_sha256']==arrsha(fixed[label]) and m['target']==target and m['spec']['batch_size']==batch
                rlog=np.asarray(m['record']['original_logits'],np.float32)[None];rmasked=legacy['masked_logits'][selected]
            independent=gate(rlog,rmasked,current['logits'],current['masked_logits'],target,tile==4);compare_record(independent,s)
            class_qa=masked_class_qa(rmasked,current['masked_logits'],tile==4)
            independent.update({'tile':tile,'label':label,'reference_batch':batch,'actual_probe_batch':batch,'new_probe_file_sha256':sha(npth),'legacy_file_sha256':sha(lp),'masked_scope':'all five classes' if tile==4 else 'legacy target only; no other classes fabricated','masked_prediction_class_qa':class_qa});result['A_source_cases'].append(independent)
        assert saved['status']==('PASS' if all(s['status']=='PASS' for s in result['A_source_cases']) else 'FAIL');result['A_source_status']=saved['status']
        result['A_full_five_masked_prediction_classes_consistent']=all(s['masked_prediction_class_qa']['status']=='PASS' for s in result['A_source_cases'] if s['tile']==4)
        if not result['A_full_five_masked_prediction_classes_consistent']:result['status']='FAIL'
    else:result['A_source_status']='NOT_COMPLETED'
    device_path=ROOT/'probes/B_device_acceptance.json'
    if device_path.exists():
        saved=json.loads(device_path.read_text());assert saved['cumulative_limit_seconds']==120 and saved['elapsed_seconds']<=120 and saved['full_CPU_2x2_final_support_acceptance']=='NOT_PERFORMED'
        assert set(saved['cases'])=={'original','perturbed'};target=int(fixed['target'][0])
        for label,s in saved['cases'].items():
            cp=Path(s['cpu_path']);mp=Path(s['mps_path']);assert sha(cp)==s['cpu_sha256'] and sha(mp)==s['mps_sha256'];batch=saved['batch'];assert s['batch']==batch
            cpu,_=probe(cp,fixed,ids,2,label,'cpu',batch);mps,_=probe(mp,fixed,ids,2,label,'mps',batch)
            independent=gate(cpu['logits'],cpu['masked_logits'],mps['logits'],mps['masked_logits'],target,True);compare_record(independent,s)
            class_qa=masked_class_qa(cpu['masked_logits'],mps['masked_logits'],True)
            independent.update({'label':label,'batch':batch,'cpu_file_sha256':sha(cp),'mps_file_sha256':sha(mp),'masked_prediction_class_qa':class_qa});result['B_device_cases'].append(independent)
        assert saved['status']==('PASS' if all(s['status']=='PASS' for s in result['B_device_cases']) else 'FAIL');result['B_device_status']=saved['status'];result['B_small_acceptance_seconds']=saved['elapsed_seconds']
        result['B_masked_prediction_classes_consistent']=all(s['masked_prediction_class_qa']['status']=='PASS' for s in result['B_device_cases'])
        if not result['B_masked_prediction_classes_consistent']:result['status']='FAIL'
    else:result['B_device_status']='NOT_COMPLETED'
    (ROOT/'audit/probes_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
