"""Readonly independent reconstruction and fixed CPU/MPS backward acceptance."""
from pathlib import Path
import hashlib,json
import numpy as np
from audit_caches import post

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def verify():
    threshold=ROOT/'probe_acceptance.json';limits=json.loads(threshold.read_text())
    assert limits['logit_rtol']==1e-3 and limits['logit_atol']==1e-4
    assert limits['gradient_and_raw_relative_l2_max']==.01 and limits['final_support_iou_min']==.95
    frozen=ROOT.parent/'output_mac_stage23/device/probe_inputs.npz'
    rows=[];cases={}
    with np.load(frozen,allow_pickle=False) as inputs:
        for name in ['original','perturbed']:
            pairs={}
            for device in ['cpu','mps']:
                path=ROOT/'probes'/f'{device}_{name}.npz'
                assert threshold.stat().st_mtime_ns<path.stat().st_mtime_ns
                with np.load(path,allow_pickle=False) as z:
                    arrays={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
                    m=json.loads(str(z['metadata'].item()))
                assert m['input']==name and m['device']==device and m['probe_acceptance_sha256']==sha(threshold)
                assert m['raw_spec']['device']==device and m['raw_spec']['precision']=='fp32'
                assert m['raw_spec']['raw_source_sha256']==sha(ROOT/'gdd_raw.py') and m['raw_spec']['utility_source_sha256']==sha(ROOT/'common.py')
                assert m['record']['device']==device and m['record']['precision']=='fp32' and m['record']['target']==int(inputs['target'][0])
                assert set(arrays)==set(m['hashes'])
                for k,a in arrays.items():assert np.isfinite(a).all() and arrsha(a)==m['hashes'][k]
                for k in ['input','gradient','raw']:
                    assert arrays[k].shape==(1,3,224,224) and arrays[k].dtype==np.float32
                assert arrays['logits'].shape==(1,5) and arrays['logits'].dtype==np.float32
                assert np.array_equal(arrays['input'],inputs[name][None])
                baseline=float(arrays['input'].min())
                assert baseline==m['record']['baseline']
                assert np.array_equal((arrays['input']-np.float32(baseline))*arrays['gradient'],arrays['raw'])
                for method in ['gdd_black_post22','gdd_black_soft15']:
                    expected=post(arrays['raw'],method)
                    for k,value in expected.items():
                        actual=arrays[f'{method}_{k}']
                        assert value.shape==actual.shape and value.dtype==actual.dtype and np.array_equal(value,actual)
                pairs[device]=arrays
                rows.append({'case':name,'device':device,'file':str(path.relative_to(ROOT)),'file_sha256':sha(path),
                             'input_sha256':arrsha(arrays['input']),'gradient_sha256':arrsha(arrays['gradient']),
                             'raw_sha256':arrsha(arrays['raw']),'all_hashes':'PASS','independent_multiplication':'byte_identical','independent_postprocessing':'all_fields_byte_identical'})
            a,b=pairs['cpu'],pairs['mps']
            close=np.isclose(a['logits'],b['logits'],rtol=1e-3,atol=1e-4)
            result={'logit_fail_count':int((~close).sum()),'logit_max_abs_difference':float(np.max(np.abs(a['logits']-b['logits']))),
                    'classification_match':int(a['logits'][0].argmax())==int(b['logits'][0].argmax())}
            for field in ['gradient','raw']:
                result[field+'_relative_l2']=float(np.linalg.norm(a[field].astype(np.float64)-b[field].astype(np.float64))/max(np.linalg.norm(a[field].astype(np.float64)),1e-8))
            for method in ['gdd_black_post22','gdd_black_soft15']:
                ac=a[method+'_final']!=0;bc=b[method+'_final']!=0
                union=int(np.count_nonzero(ac|bc));intersection=int(np.count_nonzero(ac&bc))
                result[method+'_support_iou']=intersection/max(union,1)
            result['pass']=bool(close.all() and result['classification_match'] and result['gradient_relative_l2']<=.01 and result['raw_relative_l2']<=.01
                and result['gdd_black_post22_support_iou']>=.95 and result['gdd_black_soft15_support_iou']>=.95)
            cases[name]=result
    saved=json.loads((ROOT/'probes/comparison.json').read_text())
    status='PASS' if all(c['pass'] for c in cases.values()) else 'FAIL'
    assert saved['status']==status and saved['acceptance_sha256']==sha(threshold)
    for name,row in cases.items():
        theirs=saved['cases'][name]
        for field in ['gradient_relative_l2','raw_relative_l2','gdd_black_post22_support_iou','gdd_black_soft15_support_iou','logit_max_abs_difference']:
            assert row[field]==theirs[field]
        assert row['pass']==theirs['pass']
    return {'status':status,'evidence_integrity':'PASS','acceptance_sha256':sha(threshold),'cases':cases,'records':rows,'model_forwards':0}
if __name__=='__main__':
    result=verify();(ROOT/'audit/backward_probe_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))
