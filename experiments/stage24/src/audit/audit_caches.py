"""Independent gradient multiplication, content identities and exact fixed postprocessing."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
METHODS=['gdd_black_post22','gdd_black_soft15']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def signed_top(a,frac):
    out=np.zeros_like(a);flat=out.reshape(a.shape[0],-1);source=a.reshape(a.shape[0],-1)
    for index,row in enumerate(source):
        keep=np.argsort(np.abs(row),kind='stable')[-int(np.ceil(row.size*frac)):]
        flat[index,keep]=row[keep]
    return out
def post(raw,method):
    if method==METHODS[0]:
        sparse=signed_top(raw,.25)
        smoothed=F.avg_pool2d(F.pad(torch.from_numpy(sparse),(2,2,2,2),mode='reflect'),5,stride=1).numpy()
        return {'final':signed_top(smoothed,.1)}
    if method==METHODS[1]:
        smooth=gaussian_filter(np.abs(raw[0]).astype(np.float64),sigma=(0,1,1),mode='reflect',truncate=4.)
        threshold=np.quantile(smooth,.85,method='linear')
        gate=np.maximum(1-threshold/np.maximum(smooth,1e-12),0)
        final=np.ascontiguousarray(raw[0].astype(np.float64)*gate,dtype=np.float32)[None]
        return {'s':smooth[None],'tau':np.asarray([threshold],dtype=np.float64),'gate':gate[None],'final':final}
    raise ValueError(method)

def audit(cache_dir):
    cache_dir=Path(cache_dir).resolve()
    records=[];raws={};methods={m:0 for m in METHODS};devices={}
    for path in sorted((cache_dir/'raw').rglob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)=={'input','gradient','raw','logits','metadata'},path
            m=json.loads(str(z['metadata'].item()));arrays={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
        identity=m['identity'];spec=identity['spec'];key=digest(identity)
        assert key==m['key']==path.stem and path.parent.name==key[:2]
        assert spec['method']=='gdd_black_logit_v1' and spec['precision']=='fp32' and spec['device'] in ['cpu','mps']
        assert spec['raw_source_sha256']==sha(ROOT/'gdd_raw.py') and spec['utility_source_sha256']==sha(ROOT/'common.py')
        assert spec['weight_sha256']=='eede97896edcb34840d998647cfd9db76be8ad82b3c99db7d3375781ab30e473'
        expected_packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','quantus','numpy','scipy']}
        assert spec['packages']==expected_packages
        target=np.asarray(identity['target_values'],dtype=np.dtype(identity['target_dtype']))
        assert list(target.shape)==identity['target_shape']==[1] and arrsha(target)==identity['target_sha256'] and target.dtype.kind in 'iu'
        assert int(target[0])==m['record']['target'] and 0<=int(target[0])<5
        for field,a in arrays.items():
            assert a.dtype==np.float32 and np.isfinite(a).all() and arrsha(a)==m['hashes'][field]
            assert a.shape==((1,5) if field=='logits' else (1,3,224,224))
        x=arrays['input'];g=arrays['gradient'];raw=arrays['raw'];b=float(x.min())
        assert identity['input_shape']==list(x.shape) and identity['input_dtype']==x.dtype.str and identity['input_sha256']==arrsha(x)
        assert b==m['record']['baseline'] and m['record']['device']==spec['device'] and m['record']['precision']=='fp32'
        assert np.array_equal((x-np.float32(b))*g,raw),path
        assert int(arrays['logits'][0].argmax())==m['record']['predicted_class']
        raws[key]=raw
        devices[spec['device']]=devices.get(spec['device'],0)+1
        records.append({'layer':'raw','path':str(path.relative_to(ROOT)),'file_sha256':sha(path),'key':key,'device':spec['device'],
                        'input_sha256':arrsha(x),'target_sha256':arrsha(target),'gradient_sha256':arrsha(g),'raw_sha256':arrsha(raw),
                        'reconstruction':'byte_identical'})
    for method in METHODS:
        for path in sorted((cache_dir/method).rglob('*.npz')):
            with np.load(path,allow_pickle=False) as z:
                m=json.loads(str(z['metadata'].item()));arrays={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
            identity=m['identity'];key=digest(identity);spec=identity['spec'];raw=raws[identity['raw_key']]
            assert key==m['key']==path.stem and path.parent.name==key[:2]
            assert spec['method']==method and spec['postprocess_source_sha256']==sha(ROOT/'postprocess.py')
            assert identity['raw_sha256']==arrsha(raw)
            expected=post(raw,method)
            assert set(arrays)==set(expected)
            for field,a in arrays.items():
                assert a.dtype==expected[field].dtype and a.shape==expected[field].shape and np.isfinite(a).all()
                assert arrsha(a)==m['hashes'][field] and np.array_equal(a,expected[field]),(path,field)
            records.append({'layer':method,'path':str(path.relative_to(ROOT)),'file_sha256':sha(path),'key':key,
                            'raw_key':identity['raw_key'],'final_sha256':arrsha(arrays['final']),'reconstruction':'byte_identical'})
            methods[method]+=1
    return {'status':'PASS','cache_dir':str(cache_dir.relative_to(ROOT)),'model_forwards':0,'raw_count':len(raws),'raw_by_device':devices,
            'final_counts':methods,'independent_raw_multiplication':'all byte-identical','independent_final_formula':'all fields byte-identical','records':records}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-dir',type=Path,default=ROOT/'cache')
    parser.add_argument('--output',type=Path,default=ROOT/'audit/complete_cache_audit.json');args=parser.parse_args()
    result=audit(args.cache_dir)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
if __name__=='__main__':main()
