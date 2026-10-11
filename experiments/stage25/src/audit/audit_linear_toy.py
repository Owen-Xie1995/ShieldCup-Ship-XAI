"""Exercise Stage25 real masking/reconstruction on a signed analytic linear toy."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import occlusion4_raw as core

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    # Coefficients are constant within alternating signed tiles. The one baseline
    # pixel has true effect zero, making tile-average != individual effect clear.
    x=np.zeros((1,3,224,224),np.float32);x[0,0,0,0]=-1
    coeff_tile=np.where(np.arange(9408)%2==0,.25,-.25).astype(np.float32).reshape(3,56,56)
    coeff=np.repeat(np.repeat(coeff_tile,4,axis=1),4,axis=2)
    b=float(x.min());target=3
    logits=np.asarray([[100.,5.,-5.,float(np.sum(x[0].astype(np.float64)*coeff)),1.]],dtype=np.float32)
    assert logits.argmax(1)[0]==0 and target!=int(logits.argmax(1)[0])
    effects=(x[0].astype(np.float64)-b)*coeff.astype(np.float64)
    actual=effects.reshape(3,56,4,56,4).sum(axis=(2,4)).reshape(-1)
    masked=np.repeat(logits,9408,axis=0);masked[:,target]=(logits[0,target].astype(np.float64)-actual).astype(np.float32)
    raw=core.reconstruct(logits,masked,target)
    signed_drop=logits[0,target].astype(np.float64)-masked[:,target].astype(np.float64)
    tile_sums=raw[0].astype(np.float64).reshape(3,56,4,56,4).sum(axis=(2,4)).reshape(-1)
    assert np.array_equal(tile_sums,signed_drop) and np.array_equal(signed_drop,actual)
    assert np.any(signed_drop<0) and np.any(signed_drop>0)
    assert raw[0,0,0,0]!=effects[0,0,0]
    coverage=np.zeros((3,224,224),np.uint8)
    for i in range(9408):
        c,ys,xs=core.slices(i)
        assert c==i//3136 and ys.start==((i%3136)//56)*4 and xs.start==(i%56)*4
        coverage[c,ys,xs]+=1
    assert np.array_equal(coverage,np.ones_like(coverage))
    checks=[0,1,55,56,3135,3136,6272,9407]
    made=core.mask_batch(x,np.float32(b),checks)
    for row,i in enumerate(checks):
        c,ys,xs=core.slices(i);expected=x[0].copy();expected[c,ys,xs]=b
        assert np.array_equal(made[row],expected)
        z=float(np.sum(made[row].astype(np.float64)*coeff))
        assert float(logits[0,target])-z==actual[i]
    assert not np.shares_memory(made,x) and not np.shares_memory(made[0],made[1])
    np.savez_compressed(ROOT/'audit/linear_toy.npz',input=x,target=np.asarray([target],np.int64),
        coefficients=coeff,original_logits=logits,masked_logits=masked,actual_signed_drops=actual,
        raw=raw,tile_sums=tile_sums,individual_true_effects=effects,checked_ids=np.asarray(checks,np.int64))
    result={'status':'PASS','official_classifier_forwards':0,'source_sha256':sha(ROOT/'occlusion4_raw.py'),
        'target':target,'argmax':int(logits.argmax(1)[0]),'target_is_non_argmax':True,'masks':9408,
        'masking':'actual core.mask_batch modifies exact single-channel4x4 independent copies',
        'partition':'all CHW elements covered exactly once in channel/y/x order',
        'raw_rebuild':'actual core.reconstruct; all tile totals byte-identical to analytic signed drops',
        'negative_drop_count':int(np.sum(signed_drop<0)),'positive_drop_count':int(np.sum(signed_drop>0)),
        'average_does_not_equal_individual_effect':{'first_pixel_true_effect':float(effects[0,0,0]),'assigned_tile_average':float(raw[0,0,0,0])},
        'raw_sha256':arrsha(raw),'artifact_sha256':sha(ROOT/'audit/linear_toy.npz'),
        'limitation':'Linear toy additivity does not establish additivity or per-element fidelity for nonlinear official ResNet50.'}
    (ROOT/'audit/linear_toy.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
