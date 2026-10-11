"""Analytic signed per-element black interventions, with supplied non-argmax target."""
import sys
import numpy as np
import torch
import common as c
import gdd_raw as gdd
import postprocess as post

def main():
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(108,5,bias=True))
    with torch.no_grad():
        weights=torch.linspace(-.5,.7,540).reshape(5,108);model[1].weight.copy_(weights)
        model[1].bias.copy_(torch.tensor([100.,0.,0.,0.,0.]))
    x=np.linspace(-2,3,108,dtype=np.float32).reshape(1,3,6,6);target=np.asarray([2],dtype=np.int64)
    arrays,record=gdd.compute(model,x,target,device='cpu',official_shape=False)
    expected=(x-x.min())*weights[2].numpy().reshape(x.shape)
    assert np.array_equal(arrays['gradient'],weights[2].numpy().reshape(x.shape))
    assert np.array_equal(arrays['raw'],expected)
    base=torch.full_like(torch.from_numpy(x),float(x.min()))
    with torch.no_grad():
        original=float(model(torch.from_numpy(x))[0,2]);total=float(original-model(base)[0,2])
        intervention=[]
        for i in range(x.size):
            masked=x.copy();masked.flat[i]=x.min();intervention.append(original-float(model(torch.from_numpy(masked))[0,2]))
    assert np.allclose(np.asarray(intervention).reshape(x.shape),expected,rtol=1e-4,atol=1e-5)
    assert np.isclose(expected.sum(dtype=np.float64),total,rtol=1e-5,atol=1e-5)
    assert record['target']==2 and record['predicted_class']==0
    assert all(p.grad is None for p in model.parameters())
    source=c.WORKSPACE/'output_mac_stage23/device/occlusion8_gpu.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('stage24_preserved_post22',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    rng=np.random.Generator(np.random.PCG64(1));raw=rng.normal(size=(1,3,224,224)).astype(np.float32)
    assert np.array_equal(post.process(raw,'gdd_black_post22')['final'],old.postprocess(raw))
    a=post.process(raw,'gdd_black_soft15');assert set(a)=={'final','s','tau','gate'}
    assert np.all(a['gate']>=0) and np.all(a['gate']<=1)
    c.atomic_json(c.ROOT/'tests/linear.json',{'status':'PASS','analytic_gradient_exact':True,'element_interventions_checked':108,
                 'completeness_only_for_this_linear_toy':True,'provided_non_argmax_target':2,'parameter_gradients_none':True,'post22_byte_identical':True})
    print('LINEAR_AND_POST22_PASS')

if __name__=='__main__':main()
