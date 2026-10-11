"""Current-input black-baseline difference times target-logit input gradient."""
import time
from pathlib import Path
import numpy as np
import torch
import common as c

METHOD='gdd_black_logit_v1'

def compute(model,inputs,targets,*,device,official_shape=True):
    x=np.ascontiguousarray(np.asarray(inputs));labels=np.ascontiguousarray(np.asarray(targets))
    if x.dtype!=np.float32 or x.ndim!=4 or len(x)!=1 or labels.shape!=(1,) or labels.dtype.kind not in 'iu':
        raise ValueError('Finite float32 single NCHW and original one integer target required')
    if official_shape and x.shape!=(1,3,224,224): raise ValueError('Official shape differs')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite input')
    if next(model.parameters()).device.type!=device or next(model.parameters()).dtype!=torch.float32:
        raise RuntimeError('Model actual device/precision differs')
    model.eval();baseline=float(x.min());c.sync(device);began=time.monotonic()
    with torch.inference_mode(False),torch.enable_grad():
        leaf=torch.tensor(x,device=device,dtype=torch.float32,requires_grad=True)
        logits=model(leaf)
        if logits.device.type!=device or logits.dtype!=torch.float32: raise RuntimeError('Forward device/precision differs')
        target=int(labels[0])
        if target<0 or target>=logits.shape[1]: raise ValueError('Target outside model classes')
        gradient=torch.autograd.grad(logits[0,target],leaf,create_graph=False,retain_graph=False)[0]
        if gradient.device.type!=device or gradient.dtype!=torch.float32: raise RuntimeError('Backward device/precision differs')
        g=np.ascontiguousarray(gradient.detach().cpu().numpy());z=np.ascontiguousarray(logits.detach().cpu().numpy())
        del gradient,logits,leaf
    raw=np.ascontiguousarray((x-np.float32(baseline))*g,dtype=np.float32)
    c.sync(device)
    if not all(np.isfinite(a).all() for a in (g,z,raw)): raise RuntimeError('Nonfinite gradient or raw')
    if any(p.grad is not None for p in model.parameters()): raise RuntimeError('Unexpected parameter gradient accumulation')
    return {'input':x,'gradient':g,'raw':raw,'logits':z}, {'baseline':baseline,'target':target,
          'predicted_class':int(z[0].argmax()),'device':device,'precision':'fp32',
          'wall_seconds':time.monotonic()-began,'definition':'(x-fixed_current_CHW_min)*target_logit_input_gradient'}

def specification(device,weight_sha=c.WEIGHT_SHA):
    return {'schema':1,'method':METHOD,'device':device,'precision':'fp32','weight_sha256':weight_sha,
            'raw_source_sha256':c.sha(Path(__file__)),'utility_source_sha256':c.sha(Path(c.__file__)),
            'packages':c.versions(),'baseline':'complete current CHW min treated constant',
            'output':'provided target original logit; signed gradient times baseline difference'}
