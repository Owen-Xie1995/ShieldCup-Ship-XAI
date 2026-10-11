"""Exactly two preregistered signed postprocessors; no adaptive tuning."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

METHODS=('black_occlusion4_post22','black_occlusion4_soft15')

def signed_top(a, fraction):
    flat=a.reshape(len(a),-1);out=np.zeros_like(flat)
    keep=int(np.ceil(flat.shape[1]*fraction))
    for i,row in enumerate(flat):
        chosen=np.argsort(np.abs(row),kind='stable')[-keep:];out[i,chosen]=row[chosen]
    return out.reshape(a.shape)

def process(raw,method):
    raw=np.ascontiguousarray(raw,dtype=np.float32)
    if raw.ndim!=4 or len(raw)!=1 or not np.isfinite(raw).all(): raise ValueError('Single finite NCHW raw required')
    if method=='black_occlusion4_post22':
        first=signed_top(raw,.25)
        smooth=F.avg_pool2d(F.pad(torch.from_numpy(first),(2,2,2,2),mode='reflect'),5,stride=1).numpy()
        return {'final':np.ascontiguousarray(signed_top(smooth,.10),dtype=np.float32)}
    if method=='black_occlusion4_soft15':
        s=gaussian_filter(np.abs(raw[0]).astype(np.float64),sigma=(0,1,1),mode='reflect',truncate=4.0)
        tau=float(np.quantile(s,.85,method='linear'))
        gate=np.maximum(1.0-tau/np.maximum(s,1e-12),0.0)
        final=np.ascontiguousarray(raw[0].astype(np.float64)*gate,dtype=np.float32)[None]
        return {'s':np.ascontiguousarray(s[None]),'tau':np.asarray([tau],dtype=np.float64),
                'gate':np.ascontiguousarray(gate[None]),'final':final}
    raise ValueError(method)
