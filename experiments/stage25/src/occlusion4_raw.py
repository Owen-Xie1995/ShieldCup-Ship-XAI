"""Independent copies of one-channel 4x4 black interventions; five logits saved."""
import json,time
from pathlib import Path
import numpy as np
import torch
import common as c

TILE=4;SIDE=56;COUNT=9408;AREA=16

def slices(i):
    channel,within=divmod(int(i),SIDE*SIDE);y,x=divmod(within,SIDE)
    return channel,slice(y*TILE,(y+1)*TILE),slice(x*TILE,(x+1)*TILE)

def mask_batch(x,b,ids):
    batch=np.repeat(x,len(ids),axis=0)
    for row,i in enumerate(ids):
        channel,y,z=slices(i);batch[row,channel,y,z]=b
    return batch

def reconstruct(logits,masked,target):
    values=((np.float64(logits[0,int(target)])-masked[:,int(target)].astype(np.float64))/AREA).astype(np.float32)
    return np.ascontiguousarray(np.repeat(np.repeat(values.reshape(3,SIDE,SIDE),TILE,axis=1),TILE,axis=2)[None])

def specification(device,batch):
    return {'schema':1,'method':'black_occlusion4_signed_v1','raw_source_sha256':c.sha(Path(__file__)),
            'common_source_sha256':c.sha(Path(c.__file__)),'device':device,'precision':'float32','weights_sha256':c.WEIGHT_SHA,
            'packages':c.versions(),'tile':4,'stride':4,'tile_count':COUNT,'tile_area':16,'inference_batch':batch,
            'order':'channel, y_start, x_start; starts0,4,...,220','baseline':'fixed current complete CHW minimum',
            'difference':'float64(original supplied-target logit)-float64(masked supplied-target logit)',
            'allocation':'float32(signed tile drop/16) repeated in its16elements; not exact individual interventions'}

def forward(model,batch,device):
    out=model(torch.from_numpy(batch).to(device,dtype=torch.float32))
    if out.dtype!=torch.float32 or out.device.type!=device or out.shape!=(len(batch),5):raise RuntimeError('Forward precision/device/shape differs')
    values=np.ascontiguousarray(out.detach().cpu().numpy(),dtype=np.float32)
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite five-class logits')
    return values

def compute(model,x,y,device,batch,clock,checkpoint,key,identity):
    if x.shape!=(1,3,224,224) or x.dtype!=np.float32 or y.size!=1:raise ValueError('Exact single FP32 NCHW input required')
    if next(model.parameters()).device.type!=device or next(model.parameters()).dtype!=torch.float32 or model.training:raise RuntimeError('Model must be requested device FP32 eval')
    clock.check();start=time.monotonic();b=float(x.min());done=0;prior_seconds=0.
    masked=np.empty((COUNT,5),np.float32)
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item()))
            if m['key']!=key or m['identity']!=identity or not np.array_equal(z['input'],x) or not np.array_equal(z['target'],y):raise RuntimeError('Checkpoint identity differs')
            logits=z['logits'];part=z['masked_logits'];done=len(part);masked[:done]=part;prior_seconds=m['compute_seconds']
            if logits.shape!=(1,5) or logits.dtype!=np.float32 or part.ndim!=2 or part.shape[1]!=5 or part.dtype!=np.float32 or done>COUNT or done!=m['masks_done']:raise RuntimeError('Checkpointlogitshape/dtype/countdiffers')
            if list(z['target'].shape)!=identity['target_shape'] or z['target'].dtype.str!=identity['target_dtype']:raise RuntimeError('Checkpointtargetshape/dtypediffers')
            for name in ('input','target','logits','masked_logits'):
                if c.arrsha(z[name])!=m['hashes'][name] or not np.isfinite(z[name]).all():raise RuntimeError('Checkpoint content differs')
    else:
        with torch.no_grad():logits=forward(model,x,device)
    def save():
        arrays={'input':x,'target':y,'logits':logits,'masked_logits':masked[:done].copy()}
        m={'status':'INCOMPLETE' if done<COUNT else 'COMPLETE_BATCH_CHECKPOINT','key':key,'identity':identity,
           'masks_done':done,'compute_seconds':prior_seconds+time.monotonic()-start,'hashes':{k:c.arrsha(a) for k,a in arrays.items()}}
        c.atomic_npz(checkpoint,**arrays,metadata=np.array(json.dumps(m,sort_keys=True)))
    save()
    with torch.no_grad():
        for first in range(done,COUNT,batch):
            clock.check();ids=range(first,min(first+batch,COUNT))
            masked[first:first+len(ids)]=forward(model,mask_batch(x,np.float32(b),ids),device)
            done=first+len(ids);save();clock.check()
            clock.update(masks_completed=done,masks_planned=COUNT,
                         eta_seconds=(COUNT-done)*(prior_seconds+time.monotonic()-start)/done if done else None)
    raw=reconstruct(logits,masked,int(y.reshape(-1)[0]));c.sync(device)
    arrays={'input':x,'target':y,'logits':logits,'masked_logits':masked,'raw':raw}
    record={'device':device,'precision':'float32','baseline':b,'target':int(y.reshape(-1)[0]),'predicted_class':int(logits.argmax(1)[0]),
            'inference_batch':batch,'masks_completed':COUNT,'masks_planned':COUNT,'compute_seconds':prior_seconds+time.monotonic()-start,
            'this_session_compute_seconds':time.monotonic()-start,'resumed_masks':len(part) if checkpoint.exists() and 'part' in locals() else 0,
            'memory':{k:int(getattr(torch.mps,k)()) for k in ('current_allocated_memory','driver_allocated_memory','recommended_max_memory')} if device=='mps' else {}}
    return arrays,record
