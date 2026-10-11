"""FP32 independent one-channel tile interventions, full five logits, atomic batches."""
import json,time
from pathlib import Path
import numpy as np
import torch
import common as c

def count(tile):
    if tile not in (2,4,8):raise ValueError('Only fixed tiles2/4/8')
    return 3*(224//tile)**2

def slices(i,tile):
    side=224//tile;channel,within=divmod(int(i),side*side);y,x=divmod(within,side)
    if not 0<=int(i)<count(tile):raise ValueError('Tile ID out of range')
    return channel,slice(y*tile,(y+1)*tile),slice(x*tile,(x+1)*tile)

def mask_batch(x,b,ids,tile):
    ids=list(ids);out=np.repeat(x,len(ids),axis=0)
    for row,i in enumerate(ids):
        channel,y,z=slices(i,tile);out[row,channel,y,z]=b
    return out

def reconstruct_target(logits,masked_target,target,tile):
    values=((np.float64(logits[0,int(target)])-np.asarray(masked_target).astype(np.float64))/(tile*tile)).astype(np.float32)
    side=224//tile
    return np.ascontiguousarray(np.repeat(np.repeat(values.reshape(3,side,side),tile,axis=1),tile,axis=2)[None])

def reconstruct(logits,masked,target,tile):
    return reconstruct_target(logits,masked[:,int(target)],target,tile)

def specification(device,batch,tile):
    return {'schema':1,'method':f'black_occlusion{tile}_signed_v1','raw_source_sha256':c.sha(Path(__file__)),
            'common_source_sha256':c.sha(Path(c.__file__)),'device':device,'precision':'float32','weights_sha256':c.WEIGHT_SHA,
            'packages':c.versions(),'tile':tile,'stride':tile,'tile_count':count(tile),'tile_area':tile*tile,'inference_batch':batch,
            'order':f'channel,y_start,x_start; starts0,{tile},...,{224-tile}',
            'baseline':'fixed current complete CHW minimum','difference':'float64(original supplied-target logit)-float64(masked supplied-target logit)',
            'allocation':f'float32(signed tile drop/{tile*tile}) repeated in its{tile*tile}elements; not exact individual interventions'}

def forward(model,batch,device):
    out=model(torch.from_numpy(batch).to(device,dtype=torch.float32))
    if out.dtype!=torch.float32 or out.device.type!=device or out.shape!=(len(batch),5):raise RuntimeError('Forward precision/device/shape differs')
    values=np.ascontiguousarray(out.detach().cpu().numpy(),dtype=np.float32)
    if not np.isfinite(values).all():raise RuntimeError('Nonfinite five-class logits')
    return values

def compute(model,x,y,device,batch,clock,checkpoint,key,identity,tile):
    if x.shape!=(1,3,224,224) or x.dtype!=np.float32 or y.size!=1:raise ValueError('Exact single FP32 NCHW input required')
    if next(model.parameters()).device.type!=device or next(model.parameters()).dtype!=torch.float32 or model.training:raise RuntimeError('Model must be requested device FP32 eval')
    clock.check();start=time.monotonic();b=float(x.min());done=0;prior_seconds=0.;total=count(tile);masked=np.empty((total,5),np.float32)
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item()));part=np.ascontiguousarray(z['masked_logits']);logits=np.ascontiguousarray(z['logits'])
            if m['key']!=key or m['identity']!=identity:raise RuntimeError('Checkpoint identity differs')
            done=len(part);prior_seconds=float(m['compute_seconds'])
            if logits.shape!=(1,5) or logits.dtype!=np.float32 or part.shape!=(done,5) or part.dtype!=np.float32 or done>total or done!=m['masks_done']:raise RuntimeError('Checkpoint logit shape/dtype/count differs')
            for name in ('input','target','logits','masked_logits'):
                if c.arrsha(z[name])!=m['hashes'][name] or not np.isfinite(z[name]).all():raise RuntimeError('Checkpoint content differs')
            if z['input'].dtype!=x.dtype or z['input'].shape!=x.shape or c.arrsha(z['input'])!=c.arrsha(x) or z['target'].dtype!=y.dtype or z['target'].shape!=y.shape or c.arrsha(z['target'])!=c.arrsha(y):raise RuntimeError('Checkpoint input/target differs')
            masked[:done]=part
    else:
        with torch.no_grad():logits=forward(model,x,device)
    resumed=done
    def save():
        arrays={'input':x,'target':y,'logits':logits,'masked_logits':masked[:done].copy()}
        metadata={'status':'INCOMPLETE' if done<total else 'COMPLETE_BATCH_CHECKPOINT','key':key,'identity':identity,'tile':tile,
                  'masks_done':done,'compute_seconds':prior_seconds+time.monotonic()-start,'hashes':{k:c.arrsha(a) for k,a in arrays.items()}}
        c.atomic_npz(checkpoint,**arrays,metadata=np.array(json.dumps(metadata,sort_keys=True)))
    save()
    with torch.no_grad():
        for first in range(done,total,batch):
            clock.check();ids=range(first,min(first+batch,total));masked[first:first+len(ids)]=forward(model,mask_batch(x,np.float32(b),ids,tile),device)
            done=first+len(ids);save();clock.check()
            clock.update(masks_completed=done,masks_planned=total,eta_seconds=(total-done)*(prior_seconds+time.monotonic()-start)/done if done else None)
    raw=reconstruct(logits,masked,int(y.reshape(-1)[0]),tile);c.sync(device)
    arrays={'input':x,'target':y,'logits':logits,'masked_logits':masked,'raw':raw}
    record={'device':device,'output_device':device,'precision':'float32','output_dtype':'float32','model_parameter_dtype':str(next(model.parameters()).dtype),
            'baseline':b,'target':int(y.reshape(-1)[0]),'predicted_class':int(logits.argmax(1)[0]),'inference_batch':batch,'tile':tile,
            'masks_completed':total,'masks_planned':total,'compute_seconds':prior_seconds+time.monotonic()-start,'this_session_compute_seconds':time.monotonic()-start,'resumed_masks':resumed,
            'memory':{k:int(getattr(torch.mps,k)()) for k in ('current_allocated_memory','driver_allocated_memory','recommended_max_memory')} if device=='mps' else {}}
    return arrays,record
