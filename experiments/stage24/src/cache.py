"""Separate exact-content raw and final cache layers; each hit is verified."""
import json,time
from pathlib import Path
import numpy as np
import common as c
import gdd_raw as gdd
import postprocess as post

class Cache:
    def __init__(self,root,device,method):
        self.root=Path(root);self.device=device;self.method=method
        self.raw_spec=gdd.specification(device)
        self.final_spec={'schema':1,'method':method,'postprocess_source_sha256':c.sha(Path(post.__file__)),
                         'packages':self.raw_spec['packages'],'post22':{'fractions':[.25,.10],'stable':True,'mean_size':5,'padding':'reflect'},
                         'soft15':{'sigma':[0,1,1],'mode':'reflect','truncate':4.0,'quantile':.85,'quantile_method':'linear','epsilon':1e-12}}
        self.stats={'raw_computed':0,'raw_disk_hits':0,'raw_memory_hits':0,'final_computed':0,'final_disk_hits':0,'final_memory_hits':0}
        self.memory_raw={};self.memory_final={}
        c.atomic_json(self.root/f'raw_config_{device}.json',self.raw_spec)
        c.atomic_json(self.root/f'final_config_{method}.json',self.final_spec)

    def identity(self,x,y):
        return {'spec':self.raw_spec,'input_shape':list(x.shape),'input_dtype':x.dtype.str,'input_sha256':c.arrsha(x),
                'target_shape':list(y.shape),'target_dtype':y.dtype.str,'target_sha256':c.arrsha(y),'target_values':y.tolist()}

    def path(self,layer,key): return self.root/layer/key[:2]/f'{key}.npz'

    def read_raw(self,path,identity,key):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'input','gradient','raw','logits','metadata'}: raise RuntimeError('Incomplete raw cache')
            arrays={k:np.ascontiguousarray(z[k]) for k in ('input','gradient','raw','logits')};m=json.loads(str(z['metadata'].item()))
        if m['key']!=key or m['identity']!=identity: raise RuntimeError('Raw cache identity differs')
        for k,a in arrays.items():
            if a.dtype!=np.float32 or not np.isfinite(a).all() or c.arrsha(a)!=m['hashes'][k]: raise RuntimeError(f'Invalid raw cache {k}')
        if any(arrays[k].shape!=(1,3,224,224) for k in ('input','gradient','raw')) or arrays['logits'].shape!=(1,5): raise RuntimeError('Raw shapes differ')
        b=m['record']['baseline']
        if float(arrays['input'].min())!=b or c.arrsha(arrays['input'])!=identity['input_sha256']: raise RuntimeError('Current input or baseline differs')
        if not np.array_equal((arrays['input']-np.float32(b))*arrays['gradient'],arrays['raw']): raise RuntimeError('Raw multiplication differs')
        return arrays,m

    def read_final(self,path,identity,key,raw):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item()));arrays={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
        if m['key']!=key or m['identity']!=identity: raise RuntimeError('Final cache identity differs')
        expected=post.process(raw,self.method)
        if set(expected)!=set(arrays): raise RuntimeError('Final cache fields differ')
        for k,a in arrays.items():
            if c.arrsha(a)!=m['hashes'][k] or not np.isfinite(a).all() or not np.array_equal(a,expected[k]): raise RuntimeError(f'Final cached formula differs: {k}')
        return arrays,m

    def get(self,model,x,y,deadline=None):
        if deadline is not None and time.monotonic()>=deadline: raise TimeoutError('Model deadline reached at callback boundary')
        x=np.ascontiguousarray(np.asarray(x));y=np.ascontiguousarray(np.asarray(y))
        identity=self.identity(x,y);rk=c.digest(identity);rp=self.path('raw',rk)
        if rk in self.memory_raw:
            arrays,meta=self.memory_raw[rk];self.stats['raw_memory_hits']+=1;raw_reuse='memory'
        elif rp.exists():
            arrays,meta=self.read_raw(rp,identity,rk);self.stats['raw_disk_hits']+=1;raw_reuse='disk'
        else:
            arrays,record=gdd.compute(model,x,y,device=self.device)
            meta={'key':rk,'identity':identity,'record':record,'hashes':{k:c.arrsha(a) for k,a in arrays.items()}}
            c.atomic_npz(rp,**arrays,metadata=np.array(json.dumps(meta,sort_keys=True)))
            self.read_raw(rp,identity,rk);self.stats['raw_computed']+=1;raw_reuse=None
        self.memory_raw[rk]=(arrays,meta)
        fi={'spec':self.final_spec,'raw_key':rk,'raw_sha256':meta['hashes']['raw']};fk=c.digest(fi);fp=self.path(self.method,fk)
        if fk in self.memory_final:
            fa,fm=self.memory_final[fk];self.stats['final_memory_hits']+=1;final_reuse='memory'
        elif fp.exists():
            fa,fm=self.read_final(fp,fi,fk,arrays['raw']);self.stats['final_disk_hits']+=1;final_reuse='disk'
        else:
            fa=post.process(arrays['raw'],self.method)
            fm={'key':fk,'identity':fi,'hashes':{k:c.arrsha(a) for k,a in fa.items()}}
            c.atomic_npz(fp,**fa,metadata=np.array(json.dumps(fm,sort_keys=True)))
            self.read_final(fp,fi,fk,arrays['raw']);self.stats['final_computed']+=1;final_reuse=None
        self.memory_final[fk]=(fa,fm)
        trace={'input_sha256':c.arrsha(x),'target_sha256':c.arrsha(y),'target_shape':list(y.shape),'target_dtype':y.dtype.str,
               'target':y.tolist(),'raw_key':rk,'raw_path':str(rp.resolve()),'raw_file_sha256':c.sha(rp),
               'raw_sha256':meta['hashes']['raw'],'gradient_sha256':meta['hashes']['gradient'],
               'final_key':fk,'final_path':str(fp.resolve()),'final_file_sha256':c.sha(fp),'final_sha256':fm['hashes']['final'],
               'raw_reuse':raw_reuse,'final_reuse':final_reuse,'model_gradient_computed':raw_reuse is None}
        return fa['final'],trace
