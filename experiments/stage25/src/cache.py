"""Separate content-verified 4x4 raw/final layers and atomic batch checkpoints."""
import json,time
from pathlib import Path
import numpy as np
import torch
import common as c
import occlusion4_raw as rawcode
import postprocess as post

class Cache:
    def __init__(self,root,device,method,clock,batch=64):
        self.root=Path(root);self.device=device;self.method=method;self.clock=clock;self.batch=batch
        self.stats={'raw_computed':0,'raw_disk_hits':0,'raw_memory_hits':0,'final_computed':0,'final_disk_hits':0,'final_memory_hits':0}
        self.memory_raw={};self.memory_final={};self.cold_seconds=[];self.warm_seconds=[];self.set_batch(batch)
        self.final_spec={'schema':1,'method':method,'postprocess_source_sha256':c.sha(Path(post.__file__)),
                         'packages':c.versions(),'post22':{'fractions':[.25,.10],'stable':True,'mean_size':5,'padding':'reflect'},
                         'soft15':{'sigma':[0,1,1],'mode':'reflect','truncate':4.,'quantile':.85,'quantile_method':'linear','epsilon':1e-12}}
        c.atomic_json(self.root/f'final_config_{method}.json',self.final_spec)
    def set_batch(self,batch):
        self.batch=batch;self.raw_spec=rawcode.specification(self.device,batch)
        c.atomic_json(self.root/f'raw_config_{self.device}_batch{batch}.json',self.raw_spec)
    def identity(self,x,y):
        return {'spec':self.raw_spec,'input_shape':list(x.shape),'input_dtype':x.dtype.str,'input_sha256':c.arrsha(x),
                'target_shape':list(y.shape),'target_dtype':y.dtype.str,'target_sha256':c.arrsha(y),'target_values':y.tolist()}
    def path(self,layer,key):return self.root/layer/key[:2]/f'{key}.npz'
    def read_raw(self,path,identity,key):
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'input','target','logits','masked_logits','raw','metadata'}:raise RuntimeError('Incomplete4rawfields')
            a={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'};m=json.loads(str(z['metadata'].item()))
        if m['key']!=key or m['identity']!=identity:raise RuntimeError('4rawidentitydiffers')
        expected={'input':(1,3,224,224),'logits':(1,5),'masked_logits':(9408,5),'raw':(1,3,224,224)}
        for k,value in a.items():
            if c.arrsha(value)!=m['hashes'][k] or not np.isfinite(value).all():raise RuntimeError(f'Invalid4rawhashfinite{k}')
            if k in expected and (value.shape!=expected[k] or value.dtype!=np.float32):raise RuntimeError(f'Invalid4rawshape/dtype{k}')
        if c.arrsha(a['input'])!=identity['input_sha256'] or c.arrsha(a['target'])!=identity['target_sha256']:raise RuntimeError('Rawinputtargetdiffers')
        if list(a['target'].shape)!=identity['target_shape'] or a['target'].dtype.str!=identity['target_dtype'] or a['target'].tolist()!=identity['target_values']:raise RuntimeError('Rawtargetshape/dtype/valuesdiffer')
        if float(a['input'].min())!=m['record']['baseline']:raise RuntimeError('Baseline changed')
        if not np.array_equal(rawcode.reconstruct(a['logits'],a['masked_logits'],int(a['target'].reshape(-1)[0])),a['raw']):raise RuntimeError('Bitexact4rawrebuildfailed')
        return a,m
    def read_final(self,path,identity,key,raw):
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(str(z['metadata'].item()));a={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
        if m['key']!=key or m['identity']!=identity:raise RuntimeError('Finalidentitychanged')
        expected=post.process(raw,self.method)
        if set(a)!=set(expected):raise RuntimeError('Finalfieldsdiffer')
        for k,value in a.items():
            if value.dtype!=expected[k].dtype or value.shape!=expected[k].shape or not np.isfinite(value).all() or c.arrsha(value)!=m['hashes'][k] or not np.array_equal(value,expected[k]):raise RuntimeError('Finalbitexactrebuilddiffers')
        return a,m
    def get_raw(self,model,x,y):
        self.clock.check();start=time.monotonic();x=np.ascontiguousarray(np.asarray(x));y=np.ascontiguousarray(np.asarray(y))
        identity=self.identity(x,y);key=c.digest(identity);path=self.path('raw',key);checkpoint=self.path('checkpoints',key)
        if key in self.memory_raw:
            # Memory hits receive the same content/identity/reconstruction gate.
            a,m=self.read_raw(path,identity,key);reuse='memory';self.stats['raw_memory_hits']+=1
        elif path.exists():a,m=self.read_raw(path,identity,key);reuse='disk';self.stats['raw_disk_hits']+=1
        else:
            try:a,record=rawcode.compute(model,x,y,self.device,self.batch,self.clock,checkpoint,key,identity)
            except RuntimeError as exc:
                if 'out of memory' not in str(exc).lower() or self.batch==16:raise
                nextbatch={64:32,32:16}[self.batch]
                c.atomic_json(self.root/'errors'/f'{key}.json',{'error':str(exc),'inference_batch':self.batch,'next_fixed_batch':nextbatch,
                    'identity':identity,'checkpoint':str(checkpoint),'budget_at_error_seconds':self.clock.spent()})
                if getattr(self.clock,'device_gate_passed',False):raise RuntimeError(f'OOM:batch {self.batch}→{nextbatch} requires renewed device validation before official continuation') from exc
                torch.mps.empty_cache();self.set_batch(nextbatch);return self.get_raw(model,x,y)
            m={'key':key,'identity':identity,'record':record,'hashes':{k:c.arrsha(value) for k,value in a.items()}}
            c.atomic_npz(path,**a,metadata=np.array(json.dumps(m,sort_keys=True)));self.read_raw(path,identity,key)
            checkpoint.unlink(missing_ok=True);reuse=None;self.stats['raw_computed']+=1;self.cold_seconds.append(time.monotonic()-start)
        self.memory_raw[key]=(None,m)
        if reuse:self.warm_seconds.append(time.monotonic()-start)
        self.clock.update(complete_raw_count=len(list((self.root/'raw').rglob('*.npz'))),cache_hits=self.stats['raw_disk_hits']+self.stats['raw_memory_hits'])
        return a,m,key,path,reuse
    def get(self,model,x,y):
        self.clock.check();a,m,key,path,reuse=self.get_raw(model,x,y)
        fi={'spec':self.final_spec,'raw_key':key,'raw_sha256':m['hashes']['raw']};fk=c.digest(fi);fp=self.path(self.method,fk)
        if fk in self.memory_final:fa,fm=self.read_final(fp,fi,fk,a['raw']);freuse='memory';self.stats['final_memory_hits']+=1
        elif fp.exists():fa,fm=self.read_final(fp,fi,fk,a['raw']);freuse='disk';self.stats['final_disk_hits']+=1
        else:
            fa=post.process(a['raw'],self.method);fm={'key':fk,'identity':fi,'hashes':{k:c.arrsha(v) for k,v in fa.items()}}
            c.atomic_npz(fp,**fa,metadata=np.array(json.dumps(fm,sort_keys=True)));self.read_final(fp,fi,fk,a['raw'])
            self.stats['final_computed']+=1;freuse=None
        self.memory_final[fk]=(None,fm)
        trace={'input_sha256':m['hashes']['input'],'input_shape':list(a['input'].shape),'input_dtype':a['input'].dtype.str,
               'target_sha256':m['hashes']['target'],'target_shape':list(a['target'].shape),'target_dtype':a['target'].dtype.str,'target':a['target'].tolist(),
               'raw_key':key,'raw_path':str(path.resolve()),'raw_file_sha256':c.sha(path),'raw_sha256':m['hashes']['raw'],
               'masked_logits_sha256':m['hashes']['masked_logits'],'original_logits_sha256':m['hashes']['logits'],
               'final_key':fk,'final_path':str(fp.resolve()),'final_file_sha256':c.sha(fp),'final_sha256':fm['hashes']['final'],
               'raw_reuse':reuse,'final_reuse':freuse,'model_forward_computed':reuse is None,'raw_compute_seconds':m['record']['compute_seconds'],
               'inference_batch':m['record']['inference_batch']}
        self.clock.check();return fa['final'],trace
