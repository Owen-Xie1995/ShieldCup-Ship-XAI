"""Separate tile-specific raw/final caches, with a provenance-preserving bridge."""
import json,time
from pathlib import Path
import numpy as np
import common as c
import occlusion_raw as rawcode
import postprocess as post
import legacy

METHOD_TILES={'black_occlusion8_signed':8,'black_occlusion4_soft15':4,
              'black_occlusion2_post22':2,'black_occlusion2_soft15':2}

class Cache:
 def __init__(self,root,device,method,clock,batch=64):
  if method not in METHOD_TILES:raise ValueError(method)
  self.root=Path(root);self.device=device;self.method=method;self.clock=clock;self.tile=METHOD_TILES[method]
  self.stats={'raw_computed':0,'raw_disk_hits':0,'raw_memory_hits':0,'raw_legacy_hits':0,
              'final_computed':0,'final_disk_hits':0,'final_memory_hits':0}
  self.memory_raw={};self.memory_final={};self.cold_seconds=[];self.warm_seconds=[]
  self.legacy=legacy.Legacy(c.ROOT/'reference') if device=='mps' and self.tile in (4,8) else None
  self.set_batch(batch)
  self.final_spec={'schema':2,'method':method,'tile':self.tile,'postprocess_source_sha256':c.sha(Path(post.__file__)),
                   'postprocess_alias':'black_occlusion4_soft15' if method.endswith('soft15') else 'black_occlusion4_post22',
                   'packages':c.versions(),'post22':{'fractions':[.25,.10],'stable':True,'mean_size':5,'padding':'reflect'},
                   'soft15':{'sigma':[0,1,1],'mode':'reflect','truncate':4.,'quantile':.85,'quantile_method':'linear','epsilon':1e-12}}
  c.atomic_json(self.root/f'final_config_{method}.json',self.final_spec)
 def set_batch(self,batch):
  if batch not in (64,32,16):raise ValueError('Only frozen batch64→32→16 allowed')
  self.batch=batch;self.raw_spec=rawcode.specification(self.device,batch,self.tile)
  self.raw_spec['wrapper_source_sha256']=c.sha(Path(__file__))
  self.raw_spec['legacy_reader_source_sha256']=c.sha(Path(legacy.__file__))
  c.atomic_json(self.root/f'raw_config_tile{self.tile}_{self.device}_batch{batch}.json',self.raw_spec)
 def identity(self,x,y,bridge=None):
  identity={'spec':self.raw_spec,'input_shape':list(x.shape),'input_dtype':x.dtype.str,'input_sha256':c.arrsha(x),
            'target_shape':list(y.shape),'target_dtype':y.dtype.str,'target_sha256':c.arrsha(y),'target_values':y.tolist(),
            'raw_origin':'native_stage26' if bridge is None else 'historical_compatibility_bridge'}
  if bridge is not None:
   old=bridge[1]['legacy']
   identity['legacy_binding']={k:old[k] for k in ('reuse_from_stage','original_file_sha256','original_key','original_source_sha256','tile','batch','device','precision')}
   identity['legacy_binding']['inventory_sha256']=c.sha(c.ROOT/'reference/inventory.json')
  return identity
 def path(self,layer,key):return self.root/layer/key[:2]/f'{key}.npz'
 def _bridge(self,x,y):return None if self.legacy is None else self.legacy.find(x,y,self.tile,self.batch)
 def read_raw(self,path,identity,key,bridge=None):
  with np.load(path,allow_pickle=False) as z:
   a={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'};m=json.loads(str(z['metadata'].item()))
  if m['key']!=key or m['identity']!=identity or c.digest(identity)!=key:raise RuntimeError('Raw cache identity differs')
  masked_field='masked_target_logits' if 'masked_target_logits' in a else 'masked_logits'
  expected={'input':((1,3,224,224),np.float32),'target':(tuple(identity['target_shape']),np.dtype(identity['target_dtype'])),
            'logits':((1,5),np.float32),'raw':((1,3,224,224),np.float32),
            masked_field:((3*(224//self.tile)**2,) if masked_field=='masked_target_logits' else (3*(224//self.tile)**2,5),np.float32)}
  if set(a)!=set(expected) or set(m['hashes'])!=set(a):raise RuntimeError('Raw cache fields differ')
  for k,value in a.items():
   shape,dtype=expected[k]
   if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all() or c.arrsha(value)!=m['hashes'][k]:raise RuntimeError('Raw shape/dtype/hash/finite differs '+k)
  if c.arrsha(a['input'])!=identity['input_sha256'] or c.arrsha(a['target'])!=identity['target_sha256'] or a['target'].tolist()!=identity['target_values']:raise RuntimeError('Raw full input/target bytes differ')
  record=m['record']
  if record['baseline']!=float(a['input'].min()) or record['target']!=int(a['target'][0]) or record['device']!=self.device or record['precision']!='float32' or record['inference_batch']!=self.batch or record['masks_completed']!=3*(224//self.tile)**2:raise RuntimeError('Raw model/baseline/target/tile/batch differs')
  if masked_field=='masked_target_logits':
   if self.tile!=8 or bridge is None:raise RuntimeError('Masked-target-only arrays require authentic Stage23 bridge')
   rebuilt=rawcode.reconstruct_target(a['logits'],a[masked_field],int(a['target'][0]),self.tile)
  else:rebuilt=rawcode.reconstruct(a['logits'],a[masked_field],int(a['target'][0]),self.tile)
  if not legacy.exact(rebuilt,a['raw']):raise RuntimeError('Raw signed byte reconstruction differs')
  if identity['raw_origin']=='historical_compatibility_bridge':
   if bridge is None or m.get('legacy')!=bridge[1]['legacy']:raise RuntimeError('Historical origin provenance changed')
   for k,v in bridge[0].items():
    if not legacy.exact(v,a[k]):raise RuntimeError('Bridge changed historical numerical bytes '+k)
  elif 'legacy' in m:raise RuntimeError('Native raw mislabeled historical')
  return a,m
 def read_final(self,path,identity,key,raw):
  with np.load(path,allow_pickle=False) as z:
   m=json.loads(str(z['metadata'].item()));a={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
  if m['key']!=key or m['identity']!=identity or c.digest(identity)!=key:raise RuntimeError('Final cache identity differs')
  expected=legacy.process(raw,self.method)
  if set(a)!=set(expected) or set(m['hashes'])!=set(a):raise RuntimeError('Final fields differ')
  for k,value in a.items():
   if not np.isfinite(value).all() or c.arrsha(value)!=m['hashes'][k] or not legacy.exact(value,expected[k]):raise RuntimeError('Final/intermediate byte reconstruction differs '+k)
  return a,m
 def get_raw(self,model,x,y):
  self.clock.check();start=time.monotonic();x=np.ascontiguousarray(np.asarray(x));y=np.ascontiguousarray(np.asarray(y))
  if x.shape!=(1,3,224,224) or x.dtype!=np.float32 or y.shape!=(1,) or y.dtype!=np.int64 or not np.isfinite(x).all() or not 0<=int(y[0])<=4:raise ValueError('Exact finite FP32 NCHW/int64 supplied target required')
  bridge=self._bridge(x,y);identity=self.identity(x,y,bridge);key=c.digest(identity)
  path=self.path('raw',key);checkpoint=self.path('checkpoints',key)
  if key in self.memory_raw:
   a,m=self.read_raw(path,identity,key,bridge);reuse='memory';self.stats['raw_memory_hits']+=1
  elif path.exists():
   a,m=self.read_raw(path,identity,key,bridge);reuse='disk';self.stats['raw_disk_hits']+=1
  elif bridge is not None:
   a,source,_=bridge;m={'key':key,'identity':identity,'record':source['record'],'hashes':source['hashes'],'legacy':source['legacy'],
                      'wrapper_source_sha256':c.sha(Path(__file__)),'legacy_reader_source_sha256':c.sha(Path(legacy.__file__))}
   c.atomic_npz(path,**a,metadata=np.array(json.dumps(m,sort_keys=True)));self.read_raw(path,identity,key,bridge)
   reuse='legacy';self.stats['raw_legacy_hits']+=1
  else:
   if hasattr(self.clock,'cold_guard'):self.clock.cold_guard(self)
   try:a,record=rawcode.compute(model,x,y,self.device,self.batch,self.clock,checkpoint,key,identity,self.tile)
   except RuntimeError as exc:
    if 'out of memory' in str(exc).lower():
     nextbatch={64:32,32:16}.get(self.batch)
     c.atomic_json(self.root/'errors'/f'{key}.json',{'error':str(exc),'inference_batch':self.batch,'next_fixed_batch':nextbatch,
       'identity':identity,'checkpoint':str(checkpoint),'budget_at_error_seconds':self.clock.spent(),
       'status':'RENEWED_DEVICE_GATE_REQUIRED' if nextbatch else 'FAILED_MINIMUM_BATCH'})
     raise RuntimeError(f'Out of memory: batch {self.batch}→{nextbatch} requires renewed device validation before changing batch') from exc
    raise
   m={'key':key,'identity':identity,'record':record,'hashes':{k:c.arrsha(value) for k,value in a.items()}}
   c.atomic_npz(path,**a,metadata=np.array(json.dumps(m,sort_keys=True)));self.read_raw(path,identity,key)
   checkpoint.unlink(missing_ok=True);reuse=None;self.stats['raw_computed']+=1;self.cold_seconds.append(time.monotonic()-start)
  self.memory_raw[key]=(None,m)
  if reuse:self.warm_seconds.append(time.monotonic()-start)
  self.clock.update(complete_raw_count=len(list((self.root/'raw').rglob('*.npz'))),
                    cache_hits=self.stats['raw_disk_hits']+self.stats['raw_memory_hits']+self.stats['raw_legacy_hits'])
  self.clock.check();return a,m,key,path,reuse
 def get(self,model,x,y):
  self.clock.check();a,m,key,path,reuse=self.get_raw(model,x,y)
  fi={'spec':self.final_spec,'raw_key':key,'raw_sha256':m['hashes']['raw']};fk=c.digest(fi);fp=self.path(self.method,fk)
  if fk in self.memory_final:fa,fm=self.read_final(fp,fi,fk,a['raw']);freuse='memory';self.stats['final_memory_hits']+=1
  elif fp.exists():fa,fm=self.read_final(fp,fi,fk,a['raw']);freuse='disk';self.stats['final_disk_hits']+=1
  else:
   fa=legacy.process(a['raw'],self.method);fm={'key':fk,'identity':fi,'hashes':{k:c.arrsha(v) for k,v in fa.items()}}
   c.atomic_npz(fp,**fa,metadata=np.array(json.dumps(fm,sort_keys=True)));self.read_final(fp,fi,fk,a['raw'])
   self.stats['final_computed']+=1;freuse=None
  self.memory_final[fk]=(None,fm)
  masked_field='masked_target_logits' if 'masked_target_logits' in a else 'masked_logits'
  trace={'input_sha256':m['hashes']['input'],'input_shape':list(a['input'].shape),'input_dtype':a['input'].dtype.str,
   'target_sha256':m['hashes']['target'],'target_shape':list(a['target'].shape),'target_dtype':a['target'].dtype.str,'target':a['target'].tolist(),
   'raw_key':key,'raw_path':str(path.resolve()),'raw_file_sha256':c.sha(path),'raw_sha256':m['hashes']['raw'],
   'masked_logits_sha256':m['hashes'][masked_field],'masked_logits_field':masked_field,'masked_logits_shape':list(a[masked_field].shape),
   'original_logits_sha256':m['hashes']['logits'],'final_key':fk,'final_path':str(fp.resolve()),
   'final_file_sha256':c.sha(fp),'final_sha256':fm['hashes']['final'],'raw_reuse':reuse,'final_reuse':freuse,
   'model_forward_computed':reuse is None,'raw_compute_seconds':m['record']['compute_seconds'],
   'raw_compute_seconds_scope':'historical' if 'legacy' in m else 'stage26',
   'this_run_raw_compute_seconds':m['record'].get('this_session_compute_seconds',0.) if reuse is None else 0.,
   'inference_batch':m['record']['inference_batch'],'device':self.device,'precision':'float32','tile':self.tile,
   'wrapper_source_sha256':c.sha(Path(__file__)),'raw_source_identity':m['identity']}
  if 'legacy' in m:trace['legacy_source']=m['legacy']
  self.clock.check();return fa['final'],trace
