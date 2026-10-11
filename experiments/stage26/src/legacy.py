"""Read-only historical bridge. Original source identities and bytes are preserved.

The Stage23 files contain only masked supplied-target logits. This reader never
fills the four missing columns or labels a bridge artifact as an old computation.
"""
import json, shutil
from pathlib import Path
import numpy as np
import common as c
import postprocess as post

ROOT=Path(__file__).resolve().parent
EXPECTED4={
 'occlusion4_raw.py':'cd7f21e620b92229e75dc1827d80cdd3ad341f134c18f39e3539db6c7671dcba',
 'common.py':'22c7ab24331e3982e77b266fad2e6d0b2a14fd26814ea5d9d71858aa6b9bde7d',
 'postprocess.py':'b51578f0f18e9de4211b005dbeea62405feb28f639d9ae5c6ca62b9984181fd6'}
EXPECTED8={'device/occlusion8_gpu.py':'525252f0b2a8e38f5dcd02e88d950a46f23e8bdbe9e678a408a664afbcee536a',
 'device/occlusion8_probe_frozen.py':'e4f498bf0335b4faa0c09602e5e12e4be4f5c33ce78f24777386f403b6e7295d'}

def exact(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return a.shape==b.shape and a.dtype==b.dtype and c.arrsha(a)==c.arrsha(b)

def rebuild_target(logits,masked,target,tile):
 side=224//tile
 values=((np.float64(logits[0,int(target)])-np.asarray(masked,dtype=np.float64))/(tile*tile)).astype(np.float32)
 return np.ascontiguousarray(np.repeat(np.repeat(values.reshape(3,side,side),tile,1),tile,2)[None])

def process(raw,method):
 alias='black_occlusion4_soft15' if method.endswith('soft15') else 'black_occlusion4_post22'
 return post.process(raw,alias)

def load_npz(path):
 with np.load(path,allow_pickle=False) as z:
  arrays={k:np.ascontiguousarray(z[k]) for k in z.files if k!='metadata'}
  metadata=json.loads(str(z['metadata'].item()))
 return arrays,metadata

def _source_record(src,dst,stage,relative):
 src=Path(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 source_hash=c.sha(src)
 if dst.exists():
  if c.sha(dst)!=source_hash:raise RuntimeError('Existing historical reference bytes changed')
 else:shutil.copyfile(src,dst)
 if c.sha(dst)!=source_hash:raise RuntimeError('Historical reference copy differs')
 return {'reuse_from_stage':stage,'original_path':str(src.resolve()),'original_relative':str(relative),
         'original_file_sha256':source_hash,'bytes':src.stat().st_size,
         'reference_relative':str(dst.relative_to(ROOT/'reference'))}

def _check_full_arrays(a,m,tile,x,y):
 expected={'input':((1,3,224,224),np.float32),'target':((1,),np.int64),
           'logits':((1,5),np.float32),'masked_logits':((3*(224//tile)**2,5),np.float32),
           'raw':((1,3,224,224),np.float32)}
 if set(a)!=set(expected):raise RuntimeError('Historical4 raw field set differs')
 for k,(shape,dtype) in expected.items():
  if a[k].shape!=shape or a[k].dtype!=dtype or not np.isfinite(a[k]).all() or c.arrsha(a[k])!=m['hashes'][k]:
   raise RuntimeError('Historical4 shape/dtype/hash/finite differs: '+k)
 if not exact(a['input'],x) or not exact(a['target'],y):raise RuntimeError('Historical4 input/target bytes differ')
 if not exact(rebuild_target(a['logits'],a['masked_logits'][:,int(y[0])],int(y[0]),tile),a['raw']):
  raise RuntimeError('Historical4 raw byte reconstruction differs')

def prepare_reference(root=None):
 """Prepare and verify the 9 original and 1 fixed-probe references per tile."""
 root=Path(root or ROOT/'reference')
 if root!=ROOT/'reference':raise ValueError('Stage26 canonical reference directory required')
 if (root/'inventory.json').exists():
  reader=Legacy(root)
  for record in reader.inventory['records']:reader.read(record)
  return reader.inventory
 old25=ROOT.parent/'output_mac_stage25';old23=ROOT.parent/'output_mac_stage23'
 source_records=[]
 for stage,base,expected in [(25,old25,EXPECTED4),(23,old23,EXPECTED8)]:
  for rel,h in expected.items():
   if c.sha(base/rel)!=h:raise RuntimeError('Historical program source changed '+rel)
   source_records.append(_source_record(base/rel,root/f'stage{stage}'/rel,stage,rel))
 for stage,base,rels in [(25,old25,['preregistered_method.json','environment.json','audit/probe_reference_sources.json',
       'official9/black_occlusion4_post22_mps_seed42_full9.callbacks.json',
       'official9/black_occlusion4_soft15_mps_seed42_full9.callbacks.json',
       'official9/black_occlusion4_post22_mps_seed42_full9.csv','official9/black_occlusion4_soft15_mps_seed42_full9.csv']),
       (23,old23,['device/device_environment.json','official9/black_occlusion8_signed_mps_seed42.run.json',
                 'official9/black_occlusion8_signed_mps_seed42.csv'])]:
  for rel in rels:source_records.append(_source_record(base/rel,root/f'stage{stage}'/rel,stage,rel))
 traces={method:json.loads((old25/f'official9/{method}_mps_seed42_full9.callbacks.json').read_text())['traces']
         for method in ('black_occlusion4_post22','black_occlusion4_soft15')}
 originals={t['Id']:t for t in traces['black_occlusion4_soft15'] if t['input_kind']=='original'}
 if sorted(originals)!=list(range(1,10)):raise RuntimeError('Nine historical original references missing')
 full4=[]
 for Id,t in sorted(originals.items()):
  p=Path(t['raw_path']);a,m=load_npz(p)
  if c.sha(p)!=t['raw_file_sha256']:raise RuntimeError('Historical callback raw file SHA differs')
  full4.append((f'original_Id{Id}',Id,p,a,m,t))
 with np.load(old25/'probes/fixed_inputs.npz',allow_pickle=False) as z:
  fixed={k:np.ascontiguousarray(z[k]) for k in z.files}
 perturb=np.ascontiguousarray(fixed['perturbed']);target=np.asarray([0],dtype=np.int64)
 for p in (old25/'cache/raw').rglob('*.npz'):
  a,m=load_npz(p)
  if exact(a['input'],perturb) and exact(a['target'],target):
   full4.append(('fixed_perturbed',None,p,a,m,None));break
 else:raise RuntimeError('Historical4 fixed perturbation complete raw missing')
 records=[]
 for name,Id,p,a,m,t in full4:
  rel=p.relative_to(old25)
  rec=_source_record(p,root/'stage25'/rel,25,rel)
  rec.update({'name':name,'Id':Id,'tile':4,'batch':64,'device':'mps','precision':'float32',
    'input_shape':list(a['input'].shape),'input_dtype':a['input'].dtype.str,'input_sha256':c.arrsha(a['input']),
    'target_shape':list(a['target'].shape),'target_dtype':a['target'].dtype.str,'target_sha256':c.arrsha(a['target']),
    'target_values':a['target'].tolist(),'original_source_sha256':EXPECTED4['occlusion4_raw.py'],
    'original_common_sha256':EXPECTED4['common.py'],'packages':m['identity']['spec']['packages'],
    'original_identity':m['identity'],'original_key':m['key'],'final_sources':[]})
  for method,rows in traces.items():
   match=next((v for v in rows if v['input_sha256']==rec['input_sha256'] and v['target_sha256']==rec['target_sha256']),None)
   if match:
    fp=Path(match['final_path'])
    if c.sha(fp)!=match['final_file_sha256']:raise RuntimeError('Historical final source SHA differs')
    fr=_source_record(fp,root/'stage25'/fp.relative_to(old25),25,fp.relative_to(old25));fr['method']=method
    rec['final_sources'].append(fr)
  records.append(rec)
  old8root=old23/'device/cache/black_occlusion8_signed/mps'
  for op in old8root.rglob('*.npz'):
   oa,om=load_npz(op)
   if om['input_sha256']==rec['input_sha256'] and om['target']==int(a['target'][0]) and om['spec']['batch_size']==(32 if name=='fixed_perturbed' else 64):break
  else:raise RuntimeError('Historical8 same-input complete reference missing '+name)
  orc=_source_record(op,root/'stage23'/op.relative_to(old23),23,op.relative_to(old23))
  input_path=root/'bound_inputs'/f'{rec["input_sha256"]}.npz'
  if not input_path.exists():c.atomic_npz(input_path,input=a['input'],target=a['target'])
  orc.update({k:rec[k] for k in ('name','Id','input_shape','input_dtype','input_sha256','target_shape','target_dtype','target_sha256','target_values','packages')})
  orc.update({'tile':8,'batch':om['spec']['batch_size'],'device':'mps','precision':'float32',
             'input_reference_relative':str(input_path.relative_to(root)),'input_reference_file_sha256':c.sha(input_path),
             'original_source_sha256':om['spec']['program_sha256'],'original_identity':{
                 'spec':om['spec'],'input_shape':[3,224,224],'input_dtype':a['input'].dtype.str,
                 'input_sha256':c.arrsha(a['input']),'target_dtype':a['target'].dtype.str,'target':int(a['target'][0])},
             'original_key':om['key'],'final_sources':[]})
  records.append(orc)
 inventory={'status':'PASS','schema':1,'historical_sources_read_only':True,'model_forwards':0,
            'packages':c.versions(),'reader_source_sha256':c.sha(Path(__file__)),
            'source_programs_and_evidence':source_records,'records':records,
            'limitations':['Stage23 masked logits retain only supplied-target column; no missing classes are invented.',
                          'Stage23 input bytes are bound by original SHA and cache key plus copied Stage25 complete same-input evidence.',
                          'Fixed Stage23 perturbation batch32 is probe evidence and is not eligible for batch64 formal reuse.']}
 c.atomic_json(root/'inventory.json',inventory)
 reader=Legacy(root)
 for record in records:reader.read(record)
 return inventory

class Legacy:
 def __init__(self,root=None):
  self.root=Path(root or ROOT/'reference')
  self.inventory=json.loads((self.root/'inventory.json').read_text())
  if self.inventory['status']!='PASS' or self.inventory['packages']!=c.versions():raise RuntimeError('Legacy package/reference gate differs')
  for item in self.inventory['source_programs_and_evidence']:
   if c.sha(self.root/item['reference_relative'])!=item['original_file_sha256']:raise RuntimeError('Legacy frozen source/evidence bytes changed')
  env=json.loads((self.root/'stage23/device/device_environment.json').read_text())
  if any(env['packages'].get(k)!=v for k,v in c.versions().items()):raise RuntimeError('Historical Stage23 package evidence differs')
  self.lookup={self.signature_record(r):r for r in self.inventory['records']}
 def signature_record(self,r):
  return (r['tile'],r['batch'],tuple(r['input_shape']),r['input_dtype'],r['input_sha256'],
          tuple(r['target_shape']),r['target_dtype'],r['target_sha256'],tuple(r['target_values']))
 def find(self,x,y,tile,batch):
  x=np.ascontiguousarray(x);y=np.ascontiguousarray(y)
  signature=(tile,batch,x.shape,x.dtype.str,c.arrsha(x),y.shape,y.dtype.str,c.arrsha(y),tuple(y.tolist()))
  rec=self.lookup.get(signature)
  return None if rec is None else self.read(rec,x,y)
 def read(self,r,x=None,y=None):
  path=self.root/r['reference_relative']
  if c.sha(path)!=r['original_file_sha256']:raise RuntimeError('Legacy cache file bytes changed')
  old,om=load_npz(path)
  tile=r['tile']
  if tile==4:
   a=old;x=a['input'] if x is None else x;y=a['target'] if y is None else y
   spec=om['identity']['spec']
   for rel,h in EXPECTED4.items():
    if c.sha(self.root/'stage25'/rel)!=h:raise RuntimeError('Historical4 source bytes changed on cache hit')
   if spec['raw_source_sha256']!=EXPECTED4['occlusion4_raw.py'] or spec['common_source_sha256']!=EXPECTED4['common.py'] or spec['packages']!=c.versions():raise RuntimeError('Historical4 algorithm/package differs')
   if spec['device']!='mps' or spec['precision']!='float32' or spec['inference_batch']!=r['batch'] or spec['tile']!=4 or spec['stride']!=4 or spec['tile_count']!=9408 or spec['tile_area']!=16 or spec['weights_sha256']!=c.WEIGHT_SHA:raise RuntimeError('Historical4 device/weight/tile/batch differs')
   if om['identity']!=r['original_identity'] or c.digest(om['identity'])!=om['key'] or om['key']!=r['original_key']:raise RuntimeError('Historical4 original identity differs')
   _check_full_arrays(a,om,4,x,y)
   if om['record']['baseline']!=float(x.min()) or om['record']['target']!=int(y[0]) or om['record']['device']!='mps' or om['record']['inference_batch']!=r['batch'] or om['record']['precision']!='float32':raise RuntimeError('Historical4 compute record differs')
   for f in r['final_sources']:
    fp=self.root/f['reference_relative']
    if c.sha(fp)!=f['original_file_sha256']:raise RuntimeError('Historical4 final file differs')
    fa,fm=load_npz(fp);rebuilt=process(a['raw'],f['method'])
    if fm['identity']['raw_key']!=om['key'] or fm['identity']['raw_sha256']!=c.arrsha(a['raw']) or fm['identity']['spec']['postprocess_source_sha256']!=EXPECTED4['postprocess.py'] or c.digest(fm['identity'])!=fm['key']:raise RuntimeError('Historical4 final identity differs')
    if set(fa)!=set(rebuilt):raise RuntimeError('Historical4 final fields differ')
    for k,v in fa.items():
     if not np.isfinite(v).all() or c.arrsha(v)!=fm['hashes'][k] or not exact(v,rebuilt[k]):raise RuntimeError('Historical4 final/intermediate byte reconstruction differs '+k)
   duration=om['record']['compute_seconds']
  elif tile==8:
   ip=self.root/r['input_reference_relative']
   if c.sha(ip)!=r['input_reference_file_sha256']:raise RuntimeError('Legacy8 bound input file differs')
   with np.load(ip,allow_pickle=False) as z:boundx=np.ascontiguousarray(z['input']);boundy=np.ascontiguousarray(z['target'])
   x=boundx if x is None else x;y=boundy if y is None else y
   if not exact(x,boundx) or not exact(y,boundy):raise RuntimeError('Legacy8 bound current input/target differs')
   spec=om['spec'];record=om['record']
   source_rel=next((rel for rel,h in EXPECTED8.items() if h==spec['program_sha256']),None)
   if source_rel is None or c.sha(self.root/'stage23'/source_rel)!=spec['program_sha256']:raise RuntimeError('Historical8 source bytes changed on cache hit')
   if spec['program_sha256'] not in EXPECTED8.values() or spec['program_sha256']!=r['original_source_sha256'] or spec['device']!='mps' or spec['precision']!='float32' or spec['batch_size']!=r['batch'] or spec['tile_size']!=8 or spec['tile_count']!=2352 or spec['weight_sha256']!=c.WEIGHT_SHA or spec['torch']!=c.versions()['torch'] or spec['numpy']!=c.versions()['numpy']:raise RuntimeError('Historical8 algorithm/package/device/tile/batch differs')
   if c.digest(r['original_identity'])!=om['key'] or om['key']!=r['original_key'] or om['input_sha256']!=c.arrsha(x) or om['target']!=int(y[0]):raise RuntimeError('Historical8 original identity differs')
   if set(old)!={'raw','final','masked_logits'}:raise RuntimeError('Historical8 fields differ')
   for k,shape in [('raw',(1,3,224,224)),('final',(1,3,224,224)),('masked_logits',(2352,))]:
    v=old[k]
    if v.dtype!=np.float32 or v.shape!=shape or not np.isfinite(v).all() or c.arrsha(v)!=om[k+'_sha256']:raise RuntimeError('Historical8 array shape/dtype/hash/finite differs '+k)
   logits=np.asarray(record['original_logits'],dtype=np.float32).reshape(1,5)
   if not np.isfinite(logits).all() or record['target']!=int(y[0]) or record['mask_count']!=2352 or record['input_sha256']!=c.arrsha(x) or record['original_target_logit']!=float(logits[0,int(y[0])]) or record['predicted_class']!=int(logits.argmax(1)[0]) or record['baseline']!=float(x.min()) or record['device']!='mps' or record['precision']!='float32' or record['batch_size']!=r['batch']:raise RuntimeError('Historical8 original logits/compute record differs')
   if not exact(rebuild_target(logits,old['masked_logits'],int(y[0]),8),old['raw']) or not exact(process(old['raw'],'black_occlusion8_signed')['final'],old['final']):raise RuntimeError('Historical8 raw/final byte reconstruction differs')
   a={'input':x,'target':y,'logits':logits,'masked_target_logits':old['masked_logits'],'raw':old['raw']}
   duration=record['wall_seconds']
  else:raise ValueError(tile)
  for k,val in [('input',x),('target',y)]:
   if list(val.shape)!=r[k+'_shape'] or val.dtype.str!=r[k+'_dtype'] or c.arrsha(val)!=r[k+'_sha256']:raise RuntimeError('Legacy full input/target binding differs')
  meta={'legacy':dict(r,original_metadata=om),'hashes':{k:c.arrsha(v) for k,v in a.items()},
        'record':{'device':'mps','precision':'float32','baseline':float(x.min()),'target':int(y[0]),
          'inference_batch':r['batch'],'masks_completed':3*(224//tile)**2,'masks_planned':3*(224//tile)**2,
          'compute_seconds':duration,'legacy_compute_seconds':duration,'this_session_compute_seconds':0.0,
          'predicted_class':int(a['logits'].argmax(1)[0])}}
  return a,meta,path

if __name__=='__main__':
 inventory=prepare_reference();print(json.dumps({'status':inventory['status'],'records':len(inventory['records']),'model_forwards':0}))
