"""Independent analytic signed linear toy; no official classifier forward."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
    sys.path.insert(0,str(ROOT));core=load('stage26_raw_for_independent_toy',ROOT/'occlusion_raw.py');post=load('stage26_post_for_independent_toy',ROOT/'postprocess.py')
    x=((np.arange(3*224*224,dtype=np.float32).reshape(1,3,224,224)%2048)/1024-1).astype(np.float32);b=float(x.min());target=1
    coords=np.indices((3,224,224));w=((coords[0]*3+coords[1]+coords[2]*2)%7-3).astype(np.float64)*.001
    weights=np.stack([w*(i+1) for i in range(5)]);bias=np.array([0.,0.,0.,0.,2000.])
    def evaluate(a):return ((weights*a[0].astype(np.float64)[None]).sum((1,2,3))+bias).astype(np.float32)[None]
    original=evaluate(x);assert int(original[0].argmax())!=target
    arrays={'input':x,'target':np.asarray([target],np.int64),'original_logits':original};records=[]
    for tile,count in [(2,37632),(4,9408),(8,2352)]:
        side=224//tile;effects=((x[0].astype(np.float64)-b)[None]*weights).reshape(5,3,side,tile,side,tile).sum((3,5)).reshape(5,count).T
        masked=(original.astype(np.float64)-effects).astype(np.float32)
        vals=((np.float64(original[0,target])-masked[:,target].astype(np.float64))/(tile*tile)).astype(np.float32).reshape(3,side,side)
        expected=np.empty_like(x)
        for dy in range(tile):
            for dx in range(tile):expected[0,:,dy::tile,dx::tile]=vals
        actual=core.reconstruct(original,masked,target,tile)
        assert actual.dtype==expected.dtype and actual.shape==expected.shape and sha(actual)==sha(expected)
        assert (expected<0).any() and (expected>0).any()
        for i in range(count):
            channel=i//(side*side);within=i%(side*side);y=(within//side)*tile;xx=(within%side)*tile
            c,ys,xs=core.slices(i,tile);assert (c,ys.start,ys.stop,xs.start,xs.stop)==(channel,y,y+tile,xx,xx+tile)
        # Selected actual mask construction is compared bytewise with independently edited input copies.
        ids=np.asarray([0,side-1,side,side*side-1,side*side,count-1],np.int64)
        built=core.mask_batch(x,b,ids,tile);copies=[]
        for i in ids:
            a=x.copy();channel=int(i)//(side*side);within=int(i)%(side*side);y=within//side*tile;xx=within%side*tile;a[0,channel,y:y+tile,xx:xx+tile]=np.float32(b);copies.append(a[0])
        reference=np.stack(copies);assert built.dtype==reference.dtype and built.shape==reference.shape and sha(built)==sha(reference)
        # Actual replacement evaluation equals analytic output within floating summation tolerance.
        for j,i in enumerate(ids):assert np.allclose(evaluate(reference[j:j+1])[0],masked[i],rtol=1e-6,atol=1e-5)
        exact_single=(x.astype(np.float64)-b)*weights[target][None]
        unequal=np.any(np.abs(expected.astype(np.float64)-exact_single)>1e-5)
        assert unequal,'Averaged joint intervention must not be mislabeled exact per-element effects'
        arrays[f'masked_logits_tile{tile}']=masked;arrays[f'raw_tile{tile}']=expected
        records.append({'tile':tile,'tile_count':count,'area':tile*tile,'all_tile_IDs_verified':count,'original_argmax':int(original[0].argmax()),'supplied_target':target,'positive_and_negative_present':True,'core_raw_rebuild':'byte_identical','mask_batch':'byte_identical','joint_division_rule':f'float64 target drop/{tile*tile} then float32','not_exact_individual_effect_proven':bool(unequal),'raw_sha256':sha(expected)})
    # Apply the new postprocessing to genuine historical 4x4 inputs, comparing every field.
    prior=json.loads((ROOT/'audit/prior_sources.json').read_text());p=Path(prior['device_probe_sources'][0]['stage25_raw']['path'])
    with np.load(p,allow_pickle=False) as z:raw=z['raw'];meta=json.loads(str(z['metadata'].item()))
    stage25=ROOT.parent/'output_mac_stage25';bridge=[]
    for method in ['black_occlusion4_post22','black_occlusion4_soft15']:
        matches=[]
        for q in (stage25/'cache'/method).rglob('*.npz'):
            with np.load(q,allow_pickle=False) as z:
                fm=json.loads(str(z['metadata'].item()))
                if fm['identity']['raw_key']==meta['key']:matches.append((q,{k:z[k] for k in z.files if k!='metadata'}))
        assert len(matches)==1;ref,fields=matches[0];rebuilt=post.process(raw,method);assert set(rebuilt)==set(fields)
        for k,a in fields.items():assert rebuilt[k].dtype==a.dtype and rebuilt[k].shape==a.shape and sha(rebuilt[k])==sha(a),(method,k)
        bridge.append({'method':method,'raw_source':str(p),'final_source':str(ref),'raw_file_sha256':file_sha(p),'final_file_sha256':file_sha(ref),'fields_byte_identical':list(fields)})
    np.savez_compressed(ROOT/'audit/linear_toy.npz',**arrays)
    result={'status':'PASS','model_forwards':0,'official_classifier_forwards':0,'toy_only':True,'records':records,'postprocessing_bridge':bridge,'core_source_sha256':file_sha(ROOT/'occlusion_raw.py'),'postprocess_source_sha256':file_sha(ROOT/'postprocess.py'),'toy_npz_sha256':file_sha(ROOT/'audit/linear_toy.npz')}
    (ROOT/'audit/linear_toy.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
