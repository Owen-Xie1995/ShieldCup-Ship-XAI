"""Independent local-PCG64 Stage25 sampling; never consumes global RNG."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def expected_indices():
    singles_rng=np.random.Generator(np.random.PCG64(20261001))
    single_indices=np.concatenate([singles_rng.choice(50176,128,replace=False)+c*50176 for c in range(3)]).astype(np.int64)
    subset_rng=np.random.Generator(np.random.PCG64(20261002))
    subset_indices=np.stack([subset_rng.choice(150528,50,replace=False) for _ in range(128)]).astype(np.int64)
    device_rng=np.random.Generator(np.random.PCG64(20261003))
    tile_ids=device_rng.choice(9408,128,replace=False).astype(np.int64)
    return {'single_indices':single_indices,'subset50_indices':subset_indices,'device_tile_ids':tile_ids}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compare',type=Path)
    p.add_argument('--single-field',default='single_indices');p.add_argument('--subset-field',default='subset50_indices')
    p.add_argument('--tile-field',default='device_tile_ids');args=p.parse_args()
    before=np.random.get_state();arrays=expected_indices();after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert all(np.unique(row).size==row.size for row in arrays['subset50_indices'])
    for channel in range(3):
        block=arrays['single_indices'][channel*128:(channel+1)*128]
        assert np.unique(block).size==128 and np.all(block//50176==channel)
    assert np.unique(arrays['device_tile_ids']).size==128
    path=ROOT/'audit/independent_frozen_indices.npz';np.savez_compressed(path,**arrays)
    report={'status':'PASS','model_forwards':0,'global_numpy_rng_unchanged':True,
        'rules':{'single':'PCG64(20261001), c order0..2 each128/50176 without replacement, CHW offset',
                 'subset50':'PCG64(20261002),128 sequential sets of50/150528 without replacement',
                 'device_tile':'PCG64(20261003),128/9408 tile IDs without replacement'},
        'arrays':{k:{'shape':list(a.shape),'dtype':a.dtype.str,'sha256':arrsha(a)} for k,a in arrays.items()},
        'file':str(path.relative_to(ROOT)),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if args.compare:
        mapping={'single_indices':args.single_field,'subset50_indices':args.subset_field,'device_tile_ids':args.tile_field}
        with np.load(args.compare,allow_pickle=False) as z:
            for k,field in mapping.items():
                assert z[field].dtype==arrays[k].dtype and np.array_equal(z[field],arrays[k]),(k,field)
        report['actual_frozen_indices_match']='byte_identical'
        report['actual_source']=str(args.compare)
        report['actual_source_file_sha256']=hashlib.sha256(args.compare.read_bytes()).hexdigest()
    out=ROOT/'audit/frozen_indices_independent.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
