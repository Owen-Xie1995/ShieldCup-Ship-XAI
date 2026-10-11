"""Regenerate Stage26 probes via independent local PCG64; no model/RNG change."""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    state=np.random.get_state();expected={f'tile{tile}_ids128':np.random.Generator(np.random.PCG64(seed)).choice(count,128,replace=False).astype(np.int64) for tile,seed,count in [(2,20261004,37632),(4,20261005,9408),(8,20261006,2352)]}
    np.savez_compressed(ROOT/'audit/independent_frozen_indices.npz',**expected)
    with np.load(ROOT/'probes/frozen_indices.npz',allow_pickle=False) as z:
        assert set(z.files)==set(expected)
        records={}
        for key,a in expected.items():
            saved=z[key];assert a.dtype==saved.dtype and a.shape==saved.shape and sha(a)==sha(saved)
            assert len(set(map(int,a)))==128
            records[key]={'shape':list(a.shape),'dtype':a.dtype.str,'array_sha256':sha(a),'regeneration':'byte_identical','without_replacement':True}
    after=np.random.get_state();assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    result={'status':'PASS','model_forwards':0,'global_numpy_rng_preserved':True,'generator':'numpy.Generator(PCG64)','seeds':{'tile2':20261004,'tile4':20261005,'tile8':20261006},'records':records,'frozen_file_sha256':hashlib.sha256((ROOT/'probes/frozen_indices.npz').read_bytes()).hexdigest()}
    (ROOT/'audit/frozen_indices_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
