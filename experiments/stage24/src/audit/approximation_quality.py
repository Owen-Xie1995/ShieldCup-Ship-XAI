"""Post-score diagnostic: saved Taylor sums versus saved exact 8x8 tile interventions."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    run_path=ROOT/'official9/gdd_black_post22_mps_seed42.run.json'
    run=json.loads(run_path.read_text());gate=json.loads((ROOT/'audit/reference_reuse_gate.json').read_text())
    references=gate['all_nine_current_transformed_inputs_match_historical_cache']
    per_image=[];tiles=[]
    for item in references:
        index=item['Id']-1;trace=run['callback_traces'][index*8]
        assert trace['input_sha256']==item['input_sha256'] and trace['target']==[item['target']]
        raw_path=Path(trace['raw_path']);reference_path=ROOT.parent/'output_mac_stage23'/item['matching_historical_cache']
        with np.load(raw_path,allow_pickle=False) as z:
            a0=z['raw'][0];meta=json.loads(str(z['metadata'].item()));z_gdd=float(z['logits'][0,item['target']])
        with np.load(reference_path,allow_pickle=False) as z:
            masked=z['masked_logits'].astype(np.float64);ref=json.loads(str(z['metadata'].item()))
        assert ref['input_sha256']==item['input_sha256'] and ref['target']==item['target'] and ref['spec']['device']=='mps'
        expected=a0.astype(np.float64).reshape(3,28,8,28,8).sum(axis=(2,4)).reshape(-1)
        actual=float(ref['record']['original_target_logit'])-masked
        error=expected-actual
        assert expected.shape==actual.shape==(2352,) and np.isfinite(error).all()
        nonzero=np.maximum(np.abs(expected),np.abs(actual))>1e-6
        corr=float(np.corrcoef(expected,actual)[0,1])
        relative=float(np.linalg.norm(error)/max(np.linalg.norm(actual),1e-8))
        sign=float(np.mean(np.sign(expected[nonzero])==np.sign(actual[nonzero]))) if nonzero.any() else None
        row={'Id':item['Id'],'target':item['target'],'input_sha256':item['input_sha256'],'tiles':2352,
             'pearson_signed_tile_drop':corr,'relative_l2_vs_actual_drop':relative,
             'mean_abs_logit_error':float(np.mean(np.abs(error))),'max_abs_logit_error':float(np.max(np.abs(error))),
             'sign_agreement_excluding_both_near_zero':sign,'both_nontrivial_or_one_nontrivial_tiles':int(nonzero.sum()),
             'actual_drop_range':[float(actual.min()),float(actual.max())],'approximate_drop_range':[float(expected.min()),float(expected.max())],
             'gdd_vs_stage23_original_target_logit_difference':z_gdd-float(ref['record']['original_target_logit']),
             'gdd_raw_file':str(raw_path.relative_to(ROOT)),'gdd_raw_file_sha256':sha(raw_path),
             'stage23_finite_intervention_cache':str(reference_path),'stage23_cache_sha256':sha(reference_path)}
        per_image.append(row)
        for tile in range(2352):
            channel,within=divmod(tile,784);yy,xx=divmod(within,28)
            tiles.append({'Id':item['Id'],'target':item['target'],'channel':channel,'y_start':yy*8,'x_start':xx*8,
                          'a0_sum_64_elements':float(expected[tile]),'actual_black_tile_logit_drop':float(actual[tile]),
                          'approximation_error':float(error[tile])})
    out=ROOT/'audit'
    pd.DataFrame(tiles).to_csv(out/'approximation_quality_tiles.csv',index=False)
    result={'status':'PASS','model_forwards':0,'scope':'Post-score diagnostic using preserved original inputs only; no parameter selection or new candidate',
            'comparison':'sum of current signed a0 over each 8x8 single-channel tile versus saved exact replacement of those 64 elements by current CHW min',
            'limitations':['These are collective 64-element finite interventions; they do not prove or disprove exact individual-element quality.',
                           'Taylor a0 is an endpoint derivative times a finite baseline displacement; it is not a path integral or completeness result.',
                           'Pearson here is over all deterministic 8x8 tiles, not the official FaithfulnessCorrelation random 50-element subsets.',
                           'This analysis was chosen after scores and is diagnostic only.'],
            'per_image':per_image,'tiles_csv_sha256':sha(out/'approximation_quality_tiles.csv')}
    (out/'approximation_quality.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':result['status'],'per_image':[{k:r[k] for k in ['Id','pearson_signed_tile_drop','relative_l2_vs_actual_drop','sign_agreement_excluding_both_near_zero']} for r in per_image]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
