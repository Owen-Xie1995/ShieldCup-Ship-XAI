"""Independent closed model ledger, entry estimates and frozen A/B gates."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    b=read(ROOT/'budget.json');assert b['active_session'] is None and b['limit_seconds']==10800 and b['phase_limits_seconds']=={'A':7200.,'B':3600.}
    assert b['model_seconds']<=10800 and b['phase_seconds']['A']<=7200 and b['phase_seconds']['B']<=3600
    assert abs(sum(b['phase_seconds'].values())-b['model_seconds'])<1e-7
    assert abs(sum(s['elapsed_seconds'] for s in b['sessions'])-b['model_seconds'])<1e-7
    for phase in ['A','B']:assert abs(sum(s['phase_elapsed_seconds'][phase] for s in b['sessions'])-b['phase_seconds'][phase])<1e-7
    assert abs(b['remaining_seconds']-max(10800-b['model_seconds'],0))<1e-9
    prereg=read(ROOT/'preregistered_method.json')
    for rel,h in {**prereg['program_sha256'],**prereg['frozen_sampling_files_sha256']}.items():assert sha(ROOT/rel)==h
    assert (ROOT/'preregistered_method.sha256').read_text().split()[0]==sha(ROOT/'preregistered_method.json')
    gates=[]
    for path in sorted((ROOT/'gates').glob('*.json')):
        g=read(path)
        if 'components_seconds' not in g:continue
        estimate=sum(float(v) for v in g['components_seconds'].values());assert abs(estimate-g['measured_estimate_seconds'])<1e-9
        assert g['safety_multiplier']==1.15 and abs(estimate*1.15-g['safe_estimate_seconds'])<1e-8
        if g['entry_allowed']:assert g['safe_estimate_seconds']<min(g['remaining_model_seconds'],g['remaining_phase_seconds'])
        else:assert g['safe_estimate_seconds']>=min(g['remaining_model_seconds'],g['remaining_phase_seconds'])-.02
        assert g['remaining_model_seconds']<=10800 and g['remaining_phase_seconds']<=(7200 if g['phase']=='A' else 3600)
        gates.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'phase':g['phase'],'safe_estimate_seconds':g['safe_estimate_seconds'],'entry_allowed':g['entry_allowed']})
    branches=read(ROOT/'branch_status.json');official=read(ROOT/'audit/official_runs_independent.json');result={'status':'PASS','model_forwards':0,'closed_cumulative_model_seconds':b['model_seconds'],'phase_seconds':b['phase_seconds'],'remaining_model_seconds':b['remaining_seconds'],'session_count':len(b['sessions']),'entry_cost_gates':gates,'A_confirmation':official['A_confirmation'],'B_screen':official['B_screen']}
    cp=ROOT/'confirmation_decision.json'
    if cp.exists():
        c=read(cp);assert (ROOT/'confirmation_decision.sha256').read_text().split()[0]==sha(cp)
        if official['A_confirmation']['complete']:
            a=official['A_confirmation'];baseline=next(r for r in official['runs'] if r['method']=='black_occlusion8_signed');candidate=next(r for r in official['runs'] if r['method']=='black_occlusion4_soft15')
            assert c['status']==a['status'] and c['candidate']=='black_occlusion4_soft15' and c['eval_seed']==314159 and c['device']=='mps'
            assert abs(c['baseline_mean']-baseline['formula_mean'])<1e-9 and abs(c['candidate_mean']-candidate['formula_mean'])<1e-9 and abs(c['paired_gain']-a['mean_delta'])<1e-9
            assert c['improved_images']==a['wins'] and c['seed42_historical_candidate']==82.65370613379287 and c['seed42_historical_baseline']==78.62195221134283
            assert c['meets90']==(candidate['formula_mean']>=90) and c['meets95']==(candidate['formula_mean']>=95) and c['not_generalization_or_official_qualification'] is True
            assert branches['A']['status']=='COMPLETED' and branches['A_8x8']['status']=='COMPLETED' and branches['A_4x4']['status']=='COMPLETED'
        else:assert c['status']=='SECOND_SEED_CONFIRMATION_PARTIAL_OR_NOT_ENTERED' and branches['A']['status'] in ['PARTIAL','NOT_STARTED']
        result['confirmation_decision_sha256']=sha(cp)
    else:assert branches['A']['status']!='COMPLETED'
    bp=ROOT/'single_image_decision.json'
    if bp.exists():
        s=read(bp);assert (ROOT/'single_image_decision.sha256').read_text().split()[0]==sha(bp)
        if official['B_screen']['complete_two_methods']:
            screen=official['B_screen'];assert s['status']==screen['status'] and s['selected_exploration_method']==screen['next_exploration_priority'] and s['best4_seed42_single_image_reference']==78.19467843751505
            assert s['image_count']==1 and s['Id']==1 and s['eval_seed']==42 and s['device']=='mps' and s['not_a_full9_score'] is True and s['does_not_replace_A_nine_image_candidate'] is True
            for r in official['runs']:
                if r['phase']=='B':
                    row=s['methods'][r['method']];assert abs(row['score']-r['formula_mean'])<1e-9 and abs(row['gain_vs_best4']-(r['formula_mean']-78.19467843751505))<1e-9 and row['passed']==(r['formula_mean']>78.19467843751505)
            assert branches['B']['status']=='COMPLETED' and branches['B_post22']['status']==branches['B_soft15']['status']=='COMPLETED'
        else:assert s['status'] in ['B_OFFICIAL_NOT_ENTERED_COST_GATE','B_SINGLE_IMAGE_PARTIAL'] and branches['B']['status'] in ['PARTIAL','NOT_STARTED']
        result['single_image_decision_sha256']=sha(bp)
    else:assert branches['B']['status']!='COMPLETED'
    for key in ['candidate_public5','2x2_full9','full_CPU_final_support','Windows_CUDA']:assert branches[key]['status']=='NOT_STARTED' and branches[key]['reason']
    assert not list((ROOT/'official9').glob('*.pending.csv')) and not list((ROOT/'single1').glob('*.pending.csv'))
    (ROOT/'audit/budget_decisions_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
