"""Independent persisted cumulative model timing, cost entry and fixed decisions."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=['black_occlusion4_post22','black_occlusion4_soft15']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    budget=json.loads((ROOT/'budget.json').read_text());branches=json.loads((ROOT/'branch_status.json').read_text())
    prereg_path=ROOT/'preregistered_method.json';prereg=json.loads(prereg_path.read_text())
    assert budget['limit_seconds']==prereg['budget']['model_seconds_total']==5400
    assert budget['A_limit_seconds']==prereg['budget']['A_included_seconds']==600
    assert budget['active_session'] is None,'Await final model ledger before audit'
    assert np.isclose(sum(s['elapsed_seconds'] for s in budget['sessions']),budget['model_seconds'],rtol=0,atol=1e-9)
    assert np.isclose(sum(s['A_elapsed_seconds'] for s in budget['sessions']),budget['A_seconds'],rtol=0,atol=1e-9)
    assert np.isclose(budget['remaining_seconds'],max(5400-budget['model_seconds'],0),rtol=0,atol=1e-9)
    assert budget['model_seconds']<=5400 and budget['A_seconds']<=600
    gates=[]
    for path in sorted((ROOT/'gates').glob('*.json')):
        g=json.loads(path.read_text())
        if 'components_seconds' not in g:continue
        assert g['safety_multiplier']==1.15
        assert np.isclose(g['measured_estimate_seconds'],sum(g['components_seconds'].values()),rtol=0,atol=1e-9)
        assert np.isclose(g['safe_estimate_seconds'],g['measured_estimate_seconds']*1.15,rtol=0,atol=1e-9)
        assert g['entry_allowed']==(g['safe_estimate_seconds']<g['remaining_model_seconds'])
        assert abs(g['remaining_model_seconds']+g['cumulative_model_seconds']-5400)<.1
        assert g['cumulative_model_seconds']<=budget['model_seconds']
        gates.append({'path':str(path.relative_to(ROOT)),'file_sha256':sha(path),**g})
    decision_path=ROOT/'candidate_decision.json';screen_path=ROOT/'screen_decision.json'
    comparisons=json.loads((ROOT/'audit/official_runs_independent.json').read_text())['runs']
    independent={(r['method'],r['sample_count']):r for r in comparisons}
    screen=None;decision=None
    if screen_path.exists():
        screen=json.loads(screen_path.read_text())
        assert screen['status']=='FROZEN' and screen['preregistered_method_sha256']==sha(prereg_path)
        assert (ROOT/'screen_decision.sha256').read_text().split()[0]==sha(screen_path)
        qualified=[]
        for method in METHODS:
            evidence=independent[(method,3)];saved=screen['methods'][method]
            assert saved['passed']==evidence['gate_pass'] and saved['improved_count']==evidence['improved_images']
            assert abs(saved['prefix3_mean']-evidence['official_score'])<1e-9 and saved['csv_sha256']==evidence['csv_sha256']
            if evidence['gate_pass']:qualified.append(method)
        assert qualified==screen['qualified_methods']
        for g in gates:
            if g['phase'].startswith('C_'):
                method=g['phase'].removeprefix('C_full9_');assert method in qualified
                assert g['cumulative_model_seconds']>=screen['cumulative_model_seconds']
        decision=json.loads(decision_path.read_text());assert decision['screen_decision_sha256']==sha(screen_path)
        full=[independent[(m,9)] for m in METHODS if (m,9) in independent]
        if not qualified:
            assert not full and decision['status']=='FULL9_NOT_ENTERED_SCREEN_FAIL' and decision['candidate'] is None
            assert branches['C']['status']=='NOT_STARTED' and branches['C']['reason']=='FULL9_NOT_ENTERED_SCREEN_FAIL'
        else:
            for row in full:assert row['method'] in qualified
            gains=[r for r in full if r['official_score']>78.621952211343 and r['improved_images']>=6]
            selected=max(gains,key=lambda r:(r['official_score'],r['method']==METHODS[0]))['method'] if gains else None
            assert decision['candidate']==selected
            assert decision['status']==('GAIN_SEED42_ONLY_CONFIRMATION_PENDING' if selected else ('FULL9_NO_QUALIFIED_GAIN' if full else 'FULL9_NOT_ENTERED_COST_GATE'))
            assert set(decision['completed_full9'])=={row['method'] for row in full}
            for row in full:
                saved=decision['completed_full9'][row['method']]
                assert abs(saved['score']-row['official_score'])<1e-9 and saved['improved_count']==row['improved_images']
                assert saved['meets90']==(row['official_score']>=90) and saved['meets95']==(row['official_score']>=95)
    assert branches['seed314159']['status']=='NOT_STARTED' and branches['public5']['status']=='NOT_STARTED'
    progress=json.loads((ROOT/'progress.json').read_text())
    assert abs(progress['cumulative_model_seconds']-budget['model_seconds'])<1e-9
    result={'status':'PASS','model_forwards':0,'model_seconds':budget['model_seconds'],'A_seconds':budget['A_seconds'],
        'remaining_seconds':budget['remaining_seconds'],'sessions':budget['sessions'],'one_persisted_cumulative_ledger':'PASS',
        'timing_scope':'actual model-process intervals include CPU/MPS model/probe/metric/cacheIO; prep/audit/render/package separate',
        'cost_gates':gates,'screen':screen,'candidate_decision':decision,'unused_branches':'seed314159 and candidate public5 correctly NOT_STARTED',
        'conservative_crash_charge_present':any(s['status']=='CRASH_RECOVERY_CONSERVATIVE_WALL_CHARGE' for s in budget['sessions'])}
    (ROOT/'audit/budget_decisions_independent.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','model_seconds':budget['model_seconds'],'A_seconds':budget['A_seconds'],
        'cost_gates':len(gates),'candidate':decision['candidate'] if decision else None}))
if __name__=='__main__':main()
