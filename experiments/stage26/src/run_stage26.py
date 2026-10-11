"""Single bounded native-MPS session: paired second-seed nine, then fixed 2x2 single."""
import json,sys,time,traceback
import numpy as np
import torch
import common as c
import probes as p
import run_official as official_run
from budget import Clock,BudgetStop
from cache import Cache
from legacy import Legacy

WORST4=95.3132707090117
AMETHODS=('black_occlusion8_signed','black_occlusion4_soft15')
BMETHODS=('black_occlusion2_post22','black_occlusion2_soft15')

def branch(records,name,status,reason=None,**fields):
    records[name]={'status':status,'reason':reason,**fields};c.atomic_json(c.ROOT/'branch_status.json',records)

def cost_gate(clock,name,components):
    estimate=sum(float(v) for v in components.values());safe=estimate*1.15
    record={'phase':clock.phase,'gate':name,'components_seconds':components,'measured_estimate_seconds':estimate,'safety_multiplier':1.15,
            'safe_estimate_seconds':safe,'remaining_model_seconds':clock.remaining(),'remaining_phase_seconds':clock.phase_remaining(),
            'cumulative_model_seconds':clock.spent(),'phase_model_seconds':clock.phase_spent(),'entry_allowed':safe<min(clock.remaining(),clock.phase_remaining()),'created_unix':time.time()}
    c.atomic_json(c.ROOT/'gates'/f'{name}.json',record);return record['entry_allowed']

def save_decision(name,value):
    c.atomic_json(c.ROOT/name,value);(c.ROOT/PathLikeHash(name)).write_text(c.sha(c.ROOT/name)+f'  {name}\n')
def PathLikeHash(name):return name.removesuffix('.json')+'.sha256'

def completed(method,count,seed):
    scope='full9' if count==9 else 'single1';folder=c.ROOT/('official9' if count==9 else 'single1');stem=f'{method}_mps_seed{seed}_{scope}'
    csv=folder/f'{stem}.csv';rp=folder/f'{stem}.run.json';sp=folder/f'{stem}.score.json'
    if not any(x.exists() for x in (csv,rp,sp)):return None
    if not all(x.exists() for x in (csv,rp,sp)):raise RuntimeError('Partial completed artifacts require independent recovery before rerun')
    run=json.loads(rp.read_text());score=json.loads(sp.read_text());prereg=json.loads((c.ROOT/'preregistered_method.json').read_text())
    if run['status']!='COMPLETE' or run['csv_sha256']!=c.sha(csv) or run['eval_seed']!=seed or run['image_count']!=count or run['device']!='mps' or run['program_sha256']!=prereg['program_sha256'] or run['preregistered_method_sha256']!=c.sha(c.ROOT/'preregistered_method.json'):raise RuntimeError('Completed run recovery identity differs')
    for t in run['callback_traces']:
        if c.sha(t['raw_path'])!=t['raw_file_sha256'] or c.sha(t['final_path'])!=t['final_file_sha256']:raise RuntimeError('Completed recovery cache file changed')
    return run,score

def main():
    official=c.frozen_sources();prereg=json.loads((c.ROOT/'preregistered_method.json').read_text())
    for name,h in {**prereg['program_sha256'],**prereg['frozen_sampling_files_sha256']}.items():
        if c.sha(c.ROOT/name)!=h:raise RuntimeError(f'Frozen source changed:{name}')
    if json.loads((c.ROOT/'audit/prior_sources.json').read_text())['status']!='PASS':raise RuntimeError('Prior source gate failed')
    c.check_device('mps');torch.set_num_threads(4);inputs,indices=p.read();legacy=Legacy()
    sys.path.insert(0,str(official/'vendor'))
    from kaggle.model.model import load_model
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms
    records={}
    for key in ('A','A_8x8','A_4x4','B','B_original_raw','B_post22','B_soft15'):branch(records,key,'NOT_STARTED')
    for key,reason in [('candidate_public5','Outside Stage26; Stage24 historical five-image evidence only'),('2x2_full9','Stage26 permits one image only'),('full_CPU_final_support','Only fixed small subsets allowed'),('Windows_CUDA','Not verified on this Mac')]:branch(records,key,'NOT_STARTED',reason)
    clock=Clock();finish='FAILED_OR_INTERRUPTED';source_valid=False;Aruns={};Ascores={};Bruns={};Bscores={};original_cost=None
    try:
        branch(records,'A','RUNNING');clock.update(method='official_model_load',device='cpu')
        model=load_model(str(c.WEIGHTS),device=torch.device('cpu'));model.eval();model.to('mps');model.eval();c.sync('mps');clock.check()
        source_gate=p.source_acceptance(model,inputs,indices,clock)
        if source_gate['status']!='PASS':
            branch(records,'A','FAILED','SOURCE_DEVICE_NUMERICAL_GATE_FAIL');branch(records,'B','NOT_STARTED','A_DEVICE_NUMERICAL_SOURCE_FAIL');finish='COMPLETE_WITH_SOURCE_BLOCK';return
        source_valid=True;clock.device_gate_passed=True
        dataset=ShipImageDataset(str(official/'data/test.csv'),str(official/'data/test_images'),transform=val_transforms)
        if len(dataset)!=9:raise RuntimeError('Exactly nine official inputs required')
        hits={4:0,8:0}
        for i in range(9):
            x,y=dataset[i];x=np.ascontiguousarray(x.numpy()[None]);y=np.asarray([int(y)],np.int64)
            if i==0 and (c.arrsha(x)!=c.arrsha(inputs['original']) or c.arrsha(y)!=c.arrsha(inputs['target'])):raise RuntimeError('First official input/target binding differs')
            for tile in (4,8):hits[tile]+=int(legacy.find(x,y,tile,64) is not None)
        c.atomic_json(c.ROOT/'gates/actual_original_legacy_matches.json',{'status':'PASS','matches':hits,'tested_nine_exact_inputs_targets':True,'new_unique_estimate':{str(t):54-h for t,h in hits.items()}})
        oldpath=c.WORKSPACE/'output_mac_stage23/official9/black_occlusion8_signed_mps_seed42.run.json';old8=json.loads(oldpath.read_text())
        old8_seconds=float(old8['elapsed_seconds']);historical8cold=max(old8['cold_raw_seconds']) if 'cold_raw_seconds' in old8 and old8['cold_raw_seconds'] else old8_seconds/54
        c.atomic_json(c.ROOT/'gates/historical_cost_sources.json',{'stage25_worst_cold4_seconds':WORST4,'stage23_complete_native8_seconds':old8_seconds,'stage23_source_path':str(oldpath),'stage23_source_sha256':c.sha(oldpath),'not_new_measured_cost':True})
        def cold_guard(cache):
            if cache.method=='black_occlusion8_signed':
                remaining=max(54-hits[8]-cache.stats['raw_computed'],0);rate=max([historical8cold]+cache.cold_seconds)
                components={'remaining_native8_raw_including_current':remaining*rate,'reserved_candidate4_new_raw':(54-hits[4])*WORST4,'official_metrics_IO_allowance':60.}
            elif cache.method=='black_occlusion4_soft15':
                remaining=max(54-hits[4]-cache.stats['raw_computed'],0);rate=max([WORST4]+cache.cold_seconds)
                components={'remaining_native4_raw_including_current':remaining*rate,'official_metrics_IO_allowance':30.}
            else:
                total_new=5 if getattr(clock,'B_official_entered',False) else 1;remaining=max(total_new-cache.stats['raw_computed'],0)
                rate=max([4*WORST4]+([original_cost] if original_cost is not None else [])+cache.cold_seconds)
                components={'remaining_native2_raw_including_current':remaining*rate,'official_metrics_IO_allowance':30.}
            name=f'{clock.phase}_{cache.method}_before_new_raw_{cache.stats["raw_computed"]+1:02d}'
            if not cost_gate(clock,name,components):raise BudgetStop(f'COST_GATE_STOP:{name}')
        clock.cold_guard=cold_guard
        try:
            if not cost_gate(clock,'A_initial_pair',{'remaining_new4':(54-hits[4])*WORST4,'historical_complete_MPS8_flow':old8_seconds,'probe_metric_validation_allowance':60.}):raise BudgetStop('A_NOT_ENTERED_COST_GATE')
            for method in AMETHODS:
                key='A_8x8' if method==AMETHODS[0] else 'A_4x4';branch(records,key,'RUNNING')
                if method==AMETHODS[1] and not cost_gate(clock,'A_candidate4_entry',{'remaining_new4':(54-hits[4])*WORST4,'official_metric_validation_allowance':30.}):raise BudgetStop('A_CANDIDATE_NOT_ENTERED_COST_GATE')
                prior=completed(method,9,314159)
                Aruns[method],Ascores[method]=prior if prior is not None else official_run.run(model,dataset,method,9,clock,64,314159)
                branch(records,key,'COMPLETED',score=Ascores[method]['official_score'],elapsed_seconds=Aruns[method]['elapsed_seconds'],recovered_completed=prior is not None)
            t8=Aruns[AMETHODS[0]]['callback_traces'];t4=Aruns[AMETHODS[1]]['callback_traces']
            if len(t8)!=72 or len(t4)!=72 or any(any(a[k]!=b[k] for k in ('input_sha256','input_shape','input_dtype','target_sha256','target_shape','target_dtype','target')) for a,b in zip(t8,t4)):raise RuntimeError('Second-seed paired input/target callbacks differ')
            baseline=Ascores[AMETHODS[0]]['official_score'];candidate=Ascores[AMETHODS[1]];passed=candidate['official_score']>baseline and candidate['improved_count']>=6
            decision={'status':'GAIN_CONFIRMED_TWO_SEEDS_SAME_MPS' if passed else 'SECOND_SEED_GAIN_NOT_CONFIRMED','candidate':'black_occlusion4_soft15','eval_seed':314159,'device':'mps','baseline_mean':baseline,'candidate_mean':candidate['official_score'],'paired_gain':candidate['official_score']-baseline,'improved_images':candidate['improved_count'],'paired_deltas':candidate['paired_deltas'],'threshold':'strictly higher mean than new same-MPS8 AND >=6/9 improves','seed42_historical_candidate':82.65370613379287,'seed42_historical_baseline':78.62195221134283,'meets90':candidate['official_score']>=90,'meets95':candidate['official_score']>=95,'not_generalization_or_official_qualification':True}
            save_decision('confirmation_decision.json',decision);branch(records,'A','COMPLETED',confirmation=decision['status'],phase_model_seconds=clock.phase_spent())
        except BudgetStop as exc:
            save_decision('confirmation_decision.json',{'status':'SECOND_SEED_CONFIRMATION_PARTIAL_OR_NOT_ENTERED','reason':str(exc),'completed_methods':list(Ascores),'candidate_seed42_retained':True})
            branch(records,'A','PARTIAL' if Ascores else 'NOT_STARTED',str(exc),completed_methods=list(Ascores),phase_model_seconds=clock.phase_spent())
            for key in ('A_8x8','A_4x4'):
                if records[key]['status']=='RUNNING':branch(records,key,'PARTIAL',str(exc))
        clock.switch('B');clock.update(method='B_small_device_acceptance',Id=1,input_kind='original',masks_completed=0,masks_planned=128)
        branch(records,'B','RUNNING')
        try:
            acceptance=p.device_acceptance2(model,inputs,indices,clock,64)
            if acceptance['status']!='PASS':
                branch(records,'B','FAILED','CPU_MPS_2x2_ACCEPTANCE_FAIL');finish='COMPLETE_WITH_B_DEVICE_GATE_FAIL';return
            if not cost_gate(clock,'B_original_raw_entry',{'one_native2_original':4*WORST4,'validation_allowance':15.}):raise BudgetStop('B_ORIGINAL_NOT_ENTERED_COST_GATE')
            cache=Cache(c.ROOT/'cache','mps',BMETHODS[0],clock,64);clock.update(method='black_occlusion2_signed_v1',input_kind='original',masks_completed=0,masks_planned=37632)
            before=time.monotonic();a,m,key,path,reuse=cache.get_raw(model,inputs['original'],inputs['target']);original_cost=time.monotonic()-before
            if reuse is not None:original_cost=float(m['record']['compute_seconds'])
            c.atomic_json(c.ROOT/'gates/B_original_raw_measured.json',{'status':'COMPLETE','path':str(path),'sha256':c.sha(path),'key':key,'reuse':reuse,'new_cold_seconds':original_cost if reuse is None else None,'historical_this_stage_raw_seconds':m['record']['compute_seconds'],'single_conservative_seconds':max(original_cost,4*WORST4)})
            branch(records,'B_original_raw','COMPLETED',elapsed_seconds=original_cost,reuse=reuse)
            if not cost_gate(clock,'B_single_official_entry',{'remaining5_native2_raw':5*max(original_cost,4*WORST4),'two_official_single_metrics_and_IO_allowance':60.}):raise BudgetStop('B_OFFICIAL_NOT_ENTERED_COST_GATE')
            clock.B_official_entered=True
            for method in BMETHODS:
                key='B_post22' if method==BMETHODS[0] else 'B_soft15';branch(records,key,'RUNNING')
                if Bscores and not cost_gate(clock,'B_second_warm_method_entry',{'measured_first_metric_overhead':max(Bruns[BMETHODS[0]]['elapsed_seconds']-sum(Bruns[BMETHODS[0]]['cold_raw_seconds']),0.),'validation_allowance':15.}):raise BudgetStop('B_SECOND_WARM_METHOD_NOT_ENTERED_COST_GATE')
                prior=completed(method,1,42)
                Bruns[method],Bscores[method]=prior if prior is not None else official_run.run(model,dataset,method,1,clock,64,42)
                branch(records,key,'COMPLETED',score=Bscores[method]['official_score'],elapsed_seconds=Bruns[method]['elapsed_seconds'])
            for x,y in zip(Bruns[BMETHODS[0]]['callback_traces'],Bruns[BMETHODS[1]]['callback_traces']):
                if any(x[k]!=y[k] for k in ('input_sha256','target_sha256','raw_key','raw_sha256','masked_logits_sha256','original_logits_sha256')) or y['raw_reuse'] is None:raise RuntimeError('B paired same raw/input/target source differs')
            best4=78.19467843751505;qual=[method for method,s in Bscores.items() if s['official_score']>best4]
            selected=max(qual,key=lambda method:(Bscores[method]['official_score'],method==BMETHODS[0])) if qual else None
            decision={'status':'SINGLE_IMAGE_GAIN_REQUIRES_PREFIX3_AND_FULL9' if qual else 'SINGLE_IMAGE_NO_GAIN_SCREEN','selected_exploration_method':selected,'best4_seed42_single_image_reference':best4,'methods':{method:{'score':s['official_score'],'gain_vs_best4':s['official_score']-best4,'passed':method in qual} for method,s in Bscores.items()},'image_count':1,'Id':1,'eval_seed':42,'device':'mps','not_a_full9_score':True,'threshold':'complete single-image score strictly higher than78.19467843751505; highermean then exacttiepost22','does_not_replace_A_nine_image_candidate':True}
            save_decision('single_image_decision.json',decision);branch(records,'B','COMPLETED',decision=decision['status'],phase_model_seconds=clock.phase_spent());finish='COMPLETE'
        except BudgetStop as exc:
            reason=str(exc);save_decision('single_image_decision.json',{'status':'B_OFFICIAL_NOT_ENTERED_COST_GATE' if not Bscores else 'B_SINGLE_IMAGE_PARTIAL','reason':reason,'completed_methods':list(Bscores),'not_a_full9_score':True})
            branch(records,'B','PARTIAL' if records['B_original_raw']['status']=='COMPLETED' or Bscores else 'NOT_STARTED',reason,completed_methods=list(Bscores),phase_model_seconds=clock.phase_spent())
            for key in ('B_post22','B_soft15'):
                if records[key]['status']=='RUNNING':branch(records,key,'PARTIAL',reason)
            finish='COMPLETE_WITH_B_BUDGET_STOP'
    except BaseException as exc:
        c.atomic_json(c.ROOT/'failure.json',{'error':f'{type(exc).__name__}:{exc}','traceback':traceback.format_exc(),'phase':clock.phase,'model_seconds':clock.spent(),'phase_model_seconds':clock.phase_spent(),'source_device_acceptance_passed':source_valid})
        branch(records,clock.phase,'PARTIAL' if source_valid else 'FAILED',str(exc))
        if clock.phase=='A':branch(records,'B','NOT_STARTED','A_DEVICE_NUMERICAL_SOURCE_OR_RUNTIME_FAIL')
        raise
    finally:
        clock.close(finish);print(json.dumps({'finished':finish,'budget':json.loads((c.ROOT/'budget.json').read_text())},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
