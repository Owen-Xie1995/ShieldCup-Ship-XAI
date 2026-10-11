"""One bounded native-MPS session: A diagnostic, B screen, conditional C nine."""
import argparse,json,sys,time,traceback
import numpy as np
import torch
import common as c
import diagnostic as d
import postprocess as post
import run_official as official_run
from budget import Clock
from cache import Cache

def branch(records,name,status,reason=None,**fields):
    records[name]={'status':status,'reason':reason,**fields};c.atomic_json(c.ROOT/'branch_status.json',records)

def cost_gate(clock,name,components):
    estimate=sum(float(v) for v in components.values());safe=estimate*1.15
    record={'phase':name,'components_seconds':components,'measured_estimate_seconds':estimate,'safety_multiplier':1.15,
            'safe_estimate_seconds':safe,'remaining_model_seconds':clock.remaining(),'cumulative_model_seconds':clock.spent(),
            'entry_allowed':safe<clock.remaining(),'created_unix':time.time()}
    c.atomic_json(c.ROOT/'gates'/f'{name}.json',record);return record['entry_allowed']

def main(image_count=3):
    official=c.frozen_sources();prereg=json.loads((c.ROOT/'preregistered_method.json').read_text())
    for name,h in prereg['program_sha256'].items():
        if c.sha(c.ROOT/name)!=h:raise RuntimeError(f'Frozenprogramchanged:{name}')
    for filename,h in prereg['frozen_sampling_files_sha256'].items():
        if c.sha(c.ROOT/filename)!=h:raise RuntimeError('Frozensamplingchanged')
    reference=json.loads((c.ROOT/'audit/baseline_reference.json').read_text())
    if reference['status']!='PASS':raise RuntimeError('Historicalreferencegatefailed')
    c.check_device('mps');torch.set_num_threads(4);inputs,indices,sampling=d.read_sampling()
    sys.path.insert(0,str(official/'vendor'))
    from kaggle.model.model import load_model
    from kaggle.model.dataset.ship_image_datasets import ShipImageDataset
    from kaggle.model.dataset.transforms import val_transforms
    records={};branch(records,'A','RUNNING');branch(records,'B','NOT_STARTED');branch(records,'C','NOT_STARTED')
    branch(records,'seed314159','NOT_STARTED','Excluded from Stage25; future paired MPS8x8 baseline required')
    branch(records,'public5','NOT_STARTED','Stage24 completed pair reused as historical evidence; no Stage25 candidate five-image rerun')
    clock=Clock();finish='FAILED_OR_INTERRUPTED'
    try:
        # Model loading and all probes, metric work and cache IO below are within
        # the same actual elapsed interval and its one monotonic deadline.
        model=load_model(str(c.WEIGHTS),device=torch.device('cpu'));model.eval();clock.check()
        cpu={};mps={};raw4={};rawmeta={};rawpaths={}
        for label in ('original','perturbed'):
            clock.update(phase='A',method='finite_intervention_diagnostic',device='cpu',Id=1,input_kind=label)
            cpu[label],_=d.run_probes(model,inputs[label],inputs['target'],'cpu',indices,clock,batch=64,label=label)
        model=model.to('mps');model.eval();c.sync('mps');clock.check()
        acache=Cache(c.ROOT/'cache','mps',post.METHODS[0],clock,64)
        for label in ('original','perturbed'):
            clock.update(phase='A',method='black_occlusion4_signed_v1',device='mps',Id=1,input_kind=label,masks_completed=0,masks_planned=9408)
            a,m,key,path,reuse=acache.get_raw(model,inputs[label],inputs['target'])
            raw4[label]=a;rawmeta[label]=m;rawpaths[label]={'path':str(path.resolve()),'sha256':c.sha(path),'key':key,'reuse':reuse}
        # If the second fullraw needed a smaller registered batch, rebuild the
        # first under that final identity before the subset device acceptance.
        for label in ('original','perturbed'):
            if rawmeta[label]['record']['inference_batch']!=acache.batch:
                a,m,key,path,reuse=acache.get_raw(model,inputs[label],inputs['target'])
                raw4[label]=a;rawmeta[label]=m;rawpaths[label]={'path':str(path.resolve()),'sha256':c.sha(path),'key':key,'reuse':reuse}
            mps[label],_=d.run_probes(model,inputs[label],inputs['target'],'mps',indices,clock,batch=acache.batch,label=label)
        gates={label:d.device_gate(cpu[label],mps[label],raw4[label],indices=indices,out_path=c.ROOT/'probes'/f'device_gate_{label}.json') for label in ('original','perturbed')}
        c.atomic_json(c.ROOT/'probes/device_acceptance.json',{'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'FAIL',
                      'cases':gates,'raw_sources':rawpaths,'actual_inference_batch':acache.batch,
                      'full_cpu_4x4_final_support_validation':'NOT_PERFORMED_IN_THIS_STAGE'})
        for label in ('original','perturbed'):d.build_diagnostics(label,raw4[label],raw4_metadata=rawmeta[label],mps_arrays=mps[label])
        clock.check();a_seconds=clock.A_spent();clock.finish_A()
        if not all(g['status']=='PASS' for g in gates.values()):
            branch(records,'A','FAILED','Preregistered CPU/MPS subset device gate failed',actual_seconds=a_seconds)
            branch(records,'B','NOT_STARTED','DEVICE_GATE_FAIL');branch(records,'C','NOT_STARTED','DEVICE_GATE_FAIL');finish='COMPLETE_WITH_DEVICE_GATE_FAIL';return
        clock.device_gate_passed=True
        branch(records,'A','COMPLETED',actual_seconds=a_seconds,full_raw_inputs=2,single_masks_per_input=384,group50_masks_per_input=128,
               cpu_tile_masks_per_input=128,cpu_single_masks_per_input=32,inference_batch=acache.batch)
        dataset=ShipImageDataset(str(official/'data/test.csv'),str(official/'data/test_images'),transform=val_transforms)
        if len(dataset)!=9:raise RuntimeError('Exactlynineofficialinputsrequired')
        if not np.array_equal(dataset[0][0].numpy()[None],inputs['original']) or int(dataset[0][1])!=int(inputs['target'][0]):raise RuntimeError('Officialfirstinputtargetbindingchanged')
        cold=max(acache.cold_seconds) if acache.cold_seconds else max(m['record']['compute_seconds'] for m in rawmeta.values())
        # Prefix3 has18unique inputs; A supplies exactly one matching original.
        # Historical completed same-environment official single-image fullflow
        # supplies a measured conservative metric overhead before new B exists.
        oldflow_path=c.WORKSPACE/'output_mac_stage24/official_single/gdd_black_post22_mps_seed42_image1.run.json'
        oldflow=float(json.loads(oldflow_path.read_text())['elapsed_seconds'])
        c.atomic_json(c.ROOT/'gates/historical_metric_timing_source.json',{'path':str(oldflow_path),'sha256':c.sha(oldflow_path),
                      'measured_seconds':oldflow,'scope':'Historical same-environment complete single-image GDD flow; conservative pre-B metric overhead, not new4x4 speed'})
        if not cost_gate(clock,'B_prefix3',{'17_remaining_cold_raw':17*cold,'historical_same_environment_official_metric_fullflow_3images':3*oldflow,'warm_second_method_and_validation_allowance':60.}):
            branch(records,'B','NOT_STARTED','COST_GATE_NO_BUDGET');branch(records,'C','NOT_STARTED','B_NOT_ENTERED');finish='COMPLETE_WITH_COST_GATE_STOP';return
        screens={};bruns={};bscores={};branch(records,'B','RUNNING')
        for method in post.METHODS:
            clock.check();bruns[method],bscores[method]=official_run.run(model,dataset,method,image_count,clock,acache.batch)
            s=bscores[method];screens[method]={'passed':bool(s['official_score']>reference['prefix3']['mean_score'] and s['improved_count']>=2),
                'prefix3_mean':s['official_score'],'baseline_prefix3_mean':reference['prefix3']['mean_score'],
                'improved_count':s['improved_count'],'required_improved_count':2,'paired_deltas':s['paired_deltas'],
                'csv_sha256':s['csv_sha256'],'not_a_full9_score':True}
        screen={'status':'FROZEN','rule':'mean strictly higher than historical same-MPS seed42 prefix3 AND at least2/3 improves',
                'methods':screens,'qualified_methods':[m for m in post.METHODS if screens[m]['passed']],
                'preregistered_method_sha256':c.sha(c.ROOT/'preregistered_method.json'),'cumulative_model_seconds':clock.spent()}
        c.atomic_json(c.ROOT/'screen_decision.json',screen)
        (c.ROOT/'screen_decision.sha256').write_text(c.sha(c.ROOT/'screen_decision.json')+'  screen_decision.json\n')
        branch(records,'B','COMPLETED',methods=screens)
        if not screen['qualified_methods']:
            branch(records,'C','NOT_STARTED','FULL9_NOT_ENTERED_SCREEN_FAIL')
            c.atomic_json(c.ROOT/'candidate_decision.json',{'status':'FULL9_NOT_ENTERED_SCREEN_FAIL','candidate':None,
                'scope':'A plus two prefix3 screens completed; full9 outcome remains unmeasured','screen_decision_sha256':c.sha(c.ROOT/'screen_decision.json')})
            finish='COMPLETE';return
        clock.phase='C';fullscores={};fullruns={};branch(records,'C','RUNNING')
        bcold=bruns[post.METHODS[0]]['cold_raw_seconds'];cold=max(bcold+[cold])
        newcold_total=sum(bcold);perimage_overhead=max((bruns[post.METHODS[0]]['elapsed_seconds']-newcold_total)/3,0)
        warm_perimage=bruns[post.METHODS[1]]['elapsed_seconds']/3
        for i,method in enumerate(screen['qualified_methods']):
            components={'remaining36_cold_raw':36*cold if not fullscores else 0.,
                        'full9_measured_official_metrics_and_cacheIO':9*max(perimage_overhead,warm_perimage),'source_scoring_validation_allowance':10.}
            if not cost_gate(clock,f'C_full9_{method}',components):
                branch(records,'C_'+method,'NOT_STARTED','COST_GATE_NO_BUDGET');continue
            fullruns[method],fullscores[method]=official_run.run(model,dataset,method,9,clock,acache.batch)
            # Full generator restarts at image1; first24callbacks must bind to B.
            btrace=bruns[method]['callback_traces'];ftrace=fullruns[method]['callback_traces'][:24]
            for old,new in zip(btrace,ftrace):
                if any(old[k]!=new[k] for k in ('input_sha256','target_sha256','raw_key','raw_sha256','final_sha256')) or new['raw_reuse'] is None:raise RuntimeError('Full9prefix24cache/input/sourcebindingdiffers')
            branch(records,'C_'+method,'COMPLETED',official_score=fullscores[method]['official_score'],improved_count=fullscores[method]['improved_count'])
        qualified=[m for m,s in fullscores.items() if s['official_score']>78.621952211343 and s['improved_count']>=6]
        selected=max(qualified,key=lambda m:(fullscores[m]['official_score'],m==post.METHODS[0])) if qualified else None
        decision={'status':'GAIN_SEED42_ONLY_CONFIRMATION_PENDING' if selected else ('FULL9_NO_QUALIFIED_GAIN' if fullscores else 'FULL9_NOT_ENTERED_COST_GATE'),
                  'candidate':selected,'completed_full9':{m:{'score':s['official_score'],'improved_count':s['improved_count'],'meets90':s['official_score']>=90,'meets95':s['official_score']>=95} for m,s in fullscores.items()},
                  'threshold':'mean>78.621952211343 AND >=6/9 wins; highestmean; exacttiepost22',
                  'screen_decision_sha256':c.sha(c.ROOT/'screen_decision.json'),'second_seed_and_candidate_public5':'NOT_PERFORMED'}
        c.atomic_json(c.ROOT/'candidate_decision.json',decision)
        branch(records,'C','COMPLETED' if len(fullscores)==len(screen['qualified_methods']) else 'PARTIAL',methods=list(fullscores),decision=decision['status'])
        finish='COMPLETE'
    except Exception as exc:
        c.atomic_json(c.ROOT/'failure.json',{'error':f'{type(exc).__name__}:{exc}','traceback':traceback.format_exc(),
                       'phase':clock.phase,'model_seconds':clock.spent(),'remaining_seconds':clock.remaining()})
        branch(records,clock.phase,'FAILED',f'{type(exc).__name__}:{exc}')
        print(traceback.format_exc(),flush=True)
    finally:
        c.sync('mps');clock.close(finish)
        print(json.dumps({'finished':finish,'budget':json.loads((c.ROOT/'budget.json').read_text())}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-count',type=int,choices=[3],default=3,help='Fixed three-image screening; qualified methods restart complete nine-image generator.')
    main(parser.parse_args().image_count)
