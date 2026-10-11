"""Full9 seed314159 confirmation and single1 seed42 exploration; unchanged generator."""
import contextlib,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import common as c
from cache import Cache

FIELDS=['Id','Faithfulness','Max_Sensitivity','Sparsity']
def reset_rng(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def row_scores(f):return 20*(1+f.Faithfulness.to_numpy(float))+30*np.maximum(1-f.Max_Sensitivity.to_numpy(float)/2,0)+30*f.Sparsity.to_numpy(float)

class Tee:
    def __init__(self,stream,file):self.stream=stream;self.file=file;self.nan_warning=False
    def write(self,s):
        if 'NaN 空值' in s or '自動補 0.0' in s:self.nan_warning=True
        self.stream.write(s);self.stream.flush();self.file.write(s);self.file.flush();return len(s)
    def flush(self):self.stream.flush();self.file.flush()

def run(model,dataset,method,count,clock,batch,seed=42):
    if count not in (1,9):raise ValueError('Only preregistered single1/full9 permitted')
    clock.check();official=c.frozen_sources();sys.path.insert(0,str(official/'vendor'))
    import pandas as pd
    from kaggle.submmit.submmit import generate_xai_submission
    from torch.utils.data import DataLoader,Subset
    prereg=json.loads((c.ROOT/'preregistered_method.json').read_text())
    for name,h in prereg['program_sha256'].items():
        if c.sha(c.ROOT/name)!=h:raise RuntimeError(f'Program changed after preregistration:{name}')
    if next(model.parameters()).device.type!='mps' or next(model.parameters()).dtype!=torch.float32 or model.training:raise RuntimeError('ActualMPSFP32evalrequired')
    scope='single1' if count==1 else 'full9';folder=c.ROOT/('single1' if count==1 else 'official9');folder.mkdir(exist_ok=True)
    stem=f'{method}_mps_seed{seed}_{scope}';csv=folder/f'{stem}.csv';pending=folder/f'{stem}.pending.csv';runfile=folder/f'{stem}.run.json'
    if csv.exists() or runfile.exists():raise FileExistsError('Completed official result exists; validate before reuse')
    loader=DataLoader(Subset(dataset,list(range(count))),batch_size=1,shuffle=False,num_workers=0)
    cache=Cache(c.ROOT/'cache','mps',method,clock,batch);traces=[];started=time.monotonic()
    log=folder/f'{stem}.official_stdout.log'
    tile=8 if method=='black_occlusion8_signed' else (4 if method=='black_occlusion4_soft15' else 2)
    total=3*(224//tile)**2
    clock.update(method=method,Id=1,masks_completed=0,masks_planned=total,planned_callbacks=8*count,completed_callbacks=0,eval_seed=seed,scope=scope)
    def callback(model,inputs,targets,device='cpu',**kwargs):
        clock.check()
        if device!='mps':raise RuntimeError('Officialactualdevicediffers')
        index=len(traces)+1;within=(index-1)%8
        clock.update(Id=(index-1)//8+1,input_kind='original' if within in (0,1,7) else f'metric_perturbation_{within-1}',
                     current_callback=index,masks_completed=0,masks_planned=total)
        before=time.monotonic();final,trace=cache.get(model,np.ascontiguousarray(inputs),np.ascontiguousarray(targets))
        trace.update({'callback_index':index,'Id':(index-1)//8+1,'input_kind':clock.state['input_kind'],
                      'callback_elapsed_seconds':time.monotonic()-before,'completed_unix':time.time()});traces.append(trace)
        c.atomic_json(folder/f'{stem}.callbacks.json',{'status':'RUNNING','traces':traces,'expected_callbacks':8*count})
        clock.update(completed_callbacks=len(traces));clock.check();return final
    try:
        with log.open('w') as lf:
            tee=Tee(sys.stdout,lf)
            reset_rng(seed);c.sync('mps')
            with c.official_device('mps'),contextlib.redirect_stdout(tee):
                generate_xai_submission(model,loader,callback,output_path=str(pending))
            c.sync('mps')
        clock.check()
        if tee.nan_warning:raise RuntimeError('InvalidofficialNaNfillzerowarning')
        f=pd.read_csv(pending)
        if f.columns.tolist()!=FIELDS or f.Id.tolist()!=list(range(1,count+1)) or not np.isfinite(f[FIELDS[1:]].to_numpy(float)).all():raise RuntimeError('InvalidCSVschemaorfinitevalues')
        if len(traces)!=8*count or len(set(t['raw_key'] for t in traces))!=6*count:raise RuntimeError('Officialcallback/uniqueinputcountsnotexpected')
        spec=importlib.util.spec_from_file_location('stage26_original_metrix',official/'script/metrix.py')
        scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
        solution=pd.read_csv(official/'script/solution_all_public.csv') if count==9 else pd.DataFrame({'Id':list(range(1,count+1))})
        solution_path=folder/f'{scope}_solution.csv'
        if not solution_path.exists():solution.to_csv(solution_path,index=False)
        score=float(scorer.score(solution,f.copy(),'Id'));values=row_scores(f)
        if not np.isclose(score,values.mean(),atol=1e-9,rtol=0):raise RuntimeError('Unmodifiedscorerrowformuladisagree')
        pending.replace(csv)
        if method=='black_occlusion8_signed':
            reference=f.copy();reference_path=csv;reference_kind='SELF_BASELINE_NO_GAIN_TEST'
        elif count==9:
            reference_path=folder/f'black_occlusion8_signed_mps_seed{seed}_full9.csv'
            reference=pd.read_csv(reference_path);reference_kind='STAGE26_SAME_MPS_SEED314159'
        else:
            old='black_occlusion4_post22' if method.endswith('post22') else 'black_occlusion4_soft15'
            reference_path=c.WORKSPACE/'output_mac_stage25/official9'/f'{old}_mps_seed42_full9.csv'
            reference=pd.read_csv(reference_path).iloc[:1];reference_kind='STAGE25_SAME_POST_SEED42_SINGLE_IMAGE'
        refscores=row_scores(reference);paired=values-refscores
        rows=[{'Id':int(row.Id),'F':float(row.Faithfulness),'M':float(row.Max_Sensitivity),'S':float(row.Sparsity),
               'F_points':20*(1+float(row.Faithfulness)),'M_points':30*max(1-float(row.Max_Sensitivity)/2,0),
               'S_points':30*float(row.Sparsity),'total':float(values[i]),'baseline8_total':float(refscores[i]),'paired_delta':float(paired[i])}
              for i,row in enumerate(f.itertuples(index=False))]
        score_record={'status':'UNMODIFIED_METRIX_AND_ROW_FORMULA_PASS','sample_count':count,
            'scope':'official nine-image formula' if count==9 else 'single-image exploration only; not a nine-image score',
            'official_score':score,'row_formula_scores':values.tolist(),'paired_deltas':paired.tolist(),'improved_count':int((paired>0).sum()),
            'baseline_mean':float(refscores.mean()),'reference_kind':reference_kind,'reference_path':str(reference_path),'reference_file_sha256':c.sha(reference_path),'csv_sha256':c.sha(csv),'scorer_sha256':c.SCORER_SHA,'rows':rows,
            'metric_means':{k:float(f[k].mean()) for k in FIELDS[1:]},
            'contribution_means':{'F':float((20*(1+f.Faithfulness)).mean()),'M':float((30*np.maximum(1-f.Max_Sensitivity/2,0)).mean()),'S':float((30*f.Sparsity).mean())}}
        c.atomic_json(folder/f'{stem}.score.json',score_record)
        record={'status':'COMPLETE','method':method,'device':'mps','precision':'fp32','eval_seed':seed,'image_count':count,
            'batch_size':1,'shuffle':False,'num_workers':0,'scope':scope,'phase':clock.phase,'tile':tile,'inference_batch':cache.batch,'elapsed_seconds':time.monotonic()-started,
            'cache_stats':cache.stats,'cold_raw_seconds':cache.cold_seconds,'warm_raw_seconds':cache.warm_seconds,
            'callback_traces':traces,'csv_sha256':c.sha(csv),'official_score':score,'program_sha256':prereg['program_sha256'],
            'preregistered_method_sha256':c.sha(c.ROOT/'preregistered_method.json'),
            'sources':{'weights':c.WEIGHT_SHA,'generator':c.GENERATOR_SHA,'scorer':c.SCORER_SHA,'test_csv':c.TEST_SHA},
            'rng_reset':'Python/NumPy/torch immediately before unchanged generator, after model/dataset/cache preparation',
            'actual_model_parameter_device':str(next(model.parameters()).device),'no_cpu_fallback':True,
            'cumulative_model_seconds_at_finish':clock.spent(),'phase_model_seconds_at_finish':clock.phase_spent(),'clock_deadline_monotonic':clock.deadline,
            'official_stdout_log_sha256':c.sha(log),'own_solution_sha256':c.sha(solution_path)}
        c.atomic_json(runfile,record);c.atomic_json(folder/f'{stem}.callbacks.json',{'status':'COMPLETE','traces':traces,'expected_callbacks':8*count})
        print(json.dumps({'official_run':str(runfile),'score':score,'sample_count':count,'elapsed_seconds':record['elapsed_seconds'],'cache_stats':cache.stats}),flush=True)
        return record,score_record
    except BaseException as exc:
        c.atomic_json(folder/f'{stem}.failure.json',{'status':'INCOMPLETE','error':f'{type(exc).__name__}:{exc}',
                      'callback_traces':traces,'elapsed_seconds':time.monotonic()-started,'cumulative_model_seconds':clock.spent()})
        if pending.exists():pending.replace(folder/f'{stem}.invalid.csv')
        raise
