"""One persisted global and A/B phase model ledger with monotonic deadlines."""
import contextlib,json,os,threading,time
import common as c
LIMIT=10800.;PHASE_LIMITS={'A':7200.,'B':3600.}

class BudgetStop(BaseException):
    """Abort Quantus immediately instead of allowing its Exception-to-NaN handler."""
    pass

class Clock:
    def __init__(self,phase='A'):
        self.path=c.ROOT/'budget.json';self.lock=threading.RLock();self.stop=threading.Event()
        self.previous=json.loads(self.path.read_text()) if self.path.exists() else {'sessions':[],'model_seconds':0.,'phase_seconds':{'A':0.,'B':0.}}
        old=self.previous.get('active_session')
        if old:
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('A model session is already active')
            elapsed=max(float(old['elapsed_seconds']),time.time()-old['started_unix']);extra=elapsed-old['elapsed_seconds']
            phases=dict(old['phase_elapsed_seconds']);phases[old['phase']]+=extra
            self.previous['model_seconds']+=elapsed
            for key in PHASE_LIMITS:self.previous['phase_seconds'][key]+=phases[key]
            self.previous['sessions'].append({**old,'elapsed_seconds':elapsed,'phase_elapsed_seconds':phases,'status':'CRASH_RECOVERY_CONSERVATIVE_WALL_CHARGE'})
        self.started=time.monotonic();self.started_unix=time.time();self.base=float(self.previous['model_seconds'])
        self.phase_base={k:float(self.previous['phase_seconds'][k]) for k in PHASE_LIMITS};self.phase_elapsed={k:0. for k in PHASE_LIMITS}
        self.phase=phase;self.phase_started=self.started;self.deadline=self.started+max(LIMIT-self.base,0);self.probe_deadline=None
        self.state={'status':'RUNNING','phase':phase,'PID':os.getpid(),'method':None,'Id':None,'input_kind':None,'masks_completed':0,'masks_planned':0,'complete_raw_count':0,'cache_hits':0,'eta_seconds':None}
        self.thread=threading.Thread(target=self._heartbeat,daemon=True);self.thread.start();self.update()
    def elapsed(self):return time.monotonic()-self.started
    def spent(self):return self.base+self.elapsed()
    def remaining(self):return max(LIMIT-self.spent(),0)
    def phase_spent(self,phase=None):
        phase=phase or self.phase;return self.phase_base[phase]+self.phase_elapsed[phase]+(time.monotonic()-self.phase_started if phase==self.phase else 0.)
    def phase_remaining(self):return max(PHASE_LIMITS[self.phase]-self.phase_spent(),0)
    def check(self):
        now=time.monotonic()
        if now>=self.deadline:raise BudgetStop('Stage26 cumulative10800-second deadline')
        if self.phase_spent()>=PHASE_LIMITS[self.phase]:raise BudgetStop(f'Stage26 phase{self.phase} cumulative{PHASE_LIMITS[self.phase]}-second deadline')
        if self.probe_deadline is not None and now>=self.probe_deadline:raise BudgetStop('Stage26 B small-device-acceptance120-second deadline')
    def switch(self,phase):
        with self.lock:
            if phase==self.phase:return
            now=time.monotonic();self.phase_elapsed[self.phase]+=now-self.phase_started;self.phase=phase;self.phase_started=now;self.update(phase=phase)
    @contextlib.contextmanager
    def probe_scope(self):
        self.probe_deadline=time.monotonic()+120.;began=time.monotonic()
        try:yield
        finally:
            self.last_probe_seconds=time.monotonic()-began;self.probe_deadline=None
    def update(self,**fields):
        with self.lock:
            self.state.update(fields);phase_spent={k:self.phase_spent(k) for k in PHASE_LIMITS}
            self.state.update({'phase':self.phase,'cumulative_model_seconds':self.spent(),'remaining_model_seconds':self.remaining(),'phase_model_seconds':phase_spent,'remaining_phase_seconds':self.phase_remaining(),'monotonic_deadline':self.deadline,'updated_unix':time.time(),'eta_note':'Measured completed new cold work; hardware rate may change. Warm cache is not cold speed.'})
            record={**self.previous,'model_seconds':self.base,'phase_seconds':self.phase_base,'limit_seconds':LIMIT,'phase_limits_seconds':PHASE_LIMITS,
                    'active_session':{'pid':os.getpid(),'started_unix':self.started_unix,'started_monotonic':self.started,'elapsed_seconds':self.elapsed(),'phase':self.phase,'phase_elapsed_seconds':{k:phase_spent[k]-self.phase_base[k] for k in PHASE_LIMITS}},
                    'live_model_seconds':self.spent(),'live_phase_seconds':phase_spent,'remaining_seconds':self.remaining(),'remaining_phase_seconds':self.phase_remaining(),'deadline_monotonic':self.deadline,'clock_semantics':'One interval includes model loads, all probes, metrics and verified cache/checkpoint IO; preparation/final audits separate.'}
            c.atomic_json(self.path,record);c.atomic_json(c.ROOT/'progress.json',self.state)
    def _heartbeat(self):
        while not self.stop.wait(30):self.update();print(json.dumps({'progress':self.state},ensure_ascii=False),flush=True)
    def close(self,status):
        self.stop.set();self.thread.join(timeout=2)
        with self.lock:
            now=time.monotonic();elapsed=now-self.started;self.phase_elapsed[self.phase]+=now-self.phase_started;self.phase_started=now
            phases={k:self.phase_base[k]+self.phase_elapsed[k] for k in PHASE_LIMITS}
            sessions=self.previous['sessions']+[{'pid':os.getpid(),'started_unix':self.started_unix,'elapsed_seconds':elapsed,'phase_elapsed_seconds':dict(self.phase_elapsed),'status':status}]
            c.atomic_json(self.path,{'sessions':sessions,'model_seconds':self.base+elapsed,'phase_seconds':phases,'limit_seconds':LIMIT,'phase_limits_seconds':PHASE_LIMITS,'remaining_seconds':max(LIMIT-self.base-elapsed,0),'active_session':None,'status':status,'finished_unix':time.time()})
            self.state.update({'status':status,'cumulative_model_seconds':self.base+elapsed,'remaining_model_seconds':max(LIMIT-self.base-elapsed,0),'phase_model_seconds':phases,'remaining_phase_seconds':max(PHASE_LIMITS[self.phase]-phases[self.phase],0)})
            c.atomic_json(c.ROOT/'progress.json',self.state)
