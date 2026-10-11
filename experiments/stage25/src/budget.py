"""One persisted measured model ledger and monotonic deadline across resumes."""
import json, os, threading, time
import common as c

class Clock:
    def __init__(self):
        self.path=c.ROOT/'budget.json';self.lock=threading.RLock();self.stop=threading.Event()
        self.previous=json.loads(self.path.read_text()) if self.path.exists() else {'sessions':[],'model_seconds':0.,'A_seconds':0.}
        old=self.previous.get('active_session')
        if old:
            # Crashed sessions are charged up to their durable heartbeat. Any
            # uncertainty through restart is charged conservatively as well.
            try:os.kill(old['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError('A model session is already active')
            elapsed=max(old['elapsed_seconds'],time.time()-old['started_unix'])
            self.previous['model_seconds']+=elapsed
            self.previous['A_seconds']+=elapsed if old.get('phase')=='A' else old['A_elapsed_seconds']
            self.previous['sessions'].append({**old,'elapsed_seconds':elapsed,'status':'CRASH_RECOVERY_CONSERVATIVE_WALL_CHARGE'})
        self.started=time.monotonic();self.started_unix=time.time()
        self.base=float(self.previous['model_seconds']);self.A_base=float(self.previous['A_seconds'])
        self.deadline=self.started+max(5400-self.base,0)
        self.A_started=self.started;self.A_finished=None;self.phase='A'
        self.state={'status':'RUNNING','phase':'A','PID':os.getpid(),'method':None,'Id':None,'input_kind':None,
                    'masks_completed':0,'masks_planned':0,'complete_raw_count':0,'cache_hits':0,'eta_seconds':None}
        self.thread=threading.Thread(target=self._heartbeat,daemon=True);self.thread.start();self.update()

    def elapsed(self):return time.monotonic()-self.started
    def spent(self):return self.base+self.elapsed()
    def remaining(self):return max(5400-self.spent(),0)
    def A_spent(self):return self.A_base+((self.A_finished or time.monotonic())-self.A_started)
    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Stage25 cumulative 5400-second model deadline')
        if self.phase=='A' and self.A_spent()>=600:raise TimeoutError('Stage25 A cumulative 600-second model deadline')
    def finish_A(self):
        self.A_finished=time.monotonic();self.phase='B';self.update(phase='B')
    def update(self,**fields):
        with self.lock:
            self.state.update(fields)
            self.state.update({'cumulative_model_seconds':self.spent(),'remaining_model_seconds':self.remaining(),
                               'A_model_seconds':self.A_spent(),'monotonic_deadline':self.deadline,'updated_unix':time.time(),
                               'eta_note':'Estimate from measured completed raw; hardware rate can change.'})
            record={**self.previous,'model_seconds':self.base,'A_seconds':self.A_base,'limit_seconds':5400,'A_limit_seconds':600,
                    'active_session':{'pid':os.getpid(),'started_unix':self.started_unix,'started_monotonic':self.started,
                    'elapsed_seconds':self.elapsed(),'phase':self.phase,'A_elapsed_seconds':self.A_spent()-self.A_base},
                    'live_model_seconds':self.spent(),'live_A_seconds':self.A_spent(),'remaining_seconds':self.remaining(),
                    'deadline_monotonic':self.deadline,'clock_semantics':'One active interval includes model loads, probes, official metrics and verified cache IO. Preparation and post-run audits are separate.'}
            c.atomic_json(self.path,record);c.atomic_json(c.ROOT/'progress.json',self.state)
    def _heartbeat(self):
        while not self.stop.wait(30):
            self.update();print(json.dumps({'progress':self.state},ensure_ascii=False),flush=True)
    def close(self,status):
        self.stop.set();self.thread.join(timeout=2)
        with self.lock:
            elapsed=self.elapsed();a=self.A_spent()
            sessions=self.previous['sessions']+[{'pid':os.getpid(),'started_unix':self.started_unix,
                       'elapsed_seconds':elapsed,'A_elapsed_seconds':a-self.A_base,'status':status}]
            c.atomic_json(self.path,{'sessions':sessions,'model_seconds':self.base+elapsed,'A_seconds':a,
                         'limit_seconds':5400,'A_limit_seconds':600,'remaining_seconds':max(5400-self.base-elapsed,0),
                         'active_session':None,'status':status,'finished_unix':time.time()})
            self.state.update({'status':status,'cumulative_model_seconds':self.base+elapsed,'remaining_model_seconds':max(5400-self.base-elapsed,0)})
            c.atomic_json(c.ROOT/'progress.json',self.state)
