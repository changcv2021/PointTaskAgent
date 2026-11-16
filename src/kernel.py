"""Persistent Python worker in a bubblewrap filesystem/network/PID sandbox.

Portability adaptation of the recorded experiment's kernel. The worker API,
timeouts, and persistent-cell semantics are unchanged. No API credentials are
passed to the worker. Runtime roots must contain software, not experiment data.
"""
import json,os,subprocess,selectors,signal,time,sys,shutil
from pathlib import Path
from common_run import PACKAGE_ROOT

class Kernel:
    def __init__(self,frame_dir,episode_id,out):
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.err=None;self.process=None;self.selector=None
        executable=Path(os.environ.get('PTA_PYTHON',sys.executable)).resolve()
        bwrap=os.environ.get('PTA_BWRAP') or shutil.which('bwrap')
        if not bwrap:raise RuntimeError('bubblewrap is required; install bwrap or set PTA_BWRAP. There is no unsandboxed fallback.')
        src=PACKAGE_ROOT/'src';frame=Path(frame_dir).resolve()
        cmd=[bwrap,'--die-with-parent','--new-session','--unshare-all','--clearenv']
        roots=['/usr','/lib','/lib64','/bin',str(Path(sys.prefix).resolve()),str(Path(sys.base_prefix).resolve()),str(executable.parent)]
        extra=[p for p in os.environ.get('PTA_RUNTIME_ROOTS','').split(os.pathsep) if p]
        pypath=[p for p in os.environ.get('PTA_SANDBOX_PYTHONPATH','').split(os.pathsep) if p]
        for value in extra+pypath:
            p=Path(value).resolve()
            if p==Path('/') or p==Path.home() or p==frame or p in frame.parents or p==self.out or p in self.out.parents:
                raise ValueError('Runtime mounts must not expose input/output parents or home: '+str(p))
            if any(s in str(p) for s in ['private_gt','/lidarseg','/episodes','/prepared']):
                raise ValueError('Runtime mount is not a software directory: '+str(p))
        roots+=extra+pypath
        present=sorted({str(Path(p).resolve()) for p in roots if Path(p).exists()},key=lambda p:(p.count('/'),p))
        mounted=[]
        for p in present:
            if any(Path(parent)==Path(p) or Path(parent) in Path(p).parents for parent in mounted):continue
            cmd+=['--ro-bind',p,p];mounted.append(p)
        # System aliases may differ from their resolved paths.
        for p in sorted(set(roots+['/bin','/lib','/lib64'])):
            if Path(p).is_symlink():cmd+=['--symlink',os.readlink(p),p]
        for p in ['/etc/ld.so.cache','/etc/fonts','/etc/localtime']:
            if Path(p).exists():cmd+=['--ro-bind',p,p]
        cmd+=['--ro-bind',str(src),'/app','--ro-bind',str(frame),'/input','--bind',str(self.out),'/output',
              '--tmpfs','/tmp','--proc','/proc','--dev','/dev','--chdir','/output']
        env=dict(PATH=str(executable.parent)+':/usr/bin:/bin',PYTHONPATH='/app'+(':'+':'.join(pypath) if pypath else ''),
                 PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl',
                 XDG_CACHE_HOME='/tmp/cache',TORCH_HOME='/tmp/torch')
        for k in ['LD_LIBRARY_PATH','LD_PRELOAD']:
            if os.environ.get(k):env[k]=os.environ[k]
        for k,v in env.items():cmd+=['--setenv',k,v]
        cmd+=['--',str(executable),'-u','/app/kernel_worker.py',episode_id]
        errpath=self.out/'kernel_stderr.log'
        if errpath.exists():errpath.rename(self.out/('kernel_stderr.previous_'+str(time.time_ns())+'.log'))
        self.err=errpath.open('w')
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,start_new_session=True,bufsize=0,env={'PATH':'/usr/bin:/bin'})
        self.selector=selectors.DefaultSelector();self.selector.register(self.process.stdout,selectors.EVENT_READ);self.buffer=b''
        try:
            self.initial=self.receive(600)
            if self.initial.get('status')!='READY':raise RuntimeError(str(self.initial))
        except BaseException:self.close();raise
    def receive(self,timeout):
        deadline=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.selector.select(remaining):raise TimeoutError('Kernel process deadline')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('Kernel terminated: '+(self.out/'kernel_stderr.log').read_text()[-3000:])
            self.buffer+=chunk
        line,self.buffer=self.buffer.split(b'\n',1);return json.loads(line)
    def execute(self,code,timeout=90):
        self.process.stdin.write((json.dumps(dict(code=code,timeout_s=timeout))+'\n').encode());self.process.stdin.flush();return self.receive(timeout+15)
    def close(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if self.selector is not None:self.selector.close()
        if self.err is not None:self.err.close()
