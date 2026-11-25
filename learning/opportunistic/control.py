"""Local controls for training loop only; never stop, start, or rent RunPod GPU."""
import argparse,json,os,signal,subprocess,sys
from pathlib import Path
from . import autolearn
STATE=autolearn.STATE
PID=Path('/tmp/private-brain-autolearn.pid')
def active():
    try:
        pid=int(PID.read_text())
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        if b'learning.opportunistic.autolearn' not in cmd:return None
        return pid
    except (OSError,ValueError):return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('status','pause','resume'))
    args=p.parse_args()
    STATE.mkdir(parents=True,exist_ok=True)
    if args.action=='pause':
        (STATE/'PAUSE').touch(mode=0o600,exist_ok=True)
        print(json.dumps({'pause_requested':True,'gpu_pod_shutdown':False,'effect':'next training round'}))
        return
    if args.action=='status':
        try:status=json.loads((STATE/'auto_learning_status.json').read_text())
        except (OSError,ValueError):status={}
        print(json.dumps({'controller_running':active() is not None,'pause_requested':(STATE/'PAUSE').exists(),
                          'state':status},ensure_ascii=False))
        return
    # Resume never provisions compute. It refuses CPU-only environments.
    if not autolearn.approved(STATE,json.loads((STATE/'approval.json').read_text())['gpu_pod_id']):
        # Approval would be false if PAUSE exists, which is expected for resume.
        approval=json.loads((STATE/'approval.json').read_text())
        if approval.get('approved') is not True:raise SystemExit('Not approved')
    import torch
    if not torch.cuda.is_available():raise SystemExit('GPU runtime unavailable')
    (STATE/'PAUSE').unlink(missing_ok=True)
    if active() is None:
        approval=json.loads((STATE/'approval.json').read_text())
        env=os.environ.copy();env.update({'PRIVATE_BRAIN_GPU_TRAINING_APPROVED':'YES',
                                           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        with (STATE/'controller.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-m','learning.opportunistic.autolearn',
                '--pod-id',approval['gpu_pod_id'],'--poll-seconds','20'],
                cwd=str(autolearn.ROOT),stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        PID.write_text(str(process.pid))
    print(json.dumps({'training_controller_running':active() is not None,'pause_removed':True,
                      'gpu_pod_shutdown':False}))
if __name__=='__main__':main()
