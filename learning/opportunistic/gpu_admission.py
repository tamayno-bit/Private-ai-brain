"""Non-invasive GPU admission checker; does not reserve, start, or rent GPUs."""
import json,subprocess
def decision(free_mib,total_mib,requested_mib=4096,reserve_mib=8192,primary_healthy=False,permissions_ok=False):
    # GPU training is off unless a human explicitly authorizes it per session.
    if not primary_healthy:return {'admit':False,'reason':'primary_not_verified'}
    if not permissions_ok:return {'admit':False,'reason':'training_not_approved'}
    if not isinstance(free_mib,int) or not isinstance(total_mib,int) or total_mib<=0:return {'admit':False,'reason':'unknown_gpu'}
    if free_mib-requested_mib<reserve_mib:return {'admit':False,'reason':'insufficient_vram_headroom'}
    return {'admit':True,'reason':'eligible_for_separate_gpu_worker_only','free_mib':free_mib}
def probe():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
        rows=[tuple(int(z.strip()) for z in row.split(',')) for row in p.stdout.splitlines() if row.strip()]
        return {'present':bool(rows),'gpus':[{'total_mib':a,'free_mib':b} for a,b in rows]}
    except (OSError,ValueError,subprocess.SubprocessError):
        return {'present':False,'gpus':[]}
if __name__=='__main__':
    print(json.dumps({'probe':probe(),'admission':'disabled_by_default','training_started':False}))
