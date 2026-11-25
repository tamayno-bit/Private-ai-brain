"""No-GPU unit tests for fail-closed GPU training supervision."""
import json,os,tempfile,time,unittest,io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from learning.opportunistic import primary_guard,gpu_supervisor
from learning.opportunistic.gpu_admission import decision

class PrimaryGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.primary=Path(self.temp.name)
        q=self.primary/"live_queue";q.mkdir()
        (q/"expert_alive.json").write_text(json.dumps({"ts":time.time(),"ready":True}))
    def gpu(self):
        return {"present":True,"gpus":[{"total_mib":32768,"free_mib":24576}]}
    def check(self,**kw):
        with patch.object(primary_guard,"probe",return_value=self.gpu()), patch.object(primary_guard.urllib.request,"urlopen",side_effect=lambda *args,**kwargs:io.BytesIO(b'[{"id":0,"is_processing":false}]')):
            return primary_guard.admission(None,self.primary,**kw)
    def test_denies_busy_inference_slots(self):
        with patch.object(primary_guard,"probe",return_value=self.gpu()), patch.object(primary_guard.urllib.request,"urlopen",return_value=io.BytesIO(b'[{"id":0,"is_processing":true}]')):
            result=primary_guard.admission(None,self.primary,True)
        self.assertEqual(result["reason"],"primary_inference_busy")
    def test_denies_unknown_inference_slot_state(self):
        with patch.object(primary_guard,"probe",return_value=self.gpu()), patch.object(primary_guard.urllib.request,"urlopen",side_effect=OSError("offline")):
            result=primary_guard.admission(None,self.primary,True)
        self.assertEqual(result["reason"],"primary_slots_unavailable")
    def test_denies_vision_request(self):
        q=self.primary/"vision_queue"/"requests";q.mkdir(parents=True)
        (q/"job.working").write_text("{}")
        self.assertEqual(self.check(approved=True)["reason"],"primary_queue_pending")
    def test_denies_without_approval(self):
        self.assertEqual(self.check(approved=False)["reason"],"training_not_approved")
    def test_allows_with_positive_probe_and_approval(self):
        self.assertTrue(self.check(approved=True)["admit"])
    def test_denies_pending_primary_work(self):
        q=self.primary/"coding_queue"/"requests";q.mkdir(parents=True)
        (q/"job.json").write_text("{}")
        self.assertEqual(self.check(approved=True)["reason"],"primary_queue_pending")
    def test_allows_optional_vision_cold_with_fresh_supervisor(self):
        root=self.primary/"vision_queue";root.mkdir(parents=True)
        (root/"on_demand_status.json").write_text(json.dumps({
            "timestamp":time.time(),"supervisor_online":True,"on_demand":True,
            "vision_hot":False,"active_request":False}))
        real=primary_guard.urllib.request.urlopen
        def probe(req,*a,**kwargs):
            if "8772" in str(req):raise OSError("deliberate cold vision")
            return io.BytesIO(b'[{"id":0,"is_processing":false}]')
        with patch.object(primary_guard,"probe",return_value=self.gpu()), patch.object(primary_guard.urllib.request,"urlopen",side_effect=probe):
            result=primary_guard.admission(None,self.primary,True)
        self.assertTrue(result["admit"],result)
    def test_cold_vision_without_supervisor_is_denied(self):
        root=self.primary/"vision_queue";root.mkdir(parents=True)
        (root/"on_demand_status.json").write_text(json.dumps({
            "timestamp":time.time()-30,"supervisor_online":True,"on_demand":True,
            "vision_hot":False,"active_request":False}))
        def probe(req,*args,**kwargs):
            if "8772" in str(req):raise OSError("stale cold vision")
            return io.BytesIO(b'[{"id":0,"is_processing":false}]')
        with patch.object(primary_guard,"probe",return_value=self.gpu()), patch.object(primary_guard.urllib.request,"urlopen",side_effect=probe):
            result=primary_guard.admission(None,self.primary,True)
        self.assertFalse(result["admit"])
        self.assertEqual(result["reason"],"primary_slots_unavailable")
    def test_denies_stale_heartbeat(self):
        p=self.primary/"live_queue"/"expert_alive.json"
        p.write_text(json.dumps({"ts":time.time()-40,"ready":True}))
        self.assertEqual(self.check(approved=True)["reason"],"primary_unhealthy_or_stale")
    def test_denies_low_vram(self):
        self.assertEqual(self.check(approved=True,requested_mib=20000,reserve_mib=8192)["reason"],"insufficient_vram_headroom")
    def test_denies_missing_gpu(self):
        with patch.object(primary_guard,"probe",return_value={"present":False,"gpus":[]}):
            self.assertEqual(primary_guard.admission(None,self.primary,True)["reason"],"no_gpu")
    def test_supervisor_never_starts_without_two_confirmations(self):
        args=SimpleNamespace(max_seconds=30,steps=10,approve=True,output=str(self.primary/"train"),requested_mib=4096,reserve_mib=8192,batch_size=1,grad_accum=1)
        def yes(*a):return {"admit":True}
        with patch.dict(os.environ,{"PRIVATE_BRAIN_GPU_TRAINING_APPROVED":""}):
            result=gpu_supervisor.run(args,gate=yes,popen=lambda *a,**kw:self.fail("unexpected process"))
        self.assertFalse(result["started"])
        self.assertEqual(result["reason"],"explicit_approval_required")
    def test_rejects_unreasonable_vram_reserve(self):
        args=SimpleNamespace(max_seconds=30,steps=10,requested_mib=512,reserve_mib=-10,batch_size=1,grad_accum=1)
        with self.assertRaises(ValueError):gpu_supervisor.run(args)
    def test_gpu_job_requires_working_cuda_runtime(self):
        args=SimpleNamespace(max_seconds=30,steps=10,requested_mib=512,reserve_mib=4096,batch_size=1,grad_accum=1,approve=True,output=str(self.primary/"train"))
        with patch.dict(os.environ,{"PRIVATE_BRAIN_GPU_TRAINING_APPROVED":"YES"}), patch("torch.cuda.is_available",return_value=False):
            result=gpu_supervisor.run(args,gate=lambda *a:{"admit":True},popen=lambda *a,**kw:self.fail("should not start"))
        self.assertEqual(result["reason"],"cuda_runtime_unavailable")
    def test_bad_training_budget_rejected(self):
        args=SimpleNamespace(max_seconds=3600,steps=10)
        with self.assertRaises(ValueError):gpu_supervisor.run(args)
if __name__=="__main__":unittest.main()
