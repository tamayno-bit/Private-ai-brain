"""MD search integration tests: genuine owner approval and source evidence only."""
import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from brain import md_research_agent as agent
from brain import md_report
from brain.md_plans import store
from brain.learning_plans import save_draft,approve

class MdResearchAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.digest=store(self.root,"goals.md","# بحث Python\n- تعلم البرمجة")["document"]["id"]
        save_draft(self.root,self.digest)
    def test_unapproved_plan_denied(self):
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"Python documentation",public_query_approved=True,task_index=0)
    def test_query_permission_and_private_data_denied(self):
        approve(self.root,self.digest,True)
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"Python documentation",public_query_approved=False,task_index=0)
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"/workspace/private/project files",public_query_approved=True,task_index=0)
    def test_completes_existing_original_brain_job_with_grounded_evidence(self):
        approve(self.root,self.digest,True)
        fake_id="a"*32
        def upstream(method,path,body=None,timeout=7):
            if method=="POST":
                self.assertEqual(path,"/v1/research/fast-start")
                self.assertEqual(body["approved"],True)
                self.assertEqual(body["question"],"official Python documentation")
                return 202,{"id":fake_id}
            self.assertEqual(path,"/v1/research/jobs/"+fake_id)
            return 200,{"stage":"ready","sources":[{"id":1,"title":"Python Docs","url":"https://docs.python.org/3/",
                       "snippet":"Reference documentation"}],"answer":"Python docs [1]",
                        "model_inference_used":True,"citations_valid_ids":True}
        with patch.object(agent,"_local_original",side_effect=upstream):
            reply=agent.start(self.root,self.digest,"official Python documentation",
                     pages=1,tongyi=True,public_query_approved=True,task_index=0)
            deadline=time.time()+4
            while time.time()<deadline:
                data=agent.get(self.root,reply["job_id"])
                if data["phase"] in ("ready","failed"):break
                time.sleep(.02)
        self.assertEqual(data["phase"],"ready",data.get("error"))
        self.assertTrue(data["model_inference_used"])
        self.assertEqual(data["task_index"],0)
        self.assertFalse(data["whole_md_sent_to_tongyi"])
        self.assertTrue(data["one_task_at_a_time"])
        self.assertTrue(data["citations_valid_ids"])
        self.assertEqual(data["source_count"],1)
        status=md_report.build(self.root,self.digest)
        self.assertEqual(status["research_job_count"],1)
        self.assertEqual(status["evidence_count"],1)
        self.assertEqual(status["training_status"],"not_started_from_this_md")
        self.assertFalse(status["capability_gain_verified"])
        self.assertFalse(status["privacy"]["private_document_sent_to_web"])
        self.assertFalse(status["research_evidence"][0]["results"][0]["training_rights_verified"])
    def test_rejects_missing_task_and_concurrent_next_step(self):
        approve(self.root,self.digest,True)
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"Python docs",public_query_approved=True)
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"Python docs",public_query_approved=True,task_index=99)
        # Simulated durable queued task must block another public search.
        job_id="b"*32
        agent._persist(agent._job_path(self.root,job_id),{
          "id":job_id,"plan_id":self.digest,"phase":"searching","updated_at":time.time()})
        with self.assertRaises(agent.PlanJobError):
            agent.start(self.root,self.digest,"Python docs",public_query_approved=True,task_index=0)
    def test_unknown_job_and_untrusted_requests(self):
        with self.assertRaises(agent.PlanJobError):agent.get(self.root,"../some_file")
        self.assertFalse((self.root/"md_research_jobs"/("a"*32+".json")).is_file())
if __name__=="__main__":unittest.main()
