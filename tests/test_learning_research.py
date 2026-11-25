"""Owner-approved Markdown-plan and read-only public research regression."""
import json,tempfile,unittest
from unittest.mock import patch
from brain import learning_plans,research
from brain.md_plans import store

class PlanResearchTests(unittest.TestCase):
    def test_plan_is_not_training_or_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc=store(tmp,"learn.md","# بحث المكتبات\n- فحص PyTorch\n- تدريب الموديل\n\n"+chr(96)*3+"bash\nrm -rf /\n"+chr(96)*3)
            ident=doc["document"]["id"]
            plan,created=learning_plans.save_draft(tmp,ident)
            self.assertTrue(created)
            self.assertEqual(plan["status"],"draft_pending_approval")
            self.assertEqual(len(plan["tasks"]),2)
            self.assertTrue(all(not task["code_execution_allowed"] and not task["automated_training_allowed"] for task in plan["tasks"]))
            self.assertFalse(plan["automatic_training"])
            self.assertEqual(learning_plans.read_plan(tmp,ident)["status"],"draft_pending_approval")
            with self.assertRaises(ValueError):learning_plans.approve(tmp,ident,False)
            result=learning_plans.approve(tmp,ident,True)
            self.assertEqual(result["status"],"approved_for_review_and_research")
            self.assertFalse(result["automatic_training"])
            research_result={"query":"torch","provider":"pypi","retrieved_at":123.0,"external_data_for_training":False,"results":[{"title":"PyTorch","training_rights_verified":False}]}
            evidence=learning_plans.append_research(tmp,ident,research_result)
            self.assertEqual(evidence["references"],1)
            self.assertFalse(evidence["dataset_ingested"])
            indexed=learning_plans.list_evidence(tmp,ident)
            self.assertEqual(len(indexed["evidence"]),1)
            self.assertEqual(indexed["evidence"][0]["evidence_id"],evidence["evidence_id"])
            self.assertFalse(indexed["evidence"][0]["dataset_ingested"])
    def test_evidence_refuses_unapproved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            ident=store(tmp,"a.md","# Research")["document"]["id"]
            learning_plans.save_draft(tmp,ident)
            with self.assertRaises(ValueError):
                learning_plans.append_research(tmp,ident,{"results":[]})
    def test_query_secret_guard_and_provider(self):
        for q in ("Authorization: Bearer secret","token=hidden","a",chr(10)+"hello"):
            with self.assertRaises(research.ResearchError):research.validate_query(q)
        with self.assertRaises(research.ResearchError):research.search("test","arbitrary")
        with self.assertRaises(research.ResearchError):research.pypi("../etc/passwd")
    def test_search_metadata_and_external_rights(self):
        with patch.object(research,"_json",return_value={"info":{"name":"torch","summary":"A deep learning library","version":"2.9.0","license":"BSD-3-Clause"}}) as call:
            results=research.search("torch","pypi")
        self.assertTrue(call.called)
        self.assertEqual(len(results["results"]),1)
        self.assertFalse(results["results"][0]["package_installed"])
        self.assertFalse(results["external_data_for_training"])
        self.assertEqual(results["results"][0]["version"],"2.9.0")
    def test_public_github_metadata_never_executes_code(self):
        with patch.object(research,"_json",return_value={"items":[
           {"full_name":"open/test","html_url":"https://github.com/open/test",
            "description":"A repository","stargazers_count":42,"language":"Python",
            "license":{"spdx_id":"MIT"}}]}):
            result=research.search("python test","github")
        self.assertEqual(len(result["results"]),1)
        row=result["results"][0]
        self.assertEqual(row["license_declared"],"MIT")
        self.assertFalse(row["code_executed"])
        self.assertFalse(row["repository_cloned"])
        self.assertFalse(row["training_rights_verified"])
    def test_allowlisted_host(self):
        with self.assertRaises(research.ResearchError):
            research._json("http://pypi.org/pypi/torch/json","pypi.org")
        with self.assertRaises(research.ResearchError):
            research._json("https://evil.example/","pypi.org")
if __name__=="__main__":unittest.main()
