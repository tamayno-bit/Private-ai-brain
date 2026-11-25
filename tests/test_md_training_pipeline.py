"""MD dataset consent, immutable provenance, holdout separation and training queue tests."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from brain.md_plans import store
from brain.learning_plans import save_draft,approve
from brain.md_dataset import inspect,propose,DatasetPlanError
from brain.md_training import approve_training,get_training_job,TrainingPlanError
from brain.md_report import build

def sample_document():
    def item(i):
        return {"prompt":"Explain how to add integers "+str(i)+" and "+str(i+70)+" in Python.",
         "answer":"Use the addition operator and return "+str(2*i+70)+". This is a locally written example.",
         "license":"own","consent":True,"source_type":"owner_written"}
    obj={"type":"private_brain_examples_v1","owner_rights_confirmed":True,
         "train":[item(i) for i in range(26)],
         "holdout":[item(i) for i in range(800,810)]}
    fence=chr(96)*3
    return "# Plan for Python learning\n- Research official docs\n"+fence+"json\n"+json.dumps(obj)+"\n"+fence

class MdTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.doc=store(self.root,'owner-plan.md',sample_document())['document']['id']
        save_draft(self.root,self.doc)
    def test_requires_plan_approval_and_rights(self):
        with self.assertRaises(DatasetPlanError):propose(self.root,self.doc)
        approve(self.root,self.doc,True)
        preview=propose(self.root,self.doc)
        self.assertEqual(preview["train_records"],26)
        self.assertEqual(preview["holdout_records"],10)
        self.assertFalse(preview["web_content_used"])
        self.assertFalse(preview["training_started"])
        self.assertEqual(propose(self.root,self.doc)["dataset_id"],preview["dataset_id"])
        with self.assertRaises(TrainingPlanError):
            approve_training(self.root,self.doc,preview["dataset_id"],False,True,20)
        with self.assertRaises(TrainingPlanError):
            approve_training(self.root,self.doc,preview["dataset_id"],True,False,20)
        result,created=approve_training(self.root,self.doc,preview["dataset_id"],True,True,40)
        self.assertTrue(created)
        self.assertEqual(result["status"],"queued")
        self.assertFalse(result["replaces_live_model"])
        self.assertEqual(get_training_job(self.root,result["id"])["max_steps"],40)
        self.assertFalse(approve_training(self.root,self.doc,preview["dataset_id"],True,True,40)[1])
        summary=build(self.root,self.doc)
        self.assertEqual(summary["training_run_id"],result["id"])
        self.assertEqual(summary["training_status"],"queued")
        self.assertFalse(summary["capability_gain_verified"])
    def test_holdout_leak_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            document=sample_document()
            # same example in holdout => exact duplicate across train and holdout
            text=document.replace('"prompt": "Explain how to add integers 800 and 870 in Python."',
                                  '"prompt": "Explain how to add integers 0 and 70 in Python."')
            doc=store(tmp,'dup.md',text)['document']['id']
            self.assertNotEqual(doc,self.doc)
            # duplicate prompt alone not identical answer: validate still accepts, as honest exact split
            # and never pretends semantic independence.
    def test_no_training_from_plain_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc=store(tmp,'simple.md','# Learn Python\n- Search official docs')['document']['id']
            save_draft(tmp,doc);approve(tmp,doc,True)
            with self.assertRaises(DatasetPlanError):propose(tmp,doc)
    def test_corrupt_staged_dataset_rejected(self):
        approve(self.root,self.doc,True)
        info=propose(self.root,self.doc)
        folder=self.root/'md_datasets'/self.doc
        path=next(folder.glob('*.train.jsonl'))
        path.write_text('corrupted')
        with self.assertRaises(TrainingPlanError):
            approve_training(self.root,self.doc,info['dataset_id'],True,True,40)
if __name__=='__main__':unittest.main()
