"""Local-only Tongyi MD review is owner-confirmed and non-executing."""
import json,tempfile,unittest
from pathlib import Path
from brain.md_plans import store
from brain.learning_plans import save_draft,approve
from brain.md_model_review import submit,get,list_for_plan,ModelReviewError

class LocalTongyiReviewTests(unittest.TestCase):
    def test_requires_plan_owner_consent_and_keeps_text_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc=store(tmp,'plan.md','# تعلم البرمجة\n- ابحث عن مكتبات PyTorch')['document']['id']
            save_draft(tmp,doc)
            with self.assertRaises(ModelReviewError):submit(tmp,doc,True,task_index=0)
            approve(tmp,doc,True)
            with self.assertRaises(ModelReviewError):submit(tmp,doc,False,task_index=0)
            job,created=submit(tmp,doc,True,task_index=0)
            self.assertTrue(created)
            self.assertTrue(job['private_local_inference_only'])
            self.assertFalse(job['external_web_search_used'])
            self.assertEqual(job['task_index'],0)
            self.assertFalse(job['whole_document_sent'])
            self.assertLess(job['task_characters'],120)
            self.assertFalse(job['instructions_active'])
            self.assertFalse(job['training_started'])
            same,created=submit(tmp,doc,True,task_index=0)
            self.assertFalse(created)
            self.assertEqual(same['id'],job['id'])
            payload=json.loads((Path(tmp)/'md_local_model_review'/'requests'/(job['id']+'.json')).read_text())
            self.assertIn('PyTorch',payload['content'])
            self.assertEqual(list_for_plan(tmp,doc)[0]['id'],job['id'])
            reply=Path(tmp)/'md_local_model_review'/'responses'/(job['id']+'.json')
            reply.write_text(json.dumps({'id':job['id'],'model_inference_used':True,'answer':'المهارات المطلوبة: Python','error':None}))
            refreshed=get(tmp,job['id'])
            self.assertEqual(refreshed['status'],'completed')
            self.assertIn('Python',refreshed['answer'])
            self.assertFalse(refreshed['model_output_trusted'])
    def test_invalid_review_id_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ModelReviewError):get(tmp,'../../')
if __name__=='__main__':unittest.main()
