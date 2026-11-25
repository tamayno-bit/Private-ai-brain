"""MD upload acceptance: no training, instruction activation or command execution."""
import json, tempfile, threading, unittest, http.client
from pathlib import Path
from brain.api import APIServer
from brain.md_plans import store,fetch,list_uploads,InvalidMarkdown

class MarkdownPlansTests(unittest.TestCase):
    def test_roundtrip_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            value=store(tmp,'خطة-التعلم.md','# خطة التعلم\n- تحسين البرمجة')
            ident=value['document']['id']
            self.assertEqual(value['document']['status'],'pending_owner_review')
            self.assertFalse(value['document']['training_enabled'])
            self.assertEqual(fetch(tmp,ident)['content'],'# خطة التعلم\n- تحسين البرمجة')
            self.assertEqual(len(list_uploads(tmp)),1)
            self.assertTrue(store(tmp,'خطة-التعلم.md','# خطة التعلم\n- تحسين البرمجة')['already_exists'])
            with self.assertRaises(InvalidMarkdown):store(tmp,'../escape.md','hi')
            with self.assertRaises(InvalidMarkdown):store(tmp,'bad.txt','hi')
            with self.assertRaises(InvalidMarkdown):store(tmp,'huge.md','x'*131073)
    def test_authenticated_http_upload_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            srv=APIServer(('127.0.0.1',0),'t'*48,tmp)
            thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
            def req(method,path,payload=None,auth=True):
                headers={'Content-Type':'application/json'}
                if auth:headers['Authorization']='Bearer '+'t'*48
                conn=http.client.HTTPConnection('127.0.0.1',srv.server_address[1],timeout=5)
                conn.request(method,path,body=json.dumps(payload).encode() if payload is not None else None,headers=headers)
                result=conn.getresponse();status=result.status;data=json.loads(result.read());conn.close()
                return status,data
            try:
                data={'filename':'work-plan.md','content':'# خطة\n- تطوير الموديل'}
                self.assertEqual(req('POST','/v1/brain/md-plans',data,False)[0],401)
                status,created=req('POST','/v1/brain/md-plans',data)
                self.assertEqual(status,201)
                self.assertEqual(created['document']['status'],'pending_owner_review')
                docid=created['document']['id']
                self.assertEqual(req('POST','/v1/brain/md-plans',data)[0],200)
                self.assertEqual(req('GET','/v1/brain/md-plans')[1]['documents'][0]['id'],docid)
                self.assertEqual(req('GET','/v1/brain/md-plans/'+docid)[1]['content'],data['content'])
                self.assertEqual(req('POST','/v1/brain/md-plans',{'filename':'a.md','content':'x'*131073})[0],400)
                self.assertEqual(req('POST','/v1/brain/md-plans/approve',{'approved':True})[0],404)
            finally:srv.shutdown();srv.server_close()
