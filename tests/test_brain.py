import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from brain.model import BrainTransformer,ModelConfig
from brain.tokenizer import ByteTokenizer
from brain.dataset import validate_record,load_records,token_stream,split_stream
from brain.checkpoint import load_checkpoint
from brain.train import run
from brain.infer import generate
from brain.evaluate import evaluate
from brain.memory import Memory

ROOT=Path(__file__).resolve().parents[1]
class ArchitectureTests(unittest.TestCase):
    def test_config_requires_valid_gqa(self):
        with self.assertRaises(ValueError):ModelConfig(dim=60,heads=8).validate()
    def test_forward_and_gradients(self):
        c=ModelConfig(context=12,dim=32,layers=1,heads=4,kv_heads=2)
        model=BrainTransformer(c)
        ids=torch.randint(0,260,(2,10));logits,loss=model(ids,ids)
        self.assertEqual(tuple(logits.shape),(2,10,260));loss.backward()
        self.assertIsNotNone(model.emb.weight.grad)
        self.assertGreater(float(model.emb.weight.grad.norm()),0)
    def test_causal_mask(self):
        c=ModelConfig(context=12,dim=32,layers=1,heads=4,kv_heads=2)
        model=BrainTransformer(c).eval()
        x=torch.randint(0,260,(1,9));y=x.clone();y[0,-1]=(y[0,-1]+1)%260
        with torch.no_grad():a,_=model(x);b,_=model(y)
        self.assertTrue(torch.allclose(a[:,:-1],b[:,:-1],atol=1e-5))
    def test_tokenizer_round_trip(self):
        t=ByteTokenizer();s='سلام — مرحبا 👋';self.assertEqual(t.decode(t.encode(s,bos=True,eos=True)),s)
    def test_input_validation(self):
        with self.assertRaises(ValueError):validate_record({'text':'hi','license':'own','source':'me'})
        with self.assertRaises(ValueError):validate_record({'text':'hi','license':'unknown','source':'me','consent':True})
    def test_memory_kept_separate(self):
        with tempfile.TemporaryDirectory() as d:
            mem=Memory(Path(d)/'memory.db');mem.add('Project note');self.assertEqual(mem.search('Project')[0]['note'],'Project note')
    def test_pretraining_checkpoint_resume_and_exam(self):
        with tempfile.TemporaryDirectory() as d:
            dst=Path(d)
            a=SimpleNamespace(model_config=str(ROOT/'config/model_tiny.json'),data=str(ROOT/'config/dataset_example.jsonl'),output=str(dst/'training'),resume=None,
              steps=2,batch_size=1,grad_accum=1,lr=0.001,save_every=1,log_every=1,seed=42)
            run(a)
            cp=dst/'training/latest.pt';self.assertTrue(cp.exists());self.assertTrue((dst/'training/latest.pt.json').exists())
            self.assertEqual(json.loads((dst/'training/latest.pt.json').read_text())['step'],2)
            a.resume=str(cp);a.steps=3;run(a)
            self.assertEqual(json.loads((dst/'training/latest.pt.json').read_text())['step'],3)
            out=generate(cp,'Hello',3)
            self.assertEqual(out['trained_step'],3);self.assertFalse(out['used_external_model'])
            metric=evaluate(cp,a.data);self.assertGreater(metric['perplexity'],0)
            model=BrainTransformer(ModelConfig(**json.loads(a.model_config and Path(a.model_config).read_text())))
            with self.assertRaises(ValueError):load_checkpoint(cp,model,expected_dataset='tampered')
            before=cp.read_bytes();cp.write_bytes(before+b'tampered')
            with self.assertRaises(ValueError):load_checkpoint(cp,model)
if __name__=='__main__':unittest.main()

class APITests(unittest.TestCase):
    def test_auth_and_exam_untrained(self):
        import http.client,threading
        from brain.api import APIServer
        with tempfile.TemporaryDirectory() as d:
            srv=APIServer(('127.0.0.1',0),'x'*48,d)
            thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
            try:
                con=http.client.HTTPConnection('127.0.0.1',srv.server_address[1],timeout=5)
                con.request('GET','/v1/brain/health');res=con.getresponse();self.assertEqual(res.status,401);res.read()
                con.request('GET','/v1/brain/status',headers={'Authorization':'Bearer '+'x'*48})
                res=con.getresponse();self.assertEqual(res.status,200);self.assertFalse(json.loads(res.read())['trained'])
                con.request('POST','/v1/brain/exam',body=json.dumps({'prompt':'2+2'}),headers={'Authorization':'Bearer '+'x'*48,'Content-Type':'application/json'})
                res=con.getresponse();self.assertEqual(res.status,409);res.read();con.close()
            finally:srv.shutdown();srv.server_close()
