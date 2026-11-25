"""Owner-only localhost management API, minimal stdlib server.

No public bind, no API keys inside Android APK, no automatic GPU provision.
"""
import hmac,json,os,threading,time,importlib.util,collections,urllib.request
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit

class APIServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,addr,token,root):
        self.auth_token=token;self.root=Path(root);super().__init__(addr,Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,data):
        body=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def authorized(self):
        token=self.headers.get('Authorization','')
        return hmac.compare_digest(token,'Bearer '+self.server.auth_token)
    def do_GET(self):
        if not self.authorized():return self.respond(401,{'error':'Unauthorized'})
        path=urlsplit(self.path).path
        cp=self.server.root/'training'/'latest.pt'
        if path.startswith('/v1/brain/md-plans/') and path.endswith('/report'):
            from .md_report import build
            try:return self.respond(200,{'report':build(self.server.root,path.split('/')[-2])})
            except (FileNotFoundError,ValueError,RuntimeError):return self.respond(404,{'error':'Plan report not found'})
        if path.startswith('/v1/brain/md-plans/') and path.endswith('/evidence'):
            from .learning_plans import list_evidence
            try:return self.respond(200,list_evidence(self.server.root,path.split('/')[-2]))
            except (FileNotFoundError,ValueError):return self.respond(404,{'error':'Plan not found'})
        if path.startswith('/v1/brain/md-plans/') and path.endswith('/plan'):
            from .learning_plans import read_plan
            try:return self.respond(200,{'plan':read_plan(self.server.root,path.split('/')[-2])})
            except (FileNotFoundError,ValueError):return self.respond(404,{'error':'Plan not found'})
        if path.startswith('/v1/brain/md-model-review/jobs/'):
            from .md_model_review import get,ModelReviewError
            try:return self.respond(200,{'review':get(self.server.root,path.split('/')[-1])})
            except (ModelReviewError,ValueError,OSError):return self.respond(404,{'error':'Model review not found'})
        if path.startswith('/v1/brain/md-training/jobs/'):
            from .md_training import get_training_job,TrainingPlanError
            try:return self.respond(200,{'job':get_training_job(self.server.root,path.split('/')[-1])})
            except (TrainingPlanError,OSError,ValueError):return self.respond(404,{'error':'Training job not found'})
        if path.startswith('/v1/brain/md-plans/') and path.endswith('/dataset'):
            from .md_plans import ID_RE
            digest=path.split('/')[-2]
            if not ID_RE.fullmatch(digest):return self.respond(404,{'error':'Invalid plan'})
            folder=self.server.root/'md_datasets'/digest
            records=[]
            if folder.is_dir() and not folder.is_symlink():
                for file in list(folder.glob('*.manifest.json'))[:50]:
                    if file.is_symlink():continue
                    try:
                        data=json.loads(file.read_text())
                        if data.get('document_id')==digest:
                            records.append({k:v for k,v in data.items() if not k.endswith('_file')})
                    except (OSError,ValueError):pass
            return self.respond(200,{'datasets':records})
        if path.startswith('/v1/brain/md-research/jobs/'):
            from .md_research_agent import get,PlanJobError
            try:return self.respond(200,{'job':get(self.server.root,path.split('/')[-1])})
            except (PlanJobError,ValueError,OSError):return self.respond(404,{'error':'Research job not found'})
        if path=='/v1/brain/research/providers':
            return self.respond(200,{'providers':['wikipedia','pypi','crossref','github'],
                 'mode':'read_only_public_research','auto_train':False})
        if path=='/v1/brain/md-plans':
            from .md_plans import list_uploads
            return self.respond(200,{'documents':list_uploads(self.server.root)})
        if path.startswith('/v1/brain/md-plans/'):
            from .md_plans import fetch,InvalidMarkdown
            try:return self.respond(200,fetch(self.server.root,path.split('/')[-1]))
            except (FileNotFoundError,InvalidMarkdown):return self.respond(404,{'error':'Document not found'})
            except (OSError,ValueError,RuntimeError):return self.respond(409,{'error':'Document integrity error'})
        if path=='/v1/brain/health':return self.respond(200,{'ready':True,'management_api_ready':True,'inference_ready':cp.is_file() and importlib.util.find_spec('torch') is not None,'model_checkpoint_available':cp.is_file(),'mode':'local_owner_only','integration':'sidecar_only'})
        if path=='/v1/brain/integration':
            # Safe, read-only local health probe. No auth secrets leave this service.
            try:
                with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=1.0) as response:
                    upstream_ok = response.status == 200
            except (OSError,ValueError):
                upstream_ok = False
            return self.respond(200,{'original_brain_health_reachable':upstream_ok,'integration_mode':'separate_sidecar','modified_original_brain':False,'training_autostart':False})
        if path=='/v1/brain/status':
            info={'trained':cp.is_file(),'checkpoint_present':cp.is_file(),'training_running':False,'gpu_rented_by_service':False,'torch_available':importlib.util.find_spec('torch') is not None,'integration':'sidecar_only','continuous_learning_enabled':False}
            # GPU process lives on a separate Pod. Shared status is historical,
            # not a reliable real-time running flag or proof of an active GPU lease.
            status_path=self.server.root/'gpu_supervisor_status.json'
            try:
                last=json.loads(status_path.read_text())
                if isinstance(last,dict):
                    info['gpu_training_last_result']={key:last.get(key) for key in
                        ('state','stop_reason','checkpoint_step','checkpoint_available','updated_at') if key in last}
            except (OSError,ValueError):
                pass
            observer_file=self.server.root/'gpu_observer.json'
            try:
                observer=json.loads(observer_file.read_text())
                fresh=time.time()-float(observer['updated_at'])<35
                info['gpu_observer']={
                    'online':fresh and bool(observer.get('gpu')),
                    'last_seen':observer.get('updated_at'),
                    'gpu':observer.get('gpu',[]) if fresh else [],
                    'primary':observer.get('primary',{}) if fresh else {},
                }
            except (OSError,ValueError,TypeError,KeyError):
                info['gpu_observer']={'online':False}
            # Shared on-demand Vision supervisor: cold Vision is a healthy resource
            # saving state, not a failed primary model.
            vision_file=Path('/workspace/services/original-brain-v1/vision_queue/on_demand_status.json')
            try:
                vision=json.loads(vision_file.read_text())
                fresh=0<=time.time()-float(vision['timestamp'])<8
                info['vision_resource_manager']={
                    'online':fresh and vision.get('supervisor_online') is True,
                    'on_demand':fresh and vision.get('on_demand') is True,
                    'gpu_memory_loaded':vision.get('vision_hot') is True if fresh else None,
                    'active_request':vision.get('active_request') is True if fresh else None,
                    'idle_seconds':vision.get('idle_seconds') if fresh else None,
                    'reason':vision.get('reason') if fresh else 'supervisor_offline_or_stale',
                    'last_seen':vision.get('timestamp'),
                    'gpu_pod_stopped':False}
            except (OSError,ValueError,TypeError,KeyError):
                info['vision_resource_manager']={'online':False,'on_demand':False}
            # Read-only, time-bounded background learning telemetry from the GPU Pod.
            # Production inference checkpoint is intentionally not replaced by candidates.
            try:
                learning=json.loads((self.server.root/'autolearn'/'auto_learning_status.json').read_text())
                fresh=time.time()-float(learning['updated_at'])<95
                info['autonomous_learning']={
                    'online':fresh and learning.get('learning_enabled') is True,
                    'phase':learning.get('phase') if fresh else 'offline_or_stale',
                    'step':learning.get('step'),
                    'target_step':learning.get('target_step'),
                    'best_step':learning.get('best_step'),
                    'best_loss':learning.get('best_loss'),
                    'independent_holdout_loss':learning.get('independent_holdout_loss'),
                    'patience':learning.get('patience'),
                    'last_seen':learning.get('updated_at'),
                    'candidate_promoted_to_live':False,
                }
                if fresh:
                    info['training_running']=learning.get('phase')=='training'
                    info['continuous_learning_enabled']=learning.get('learning_enabled') is True
            except (OSError,ValueError,TypeError,KeyError):
                info['autonomous_learning']={'online':False,'phase':'no_status'}
            if cp.is_file():
                manifest_path=Path(str(cp)+'.json')
                if manifest_path.exists():
                    try:
                        manifest=json.loads(manifest_path.read_text());info.update({'step':manifest.get('step'),'model_config':manifest.get('model_config')})
                    except (OSError,ValueError):info['checkpoint_metadata_error']=True
            return self.respond(200,info)
        if path=='/v1/brain/reports':
            fp=self.server.root/'training'/'metrics.jsonl'
            events=[]
            if fp.exists():
                with fp.open(encoding='utf-8') as log:
                    for line in collections.deque(log,maxlen=100):
                        try:events.append(json.loads(line))
                        except json.JSONDecodeError:pass
            return self.respond(200,{'events':events,'metrics_type':'training_and_holdout_loss_not_intelligence_score'})
        return self.respond(404,{'error':'Not found'})
    def do_POST(self):
        if not self.authorized():return self.respond(401,{'error':'Unauthorized'})
        route=urlsplit(self.path).path
        if route.startswith('/v1/brain/md-plans/') and route.endswith('/model-review'):
            from .md_model_review import submit,ModelReviewError
            from .md_plans import ID_RE
            parts=route.split('/')
            if len(parts)!=6 or not ID_RE.fullmatch(parts[4]):
                return self.respond(404,{'error':'Review route not found'})
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=200):return self.respond(413,{'error':'Invalid request size'})
                data=json.loads(self.rfile.read(n))
                if not isinstance(data,dict) or set(data)!={'approved','task_index'}:
                    return self.respond(400,{'error':'Owner approval and one task index required'})
                result,created=submit(self.server.root,parts[4],data['approved'],data['task_index'])
                return self.respond(202 if created else 200,{'review':result})
            except (ModelReviewError,ValueError,TypeError):
                return self.respond(400,{'error':'MD local model review not approved'})
        if route.startswith('/v1/brain/md-plans/') and (
             route.endswith('/dataset/propose') or route.endswith('/dataset/approve')):
            from .md_dataset import propose,DatasetPlanError
            from .md_training import approve_training,TrainingPlanError
            from .md_plans import ID_RE
            parts=route.split('/')
            if len(parts)!=7 or not ID_RE.fullmatch(parts[4]) or parts[5]!='dataset':
                return self.respond(404,{'error':'Dataset route not found'})
            doc=parts[4]
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=2000):return self.respond(413,{'error':'Invalid request size'})
                body=json.loads(self.rfile.read(n))
                if not isinstance(body,dict):return self.respond(400,{'error':'Invalid JSON'})
                if route.endswith('/propose'):
                    if set(body)!={'approved'} or body.get('approved') is not True:
                        return self.respond(400,{'error':'Dataset inspection approval required'})
                    return self.respond(200,{'dataset':propose(self.server.root,doc)})
                if set(body)!={'dataset_id','training_approved','rights_reviewed','max_steps'}:
                    return self.respond(400,{'error':'Invalid dataset approval'})
                result,created=approve_training(self.server.root,doc,body['dataset_id'],
                    body['training_approved'],body['rights_reviewed'],body['max_steps'])
                return self.respond(202 if created else 200,{'job':result})
            except (DatasetPlanError,TrainingPlanError,ValueError,TypeError):
                return self.respond(400,{'error':'Invalid, unapproved or unlicensed MD dataset'})
            except (OSError,RuntimeError):return self.respond(409,{'error':'Dataset preparation unavailable'})
        if route=='/v1/brain/md-research/start':
            from .md_research_agent import start,PlanJobError
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=5000):return self.respond(413,{'error':'Invalid request size'})
                payload=json.loads(self.rfile.read(n))
                if not isinstance(payload,dict) or set(payload)-{'plan_id','query','pages','tongyi','public_query_approved','task_index'}:
                    return self.respond(400,{'error':'Unexpected research parameters'})
                response=start(self.server.root,payload.get('plan_id'),payload.get('query'),
                    payload.get('pages',1),payload.get('tongyi',True),
                    payload.get('public_query_approved',False),payload.get('task_index'))
                return self.respond(202,response)
            except (PlanJobError,ValueError,TypeError):return self.respond(400,{'error':'Invalid or unapproved MD research'})
            except (OSError,RuntimeError):return self.respond(503,{'error':'Research service unavailable'})
        if route=='/v1/brain/research':
            from .research import search,ResearchError
            from .learning_plans import append_research
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=6000):return self.respond(413,{'error':'Invalid request size'})
                body=json.loads(self.rfile.read(n))
                if not isinstance(body,dict) or body.get('approved') is not True:
                    return self.respond(400,{'error':'Explicit research approval required'})
                if set(body)-{'query','provider','approved','plan_id'}:
                    return self.respond(400,{'error':'Unexpected research parameters'})
                result=search(body.get('query'),body.get('provider','wikipedia'))
                if body.get('plan_id'):
                    result['plan_evidence']=append_research(self.server.root,body['plan_id'],result)
                return self.respond(200,result)
            except (ValueError,TypeError,ResearchError):return self.respond(400,{'error':'Invalid or unavailable research request'})
            except FileNotFoundError:return self.respond(404,{'error':'Learning plan not found'})
            except OSError:return self.respond(503,{'error':'Research temporarily unavailable'})
        if route.startswith('/v1/brain/md-plans/') and (
            route.endswith('/analyze') or route.endswith('/approve')):
            from .learning_plans import save_draft,approve
            from .md_plans import InvalidMarkdown
            document_id=route.split('/')[-2]
            from .md_plans import ID_RE
            if len(route.split('/'))!=6 or not ID_RE.fullmatch(document_id):
                return self.respond(404,{'error':'Plan not found'})
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=200):return self.respond(413,{'error':'Invalid request size'})
                body=json.loads(self.rfile.read(n))
                if not isinstance(body,dict) or set(body)!={'approved'} or body['approved'] is not True:
                    return self.respond(400,{'error':'Explicit request approval required'})
                if route.endswith('/analyze'):
                    plan,created=save_draft(self.server.root,document_id)
                    return self.respond(201 if created else 200,{'plan':plan})
                return self.respond(200,{'plan':approve(self.server.root,document_id,True)})
            except (InvalidMarkdown,ValueError,TypeError):return self.respond(400,{'error':'Invalid learning plan request'})
            except FileNotFoundError:return self.respond(404,{'error':'Document or plan not found'})
            except OSError:return self.respond(409,{'error':'Learning plan unavailable'})
        if route=='/v1/brain/md-plans':
            from .md_plans import store,InvalidMarkdown
            try:
                n=int(self.headers.get('Content-Length','-1'))
                if not (0<n<=200000):return self.respond(413,{'error':'Invalid request size'})
                payload=json.loads(self.rfile.read(n))
                if not isinstance(payload,dict) or set(payload)!={'filename','content'}:
                    raise InvalidMarkdown('Invalid upload fields')
                answer=store(self.server.root,payload['filename'],payload['content'])
                return self.respond(200 if answer['already_exists'] else 201,answer)
            except (ValueError,TypeError,InvalidMarkdown,UnicodeError):
                return self.respond(400,{'error':'Invalid Markdown document'})
            except (OSError,RuntimeError):
                return self.respond(409,{'error':'Failed to save verified document'})
        if urlsplit(self.path).path!='/v1/brain/exam':return self.respond(404,{'error':'Not found'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not (0<length<=12000):return self.respond(413,{'error':'Invalid request size'})
            body=json.loads(self.rfile.read(length));prompt=body['prompt']
            if not isinstance(prompt,str) or not 1<=len(prompt)<=2000:raise ValueError('Invalid prompt')
            tokens=body.get('max_new_tokens',48)
            if type(tokens) is not int or not 1<=tokens<=128:raise ValueError('Invalid token count')
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):return self.respond(400,{'error':'Invalid JSON'})
        cp=self.server.root/'training'/'latest.pt'
        if not cp.exists():return self.respond(409,{'error':'No trained checkpoint. Train explicitly before exam.'})
        # Explicit lock prevents simultaneous inference from exhausting CPU.
        if not EXAM_LOCK.acquire(blocking=False):return self.respond(429,{'error':'Exam busy'})
        try:
            if importlib.util.find_spec('torch') is None:return self.respond(503,{'error':'PyTorch inference dependency not yet installed'})
            from .infer import generate
            self.respond(200,generate(cp,prompt,tokens))
        except Exception:return self.respond(500,{'error':'Inference failed; inspect private server logs'})
        finally:EXAM_LOCK.release()

EXAM_LOCK=threading.Lock()
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8877)
    p.add_argument('--state-root',default='state');args=p.parse_args()
    if args.host not in ('127.0.0.1','::1'):raise SystemExit('Public bind disabled; use an authenticated TLS gateway for Android.')
    token=os.getenv('PRIVATE_BRAIN_API_TOKEN','')
    if len(token)<32:raise SystemExit('PRIVATE_BRAIN_API_TOKEN must be at least 32 characters; service will not start without it')
    APIServer((args.host,args.port),token,args.state_root).serve_forever()
if __name__=='__main__':main()
