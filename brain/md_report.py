"""Owner-facing evidence-based progress report, never invents learning outcomes."""
import json,time
from pathlib import Path
from .learning_plans import read_plan,list_evidence
from .md_research_agent import JOBS,STATES

def build(root,plan_id):
    root=Path(root)
    plan=read_plan(root,plan_id)
    gathered=list_evidence(root,plan_id)
    records=gathered.get("evidence",[])
    jobs=[]
    directory=root/JOBS
    if directory.is_dir() and not directory.is_symlink():
        for path in directory.glob("*.json"):
            if path.is_symlink():continue
            try:
                item=json.loads(path.read_text())
                if item.get("plan_id")==plan_id and item.get("phase") in STATES:
                    jobs.append({k:item.get(k) for k in
                        ("id","task_index","query","phase","source_count","model_inference_used","citations_valid_ids",
                         "created_at","updated_at","evidence_id","error")})
            except (OSError,ValueError,TypeError):continue
    jobs.sort(key=lambda x:x.get("created_at") or 0,reverse=True)
    # A supervised MD pilot can use several verified read-only adapters and
    # produce staged evidence without invoking the asynchronous Tongyi agent.
    # Never equate completed research with completed training.
    pilot_data={}
    pilot_path=root/"md_plan_executions"/(plan_id+".json")
    if pilot_path.is_file() and not pilot_path.is_symlink():
        try:
            candidate=json.loads(pilot_path.read_text())
            if candidate.get("plan_id")==plan_id and isinstance(candidate.get("tasks"),list):
                pilot_data=candidate
        except (OSError,ValueError,TypeError):pass
    pilot_tasks={x.get("task_index"):x for x in pilot_data.get("tasks",[])
                 if isinstance(x,dict) and type(x.get("task_index")) is int}
    owner_data=root/"autolearn"/"auto_learning_status.json"
    try:
        autolearn=json.loads(owner_data.read_text())
    except (OSError,ValueError):autolearn={}
    from .md_training import plan_training_jobs
    from .md_model_review import list_for_plan
    model_reviews=list_for_plan(root,plan_id)
    training_jobs=plan_training_jobs(root,plan_id)
    latest=training_jobs[0] if training_jobs else None
    # Present task-by-task progress based only on persisted real executions.
    per_task=[]
    for index,task in enumerate(plan.get("tasks",[])):
        matching_jobs=[r for r in jobs if r.get("task_index")==index]
        matching_reviews=[r for r in model_reviews if r.get("task_index")==index]
        last_research=matching_jobs[0] if matching_jobs else None
        last_review=matching_reviews[0] if matching_reviews else None
        completed_research=any(r.get("phase")=="ready" for r in matching_jobs)
        completed_review=any(r.get("status")=="completed" for r in matching_reviews)
        pilot=pilot_tasks.get(index)
        per_task.append({"task_index":index,"goal":task.get("goal"),
            "kind":task.get("kind"),
            "workflow_status":pilot.get("status") if pilot else "not_started",
            "workflow_summary":pilot.get("summary") if pilot else None,
            "workflow_evidence":pilot.get("sources",[]) if pilot else [],
            "research_job_id":last_research.get("id") if last_research else None,
            "research_status":last_research.get("phase") if last_research else "not_started",
            "review_status":last_review.get("status") if last_review else "not_started",
            "research_completed":completed_research,"model_review_completed":completed_review,
            "training_completed_from_task":False})
    # The existing scratch arithmetic curriculum has no relation to this plan:
    # do not count its steps as learning from uploaded Markdown.
    return {"plan_id":plan_id,"plan_status":plan["status"],"generated_at":time.time(),
        "plan_tasks":plan.get("tasks",[]),"task_progress":per_task,
        "task_count":len(per_task),"tasks_with_completed_research":sum(t["research_completed"] for t in per_task),
        "whole_document_sent_to_tongyi":False,
        "task_dispatch_mode":"one_task_at_a_time_owner_approved",
        "pilot_execution_present":bool(pilot_data),
        "pilot_execution_details":[pilot_tasks[k] for k in sorted(pilot_tasks)],
        "pilot_steps_recorded":len(pilot_tasks),
        "pilot_outcomes":[{"task_index":k,"status":v.get("status"),
            "summary":v.get("summary"),"kind":v.get("kind")}
             for k,v in sorted(pilot_tasks.items())],
        "research_job_count":len(jobs),
        "research_jobs":jobs[:100],"evidence_count":len(records),
        "research_evidence":records[:50],
        "local_tongyi_reviews":model_reviews[:10],
        "local_model_review_count":len(model_reviews),
        "dataset_status":"not_imported",
        "training_status":latest["status"] if latest else "not_started_from_this_md",
        "training_run_id":latest["id"] if latest else None,
        "training_jobs":training_jobs,
        "baseline_functional_benchmark":latest.get("baseline_evaluation") if latest else None,
        "post_training_functional_benchmark":latest.get("final_evaluation") if latest else None,
        "capability_gain_verified":latest.get("capability_gain_verified") is True if latest else False,
        "existing_separate_scratch_curriculum":{"step":autolearn.get("step"),
            "phase":autolearn.get("phase"),"related_to_this_plan":False},
        "next_required":"review research provenance/licenses, approve a dataset and a separate bounded training run",
        "privacy":{"private_document_sent_to_web":False,
            "external_training_data_used":False}}
