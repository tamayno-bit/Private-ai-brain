# MD learning workflow — pilot review and remaining gates (2026-11-23)

## Delivered to the phone
Pilot Markdown file: `original-brain-md-pilot-v1.md`, 8 numbered tasks (Python official documentation, PyPI, GitHub, research paper, comparison, original training data proposal, coding evaluation, evidence-based report). This is an owner plan, NOT a training dataset and does not approve training.

## Server work completed
- MD upload, immutable document storage, authenticating through the Android device gateway.
- Local deterministic plan extraction, now prioritizing numbered steps so document headings and safety rules are not incorrectly treated as work items; 8-step parser test PASS.
- Plan draft/review/approval.
- Per-task index required for local Tongyi model review; only one goal sent, not the complete document.
- Per-task index required for public research, which uses existing Original Brain search/Tongyi backend and a separately approved public query.
- One active research job per process; stop another job while a task is already active for that plan.
- Per-task report with independently recorded research and review statuses; no fabricated training.
- Optional owner-authored JSON training examples held in Markdown must have independent holdout, rights and owner training approval. GPU training scheduler preserves primary workload VRAM headroom.
- Local CPU and GPU pods remain RUNNING, and Original Brain coding, vision, Tongyi services were healthy.
- Full Private AI Brain unit suite: 50/50 passed after numbered-step parser adjustment.
- Standalone training pipeline smoke: explicit owner-authored synthetic 26/10 train/holdout dataset was accepted, baseline exact-match evaluated at 0/8, but GPU training was correctly deferred because 4765MiB free VRAM was less than 4096MiB reserved for primary + 768MiB estimated training. Test-only job marked cancelled; the GPU and primary services were not stopped.
- API sidecar and gateway restarted and integration verified. No Android APK built here.

## Not yet complete
- **Android APK integration must be implemented/tested by Codex**. The server API does not mean the installed app button is already wired.
- True automatic end-to-end progression through all MD tasks has not been activated; current model review and public web research require owner-approved task submissions one step at a time.
- No automatic ingestion of public-source content into model weights, pending rights/provenance/quality verification.
- Functional improvement from an MD-directed GPU training job has NOT yet been demonstrated because the available GPU headroom did not meet the primary reserve.
- Full Pod reboot recovery and live concurrent-primary preemption require more tests.

## User contract for APK
- First POST /v1/brain/md-plans (upload).
- POST /v1/brain/md-plans/{id}/analyze with {"approved":true}; GET .../plan shows each task_index.
- POST /v1/brain/md-plans/{id}/approve with {"approved":true}; approving the plan is not permission to install packages or train.
- For a selected task: POST /v1/brain/md-plans/{id}/model-review with {"approved":true,"task_index":N}; GET /v1/brain/md-model-review/jobs/{job_id}.
- For an approved public search: POST /v1/brain/md-research/start with {"plan_id":"id","task_index":N,"query":"public-only","pages":1,"tongyi":true,"public_query_approved":true}; GET /v1/brain/md-research/jobs/{job_id}.
- GET /v1/brain/md-plans/{id}/report displays task_progress and real states.
- Do NOT mark all tasks complete when just one is processed. Do not upload the whole MD to Tongyi.
- GPU Pod MUST remain running until owner separately requests shutdown.
