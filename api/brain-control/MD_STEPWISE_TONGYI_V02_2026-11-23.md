# MD Step-by-Step Processing — Contract V02
Date: 2026-11-23.
Owner decision: NEVER submit the whole Markdown file to Tongyi in one batch.
Original Brain is the orchestrator and Tongyi is a single-task research/review tool.

## Correct flow
1. POST /v1/brain/md-plans saves the raw MD privately as immutable data. NO Tongyi inference.
2. POST /v1/brain/md-plans/{sha256}/analyze with {"approved":true} extracts **structural** title/list tasks locally, without Tongyi. GET /v1/brain/md-plans/{sha256}/plan provides ordered tasks indexed 0..N-1. Draft extraction is heuristic; owner verifies tasks.
3. POST /v1/brain/md-plans/{sha256}/approve with {"approved":true} authorizes the plan for review and research, NOT arbitrary commands/training.
4. For local Tongyi analysis, POST /v1/brain/md-plans/{sha256}/model-review with {"approved":true,"task_index":N}. It submits ONLY the chosen task goal, at most 180 characters, through private persistent queue on same network volume, to CPU Tongyi. No raw MD, no web request. At most one unfinished task per plan. GET /v1/brain/md-model-review/jobs/{job_id} returns genuine result with model_inference_used, status, task_index and "whole_document_sent":false. Next task can proceed only after prior one completes or fails; previous completed results are idempotent.
5. For an approved public web research query concerning a single task, POST /v1/brain/md-research/start with {"plan_id":"<sha256>","task_index":N,"query":"public topic only","public_query_approved":true,"pages":1,"tongyi":true}. This forwards ONLY the separately reviewed PUBLIC query to Original Brain's existing V23.3 research API; never full MD. One active job per plan (and at most one active job from MD agent in current CPU process). GET /v1/brain/md-research/jobs/{job_id} polls bounded sources and Tongyi outcome. Model-generated answer is not required for saved source evidence; model_inference_used is explicitly false if unavailable.
6. GET /v1/brain/md-plans/{sha256}/report returns task_progress[] with each index, genuine research/review state, training status, evidence, completion counts and "whole_document_sent_to_tongyi":false. Only mark each task done after its actual job completes.
7. Owner-written example dataset and separate explicitly approved GPU training remain independent paths with independent holdout; web research results are not automatically licensed or trained.
8. Sequential work can be resumed after disconnect because task jobs and evidence are persisted. Do not claim any continuous auto-advance beyond the owner-approved task in this version.

## Android client implications for Codex
- Replace "send entire plan to Tongyi" interaction with ordered task display. User selects/approves one task, sees its status and response, then continues to next task.
- Each POST model-review MUST include "task_index"; missing task_index is rejected 400. Never place MD content in the request; it is read locally server-side and only the selected goal is passed to Tongyi.
- Each POST MD research MUST include "task_index" and a separately approved PUBLIC query, not raw MD content. Guard against personal/private data in public query.
- Report task_progress statuses directly; don't mark MD file fully processed when only one goal finished.
- Paths require existing authenticated Brain Control token.
- Ordinary docs and searches never run shell commands, install dependencies, start GPU training or change live model weights.

## Acceptance and limitations
- Python full suite: 48 / 48 PASS.
- Android gateway smoke: missing task_index rejected 400; selecting task_index=1 queued exactly 16 characters (not whole MD).
- A task-only local Tongyi inference eventually produced a genuine Arabic answer, no public web search, no whole Markdown transfer.
- Original Brain coding / vision / Tongyi health probes: all 200. GPU kept RUNNING.
- Previous live research job predates task_index requirements; old evidence remains historical and is not counted as task-indexed completion.
- This is serial per-task owner-confirmed processing, not yet automatic full-plan traversal or a demonstration that the scratch model learned from web research.
