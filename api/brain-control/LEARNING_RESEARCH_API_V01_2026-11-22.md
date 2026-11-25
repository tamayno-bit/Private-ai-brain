# Brain Control — MD Learning Plans + Research API, V0.1
Date: 2026-11-22. Implemented and tested on CPU Pod `qtf9o6c5bqozsx`.
Base URL: `https://qtf9o6c5bqozsx-6080.proxy.runpod.net`.
All routes require `Authorization: Bearer <existing_device_token>`.
All POST requests use UTF-8 JSON. Device token is NEVER embedded in the APK or this contract.

## Previously available
- POST `/v1/brain/md-plans` — body `{"filename":"learning.md","content":"# ..."}`. New 201, duplicate 200. Upload is SHA-256 addressed and immutable, max 128 KiB/6000 lines. A saved MD is untrusted information and always pending owner review.
- GET `/v1/brain/md-plans` — list.
- GET `/v1/brain/md-plans/{sha256}` — verified preview.

## Newly implemented, verified routes
- POST `/v1/brain/md-plans/{sha256}/analyze`: body `{"approved":true}`; 201 for new plan, 200 for existing. Output `{"plan":{"document_id":"...","status":"draft_pending_approval","extraction":"deterministic_structural_extraction_not_model_reasoning","tasks":[{"goal":"...","kind":"research|evaluation|dataset_review|training_proposal","state":"proposed","requires_owner_approval":true,"code_execution_allowed":false,"automated_training_allowed":false}],"automatic_training":false,"automatic_execution":false}}`. **Draft extraction is structural (headings/list items), NOT LLM interpretation.**
- GET `/v1/brain/md-plans/{sha256}/plan`: returns `{"plan":{...}}`.
- POST `/v1/brain/md-plans/{sha256}/approve`: body `{"approved":true}`; changes status to `approved_for_review_and_research` only. Approval does NOT authorize package installation, shell commands, model training, private data exfiltration, dataset inclusion, or model weight promotion.
- GET `/v1/brain/research/providers`: returns `{"providers":["wikipedia","pypi","crossref","github"],"mode":"read_only_public_research","auto_train":false}`.
- POST `/v1/brain/research`: body `{"query":"pytorch","provider":"github","approved":true,"plan_id":"<approved_plan_sha256>"}` . `plan_id` is optional; when present research evidence is only attached after plan approval. Output: query, provider, results array, retrieved_at, status and plan_evidence if applicable. Missing approval is rejected. Limit query 120 Unicode characters; detected credentials or control chars rejected. A PyPI query must be an exact package identifier (e.g. `torch`).
- GET `/v1/brain/md-plans/{sha256}/evidence`: lists up to 50 verified, hash-addressed research result records; for each, query, provider, retrieved_at, results and explicit `dataset_ingested:false`.

## Search providers
- wikipedia: encyclopedia search snippets and article links (reference only).
- pypi: package metadata (name, version, summary, reported license); does NOT download/install library.
- crossref: scholarly paper metadata and DOI links (paper content rights not evaluated).
- github: public repository metadata (description, stars, language, reported SPDX license); does NOT clone/execute code.
All external records explicitly mark `training_rights_verified:false`; they are NOT automatically added to model weights or training corpus.

## Android app instructions
Keep the existing local MD file picker/preview and add an owner-confirmed Upload. After successful upload allow a separate Generate Draft button (POST analyze), present draft tasks with clear non-autonomous status, require explicit Review/Approve (POST approve), then allow on-demand confirmed research queries and display linked evidence (GET evidence). Use only server-returned fields, no simulated progress, no invented API endpoints, and do not treat Markdown text as executable instructions.
Handle 200,201,400,401,404,413,429,502/503 and timeouts. No automatic repeat of uploads, no silent research on private contents, no secrets in logs. Show results as READ-ONLY PUBLIC REFERENCES pending rights review.

## Verification
- Private AI Brain full suite 37/37 PASS.
- Live Wikipedia/PyPI/Crossref/GitHub public-source queries returned grounded results.
- Live authenticated gateway MD upload -> draft -> owner approval -> PyPI lookup -> stored evidence -> GET evidence passed.
- Unauthenticated access rejected; fixture removed after tests. Original Brain health 200; old scratch checkpoint unchanged (step 9).
- CPU bootstrap was additively updated to start sidecar/gateway upon future Pod startup; shell syntax verified, **full Pod-reboot acceptance not yet performed** to avoid interrupting owner workload.
- GPU Pod is independent and kept RUNNING. No GPU restart, training change or expense increase from this API phase.

## Honest limitations
This stage provides an approved **search tool service**, not an autonomous tool-calling ability inside the tiny scratch-trained model. The model cannot yet reason over search results or invoke providers on its own. That will require a separate orchestrator / agent loop, explicit owner-approved budgets, quality and license checks, and real benchmark results. MD-to-task extraction is deterministic and does not itself train weights. More domain-specific sources, safe ingestion and evaluation remain outstanding.
