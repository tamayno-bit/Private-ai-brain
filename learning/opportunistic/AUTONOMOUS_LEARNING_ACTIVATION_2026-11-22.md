# Private AI Brain — Automatic GPU Learning Activated
Date: 2026-11-22 (UTC)

## User authorization
User approved automatic learning using the existing rented GPU and explicitly asked GPU Pod to remain running until they request shutdown. This controller does not stop, create, or rent Pods.

## Live service
- GPU Pod: `4yig5e3wleldkw` (RTX PRO 4500 Blackwell, 32GB, RunPod $0.72/hour).
- CPU Pod: `qtf9o6c5bqozsx` continues serving Original Brain / Brain Control.
- GPU Pod original coding and vision engines on ports 8770 and 8772; both HTTP health 200 at activation.
- Dedicated autolearn Python loop on GPU Pod, low nice priority, 20-second polling. PID in `/tmp/private-brain-autolearn.pid`.
- Explicit per-Pod consent in `/workspace/private-ai-brain-state/autolearn/approval.json`.
- No scheduled GPU stop created; if RunPod restarts GPU Pod, controller needs relaunch until autostart is integrated.

## Candidate training
- Old production-facing scratch checkpoint preserved unchanged at `/workspace/private-ai-brain-state/training/latest.pt` (step 9).
- Separate candidate with random original initialization: `/workspace/private-ai-brain-state/autolearn/runs/arithmetic-604-ba4469e2424b/latest.pt`.
- Model settings: 260 byte-token vocabulary, context 64, dimension 128, four layers, 4 attention heads, grouped KV heads 2.
- No model from Qwen/Llama used as base weights, and no external chat/teacher output harvested for training.
- A deterministic local arithmetic corpus has 3,600 licensed-own examples. A disjoint-seeded held-out corpus has 3,385 examples, after removing 215 exact duplicate examples. The earlier overlapping evaluation metrics were deliberately separated and are not comparable to new results.
- Separate best checkpoint and manifest: `best_disjoint_v1.pt` and `best_disjoint_v1.pt.json`.
- Dynamic curriculum step budget (1045 for current snapshot) based on approximate two passes through the existing corpus; stop earlier after four checks without sufficient independent loss improvement. Budget applies to this curriculum, not the overall user's long-term project.
- Every 10 steps the controller supervises work; checkpoint every 5 steps and on graceful SIGTERM; lock prevents competing supervisors; admission requires fresh primary heartbeat, idle language+vision slots, no live/coding/vision queue files, at least 4GB primary VRAM reserve and estimated 768MB for the candidate.
- GPU model is NOT automatically promoted to the user's active chat model. Improvement is only byte-level independent holdout loss, not functional arithmetic/coding capability.

## Current status at verification
- Candidate successfully trained through at least step 110, independent held-out token loss 1.83654 (not accuracy).
- Controller PID alive, stage `trained_candidate` between training chunks.
- Primary coding and vision engines each healthy (HTTP 200).
- GPU Pod verified RUNNING. No auto-shutdown or extra GPU Pod launched.
- Test suite 29/29 passed following disjoint-holdout correction.

## Safe controls
On the GPU Pod:
- Status: `cd /workspace/services/private-ai-brain && python -m learning.opportunistic.control status`
- Pause training only: `cd /workspace/services/private-ai-brain && python -m learning.opportunistic.control pause`
- Resume training only: `cd /workspace/services/private-ai-brain && python -m learning.opportunistic.control resume`
These commands do not stop or terminate GPU rental. Pausing takes effect between supervised chunks, not instantaneously.

Status via authenticated Brain Control API:
`GET /v1/brain/status` includes `autonomous_learning`, `gpu_observer`, and `gpu_training_last_result`. Android APK may need UI update to display new keys.

## Limitations and follow-up
- Although inference slots are monitored every two seconds during supervised training, priority arbitration is not an atomic GPU scheduler. A fresh primary request can begin between checks; production preemption load testing is still required.
- Auto-learning pauses when no new approved datasets or test improvement exists; it should not overtrain forever on toy arithmetic.
- More independently licensed and vetted code/math/vision corpora, safety tests, curriculum promotion tests, and actual evaluation of answer correctness are needed for useful intelligence.
- External model outputs may only enter training after permission, provenance and quality validation; teacher output capture is not activated.
