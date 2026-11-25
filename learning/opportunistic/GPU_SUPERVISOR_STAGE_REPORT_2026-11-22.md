# Private AI Brain — GPU Opportunistic Supervisor v0.1

Date: 2026-11-22. Installed at:
`/workspace/services/private-ai-brain/learning/opportunistic/`.

## Added
- `gpu_admission.py`: GPU memory probe and default-deny capacity check.
- `primary_guard.py`: requires fresh Original Brain GPU-worker heartbeat (`live_queue/expert_alive.json`), no queued live/coding jobs, sufficient available VRAM, and explicit approval.
- `gpu_supervisor.py`: opt-in bounded job runner; does **not** create or rent a GPU Pod. Uses a single-instance file lock, runs a discrete training job, polls for primary preemption or time budget, sends SIGTERM to the training subprocess, waits for checkpoint and writes `gpu_supervisor_status.json`.
- `teacher_queue.py`: quarantines potential examples. No automatic model-output harvesting. Human approval, rights/consent and independent quality validation required before export.
- `tests/test_gpu_opportunistic.py`: eight safety tests.

## Verification
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 16/16 passed, CPU-only.
- `python -m learning.opportunistic.gpu_supervisor --preflight`: `no_gpu` as expected on the CPU Pod, `training_started:false`.
- Original Brain `http://127.0.0.1:8765/health`: HTTP 200.
- No GPU Pod started; no GPU training job started; no changes to Original Brain scripts.
- Existing scratch training checkpoint remains untouched: `/workspace/private-ai-brain-state/training/latest.pt` step 6.

## Safety and non-goals
- **Not yet production-ready simultaneous GPU sharing.** Queue and heartbeat checks cannot detect an already executing in-flight request after it has been dequeued. Free VRAM also cannot guarantee available compute throughput. Actual primary-vs-secondary contention must be tested during an explicitly authorized GPU session, and ideally requires a busy/idle handshake in the GPU worker before automatic scheduling.
- **No automatic GPU rental, no continuous training, and no activation from Brain Control at this stage.**
- The model's existing scratch weights and demo corpus prove training mechanics, not intelligence. External-model terms/licenses must be verified for each source; generated responses are not automatically permissible training data.
- GPU image/PyTorch/CUDA compatibility, checkpoint interruption under a real GPU workload, and distributed training remain unverified.

## Next step
Implement primary-workload in-flight busy/idle signaling in a reviewable additive manner (or a separate GPU worker lease), then test GPU training preemption and checkpoint resume under a user-authorized short GPU session. Add read-only status to Brain Control after the server path has been vetted.
