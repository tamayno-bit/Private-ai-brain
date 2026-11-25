# Codex integration request (execute only when user asks)

Do **not** integrate yet if another agent is still editing Original Brain. User wants a safe additive module.

1. Identify canonical Original Brain root and current live processes, git/worktree status, data locations, active tasks and route contracts. Preserve old behavior.
2. Verify full externally stored backups of original sources, state, Android demo project symlinks and owner authentication files **without displaying or archiving secrets in plaintext**.
3. Create a staging folder *outside* the existing backend process directory. Unpack this module as a sidecar. Do not overwrite or alter server.py, Studio assets, Linux Desktop, Android Builder.
4. Establish fresh venv and pinned toolchain for scratch training. Verify Python, device support and disk capacity. Re-run pytest, CPU training/eval/checkpoint restoration, low-resource probes.
5. Bind Brain API only to loopback; configure reverse proxy with TLS, user-device auth, rate limiting and audit trail. Keep distinct from `/api/chat` and `/v1/chat/completions` and `coding_api.token`.
6. Stage controlled app integration with feature flag OFF by default; enable after end-to-end and rollback tests. Limit CPU usage and ensure no interference with Studio workload.
7. Do NOT enable unattended training, GPU provisioning, third-party dataset imports, external-model distillation, remote MD execution or automatic model promotion. Obtain explicit consent for resource policy and dataset licensing first.
8. Ensure persistent state, code and checkpoint backup/restore works on a *clean compatible Linux* machine before calling it portable.
9. Hand user a truthful checklist (working/experimental/planned) and exact test results. If GPU tests were not run, say so.

Expected module directory: `modules/private-ai-brain/` **after** unified parent exists, but don't move the main app simply to satisfy this structure.
