# RunPod sidecar integration — v0.2.1 (additive)

## Existing Original Brain, verified on 2026-11-22

- Existing CPU Pod starts `python -u /workspace/services/original-brain-v1/server.py` from `original-brain-desktop/bootstrap_api_desktop.sh`.
- Existing API: `127.0.0.1:8765/health` is public local unauthenticated health (`200`); `/api/status` and `/api/chat` use the existing owner bearer token; other coding routes use a separate key.
- `studio`, Linux desktop, Android Builder and existing queue **must not be modified or restarted** for this integration.
- The Running Pod uses image `python:3.11-bookworm` with PyTorch **not installed**. This sidecar can expose read-only status and integration health without PyTorch; model inference needs a separate compatible PyTorch environment and a **trained** checkpoint.

## Safe destination and state

1. Keep existing backend untouched; unpack at `/workspace/services/private-ai-brain/`, not inside `original-brain-v1`.
2. Set `PRIVATE_BRAIN_STATE_ROOT=/workspace/private-ai-brain-state`; permissions 0700. Token is secret and never included in source ZIP. Directory is on the persistent network volume; include it in encrypted full backup.
3. Python standard-library status API binds **only 127.0.0.1:8877**, never to `0.0.0.0`. A future mobile TLS gateway can authenticate separately; APK does not hold server-side master token.
4. Run `bash integration/start_sidecar.sh` only when user authorizes starting service. It does **not** start training, change model weights, rent a GPU, or alter Original Brain.
5. Status API can report availability without PyTorch, but exam inference returns 409 until a trained checkpoint exists, or 503 if PyTorch is missing.
6. Audit using `bash integration/verify_sidecar.sh`; keep existing `/health` at 8765 online.
7. Rollback by stopping this sidecar process; keep checkpoint/state for safety; do not delete or mutate any existing Original Brain files. See `integration/stop_sidecar.sh`.

## Beyond staging

- The sidecar health endpoint is not yet a fully integrated command/chat route in Original Brain Studio. Exposing Brain Control to Android requires authenticated TLS, rate-limiting and mobile access policy. Model training requires a pinned PyTorch environment and a verified dataset; currently **OFF**.
- CUDA, `torchrun` DDP, FSDP2 and Megatron require distinct real GPU integration tests and topology checks before activation. They are **NOT** certified here.
- Canonical future layout (after parent unification): `original-brain/modules/private-ai-brain/`. Never relocate the running backend just to match this conceptual path.
