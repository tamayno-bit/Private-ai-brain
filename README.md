# Private AI Brain — Scalable Scratch Model Platform v0.2.1

An **independent, opt-in, scratch-weight neural model research project** designed for later integration into Original Brain. No Qwen, Llama, OpenAI, or other pretrained model weights. The included model is a **randomly initialized decoder-only Transformer**, implemented in PyTorch; its initial knowledge is **zero**. It is **not a capable assistant** without significant licensed data, training and independent benchmarks.

## Contains

- `brain/model.py`: real neural autoregressive Transformer, GQA, RoPE, RMSNorm, SwiGLU, SDPA, tied embeddings.
- `brain/tokenizer.py`: lossless versioned UTF-8 byte tokenizer (starter; train a scalable subword tokenizer for larger runs).
- `brain/dataset.py`: explicit provenance/permission validation and disjoint token holdout.
- `brain/train.py`: explicit scratch training with AdamW, gradient clipping, gradient accumulation, optional `torchrun` DDP (GPU, **not yet validated on a live multi-GPU host**), stop-at-step checkpointing.
- `brain/checkpoint.py`: atomic (replace) model + optimizer + RNG + sampling state checkpoint and SHA-256 sidecar manifest.
- `brain/evaluate.py`: held-out next-token loss/perplexity (not a chat-ability score).
- `brain/infer.py`: exam using **only these model weights**; prompts are **not** automatically added as supervised answers.
- `brain/memory.py`: separate SQLite user-memory notes (never automatically used as training data).
- `brain/api.py`: owner-gated, localhost-only API for future Brain Control Android integration.
- `integration/`: additive-only deployment plan for the *existing* Original Brain instance. **No auto-patching** of Original Brain.

## Requirement and environmental constraints

- Linux, Python >=3.11, PyTorch >=2.4, with a PyTorch build suited to the device. `pip install -e '.[test]'`.
- `torchrun` DDP is an **initial extension point**, not a tested FSDP/Tensor/Pipeline/Multi-node stack. Those are roadmap items; no claims that arbitrary model sizes already run.
- CPU smoke run is small by design. Real scratch pretraining needs extensive licensed data and substantial GPUs. Two JSON files are **illustrative**, not performance-validated production architectures.
- User consent & dataset license must be documented; not all machine-generated/proprietary outputs may be trained on.
- No paid GPU rental or backend changes are triggered by any file in this ZIP.

## Quickstart (in isolated test directory)

```bash
cd private-ai-brain-scalable
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
python -m brain.train --data config/dataset_example.jsonl --model-config config/model_tiny.json --output state/training --steps 8 --save-every 4 --batch-size 2
python -m brain.evaluate --checkpoint state/training/latest.pt --data config/dataset_example.jsonl
python -m brain.infer --checkpoint state/training/latest.pt --prompt 'Hello' --tokens 30
# To resume: --resume state/training/latest.pt --steps 15 (same dataset/config/world size)
```

These steps run an educational gradient-flow check, **not** a trained conversational AI. The `state/` directory is excluded from repository packaging by design; if real weights, memory, data or checkpoints are generated, **include them in external encrypted backups**.

## Optional owner-only API (local)

```bash
export PRIVATE_BRAIN_API_TOKEN='REPLACE_WITH_32_PLUS_RANDOM_CHARACTERS'
python -m brain.api --host 127.0.0.1 --port 8877 --state-root state
# GET /v1/brain/health, /v1/brain/status, /v1/brain/reports
# POST /v1/brain/exam {"prompt":"Hello","max_new_tokens":32}
```

API requires `Authorization: Bearer <secret>` on every endpoint. Bind to loopback only; expose to Android via a **separately authenticated TLS proxy with per-device access control**, not a secret baked into APK. The API is diagnostic/exam-only: no public training endpoint, no remote file upload or remote command execution.

## Safety and integration

Read `integration/CODEX_INTEGRATION_PROMPT.md` and `docs/ARCHITECTURE.md`. First review Original Brain's current state and take verified backups; do **not** move, rename, override or patch running `server.py`, Studio or Android Lab. Stage this module outside the running service, pin exact tool/library versions and verify tests. No automatic training from live chats, exams or MD directives. An MD directive is a **proposal** requiring explicit approval, verified datasets and cost authorization.

## Current version limitations

- Initial byte tokenizer is not efficient for real large-model training; dedicated learned tokenizer and multilingual corpus are planned.
- This ZIP **does not** contain a pretrained model, robust chat instruction tuning, image/audio model, production FSDP, continuous learning daemon, live GPU performance results, Android APK or trained checkpoints.
- Accuracy, reasoning, coding, vision and self-improvement **must never be claimed** based on a loss curve alone.
- Checkpoint model-configuration and dataset matching are validated for resume; moving between GPU counts requires explicit migration procedures.

## Safe RunPod staging (2026-11-22)

Read `integration/RUNPOD_SIDECAR.md`. This unit may run an isolated, owner-authenticated **status API without installing PyTorch on the main Original Brain Pod**. A separately approved install/test step is needed for trained inference and GPU workloads.
