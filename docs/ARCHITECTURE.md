# Architecture decision — genuine scratch training, modular future growth

## Current, grounded Original Brain integration

Read-only RunPod inspection (2026-11-22) confirmed existing Python `http.server` on port 8765, `runtime_config.py` setting `/workspace/services/original-brain-v1`, separate Desktop and Android Builder services, and existing `secret.token` and `coding_api.token` auth. **The active Original Brain server is not changed or monkey-patched** by this package. Proposed staging path after owner approves: `/workspace/original-brain/modules/private-ai-brain/` if unified parent exists; otherwise sidecar `/workspace/services/private-ai-brain/`. No assumptions that parent has already been created.

## Target design

1. Model: original, randomly initialized decoder-only Transformer family; no foreign pretrained weights. Torch modules expose swappable tokenizer, attention backend and architecture config. Grow through *new model runs* or researched weight transfer; don't assume arbitrary resizing can preserve learned weights.
2. Data: versioned, licensed, opt-in corpora, source provenance, quality filters, deduplication, deterministic train/validation/test partitions at **document level** for production. Current demo uses disjoint token windows, not a comprehensive contamination-safe benchmark.
3. Training: actual backpropagation, optimizer state, explicit cost admission, durable checkpoint/recovery. Current single-process CPU/CUDA and basic DDP are proof-level implementation. Production expansion: FSDP2/TorchTitan and/or Megatron parallelism adapters, object-store checkpoint sharding, multi-node orchestration, throughput tests.
4. Evaluation: isolate untouched held-out test sets and domain benchmarks, prevent leaks, compare model versions and regression gate model promotion.
5. Brain Control: Android client *separate project* connects through secured gateway. Expose training metrics, checkpoints, exam chat with locally generated text only, MD proposal queue, verified training job state. Avoid fake progress percentages.
6. Security: separate privileges and budgets, license validation, confidential chat isolation, no raw production logs copied into training, zero GPU rental without owner approval, no unaudited automatic model promotion.
7. Portability: source and original data/checkpoints belong in portable backup. Python, Android SDK, CUDA, third-party libraries and externally obtainable weights are restored by verified dependency manifests; unique modified source weights must be preserved.

## No foreign models guarantee

This package downloads **no pretrained model** and defines its architecture and initial parameters locally with RNG initialization. It *does* use PyTorch and general mathematical methods from published research. The software framework licenses and data rights must be respected; tech independence does not automatically confer exclusive legal copyright rights. Future tool integrations must preserve a provenance boundary preventing unauthorized training from teacher-model answers.

## Interfaces / staged deployment

- Backend candidate: `python -m brain.api --host 127.0.0.1 --port 8877 --state-root <persistent path>`.
- Existing `/workspace/services/original-brain-v1/server.py` keeps all its routes and auth. No `/studio` changes.
- New Android app later can call a proper TLS gateway; no API keys embedded in APK.
- Backend future model education is not the same as the existing Original Brain CPU Naive-Bayes router; preserve both and label them accurately.

## Capability maturity

CURRENT: scratch model, byte tokenizer, explicit CPU/CUDA training loop, preliminary DDP, signed-off dataset provenance metadata, simple checkpoint, inference, memory notes, small API. TARGET: scalable subword training, advanced distributed strategies, robust data pipeline, real evaluators, continual training orchestration, full crash recovery and deployed Android.
