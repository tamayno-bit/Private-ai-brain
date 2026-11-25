# Opportunistic learning from other models — 2026-11-22

## What is implemented
- `learning/opportunistic/teacher_queue.py`: local staging quarantine for manually supplied prompt/response pairs. Never scraped automatically; no import from external apps. Export requires status `approved`, positive consent, permitted license, independently verified quality, and explicit permission for training on external-model outputs.
- `learning/opportunistic/gpu_admission.py`: read-only GPU probe and fail-closed resource admission policy, requiring primary workload health, explicit authorization and 8GiB default VRAM reserve. This checker **does not launch or control jobs**, lease a GPU, or provide process priority enforcement.
- Both modules are independent of live Original Brain service and its GPU startup script. No GPU rental or model training was launched in this change.

## Verified
- Unapproved example rejected.
- External teacher example without training permission rejected.
- Explicitly approved eligible example export succeeds.
- GPU admission denied without permissions and when memory insufficient; positive simulation with adequate VRAM; no GPU is exposed on the CPU Pod.
- Original Brain /health returned HTTP 200.

## Next integration (not yet implemented)
1. Track licenses and teacher provider ToS in a signed evidence registry, with human review and privacy/secret filtering.
2. Test checker on an authorized already-running GPU Pod; don't assume memory headroom from free VRAM alone.
3. Implement a GPU worker queue and preemption-safe checkpoint handoff with a separate GPU process; measure runtime interference with Original Brain coding and vision inference.
4. Add consented opt-in capture from production tasks, automatic validation/tests where appropriate, dataset versioning, and user-facing status in Brain Control; no autonomous collection pending authorization and safety checks.
5. Prioritize the Original Brain inference and coding workloads; stop/pause learning on overload and prior to any GPU shutdown.
