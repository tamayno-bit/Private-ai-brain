# Brain Control Android API — v0.2.1 staged sidecar

All routes require `Authorization: Bearer <owner token>`, localhost only. For mobile, put a TLS gateway/authentication proxy in front and do not package the owner token in the APK.

- `GET /v1/brain/health` -> `{ready, management_api_ready, inference_ready, model_checkpoint_available, mode, integration}`
- `GET /v1/brain/integration` -> `{original_brain_health_reachable, integration_mode, modified_original_brain, training_autostart}`
- `GET /v1/brain/status` -> `{trained,checkpoint_present,training_running,gpu_rented_by_service,step?,model_config?}`
- `GET /v1/brain/reports` -> `{events:[{step,train_loss,val_loss_sample,time}],metrics_type}`
- `POST /v1/brain/exam` JSON `{prompt:string,max_new_tokens:1..128}` -> `{answer,model,trained_step,used_external_model,note}`

No upload-MD or start-training API is implemented in v0.2: adding these requires explicit approval and protection against prompt injection, untrusted file execution and unauthorized expenditure. Exam never changes model weights.
