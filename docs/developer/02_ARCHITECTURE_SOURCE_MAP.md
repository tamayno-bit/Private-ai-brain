# 02 — العمارة وخريطة الكود

## الصورة العامة
```text
Brain Control (Android)
    -> HTTPS / device-auth gateway (:6080 عند التشغيل)
       -> Private AI Brain API (loopback :8877)
          -> status / exam / MD plans / research
          -> model checkpoint (live legacy)
Original Brain Studio (CPU, :8765) <- integration checks/sidecar (not merged)
RunPod shared /workspace volume <- source/state/checkpoints
GPU Pod (explicitly rented/start) -> gated training workers -> candidate checkpoints
```

## ملفات محورية (مطابقة للمصدر المرفوع)
- `brain/model.py`: decoder-only Transformer، RoPE، RMSNorm، SwiGLU، attention، إعدادات ModelConfig.
- `brain/tokenizer.py`: ByteTokenizer، مفردات 260 رمز تقريباً؛ ليس tokenizer متعدد اللغات محسّن.
- `brain/dataset.py`: تحقق provenance وtoken stream وفصل تدريب/تحقق داخلي.
- `brain/train.py`: optimizer AdamW، backprop، batch/grad accumulation، resume، SIGTERM.
- `brain/checkpoint.py`: حفظ آمن وmetadata وبصمة.
- `brain/infer.py`: إنتاج من الأوزان المختارة في CLI.
- `brain/evaluate.py`: قياس loss، ولا يساوي تقييم المحادثة.
- `brain/memory.py`: ملاحظات SQLite مستقلة ولا تدخل التدريب تلقائياً.
- `brain/api.py`: localhost-only HTTP API مع Bearer auth.
- `integration/brain_control_gateway.py`: جسر محمي إلى تطبيق Android.
- `brain/md_plans.py` / `brain/learning_plans.py`: استقبال MD معزول، استخراج مهام، موافقة مراجعة/بحث.
- `brain/md_dataset.py` / `brain/md_training.py`: تحقق حقوق وبيانات وجداول تجارب مرشحة.
- `learning/opportunistic/md_gpu_worker.py` / `gpu_supervisor.py`: فحص الموارد، تشغيل تدريب محدود، حفظ/استئناف.
- `learning/opportunistic/autolearn.py`: تجربة حسابيات اصطناعية، تتوقف عند التشبع.
- `learning/opportunistic/independent_eval.py`, `md_functional_eval.py`: تقييم loss ومطابقة الإجابات.
- `tests/`: فحوص الوحدة والتكامل التجريبية.

**تنبيه:** بعض وثائق المستودع الأصلية تعود إلى 22 نوفمبر وتذكر أن Android وCUDA غير مجرَّبين. جرى لاحقاً تدريب CUDA بالفعل، لكن هذا لا يثبت جاهزية الموديل للمحادثة ولا نجاح التشغيل متعدد GPU.
