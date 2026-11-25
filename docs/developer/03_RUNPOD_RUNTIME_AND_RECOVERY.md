# 03 — التشغيل الفعلي وحفظ البيانات

## حالة آخر فحص بتاريخ 25 نوفمبر 2026
| مورد | المعرف | الحالة |
|---|---|---|
| CPU / Studio | `mvez7af2sigq3a` | RUNNING |
| GPU | `m8on9qzmiugxfl` | EXITED (أوقفها المالك) |
| Network Volume مشتركة | `kkx03k7wdm` | مركبة في `/workspace`، 120GB سعة مسجلة |

كانت بطاقة GPU هي NVIDIA RTX PRO 4500 Blackwell (ذاكرة فعالة ~32GB) بسعر GPU سابق $0.72/ساعة. **إيقاف Pod يوقف حساب الحوسبة ولا يوقف رسوم التخزين.** عدم تشغيل GPU دون إذن هو المبدأ الأول.

## المسارات
- source: `/workspace/services/private-ai-brain/`
- private state: `/workspace/private-ai-brain-state/`
- legacy active inference: `/workspace/private-ai-brain-state/training/latest.pt`
- conversation candidate: `/workspace/private-ai-brain-state/conversation_experiment_v1/latest.pt`
- arithmetic candidate: `/workspace/private-ai-brain-state/autolearn/runs/arithmetic-604-ba4469e2424b/`
- Original Brain Studio: `/workspace/services/original-brain-v1/`

هذه المسارات مواقع **بيانات حية وليست ملفات يجب وضعها في Git**.

## الاسترجاع
1. إثبات بصمات source والموديل والداتا وإعدادات tokenizer.
2. نسخ الأوزان **والـoptimizer وملفات .pt.json** وبيانات train/holdout خارج المنصة في موقع مشفر.
3. تثبيت التبعيات المعلنة في `pyproject.toml` في venv جديدة مع تثبيت نسخ متوافقة.
4. اختبار قراءة checkpoint والحساب على CPU بموارد مناسبة قبل تفعيل GPU.
5. التحقق من authorization/health عبر loopback بدون إخراج token إلى السجلات.
6. لا تَستبدِل الأوزان الحية أو تربط Android بنسخة جديدة دون قبول مستقل.

**GitHub هنا مصدر كود فقط؛ لا يمثل نسخاً احتياطياً تاماً للبيانات والأوزان أو منصة RunPod.**
