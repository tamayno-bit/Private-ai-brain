# 04 — الأوزان، البيانات، التدريب والـCheckpoints

## تجارب مؤكدة من الحالة المحفوظة (وليست قدرات مثبتة)
| Run | Checkpoint | الخطوة | ملاحظات |
|---|---|---:|---|
| legacy exam | `training/latest.pt` | 9 | المصدر الذي يقرأه API للامتحان |
| arithmetic autolearn | `autolearn/runs/arithmetic-604-ba4469e2424b/latest.pt` | 390 | `idle_curriculum_saturated`، loss ≈ 1.1349 |
| conversation experiment v1 | `conversation_experiment_v1/latest.pt` | 100 | 468 أمثلة تدريب اصطناعية، 20 holdout؛ 0/8 أجوبة مطابقة في الاختبار الأول |

## تفاصيل مرشح المحادثة
- model config: vocab_size=260، context=128، dim=128، layers=4، heads=4، kv_heads=2، dropout=0.
- corpus تجريبي منظم بصيغة `Question: ...\nAnswer: ...`، وليس دليلاً على instruction-following أو retention متعدد الأدوار.
- train loss هبط تقريباً من 5.56 إلى 2.10 خلال 100 خطوة.
- holdout loss ~2.07؛ exact-match 0 من أصل 8. أول رد مباشر على السلام لم يكن مفهوماً.
- النتيجة: **التدريب التقني شغّال؛ جودة المحادثة غير ناجحة بعد**.

## ضوابط التشغيل
- موافقة منفصلة على dataset وحقوقها وحدود تدريب GPU المدفوع.
- فصل بيانات train/validation/test مع manifest وبصمة وتجنب contamination.
- نموذج Candidate لا يستبدل نسخة Live تلقائياً.
- Checkpoints متوافقة فقط مع architecture/tokenizer/dataset الصحيحين؛ ترقية السياق والبنية ليست resume مضموناً.
- لا تجمع محتوى خاص أو مخرجات موديلات خارجية لا تملك حق استخدامها.
- بيانات الاختبار المستقلة لا تدخل التدريب.

## المشكلة المعروفة
API `POST /v1/brain/exam` يقرأ مسار `state-root/training/latest.pt`، وبالتالي التطبيق لا يرى المرشح الجديد. المطلوب مستقبلاً Model Registry + explicit selection مع اختبارات وحماية، وليس نسخ ملف candidate فوق legacy.
