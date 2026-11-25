# 05 — Workflow MD وواجهات التطبيق

## لماذا MD ليس تدريباً تلقائياً؟
الرفع يضع النص في عزل `pending_owner_review` مع SHA256؛ تحليل مهام مرقمة يخلق `draft_pending_approval`. اعتماد الخطة يتحول إلى `approved_for_review_and_research` ولا يعطي إذناً ضمنياً للتدريب أو لاستخدام محتوى خارجي.

## تدفق المحاولة الرسمي
```text
upload .md -> sha256 immutable quarantine
    -> parse numbered steps -> draft -> owner approval for review/research
    -> inspect owner-written licensed training JSON in MD
    -> dataset proposal -> independent rights/training approval
    -> GPU worker admission -> candidate checkpoint -> independent eval
    -> human review (never automatic live promotion)
```

## قيود schema الحالية
`brain/md_dataset.py` يطلب fenced JSON من النوع `private_brain_examples_v1`، train ≥12، holdout ≥8، source_type `owner_written` وحقول consent/license. رفض استيراد مثال بدون الحق أو موافقة صريحة مقصود. خطة MD السابقة للمحادثة **هي خطة، وليست corpus صالحاً للتدريب المباشر**.

## Backend
- loopback `127.0.0.1:8877`، كل route مصادق عليه.
- `GET /v1/brain/health`، `GET /v1/brain/status`، `GET /v1/brain/reports`.
- `POST /v1/brain/exam` يأخذ prompt وmax_new_tokens.
- MD routes تبدأ بـ `/v1/brain/md-plans`، راجع `brain/api.py` قبل اختيار الطلب والحقول.
- بوابة Android على منفذ `6080` أثناء التشغيل؛ لا تضع رمز الإدارة داخل APK.

## أين مصدر Android؟
يوجد Backend/Gateway وتعاقد API في هذا المستودع؛ **لا توجد هنا شجرة مشروع Android Studio الكاملة الخاصة بـBrain Control**. يجب استيرادها بشكل منفصل بموافقة المالك ومراجعة التوافق قبل نشر APK جديدة.

## الأمن
لا تقبل MD كمصدر أوامر shell؛ لا تعليمات من صفحات الويب كأوامر؛ لا بحث عن بيانات المستخدم أو استدعاء APIs مدفوعة دون السماح المناسب؛ لا تعريض API المحلي علناً بلا TLS وتوكن لكل جهاز.
