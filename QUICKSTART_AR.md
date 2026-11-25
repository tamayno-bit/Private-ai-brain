# البداية — Private AI Brain

هاد الحزمة **ماشي موديل قوي مدرب**. فيها Transformer حقيقي بأوزان عشوائية، كود تدريب صحيح بالمبدأ، امتحان، وCheckpoints. كتخدم بمعزل عن Original Brain الحالي.

بعد تثبيت Python وPyTorch في بيئة متوافقة:

```bash
python -m pip install -e '.[test]'
pytest -q
python -m brain.train --data config/dataset_example.jsonl --model-config config/model_tiny.json --output state/training --steps 8 --save-every 4 --batch-size 2
python -m brain.evaluate --checkpoint state/training/latest.pt --data config/dataset_example.jsonl
python -m brain.infer --checkpoint state/training/latest.pt --prompt 'سلام' --tokens 32
```

تقدر تعاود التدريب من آخر Checkpoint بـ `--resume state/training/latest.pt --steps 16` مع نفس البيانات والإعدادات.

**مهم:** ما كيتدربش تلقائياً على أي سؤال من الشات. السؤال ديال الامتحان كيختبر الأوزان فقط. التدريب المستمر وتحميل ملفات MD وتطبيق Android ميزات مستقبلية مازال خاصها تطوير وحماية.

الدمج مع Original Brain خاصو يتدار فقط بعد ما يسالي الوكيل الآخر، مع Backup قبل الدمج، ومراجعة `integration/CODEX_INTEGRATION_PROMPT.md`.
