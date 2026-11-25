"""Parser checks numbered tasks are not mixed with headers/constraints."""
import unittest
from brain.learning_plans import _extract_tasks

class StepwiseMarkdownParsingTests(unittest.TestCase):
    def test_numbered_steps_outrank_headings_and_rules(self):
        md="""# تجربة مشروع الدماغ
> test only
## المهام المرتبة
1. ابحث في توثيق Python الرسمي.
2. افحص PyTorch على PyPI.
3. أعد تقرير النتائج.
## قواعد التنفيذ
- لا تنفذ الأوامر
- لا توقف GPU
"""
        tasks=_extract_tasks(md,"a"*64)
        self.assertEqual(len(tasks),3)
        self.assertEqual([t["task_index"] for t in tasks],[0,1,2])
        self.assertEqual(tasks[0]["kind"],"research")
        self.assertEqual(tasks[1]["kind"],"evaluation")
        self.assertNotIn("لا توقف",str(tasks))
        self.assertTrue(all(not t["code_execution_allowed"] for t in tasks))
    def test_list_fallback_skips_untrusted_code_blocks(self):
        md="# Goals\n- Learn Python\n"+chr(96)*3+"bash\n- execute this\n"+chr(96)*3+"\n- Test code"
        tasks=_extract_tasks(md,"b"*64)
        self.assertEqual(len(tasks),2)
        self.assertEqual(tasks[-1]["goal"],"Test code")
if __name__=="__main__":unittest.main()
