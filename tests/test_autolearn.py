"""Safety tests for opt-in autonomous training, with no GPU or real training."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from learning.opportunistic import autolearn,synthetic_curriculum,disjoint_holdout

class AutoLearnTests(unittest.TestCase):
    def test_synthetic_curriculum_deterministic(self):
        a=synthetic_curriculum.examples(604);b=synthetic_curriculum.examples(604)
        self.assertEqual(a,b);self.assertEqual(len(a),3600)
        self.assertTrue(all(r["license"]=="own" and r["consent"] for r in a))
    def test_independent_seeds(self):
        self.assertNotEqual(synthetic_curriculum.examples(604),synthetic_curriculum.examples(605))
    def test_immutable_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,m=synthetic_curriculum.create(tmp,604)
            self.assertEqual(synthetic_curriculum.create(tmp,604)[1],m)
            p.write_text("corrupt")
            with self.assertRaises(RuntimeError):synthetic_curriculum.create(tmp,604)
    def test_holdout_has_zero_exact_example_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr,_=synthetic_curriculum.create(tmp,604)
            hold,info=disjoint_holdout.create_disjoint(tmp,604,605)
            train={json.loads(x)["text"] for x in tr.read_text().splitlines()}
            verify={json.loads(x)["text"] for x in hold.read_text().splitlines()}
            self.assertFalse(train & verify)
            self.assertEqual(info["rows"],len(verify))
            self.assertGreater(info["removed_overlapping_rows"],0)
    def test_new_benchmark_does_not_overwrite_legacy_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"latest.pt").write_bytes(b"new")
            (p/"latest.pt.json").write_text("{}")
            (p/"best.pt").write_bytes(b"OLD")
            autolearn.best_update(p,2.1,50,"disjoint_v1")
            self.assertEqual((p/"best.pt").read_bytes(),b"OLD")
            self.assertEqual((p/"best_disjoint_v1.pt").read_bytes(),b"new")
    def test_only_authorized_gpu_pod_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/"approval.json").write_text(json.dumps({"approved":True,"gpu_pod_id":"other"}))
            self.assertFalse(autolearn.approved(p,"gpu"))
            self.assertTrue(autolearn.approved(p,"other"))
            (p/"PAUSE").touch()
            self.assertFalse(autolearn.approved(p,"other"))
    def test_no_live_checkpoint_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/"latest.pt").write_bytes(b"a")
            (p/"latest.pt.json").write_text("{}")
            result=autolearn.best_update(p,3.0,10)
            self.assertEqual(result["best_step"],10)
            self.assertFalse(result["candidate_promoted_to_live"])
            self.assertTrue((p/"best.pt").is_file())
            (p/"latest.pt").write_bytes(b"b")
            newer=autolearn.best_update(p,4.0,20)
            self.assertEqual(newer["best_step"],10)
            self.assertEqual((p/"best.pt").read_bytes(),b"a")
    def test_fail_closed_no_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(autolearn,"prepared",side_effect=AssertionError("preparation must not run")):
                result=autolearn.one(Path(tmp),"gpu")
            self.assertEqual(result["phase"],"paused_or_revoked")
if __name__=="__main__":unittest.main()
