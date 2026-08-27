import json
import tempfile
import unittest
from pathlib import Path

from scratchpad.score_epoch_captions import MODELS, needs_scoring


class NeedsScoringTest(unittest.TestCase):
    def test_requires_all_eight_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            self.assertTrue(needs_scoring(path))
            path.write_text(json.dumps({name: 1 for name in (
                "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR",
                "ROUGE_L", "CIDEr", "SPICE",
            )}))
            self.assertFalse(needs_scoring(path))

    def test_supports_small_baseline_outputs(self):
        prefix, _ = MODELS["small"]

        self.assertEqual(prefix, "small_solo")


if __name__ == "__main__":
    unittest.main()
