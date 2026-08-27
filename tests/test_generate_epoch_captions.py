import unittest
from pathlib import Path

from scratchpad.generate_epoch_captions import MODELS, missing_epochs


class MissingEpochsTest(unittest.TestCase):
    def test_skips_existing_caption_results(self):
        existing = {2, 5, 19}

        self.assertEqual(missing_epochs(range(6), existing), [0, 1, 3, 4])

    def test_supports_small_baseline_outputs(self):
        checkpoint_dir, _, output_prefix = MODELS["small"]

        self.assertEqual(checkpoint_dir.name, "pt_smallreg_minilm_baseline")
        self.assertEqual(output_prefix, "small_solo")


if __name__ == "__main__":
    unittest.main()
