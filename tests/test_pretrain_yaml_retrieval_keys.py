import unittest

import yaml


class PretrainYamlRetrievalKeysTest(unittest.TestCase):
    def setUp(self):
        with open("configs/pretrain.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def test_has_all_val_retrieval_keys(self):
        expected_keys = [
            "val_retrieval_enabled",
            "val_retrieval_ann_root",
            "val_retrieval_image_root",
            "val_retrieval_split",
            "val_retrieval_batch_size",
            "val_retrieval_num_workers",
            "k_test",
            "val_retrieval_itc_interval_steps",
            "val_retrieval_itc_epoch_end",
            "val_retrieval_itm_interval_steps",
            "val_retrieval_itm_epoch_end",
        ]
        for key in expected_keys:
            self.assertIn(key, self.config, f"missing config key: {key}")

    def test_itm_interval_is_coarser_than_itc_interval(self):
        self.assertGreater(
            self.config["val_retrieval_itm_interval_steps"],
            self.config["val_retrieval_itc_interval_steps"],
        )


if __name__ == "__main__":
    unittest.main()
