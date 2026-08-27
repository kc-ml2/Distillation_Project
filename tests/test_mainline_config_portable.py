import unittest
from pathlib import Path

import yaml


class MainlineConfigTest(unittest.TestCase):
    def test_paths_are_relative_to_clone_root(self):
        config = yaml.safe_load(Path("configs/pretrain_mainline.yaml").read_text())
        paths = [
            *config["train_file"],
            config["image_root_coco"],
            config["image_root_vg"],
            config["output_dir"],
            config["val_loss_file"],
            config["val_loss_image_root"],
            config["val_retrieval_ann_root"],
            config["val_retrieval_image_root"],
            config["val_caption_ann_root"],
            config["val_caption_image_root"],
            config["val_caption_gt_root"],
            config["teacher"]["checkpoint"],
        ]

        self.assertFalse([path for path in paths if Path(path).is_absolute()])


if __name__ == "__main__":
    unittest.main()
