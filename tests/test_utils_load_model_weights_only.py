import os
import tempfile
import unittest

import torch
import torch.nn as nn

import utils


class LoadModelWeightsOnlyTest(unittest.TestCase):
    def test_strips_module_prefix_and_loads_weights(self):
        source = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.bias.fill_(1.5)

        ddp_style_state_dict = {f"module.{k}": v for k, v in source.state_dict().items()}
        checkpoint = {"model": ddp_style_state_dict, "epoch": 7}

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pth")
            torch.save(checkpoint, ckpt_path)

            target = nn.Linear(2, 2)
            utils.load_model_weights_only(target, ckpt_path)

        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))

    def test_accepts_raw_state_dict_without_model_key(self):
        source = nn.Linear(2, 2)
        with torch.no_grad():
            source.weight.fill_(-2.0)
            source.bias.fill_(0.5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "checkpoint.pth")
            torch.save(source.state_dict(), ckpt_path)

            target = nn.Linear(2, 2)
            utils.load_model_weights_only(target, ckpt_path)

        self.assertTrue(torch.equal(target.weight, source.weight))
        self.assertTrue(torch.equal(target.bias, source.bias))


if __name__ == "__main__":
    unittest.main()
