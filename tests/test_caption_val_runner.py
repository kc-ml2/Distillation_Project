import unittest
from unittest import mock

import torch
import torch.nn as nn

import data.eval_validation_caption as evc


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def generate(self, image, **kwargs):
        return ["a caption"] * image.size(0)


def fake_loader():
    # (image, image_id) 배치 하나
    return [(torch.zeros(2, 3, 8, 8), torch.tensor([10, 11]))]


class CaptionValRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model.train()
        self.config = {
            "val_caption_mid_interval_steps": 50,
            "val_caption_epoch_end": True,
            "val_caption_use_spice": True,
            "val_caption_gt_root": "/tmp/gt",
            "val_caption_split": "val",
            "output_dir": "/tmp",
            "caption_score_cpu_list": None,
        }
        self.runner = evc.CaptionValRunner(
            data_loader=fake_loader(), device="cpu", config=self.config, writer=None)

    def _patch_score(self):
        # coco_caption_eval을 mock: .eval dict를 가진 객체 반환
        m = mock.patch("data.eval_validation_caption.coco_caption_eval")
        mock_eval = m.start()
        self.addCleanup(m.stop)
        obj = mock.Mock()
        obj.eval = {"CIDEr": 1.0, "METEOR": 0.3, "SPICE": 0.2}
        mock_eval.return_value = obj
        return mock_eval

    def test_mid_runs_light_only_at_interval(self):
        mock_eval = self._patch_score()
        # iteration != mid -> 실행 안 함
        self.runner.val_caption_during_train(self.model, epoch=0, iteration=49, global_step=49)
        self.assertEqual(mock_eval.call_count, 0)
        # iteration == mid -> 실행, use_spice=False
        self.runner.val_caption_during_train(self.model, epoch=0, iteration=50, global_step=50)
        self.assertEqual(mock_eval.call_count, 1)
        self.assertFalse(mock_eval.call_args.kwargs["use_spice"])

    def test_epoch_end_uses_spice_and_restores_train(self):
        mock_eval = self._patch_score()
        self.model.train()
        metrics = self.runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        self.assertEqual(mock_eval.call_count, 1)
        self.assertTrue(mock_eval.call_args.kwargs["use_spice"])
        self.assertEqual(metrics["CIDEr"], 1.0)
        self.assertTrue(self.model.training)

    def test_epoch_end_respects_disabled(self):
        mock_eval = self._patch_score()
        self.config["val_caption_epoch_end"] = False
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        self.assertEqual(mock_eval.call_count, 0)

    def test_scoring_failure_is_non_fatal(self):
        import data.eval_validation_caption as evc
        m = mock.patch("data.eval_validation_caption.coco_caption_eval",
                       side_effect=RuntimeError("java boom"))
        m.start(); self.addCleanup(m.stop)
        self.model.train()
        # must not raise, must return {}, must restore train mode
        result = self.runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        self.assertEqual(result, {})
        self.assertTrue(self.model.training)

    def test_epoch_end_writes_tensorboard_scalars(self):
        self._patch_score()
        writer = mock.Mock()
        runner = evc.CaptionValRunner(
            data_loader=fake_loader(), device="cpu", config=self.config, writer=writer)
        runner.run_epoch_end(self.model, epoch=0, global_step=37346)
        written = {c.args[0] for c in writer.add_scalar.call_args_list}
        self.assertIn("val_caption/CIDEr", written)
        self.assertIn("val_caption/SPICE", written)


class BuildCaptionRunnerFactoryTest(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        runner = evc.build_pretrain_caption_val_runner(
            config={"val_caption_enabled": False}, device="cpu", writer=None)
        self.assertIsNone(runner)


if __name__ == "__main__":
    unittest.main()
