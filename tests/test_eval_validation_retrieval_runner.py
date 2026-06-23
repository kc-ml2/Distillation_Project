import unittest
from unittest import mock

import torch.nn as nn

import data.eval_validation_retrieval as evr


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


class RetrievalValRunnerTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.model.train()

        fake_dataset = mock.Mock()
        fake_dataset.txt2img = {0: 0}
        fake_dataset.img2txt = {0: [0]}

        self.fake_loader = mock.Mock()
        self.fake_loader.dataset = fake_dataset

        self.config = {
            "val_retrieval_itc_interval_steps": 1000,
            "val_retrieval_itm_interval_steps": 50,
            "val_retrieval_itc_epoch_end": True,
            "val_retrieval_itm_epoch_end": True,
        }

        self.runner = evr.RetrievalValRunner(
            data_loader=self.fake_loader,
            device="cpu",
            config=self.config,
            writer=None,
        )

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_step_interval_gating(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        # ITC: global_step이 itc_interval(1000)의 배수일 때마다 반복
        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=999, global_step=1000)
        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 0)

        # ITM: iteration(에폭 로컬 step)이 itm_interval(50)과 같아지는 순간 1번만
        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=50, global_step=1050)
        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 1)

        # 같은 에폭에서 다시 50을 지나가도 (이미 지나갔으므로 발생 안 함, 다른 값이면 트리거 안 됨)
        self.runner.val_retrieval_during_train(self.model, epoch=0, iteration=51, global_step=1051)
        self.assertEqual(mock_itm.call_count, 1)

        # 다음 에폭에서 iteration이 0부터 다시 시작해 50에 도달하면 다시 1번 트리거
        self.runner.val_retrieval_during_train(self.model, epoch=1, iteration=50, global_step=2050)
        self.assertEqual(mock_itm.call_count, 2)

        self.runner.val_retrieval_during_train(self.model, epoch=1, iteration=0, global_step=2001)
        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 2)

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_epoch_end_runs_both_tiers_and_restores_train_mode(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        self.model.train()
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37300)

        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 1)
        self.assertTrue(self.model.training)

    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.itm_eval")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itm")
    @mock.patch("data.eval_validation_retrieval.eval_validation_tool.evaluate_retrieval_itc")
    def test_epoch_end_respects_disabled_flags(self, mock_itc, mock_itm, mock_eval):
        mock_itc.return_value = (None, None)
        mock_itm.return_value = (None, None)
        mock_eval.return_value = {"r_mean": 0.0}

        self.config["val_retrieval_itm_epoch_end"] = False
        self.runner.run_epoch_end(self.model, epoch=0, global_step=37300)

        self.assertEqual(mock_itc.call_count, 1)
        self.assertEqual(mock_itm.call_count, 0)


class BuildRunnerFactoryTest(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        runner = evr.build_pretrain_retrieval_val_runner(
            config={"val_retrieval_enabled": False}, device="cpu", writer=None)
        self.assertIsNone(runner)


if __name__ == "__main__":
    unittest.main()
