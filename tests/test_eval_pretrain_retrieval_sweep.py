import json
import os
import tempfile
import unittest
from unittest import mock

import eval_pretrain_retrieval_sweep as sweep


class CheckpointPathForEpochTest(unittest.TestCase):
    def test_zero_pads_epoch_to_two_digits(self):
        path = sweep.checkpoint_path_for_epoch("/ckpt_dir", 0)
        self.assertEqual(path, "/ckpt_dir/checkpoint_00.pth")

    def test_handles_two_digit_epoch(self):
        path = sweep.checkpoint_path_for_epoch("/ckpt_dir", 19)
        self.assertEqual(path, "/ckpt_dir/checkpoint_19.pth")


class BuildCustomScalarsLayoutTest(unittest.TestCase):
    def test_groups_txt_and_img_recall_separately_per_tier(self):
        layout = sweep.build_custom_scalars_layout(tag_prefix="checkpoint_sweep")

        itc_txt = layout["ITC"]["txt_recall"][1]
        itc_img = layout["ITC"]["img_recall"][1]
        self.assertEqual(itc_txt, [
            "checkpoint_sweep/itc/txt_r1",
            "checkpoint_sweep/itc/txt_r5",
            "checkpoint_sweep/itc/txt_r10",
        ])
        self.assertEqual(itc_img, [
            "checkpoint_sweep/itc/img_r1",
            "checkpoint_sweep/itc/img_r5",
            "checkpoint_sweep/itc/img_r10",
        ])

    def test_summary_combines_r_mean_across_tiers(self):
        layout = sweep.build_custom_scalars_layout(tag_prefix="checkpoint_sweep")
        r_mean = layout["Summary"]["r_mean"][1]
        self.assertEqual(r_mean, [
            "checkpoint_sweep/itc/r_mean",
            "checkpoint_sweep/itm/r_mean",
        ])

    def test_omits_baseline_tags_by_default(self):
        layout = sweep.build_custom_scalars_layout(tag_prefix="checkpoint_sweep")
        itc_txt = layout["ITC"]["txt_recall"][1]
        self.assertEqual(len(itc_txt), 3)

    def test_includes_baseline_tags_in_every_chart_when_requested(self):
        layout = sweep.build_custom_scalars_layout(
            tag_prefix="checkpoint_sweep", include_baseline=True, baseline_name="official")

        itc_txt = layout["ITC"]["txt_recall"][1]
        self.assertEqual(itc_txt, [
            "checkpoint_sweep/itc/txt_r1",
            "checkpoint_sweep/itc/txt_r5",
            "checkpoint_sweep/itc/txt_r10",
            "checkpoint_sweep/itc/official_txt_r1",
            "checkpoint_sweep/itc/official_txt_r5",
            "checkpoint_sweep/itc/official_txt_r10",
        ])

        r_mean = layout["Summary"]["r_mean"][1]
        self.assertEqual(r_mean, [
            "checkpoint_sweep/itc/r_mean",
            "checkpoint_sweep/itm/r_mean",
            "checkpoint_sweep/itc/official_r_mean",
            "checkpoint_sweep/itm/official_r_mean",
        ])


class WriteMetricsToTensorboardTest(unittest.TestCase):
    def test_writes_one_scalar_per_metric_with_prefixed_tag(self):
        writer = mock.Mock()
        metrics = {"txt_r1": 51.84, "r_mean": 66.42}

        sweep.write_metrics_to_tensorboard(writer, "itc", metrics, global_step=7, tag_prefix="checkpoint_sweep")

        writer.add_scalar.assert_any_call("checkpoint_sweep/itc/txt_r1", 51.84, 7)
        writer.add_scalar.assert_any_call("checkpoint_sweep/itc/r_mean", 66.42, 7)
        self.assertEqual(writer.add_scalar.call_count, 2)


class WriteConstantBaselineToTensorboardTest(unittest.TestCase):
    def test_repeats_each_metric_at_every_step_in_range(self):
        writer = mock.Mock()
        metrics = {"r_mean": 55.0, "txt_r1": 60.0}

        sweep.write_constant_baseline_to_tensorboard(
            writer, "itc", metrics, epoch_range=range(0, 3),
            tag_prefix="checkpoint_sweep", baseline_name="official",
        )

        writer.add_scalar.assert_any_call("checkpoint_sweep/itc/official_r_mean", 55.0, 0)
        writer.add_scalar.assert_any_call("checkpoint_sweep/itc/official_r_mean", 55.0, 1)
        writer.add_scalar.assert_any_call("checkpoint_sweep/itc/official_r_mean", 55.0, 2)
        self.assertEqual(writer.add_scalar.call_count, 6)


class RunSweepTest(unittest.TestCase):
    def setUp(self):
        self.model = mock.Mock()
        self.loader = mock.Mock()
        self.loader.dataset.txt2img = {0: 0}
        self.loader.dataset.img2txt = {0: [0]}
        self.device = "cpu"
        self.config = {}
        self.writer = mock.Mock()

    @mock.patch("eval_pretrain_retrieval_sweep.utils.is_main_process", return_value=True)
    @mock.patch("eval_pretrain_retrieval_sweep.utils.load_model_weights_only")
    @mock.patch("eval_pretrain_retrieval_sweep.eval_validation_tool.itm_eval")
    @mock.patch("eval_pretrain_retrieval_sweep.eval_validation_tool.evaluate_retrieval_itc")
    def test_iterates_epoch_range_inclusive_with_correct_global_step(
        self, mock_itc, mock_itm_eval, mock_load_weights, mock_is_main,
    ):
        mock_load_weights.return_value = self.model
        mock_itc.return_value = (None, None)
        mock_itm_eval.return_value = {"r_mean": 50.0}

        results = sweep.run_sweep(
            model=self.model, loader=self.loader, device=self.device, config=self.config,
            checkpoint_dir="/ckpt_dir", start_epoch=0, end_epoch=2, tier="itc",
            writer=self.writer, results={},
        )

        self.assertEqual(mock_itc.call_count, 3)
        self.assertEqual(list(results["itc"].keys()), [0, 1, 2])

        called_steps = [call.args[2] for call in self.writer.add_scalar.call_args_list]
        self.assertEqual(called_steps, [0, 1, 2])

    @mock.patch("eval_pretrain_retrieval_sweep.utils.is_main_process", return_value=True)
    @mock.patch("eval_pretrain_retrieval_sweep.utils.load_model_weights_only")
    @mock.patch("eval_pretrain_retrieval_sweep.eval_validation_tool.itm_eval")
    @mock.patch("eval_pretrain_retrieval_sweep.eval_validation_tool.evaluate_retrieval_itm")
    def test_itm_tier_calls_evaluate_retrieval_itm_not_itc(
        self, mock_itm, mock_itm_eval, mock_load_weights, mock_is_main,
    ):
        mock_load_weights.return_value = self.model
        mock_itm.return_value = (None, None)
        mock_itm_eval.return_value = {"r_mean": 70.0}

        sweep.run_sweep(
            model=self.model, loader=self.loader, device=self.device, config=self.config,
            checkpoint_dir="/ckpt_dir", start_epoch=5, end_epoch=5, tier="itm",
            writer=self.writer, results={},
        )

        self.assertEqual(mock_itm.call_count, 1)

    def test_rejects_unknown_tier(self):
        with self.assertRaises(ValueError):
            sweep.run_sweep(
                model=self.model, loader=self.loader, device=self.device, config=self.config,
                checkpoint_dir="/ckpt_dir", start_epoch=0, end_epoch=0, tier="bogus",
                writer=self.writer, results={},
            )


class SaveResultsTest(unittest.TestCase):
    def test_merges_with_existing_file_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "checkpoint_sweep_val.json")

            sweep.save_results(out_path, {"itc": {0: {"r_mean": 44.17}}})
            sweep.save_results(out_path, {"itm": {0: {"r_mean": 50.0}}})

            with open(out_path) as f:
                saved = json.load(f)

            self.assertEqual(set(saved.keys()), {"itc", "itm"})
            self.assertEqual(saved["itc"]["0"]["r_mean"], 44.17)
            self.assertEqual(saved["itm"]["0"]["r_mean"], 50.0)

    def test_overwrites_same_tier_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "checkpoint_sweep_val.json")

            sweep.save_results(out_path, {"itc": {0: {"r_mean": 1.0}}})
            sweep.save_results(out_path, {"itc": {0: {"r_mean": 2.0}}})

            with open(out_path) as f:
                saved = json.load(f)

            self.assertEqual(saved["itc"]["0"]["r_mean"], 2.0)

    def test_adding_baseline_key_preserves_existing_epochs_in_same_tier(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = os.path.join(tmp_dir, "checkpoint_sweep_val.json")

            sweep.save_results(out_path, {"itc": {0: {"r_mean": 1.0}, 1: {"r_mean": 2.0}}})
            sweep.save_results(out_path, {"itc": {"official": {"r_mean": 50.0}}})

            with open(out_path) as f:
                saved = json.load(f)

            self.assertEqual(set(saved["itc"].keys()), {"0", "1", "official"})
            self.assertEqual(saved["itc"]["0"]["r_mean"], 1.0)
            self.assertEqual(saved["itc"]["official"]["r_mean"], 50.0)


if __name__ == "__main__":
    unittest.main()
