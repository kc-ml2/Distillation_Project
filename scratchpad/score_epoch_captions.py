"""CPU-only COCO caption scoring for all generated checkpoint epochs."""

import argparse
import json
from pathlib import Path

import yaml

from data.eval_validation_caption import cpu_affinity, parse_cpu_list
from data.utils import coco_caption_eval


ROOT = Path(__file__).resolve().parents[1]
METRICS = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"}
MODELS = {
    "base": ("base_nodecay", ROOT / "output/caption_zeroshot/base_nodecay_ep02/config.yaml"),
    "lm": ("small_distill", ROOT / "output/caption_zeroshot/small_distill_ep02/config.yaml"),
    "small": ("small_solo", ROOT / "output/caption_zeroshot/small_solo_ep02/config.yaml"),
}


def needs_scoring(path):
    try:
        return not METRICS.issubset(json.loads(path.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--cpu-list")
    args = parser.parse_args()

    prefix, template = MODELS[args.model]
    config = yaml.safe_load(template.read_text())
    for epoch in range(20):
        root = ROOT / f"output/caption_zeroshot/{prefix}_ep{epoch:02d}"
        predictions = root / "result/val_epoch0.json"
        output = root / "val_metrics_with_spice.json"
        if not needs_scoring(output):
            continue
        with cpu_affinity(parse_cpu_list(args.cpu_list)):
            metrics = coco_caption_eval(config["coco_gt_root"], str(predictions), "val", use_spice=True).eval
        output.write_text(json.dumps(metrics, indent=2))
        print(f"{args.model} epoch {epoch:02d}: {metrics['CIDEr']:.4f} / {metrics['SPICE']:.4f}", flush=True)


if __name__ == "__main__":
    main()
