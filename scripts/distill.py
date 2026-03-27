#!/usr/bin/env python
"""Stage 2: Distill student model from the hold-out dataset."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anti_distill.config import Settings
from anti_distill.dataset import HoldoutDataset
from anti_distill.student import StudentDistiller

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    settings = Settings()

    dataset = HoldoutDataset(settings.dataset_path)
    hf_dataset = dataset.to_hf_dataset()
    print(f"Loaded {len(hf_dataset)} samples from {settings.dataset_path}")

    distiller = StudentDistiller(settings)
    distiller.distill(hf_dataset)
    print(f"Student model saved to {settings.output_dir}")


if __name__ == "__main__":
    main()
