
# ❀ ---------------------------------------------- ❀
# Script : 01_extract_events_tsv.py
# Author  : LizbethMG
# Goal: Extract TTL events from BrainVision .vhdr files and save as BIDS-like events.tsv
# ❀ ---------------------------------------------- ❀

"""
Summary of TTLs and their times. 
Used for visual inspection and sanity checks (raw data, and calibrated)

"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from collections import Counter
import pandas as pd
import mne
import sys
import os

# This adds the parent directory to your Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Your config lives in config/config.py
from config.config import TTL_MAP, DATASET_DIR, PREPROC_DIR, PHASES_TTL
from src.helper_functions import extract_ttl_events

# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"

# BrainVision "Stimulus" marker strings look like: "Stimulus/S  3"
BV_STIM_RE = re.compile(r"^Stimulus/S\s+\d+$")


def bids_eeg_dir(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" /"excel"

def bids_vhdr_path(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "eeg"/ f"{sub}_task-{TASK}.vhdr"


def write_qc_summary(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write a simple QC summary alongside events.tsv:
    - counts per trial_type
    - counts per raw value
    """
    counts_trial_type = Counter(df["trial_type"].tolist())
    counts_value = Counter(df["value"].tolist())

    qc_lines = []
    qc_lines.append("=== QC SUMMARY: COUNTS BY trial_type ===")
    for k, v in counts_trial_type.most_common():
        qc_lines.append(f"{k}\t{v}")

    qc_lines.append("\n=== QC SUMMARY: COUNTS BY raw value (BrainVision string) ===")
    for k, v in counts_value.most_common():
        qc_lines.append(f"{k}\t{v}")

    out_path.write_text("\n".join(qc_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract BrainVision TTL markers to BIDS-like events.tsv"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-028 sub-029. Default: all dataset/sub-*",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Discover subjects from dataset folder
    all_subjects = sorted([p.name for p in DATASET_DIR.glob("sub-*") if p.is_dir()])
    if not all_subjects:
        raise RuntimeError(
            f"No subjects found in {DATASET_DIR}. Expected folders like dataset/sub-001/"
        )

    # Select subset if requested
    subjects = args.sub if args.sub else all_subjects

    # Validate selection
    missing = [s for s in subjects if s not in all_subjects]
    if missing:
        raise RuntimeError(f"Requested subjects not found in dataset: {missing}")

    for sub in subjects:

        vhdr = bids_vhdr_path(sub)
        if not vhdr.exists():
            print(f"[SKIP] Missing vhdr for {sub}: {vhdr}")
            continue

        # Load without preload (fast). We only need annotations.
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")

        df = extract_ttl_events(raw)

        # Hard fail if UNKNOWN TTLs exist: prevents silent mislabeling
        unknown = df[df["trial_type"] == "UNKNOWN"]
        if len(unknown) > 0:
            examples = unknown["value"].value_counts().head(10).to_dict()
            raise RuntimeError(
                f"{sub}: Found UNKNOWN TTLs (not in TTL_MAP).\n"
                f"Examples (value:count): {examples}\n"
                f"Fix: add the missing codes to TTL_LABELS in config/config.py."
            )

        # --- Output folder ---
        out_dir = PREPROC_DIR / sub / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir/ f"{sub}_ses-{SES}_task-{TASK}_run-{RUN}_events.tsv"
        df.to_csv(out_path, sep="\t", index=False)

        out_qc = out_dir / f"{sub}_ses-{SES}_task-{TASK}_run-{RUN}_events_qc.txt"
        write_qc_summary(df, out_qc)

        print(f"[OK] Saved: {out_path}")

    print("[DONE] TTL extraction complete.")


if __name__ == "__main__":
    main()
