# -----------------------------------------------------------
# Script: _02_realign_excel_eeg.py
# Function: Realign the Excel file times with EEG recording times (based on the green led)
# Author: Martyna 
# Goal: Realign the EEG data to the green led and collect all the TTLs.
# ------------------------------------------------------------

"""
Recalibrated EXCEL files to EEG files. 
EXCEL files starts with 0, which represents green led. EEG file starts before so the data has to be cropped to the green led as well (Stimulus/S  2).That way, the manualy annotated tic intervals in the Excel file will be aligned with the EEG recording.
- After alignment we can visually inspect the raw data and the annotations to check if the alignment is correct (phase intervals) --> the start of phase in EXCEL file corresponds to the start of the phase in EEG file, so further resynchronization is unnecessary --> we add phase intervals from the EXCEL file
- Plot the recalibrated raw data and events using 01_extract_events_tsv.py --recalibrated 
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
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mne
from config.config import PATIENTS, PREPROC_DIR, DATASET_DIR, ICA_EXCLUSIONS
from src.align_eeg_excel import recalibrate_from_first_event
from src.helper_functions import extract_ttl_events


# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"

# BrainVision "Stimulus" marker strings look like: "Stimulus/S  3"
BV_STIM_RE = re.compile(r"^Stimulus/S\s+\d+$")


def bids_eeg_dir(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "excel"

def bids_vhdr_path(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "eeg"/ f"{sub}_task-{TASK}.vhdr"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realign the EEG data to the green led and collect all the TTLs"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-028 sub-029. Default: all dataset/sub-*",
    )
    return parser.parse_args()

def phases_from_excel(excel_file: Path, fps:int):
    """
    Function to exctract the start and end of each phase from the EXCEL file to compare with recalibrated EEG data.
    """
    df = pd.read_excel(excel_file)
    if fps == 30:
        time_col = "Time (30 fps) s"
        min_absence_frames = 30
    elif fps == 25:
        time_col = "Time (25 fps) s"
        min_absence_frames = 25
    else:
        raise ValueError(f"fps must be 30 or 25, got {fps}")

    led_col = "LED_PHASES"
    if led_col not in df.columns:
        raise ValueError(f"[ERROR] Missing required column: {led_col}")
    
    phases_led = [
            'LED_KP', 'LED_EC', 'LED_EO', 'LED_FREE', 'LED_MIM', 'LED_SUP'
        ]
    rows = []
    for phase in phases_led:
        phase_rows = df[df["LED_PHASES"] == phase][time_col].values
        if len(phase_rows) == 0:
            print(f"[WARN] {phase}: no row found in Excel — skipping")
            continue
        rows.append({
            "onset":      phase_rows[0],
            "trial_type": phase,
            "value":      phase,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    # --- Select patients ---
    if args.sub:
        subjects = {k: v for k, v in PATIENTS.items() if k in args.sub}
        missing = [s for s in args.sub if s not in PATIENTS]
        if missing:
            raise RuntimeError(f"Subjects not found in config: {missing}")
    else:
        subjects = PATIENTS

    for sub, cfg in subjects.items():
        print(f"\n{'='*60}")
        print(f"[START] Aligning {sub}")
        print(f"{'='*60}")

        # --- Output folder ---
        out_dir = PREPROC_DIR / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        realign_dir = PREPROC_DIR /sub / "realign"
        realign_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load raw EEG ---
        print(f"\n{'='*60}")
        print(f"[1/3] Loading raw EEG: ")
        print(f"\n{'='*60}")
        vdhr = bids_vhdr_path(sub)
        raw = mne.io.read_raw_brainvision(vdhr, preload= True, verbose="ERROR")

        # --- 2. Recalibrate EEG to S2 ---
        print(f"\n{'='*60}")
        print(f"[2/3] Recalibrating EEG to Stimulus/S  2...")
        print(f"\n{'='*60}")
        #raw = mne.io.read_raw_fif(raw_fif_path, preload=True)
        raw_realigned = recalibrate_from_first_event(raw, target_stim="Stimulus/S  2")

        # ---- 3. TTL extraction ---
        print(f"\n{'='*60}")
        print(f"[3/3] TTL extraction of recalibrated data")
        print(f"\n{'='*60}")
        df = extract_ttl_events(raw_realigned)

        # --- 4. Add EXCEL phase times ---
        df_phases = phases_from_excel(excel_file = cfg.excel_path, fps = cfg.fps)
        df_combined = pd.concat([df, df_phases], ignore_index=True)
        df_combined = df_combined.sort_values("onset").reset_index(drop=True)

        print(f"df_combined:{df_combined}")

        out_path_combined = realign_dir / f"{sub}_ses-{SES}_task-{TASK}_run-{RUN}_events_recalibrated_with_phases.tsv"
        df_combined.to_csv(out_path_combined, sep="\t", index=False)
        print(f"[OK] Saved combined TTL + phases: {out_path_combined}")

        recal_path = realign_dir / f"{sub}_ses-01_task-tictrack_aligned_raw.fif"
        raw_realigned.save(recal_path, overwrite = True)
        print(f"[OK] Saved recalibrated raw: {recal_path}")

        

        # ======== [DEBUG] ==========
        # In a test script
        raw_orig = mne.io.read_raw_brainvision(vdhr, preload=True)
        raw_realigned_2 = mne.io.read_raw_fif(recal_path, preload=True)
        # Pick one channel and compare a window
        ch = "F7"
        ch_idx = raw_orig.ch_names.index(ch)

        t_start = 151.738  # where green LED was in original
        sfreq = raw_orig.info["sfreq"]
        n_samples = int(50 * sfreq)  # 5 seconds

        orig_data = raw_orig.get_data(picks=ch_idx)[0]
        realigned_data = raw_realigned_2.get_data(picks=ch_idx)[0]

        # These two windows should look identical
        orig_window = orig_data[int(t_start * sfreq): int(t_start * sfreq) + n_samples]
        realigned_window = realigned_data[:n_samples]
        orig_window_dc      = orig_window - orig_window.mean()
        realigned_window_dc = realigned_window - realigned_window.mean()

        time = np.arange(n_samples) / sfreq
        plt.plot(time, orig_window_dc, label="raw data from green LED")
        plt.plot(time, realigned_window_dc, label="realigned from t=0", linestyle="--")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (V)")
        plt.title("EEG Cz channel: Original vs Realigned segment")
        plt.legend()
        plt.savefig("compare.png")

        print(f"[OK] Saved: {out_path_combined}")
    print("[DONE] TTL extraction complete.")




        



if __name__ == "__main__":
    main()
