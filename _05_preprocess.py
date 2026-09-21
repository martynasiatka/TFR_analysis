#  ------------------------------------------------------- #
# Script : 05_preprocess.py
# Function : Run full preprocessing pipeline for all patients
# Author  : Martyna
#  ------------------------------------------------------- #
"""
To preprocess the EEG data, we will run the following steps for each patient:
1. Load the recalibrated raw EEG data
2. Filter and apply montage
3. Detect bad channels with RANSAC
4. Load epochs for ICA (using no-tic epochs)
5. Reject bad epochs using FASTER method
6. Apply ICA to remove artifacts
7. Re-reference to REST

"""
from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mne
from config.config import PATIENTS, PREPROC_DIR, DATASET_DIR, ICA_EXCLUSIONS, EPOCH_EXT_PARAMS, PHASES_TTL
from src.preproc import (
    preprocess_raw,
    Ransac_bad_channel_detection,
    rejection_threshold_std,
    apply_ICA,
    apply_rest_reference
)
from src.helper_functions import plot_raw #build_phases_dict
from src.manual_tic_epochs import no_tic_gaps
# Dataset constants (1 session, 1 run)
TASK = "tictrack"
SES = "01"
RUN = "01"


def bids_eeg_dir(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "excel"

def bids_vhdr_path(sub: str) -> Path:
    return DATASET_DIR / sub / f"ses-{SES}" / "eeg"/ f"{sub}_task-{TASK}.vhdr"

def build_phases_dict(raw: mne.io.BaseRaw) -> dict:
    phases_dict = {}
    for phase_name in PHASES_TTL.keys():
        start = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
                 if desc == f"start_{phase_name}"]
        end   = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
                 if desc == f"end_{phase_name}"]
        if start and end:
            phases_dict[phase_name] = (start[0], end[0])
        else:
            print(f"[WARN] Missing annotation for phase '{phase_name}'")
            phases_dict[phase_name] = None
    return phases_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for TicTrack EEG patients"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-BB28 sub-BC29. Default: all patients",
    )
    return parser.parse_args()


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
        print(f"[START] Processing {sub}")
        print(f"{'='*60}")

        # --- Output folders ---
        out_dir = PREPROC_DIR / sub
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = out_dir / "preprocessing"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load recalibrated raw EEG ---
        fif_path = PREPROC_DIR / sub / "realign" / f"{sub}_ses-01_task-tictrack_aligned_annotated_raw.fif"
        print(f"\n{'='*60}")
        print(f"[1/7] Loading EEG: {fif_path}")
        print(f"\n{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    
        # --- 2. Preprocess (filter + montage) ---
        print(f"\n{'='*60}")
        print(f"[2/7] Filtering and applying montage...")
        print(f"\n{'='*60}")
        raw = preprocess_raw(raw, sub, cfg.montage, plots_dir)

        # --- 3. RANSAC bad channel detection ---
        print(f"\n{'='*60}")
        print(f"[3/7] Detecting bad channels with RANSAC...")
        print(f"\n{'='*60}")
        raw, bad_channels = Ransac_bad_channel_detection(raw, sub, plots_dir)

        # --- 4. Epoch rejection (FASTER) ---
        print(f"\n{'='*60}")
        print(f"[4/7] Rejecting bad epochs (FASTER)...")
        print(f"\n{'='*60}")
        # load epochs for ICA
        # --- Output folder ---
        load_out_dir = PREPROC_DIR / sub / "tics_manual"
        load_out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = load_out_dir / f"{sub}_ses-01_task-tictrack_tic_epoch_manual.xlsx"
        tics_df = pd.read_excel(tsv_path)

        phases_dict = build_phases_dict(raw)
        phase_boundaries = {
            phase_name: {"start": interval[0], "end": interval[1]}
            for phase_name, interval in phases_dict.items()
            if interval is not None
        }
        used_phases = ["PHASE_EC", "PHASE_EO", "PHASE_FREE"]

        no_tic_epochs, _ = no_tic_gaps(raw, tics_df, phase_boundaries  = phase_boundaries,epoch_duration = EPOCH_EXT_PARAMS["random_epoch_duration"] , min_gap = EPOCH_EXT_PARAMS["min_gap"], used_phases = used_phases)
        epochs_temp = rejection_threshold_std(raw, sub, plots_dir, epochs = no_tic_epochs)

        # --- 5. ICA ---
        print(f"\n{'='*60}")
        print(f"[5/7] Applying ICA...")
        print(f"\n{'='*60}")
        raw = apply_ICA(epochs_temp, raw, sub, ICA_EXCLUSIONS, plots_dir)

        # --- 6. Re-reference to REST ---
        print(f"\n{'='*60}")
        print(f"[6/7] Re-referencing to REST...")
        print(f"\n{'='*60}")
        raw = apply_rest_reference(raw, sub, plots_dir)

        # --- Save preprocessed raw ---
        out_path = out_dir /"preprocessing" / f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif"
        raw.save(out_path, overwrite=True)
        print(f"[OK] Saved preprocessed data: {out_path}")
        print(f"[OK] Saved plots to: {plots_dir}")

        # --- 7. Plot raw data ---
        # --- Output folder ---
        plot_dir = PREPROC_DIR / sub / "preprocessing"/"plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"[7/7] Plotting annotated raw by phase...")
        print(f"\n{'='*60}")
        fif_path = PREPROC_DIR / sub / "preprocessing" / f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif"
        raw_proc = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

        phases_dict = build_phases_dict(raw)
        n_channels = len( mne.pick_types(raw.info, meg=False, eeg=True))
        plot_raw(
            raw         = raw_proc,
            phases_dict = phases_dict,
            sub_id      = sub,
            plot_dir    = plot_dir,
            #window_sec    = 30.0,
            n_channels  = n_channels,
        )
        print(f"[OK] Saved raw data plots: {plot_dir.name}")

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()

