# --------------------------------------- #
# Script: _04_manual_epochs.py
# Function: Create epochs of the manually annotated tic events
# Author: Martyna
# -------------------------------------- #

"""
Fist load the manually annotated urge events. File created manually sub-XX_ses-01_task-tictrack_tic_epoch_manual with start and end which is either the reported premonitory urge or start of the tic. 
File format:
Nb of tic	start	end	phase	tic_type	annot_type
1	63,06666667	71,23333333	PHASE_EC	expressed	video_start
2	74,13333333	76,03333333	PHASE_EC	expressed	video_start
3	109,3333333	120,7666667	PHASE_EC	expressed	video_start

Based on those events we create epochs of tic (in reality its tic + urge) and no tic (in reality no tic + no urge)intervals. The epoch is extracted around the start and defined pre- and post- interval. The epochs will be used for ICA.

"""

from __future__ import annotations


import matplotlib
matplotlib.use('Agg')
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import Counter
import pandas as pd
import mne
import sys
import os



# This adds the parent directory to your Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config.config import EPOCH_EXT_PARAMS, DATASET_DIR, PREPROC_DIR, PATIENTS, PHASES_TTL
from src.manual_tic_epochs import no_tic_gaps, create_tic_epochs
TASK = "tictrack"
SES = "01"
RUN = "01"


def build_phases_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Build phases_dict from decoded annotations in aligned_annotated_raw.fif.
    Uses 'start_PHASE_X' / 'end_PHASE_X' annotations written by 04_merge_events.py.
    """
    phase_names = ["PHASE_KP", "PHASE_EC", "PHASE_EO", "PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

    ann_dict = {}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        ann_dict.setdefault(desc, []).append(onset)

    phases_dict = {}
    for phase in phase_names:
        start_key = f"start_{phase}"  # e.g. 'start_PHASE_EC'
        end_key   = f"end_{phase}"    # e.g. 'end_PHASE_EC'

        start = ann_dict.get(start_key, [])
        end   = ann_dict.get(end_key,   [])

        if start and end:
            phases_dict[phase] = (start[0], end[0])
        else:
            print(f"[WARN] Missing annotation for '{phase}': "
                  f"start('{start_key}')={start}, end('{end_key}')={end}")
            phases_dict[phase] = None

    return phases_dict



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract manually annotated epochs"
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

        # --- 1. Load recalibrated EEG ---
        fif_path = PREPROC_DIR / sub / "realign" / f"{sub}_ses-01_task-tictrack_aligned_annotated_raw.fif"
        print(f"\n{'='*60}")
        print(f"[1/5] Loading recalibrated EEG: {fif_path}")
        print(f"\n{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
        
        print("[DEBUG] Annotations found in raw:")
        for desc in sorted(set(raw.annotations.description)):
            print(f"  '{desc}'")

        # --- 2. Load manual tic annotation ---
        # --- Output folder ---
        out_dir = PREPROC_DIR / sub / "tics_manual"
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / f"{sub}_ses-01_task-tictrack_tic_epoch_manual.xlsx"
        print(f"\n{'='*60}")
        print(f"[2/5] Loading manual tic annotations : {tsv_path}")
        print(f"\n{'='*60}")
        tics_df = pd.read_excel(tsv_path)

        phases_dict = build_phases_dict(raw)
        phase_boundaries = {
            phase_name: {"start": interval[0], "end": interval[1]}
            for phase_name, interval in phases_dict.items()
            if interval is not None
        }

        # -------------------- NO TIC ------------------- #
        print(f"[DEBUG] epoch_duration = {EPOCH_EXT_PARAMS['epoch_duration']}")
        no_tic_epochs, phases_nt= no_tic_gaps(raw, tics_df, phase_boundaries  = phase_boundaries,epoch_duration = EPOCH_EXT_PARAMS["epoch_duration"] , min_gap = EPOCH_EXT_PARAMS["min_gap"], used_phases = None)

        # --- 3. Build DataFrame from the epochs in between tics ---
        print(f"\n{'='*60}")
        print(f"[3/5] Creating summary for the no_tic epochs : {tsv_path}")
        print(f"\n{'='*60}")
        no_tic_metadata = pd.DataFrame({
            "onset":    no_tic_epochs.events[:, 0] / raw.info["sfreq"],  # convert samples back to seconds
            "duration": EPOCH_EXT_PARAMS["epoch_duration"],   
            "phase": phases_nt,

            })
        no_tic_epochs.metadata = no_tic_metadata
        print(f"[DEBUG] no_tic_epochs tmax = {no_tic_epochs.tmax}")


        # # --- Print epoch counts per phase ---
        print("\n[INFO] No-tic epoch counts per phase:")
        print(no_tic_metadata["phase"].value_counts().to_string())
        print(f"       Total: {len(no_tic_metadata)}\n")

        # --- Output folder ---
        tsv_dir = PREPROC_DIR / sub / "tics_manual"/"no_tic"
        tsv_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = tsv_dir / f"{sub}_ses-{SES}_task-{TASK}_run-{RUN}_no_tic_epochs.tsv"
        no_tic_metadata.to_csv(tsv_path, sep="\t", index=False)
        print(f"OK] Saved TSV: {tsv_path}")

        # --- 4. Plot the raw no_tic epochs ---

        # --- Plot raw epochs ---
        print(f"\n{'='*60}")
        print(f"[4/5] Plotting no-tic epochs for {sub}...")
        print(f"{'='*60}")
        fig = no_tic_epochs.plot(
            n_epochs   = 5,
            n_channels = 20,
            title      = f"No-tic gap epochs — {sub}",
            show = False,
        )
        fig.savefig(tsv_dir / f"{sub}_ses-{SES}_task-{TASK}_no_tic_epochs.png", dpi=150)
        print(f"[OK] Plot saved: {tsv_dir / f'{sub}_ses-{SES}_task-{TASK}_no_tic_epochs.png'}")

        # --- 5. Build DataFrame from the epochs ---
        # --- Output folder ---
        # -------------------- TIC ------------------- #
        out_dir = PREPROC_DIR / sub / "tics_manual"/ "tic"
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / f"{sub}_ses-{SES}_task-{TASK}_run-{RUN}_tic_epochs.tsv"

        print(f"\n{'='*60}")
        print(f"[5/5] Creating summary for the tic epochs : {tsv_path}")
        print(f"\n{'='*60}")

        tic_epochs, epochs_phase, epochs_type,  epochs_annot_type = create_tic_epochs(raw, tics_df, phase_boundaries  = phase_boundaries)
        tic_metadata = pd.DataFrame({
            "onset": tic_epochs.events[:, 0]/raw.info["sfreq"],
            "phase": epochs_phase,
            "type":epochs_type,
            "annot_type": epochs_annot_type,

        })

        # # --- Print epoch counts per phase ---
        print("\n[INFO] Tic epoch counts per phase:")
        print(tic_metadata["phase"].value_counts().to_string())
        print(f"       Total: {len(tic_metadata)}\n")

        tic_metadata.to_csv(tsv_path, sep="\t", index=False)
        print(f"OK] Saved TSV: {tsv_path}")

        for i in range(len(tic_epochs)):
            fig = tic_epochs[i].plot(
                n_epochs   = 1,
                n_channels = 20,
                title      = f"Pre-Tic gap epochs — {sub}",
                show = False
            )
            fig.savefig(out_dir / f"{sub}_ses-{SES}_task-{TASK}_pre_tic_epochs_{i}.png", dpi=150)
            print(f"[OK] Plot saved: {out_dir / f'{sub}_ses-{SES}_task-{TASK}_pre_tic_epochs.png'}")


    print("\n[DONE] Epoch creation complete.")



if __name__ == "__main__":
    main()