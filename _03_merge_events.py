# -----------------------------------------------------------
# Script: _03_merge_events.py
# Function: Merge EEG TTL events and Excel tic annotations into one tidy .tsv file with event, time, phase
# Author: Martyna 
# ------------------------------------------------------------
"""
Merging EEG TTL events with tics extracted from the Excel file. 
The rules of the tic extraction are defined in the extract_tics_from_excel function in excel_tic_extraction.py.
Creating one file with TTL events and tic events, sorted by time, and with the phase label for each event.
Further, adding annotations (TTLs and EXCEL) to the raw data for visual inspection and later usage in the analysis.
Here, we also create a file with the delay between KEY_D and the start of each tic, which will be used in the analysis to check if tics are more likely to occur after a certain delay from the KEY_D event.
"""

from __future__ import annotations

import sys
import os
import argparse
import pandas as pd
import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import PATIENTS, PREPROC_DIR, PHASES_TTL, TTL_MAP
from src.helper_functions import plot_raw, compute_key_d_to_start_delays
from src.excel_tic_extraction import extract_tics_from_excel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge EEG TTL events and Excel tic annotations into one .tsv file"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-BB28 sub-BC29. Default: all patients",
    )
    return parser.parse_args()


def build_phases_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Build phases_dict from recalibrated EEG annotations.
    """
    phases_dict = {}
    for phase_name, t in PHASES_TTL.items():
        start = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description) if desc == t["start"]]
        end   = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description) if desc == t["end"]]
        if start and end:
            phases_dict[phase_name] = (start[0], end[0])
        else:
            print(f"[WARN] Missing TTL for phase '{phase_name}'")
            phases_dict[phase_name] = None
    return phases_dict


def get_phase(time: float, phases_dict: dict) -> str | None:
    """
    Find which phase a given time belongs to.
    """
    items = list(phases_dict.items())
    for i, (phase_name, interval) in enumerate(items):
        if interval is None:
            continue
        start_t, end_t = interval
        # last phase uses <=, all others use 
        if i == len(items) - 1:
            if start_t <= time <= end_t:
                return phase_name
        else:
            if start_t <= time < end_t:
                return phase_name
    return None

def add_annotations(raw: mne.io.BaseRaw, df: pd.DataFrame):
    """
    Add annotations from merged events (TTL & EXCEL) to the raw recalibrated data.
    Visual feedback (FB) was added as a control and will not be needed in the analysis. That is why, for cleaner data we will ommit it in the annotations in the raw data.
    """
    # --- Remove FB_ events ---
    df_clean = df[~df["event"].str.startswith("FB_")].copy()
    print(f"[INFO] Annotations after removing FB_: {len(df_clean)} (removed {len(df) - len(df_clean)})")

    # TTLs already in the raw data -- remove them 
    print(f"[INFO] Clearing {len(raw.annotations)} existing raw annotations (Stimulus/S markers)")
    raw.set_annotations(mne.Annotations([], [], []))

    # --- Build and merge annotations ---
    new_annotations = mne.Annotations(
        onset       = df_clean["time"].values,
        duration    = np.zeros(len(df_clean)),
        description = df_clean["event"].values.astype(str),
    )
    raw.set_annotations(raw.annotations + new_annotations)
    print(f"[INFO] Total annotations on raw: {len(raw.annotations)}")

    return raw

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
        print(f"[START] Merging events for {sub}")
        print(f"{'='*60}")

        # --- Output folder ---
        out_dir = PREPROC_DIR / sub / "tics"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Load recalibrated EEG ---
        fif_path = PREPROC_DIR / sub / "realign" / f"{sub}_ses-01_task-tictrack_aligned_raw.fif"
        
        print(f"\n{'='*60}")
        print(f"[1/4] Loading recalibrated EEG: {fif_path}")
        print(f"\n{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

        # --- 2. Build phases dict ---
        print(f"\n{'='*60}")
        print(f"[2/4] Building phases dictionary...")
        print(f"\n{'='*60}")
        phases_dict = build_phases_dict(raw)

        # Add phase boundary events for later usage 
        rows = []

        for phase_name, interval in phases_dict.items():
            start_time, end_time = interval

            rows.append({
                "event":  f"start_{phase_name}",
                "time":   start_time,
                "phase":  phase_name,
            })
            rows.append({
                "event":  f"end_{phase_name}",
                "time":   end_time,
                "phase":  phase_name,
            })


        # --- 3. Extract TTL events from EEG ---
        print(f"\n{'='*60}")
        print(f"[3/4] Extracting TTL events...")
        print(f"\n{'='*60}")
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            trial_type = TTL_MAP.get(desc, None)
            if trial_type is None:
                continue
            phase = get_phase(onset, phases_dict)
            rows.append({
                "event":  trial_type,
                "time":   onset,
                "phase":  phase,
                "source": "EEG"
            })

        # --- 4. Extract tics from Excel ---
        print(f"\n{'='*60}")
        print(f"[4/4] Extracting tics from Excel: {cfg.excel_path}")
        print(f"\n{'='*60}")
        tics = extract_tics_from_excel(
            excel_file=cfg.excel_path,
            fps=cfg.fps,
            min_absence_frames=30,
        )
        print(f"[INFO] Found {len(tics)} tics for {sub}")

        for i, (start_time, end_time) in enumerate(tics, start=1):
            phase = get_phase(start_time, phases_dict)
            rows.append({
                "event":  f"start_{i}",
                "time":   start_time,
                "phase":  phase,
                "source": "Excel"
            })
            rows.append({
                "event":  f"end_{i}",
                "time":   end_time,
                "phase":  phase,
                "source": "Excel"
            })

        # --- Sort by time and save ---
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


        # ===== CALCULATE THE DIFFERENCE BETWEEN D & START_ ==== #
        pairs = compute_key_d_to_start_delays(df, max_delay=3.0)
        out_path = out_dir / f"{sub}_ses-01_task-tictrack_start_diff.tsv"
        pairs.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] Saved start to KEY_D differences: {out_path}")


        out_path = out_dir / f"{sub}_ses-01_task-tictrack_merged_events.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"[OK] Saved: {out_path}")
        print(f"[OK] Total rows: {len(df)} ({len(tics)} tics + {len(rows) - len(tics)*2} TTLs)")


        # --- 5. Add annotations to realigned raw data ---
        raw = add_annotations(raw, df)
        out_fif = PREPROC_DIR / sub / "realign" / f"{sub}_ses-01_task-tictrack_aligned_annotated_raw.fif"
        raw.save(out_fif, overwrite=True)
        print(f"[OK] Saved annotated raw: {out_fif.name}")



        # --- 6. Plot the realigned raw data ---
        # print(f"\n{'='*60}")
        # print(f"[5/5] Plotting annotated raw by phase...")
        # print(f"\n{'='*60}")
        # plot_dir = PREPROC_DIR / sub/"realign" / "plots"
        # plot_dir.mkdir(parents=True, exist_ok=True)

        # plot_raw(
        #     raw         = raw,
        #     phases_dict = phases_dict,
        #     sub_id      = sub,
        #     plot_dir    = plot_dir,
        #     window_sec    = 30.0,
        #     n_channels  = 20,
        # )

    print("\n[DONE] Merge complete.")


if __name__ == "__main__":
    main()

"""

Output `.tsv` will look like:

event              time      phase        source
PHASE_KP_INS       0.000     None         EEG
PHASE_KP           9.500     PHASE_KP     EEG
KEY_D              12.300    PHASE_KP     EEG
PHASE_EC_INS       45.000    None         EEG
PHASE_EC           54.500    PHASE_EC     EEG
tic_1_start        60.033    PHASE_EC     Excel
tic_1_end          60.800    PHASE_EC     Excel
tic_2_start        72.100    PHASE_FREE   Excel
tic_2_end          72.567    PHASE_FREE   Excel
"""