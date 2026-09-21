# ----------------------------------------------- #
# Script: _06_plot_epochs.py
# Function: Plot the extracted epochs
# Author: Martyna
# ----------------------------------------------- #

"""
Here we are creating the epochs again, however, now based on the preprocessed raw data (after ICA and bad channel interpolation) and the manually annotated tic events.
Plot the epochs for each phase. 

"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    EPOCH_EXT_PARAMS, PREPROC_DIR, PATIENTS, PHASES_TTL,
    CHANNELS_32, CHANNELS_64,
    ROI_LIST_32, ROI_LIST_64,
    ROI_COLORS, ANNOTATION_COLORS, ANNOTATION_COLORS_DEFAULT
)
from src.manual_tic_epochs import create_tic_epochs, no_tic_gaps, plot_eo_ec_spectrum_no_tic_by_roi


TASK   = "tictrack"
SES    = "01"
RUN = "01"
PHASES = ["PHASE_EO", "PHASE_EC", "PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

def build_phases_dict(raw: mne.io.BaseRaw) -> dict:
    """
    Build phases_dict from annotations that use start_PHASE_X / end_PHASE_X naming.
    Works with the merged_events.tsv annotations (not BrainVision TTL codes).
    """
    descriptions = set(raw.annotations.description)
    phases_dict  = {}
 
    for phase_name in PHASES:
        start_label = f"start_{phase_name}"   # e.g. "start_PHASE_FREE"
        end_label   = f"end_{phase_name}"     # e.g. "end_PHASE_FREE"
 
        start = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
                 if desc == start_label]
        end   = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
                 if desc == end_label]
 
        if start and end:
            phases_dict[phase_name] = (start[0], end[0])
        else:
            print(f"[WARN] Missing annotation for phase '{phase_name}' "
                  f"(looked for '{start_label}' / '{end_label}')")
            phases_dict[phase_name] = None
 
    return phases_dict

def plot_raw_epochs_with_roi(raw, epochs, epochs_phase, epochs_type, epochs_annot_type,
                             patient, patient_id, phase_name,
                             out_dir: Path,
                             scalings=20e-6):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select channels and ROIs based on montage
    if patient.montage == "standard_1020":
        channels_to_use = CHANNELS_32
        roi_lists       = ROI_LIST_32
    elif patient.montage == "standard_1005":
        channels_to_use = CHANNELS_64
        roi_lists       = ROI_LIST_64
    else:
        raise ValueError(f"Unknown montage: {patient.montage}")

    # Pick ROI channels
    available_channels = [ch for ch in channels_to_use if ch in epochs.ch_names]
    if not available_channels:
        print(f"[SKIP] No ROI channels available in epochs")
        return

    epochs_roi = epochs.copy().pick(available_channels)

    # Channel -> ROI colour map
    channel_colors = {}
    for roi_name, roi_channels in roi_lists.items():
        for ch in roi_channels:
            if ch in available_channels:
                channel_colors[ch] = ROI_COLORS[roi_name]

    # Annotation colour map
    all_events, event_id_from_annot = mne.events_from_annotations(raw, verbose=False)
    annotation_colors = {}
    for desc, int_id in event_id_from_annot.items():
        if desc in ANNOTATION_COLORS:
            annotation_colors[int_id] = ANNOTATION_COLORS[desc]
        elif desc.startswith("start_"):
            annotation_colors[int_id] = "green"
        elif desc.startswith("end_"):
            annotation_colors[int_id] = "red"
        else:
            annotation_colors[int_id] = ANNOTATION_COLORS_DEFAULT


    # Keep only epochs that belong to this phase
    phase_indices = [i for i, p in enumerate(epochs_phase) if p == phase_name]
    if not phase_indices:
        print(f"[SKIP] No epochs found for phase '{phase_name}'")
        return

    n_phase    = len(phase_indices)
    has_events = len(all_events) > 0

    print(f"  [INFO] {n_phase} epochs to plot for {phase_name}")

    for plot_num, epoch_idx in enumerate(phase_indices, start=1):

        tic_type   = epochs_type[epoch_idx]
        annot_type = epochs_annot_type[epoch_idx]

        title = (f"{patient_id}  |  {phase_name}  |  {tic_type}"
                 f"  |  epoch {plot_num}/{n_phase}  [{annot_type}]")

        fig = epochs_roi[epoch_idx].plot(
            n_epochs    = 1,
            scalings    = scalings,
            n_channels  = len(available_channels),
            title       = title,
            show        = False,
            events      = all_events          if has_events else None,
            event_id    = event_id_from_annot if has_events else None,
            event_color = annotation_colors   if has_events else None,
        )

        ax = fig.axes[0]

        # Colour channel labels by ROI
        for tick_label in ax.get_yticklabels():
            ch_name = tick_label.get_text()
            if ch_name in channel_colors:
                tick_label.set_color(channel_colors[ch_name])
                tick_label.set_weight("bold")


        # Save
        fname = (f"{patient_id}_ses-{SES}_task-{TASK}"
                 f"_{phase_name}_{tic_type}_epoch{plot_num:03d}.png")
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved -> {fname}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the epochs extracted manually"
    )
    parser.add_argument(
        "--sub",
        nargs="*",
        default=None,
        help="Subjects to process, e.g. --sub sub-028 sub-029. Default: all dataset/sub-*",
    )
    return parser.parse_args()

def main():
    args   = parse_args()

    if args.sub:
        subjects = {k: v for k, v in PATIENTS.items() if k in args.sub}
        missing  = [s for s in args.sub if s not in PATIENTS]
        if missing:
            raise RuntimeError(f"Subjects not found in config: {missing}")
    else:
        subjects = PATIENTS

    for sub, cfg in subjects.items():

        # --- 1. Load preprocessed raw ---
        fif_path = (PREPROC_DIR / sub / "preprocessing" /
                    f"{sub}_ses-01_task-tictrack_preprocessed_raw.fif")
        print(f"\n{'='*60}")
        print(f"[1/4] Loading preprocessed raw: {fif_path}")
        print(f"{'='*60}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

        # --- 2. Add annotations to the raw preproccesed data 
        # --- Load recalibrated events and add to preprocessed raw ---
        out_dir = PREPROC_DIR / sub / "tics"
        events_path = out_dir / f"{sub}_ses-01_task-tictrack_merged_events.tsv"
        print(f"[INFO] Loading recalibrated events: {events_path}")
        events_df = pd.read_csv(events_path, sep="\t")
 
        # Remove FB_ events
        events_df = events_df[~events_df["event"].str.startswith("FB_")].copy()
        print(f"[INFO] {len(events_df)} annotations after removing FB_ events")
 
        # Build and set annotations on preprocessed raw
        new_annotations = mne.Annotations(
            onset       = events_df["time"].values,
            duration    = np.zeros(len(events_df)),
            description = events_df["event"].values.astype(str),
        )
        raw.set_annotations(new_annotations)
        print(f"[INFO] {len(raw.annotations)} annotations added to preprocessed raw")
        print("[DEBUG] Annotation descriptions:", set(raw.annotations.description))

        # --- 2. Load manual tic annotation Excel ---
        tsv_path = (PREPROC_DIR / sub / "tics_manual" /
                    f"{sub}_ses-01_task-tictrack_tic_epoch_manual.xlsx")
        print(f"[2/4] Loading manual annotations: {tsv_path}")
        tics_df = pd.read_excel(tsv_path)

        phases_dict = build_phases_dict(raw)
        phase_boundaries = {
            phase_name: {"start": interval[0], "end": interval[1]}
            for phase_name, interval in phases_dict.items()
            if interval is not None
        }

        # --- 3. Create epochs from preprocessed raw & plot ---
        print(f"[3/4] Creating and plotting epochs for {sub}...")
        tic_epochs, epochs_phase, epochs_type, epochs_annot_type = create_tic_epochs(
            raw, tics_df, phase_boundaries=phase_boundaries
        )
        metadata = pd.DataFrame({
            "phase": epochs_phase,
            "tic_type": epochs_type,
            "annot_type": epochs_annot_type
        })

        tic_epochs.metadata = metadata

        for phase in phases_dict:
            epochs_dir = PREPROC_DIR / sub / "epochs" / phase
            plot_raw_epochs_with_roi(
                raw               = raw,
                epochs            = tic_epochs,
                epochs_phase      = epochs_phase,
                epochs_type       = epochs_type,
                epochs_annot_type = epochs_annot_type,
                patient           = cfg,
                patient_id        = sub,
                phase_name        = phase,
                out_dir           = epochs_dir,
            )
        
        # --- 4. Save the epochs as fif file 

        epochs_fif_path = PREPROC_DIR / sub / "tics_manual"/ "tic" / f"{sub}_ses-01_task-tictrack_tic_epo.fif"
        tic_epochs.save(epochs_fif_path, overwrite=True)
        print(f"[4/5] Saving raw data for each epoch {epochs_fif_path}...")

        # --- 5. Save random epochs from preprocessed data ----

        print("\n[DONE] All epochs plotted.")
        print(f"\n{'='*60}")
        print(f"[4/4] Creating random epochs ...")
        print(f"{'='*60}")


        # --- Output folder ---
        # Loading manual tic annotations
        out_dir = PREPROC_DIR / sub / "tics_manual"
        out_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_dir / f"{sub}_ses-01_task-tictrack_tic_epoch_manual.xlsx"
        tics_df = pd.read_excel(tsv_path)

        phases_dict = build_phases_dict(raw)
        phase_boundaries = {
            phase_name: {"start": interval[0], "end": interval[1]}
            for phase_name, interval in phases_dict.items()
            if interval is not None
        }
        no_tic_epochs, phase_nt  = no_tic_gaps(raw, tics_df, phase_boundaries  = phase_boundaries, epoch_duration = EPOCH_EXT_PARAMS["random_epoch_duration"] , min_gap = EPOCH_EXT_PARAMS["min_gap"])

        metadata_nt = pd.DataFrame({
            "phase": phase_nt
        })
        no_tic_epochs.metadata = metadata_nt
         
        no_epochs_fif = PREPROC_DIR / sub / "tics_manual" /"no_tic" / f"{sub}_ses-01_task-tictrack_no_tic_epo.fif"
        no_tic_epochs.save(no_epochs_fif, overwrite=True)


        # ==== Alpha spectrum for no-tic epochs by ROI ==== #
        spectra_out_dir = PREPROC_DIR / sub / "spectra" / "no_tic_EO_EC_by_roi"

        plot_eo_ec_spectrum_no_tic_by_roi(
            no_tic_epochs = no_tic_epochs,
            patient       = cfg,
            patient_id    = sub,
            out_dir       = spectra_out_dir,
        )

    print("\n[DONE] All epochs saved.")





if __name__ == "__main__":
    main()
