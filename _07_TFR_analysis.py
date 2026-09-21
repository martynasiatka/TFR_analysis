# ------------------------------------------------ #
# Script: _07_TFR_analysis.py
# Function: Time_Frequency Analysis
# Author: Martyna
# ------------------------------------------------ #

"""
This runs a time-frequency (TFR) analysis pipeline for tic-related EEG data, per subject, phase, and tic type.

For each subject it:
  1. Loads pre-computed "tic" epochs (time-locked to premonitory urge onsets report or tic start) and "no-tic" epochs
     (baseline segments), both with metadata containing "phase" and "tic_type".
  2. For each experimental phase (PHASE_EO, PHASE_EC, PHASE_FREE, PHASE_MIM, PHASE_SUP):
       - selects matching no-tic epochs to use as a baseline 
       - for each tic type, selects the corresponding tic epochs and computes a
         normalized TFR per ROI (tfr_per_ROI_normalized), comparing tic vs. baseline
       - plots and saves the averaged TFR per phase/tic-type, plus a plot for every
         single epoch individually
       - computes and saves a TFR for the no-tic/baseline epochs themselves
  3. Collects PHASE_FREE | "expressed" tic results across subjects to build:
       - a grand average TFR (mean of per-subject TFRs)
       - a grand concatenated average TFR (mean across all epochs from all subjects)
     and saves both as summary plots.
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
    PREPROC_DIR, PATIENTS, PHASES_TTL,
    TFR_PARAMS
)
from src.time_frequency_analysis import tfr_per_ROI_normalized, plot_trf_roi

TASK   = "tictrack"
SES    = "01"
RUN = "01"
PHASES = ["PHASE_EO", "PHASE_EC", "PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-frequency analysis for each subject, phase, and condition"
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
    
    grand_avg_tfrs = []
    grand_concat_tfrs = []

    for sub, cfg in subjects.items():

        # --- 1. Load preprocessed epochs data 
        print(f"\n{'='*60}")
        print(f"[1/4] Loading epochs data ...")
        print(f"{'='*60}")
        # -------------- Pre-tic epochs -----------
        epochs_fif_path = PREPROC_DIR / sub / "tics_manual"/ "tic" / f"{sub}_ses-01_task-tictrack_tic_epo.fif"
        tic_epochs = mne.read_epochs(epochs_fif_path, preload = True, verbose = "ERROR")

        if tic_epochs.metadata is None:
            print("[ERROR] Epochs must contain metadata with phase and tic_type")
        phases = tic_epochs.metadata["phase"].unique()
        tic_types = tic_epochs.metadata["tic_type"].unique()

        print(f"[INFO] Phases: {phases}")
        print(f"[INFO] Tic types: {tic_types}")

        # ------------- No-tic epochs-----------
        no_epochs_fif = PREPROC_DIR / sub / "tics_manual" /"no_tic" / f"{sub}_ses-01_task-tictrack_no_tic_epo.fif"
        no_tic_epochs = mne.read_epochs(no_epochs_fif, preload = True, verbose = "ERROR")

        print(f"[INFO] Loaded {len(tic_epochs)} tic epochs")
        print(f"[INFO] Loaded {len(no_tic_epochs)} no-tic epochs")

        # -- 2. Time-Frequency Analysis 
        print(f"\n{'='*60}")
        print(f"[2/4] Time-Frequency Analysis ...")
        print(f"{'='*60}")

        # # choose the phase for no_tic epochs 
        # baseline_phase = args.baseline_phase
        # if baseline_phase == "ALL":
        #     random_epochs = no_tic_epochs
        # else:
        #     if no_tic_epochs.metadata is None:
        #         print("[ERROR] No metadata found in no_tic_epochs")

        #     sel = no_tic_epochs.metadata["phase"] == baseline_phase
        #     random_epochs = no_tic_epochs[sel]

        #     print(f"[INFO] Baseline: {baseline_phase} ({len(random_epochs)} epochs)")

        print(f"\n{'='*60}")
        print(f"[3/4] Plotting TFR Analysis results...")
        print(f"{'='*60}")

        for phase in phases:
            sel = no_tic_epochs.metadata["phase"] == phase
            random_epochs = no_tic_epochs[sel]
            print(f"[INFO] Baseline: {phase} ({len(random_epochs)} no-tic epochs)")
            if len(random_epochs) == 0:
                print(f"[WARN] No no-tic epochs for {phase}, falling back to PHASE_FREE baseline.")
                sel_fallback = no_tic_epochs.metadata["phase"] == "PHASE_FREE"
                random_epochs = no_tic_epochs[sel_fallback]
                if len(random_epochs) == 0:
                    print(f"[ERROR] No PHASE_FREE epochs found either. Skipping {phase}.")
                    continue
            else:
                random_epochs = random_epochs

            for tic in tic_types: 
                print(f"\n--- {phase} | {tic} ---")
                # choose the tic type for each phase
                select = (
                    (tic_epochs.metadata["phase"] == phase) & 
                    (tic_epochs.metadata["tic_type"] == tic)
                )
                tic_epochs_sel = tic_epochs[select]
                if len(tic_epochs_sel) == 0:
                    print(f"  [SKIP] No epochs for {phase} | {tic}, skipping.")
                    continue
                print(f"  {phase} | {tic}: tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}, n={len(tic_epochs_sel)}")
                print(f"  pre_tic tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}")
                print(f"  random  tmin={random_epochs.tmin},  tmax={random_epochs.tmax}")

                # Compute TFR
                roi_tfr, freqs, times, n , X = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel, 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )
                
                # ====== Add average aross patients ====== #
                if phase == "PHASE_FREE" and tic == 'expressed':
                    grand_avg_tfrs.append(roi_tfr)
                    grand_avg_freqs = freqs
                    grand_avg_times = times

                    grand_concat_tfrs.append(X)


            # -- 3. Plotting the TFR results
                fig = plot_trf_roi(roi_tfr, freqs, times, n, epoch_type='pre_tic', vmin=None, vmax=None)

                # save the ouputs 
                out_dir = PREPROC_DIR / sub / "tfr" / phase
                out_dir.mkdir(parents=True, exist_ok=True)

                fname = f"{sub}_{phase}_{tic}_tfr.png"
                fig.savefig(out_dir / fname, dpi=150)
                plt.close(fig)

                #--- 4. Plot each epochs seperately 
                print(f"\n{'='*60}")
                print(f"[4/4] Plotting each epoch separately  ...")
                print(f"{'='*60}")
                for i in range(len(tic_epochs_sel)):

                    # -- Time-Frequency--
                    sin_tfr, sin_freqs, sin_times, sin_n, _  = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel[i], 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )

                    # -- Plot each epoch ---
                    fig = plot_trf_roi(sin_tfr, sin_freqs, sin_times, sin_n, epoch_type='pre_tic', vmin=None, vmax=None)

                    # --- output dir ---
                    out_dir = PREPROC_DIR / sub / "tfr" / 'single_epoch'/ phase
                    out_dir.mkdir(parents=True, exist_ok=True)

                    fname = f"{sub}_{phase}_{tic}_{i}_tfr.png"
                    fig.savefig(out_dir / fname, dpi=150)
                    plt.close(fig)


            # -- 4. Analysis for no_tic epochs ---
            roi_tfr_nt, freqs_nt, times_nt, n_nt, _  = tfr_per_ROI_normalized(
                patient = cfg, pre_tic_epochs = random_epochs, 
                random_epochs = random_epochs, epoch_type='random', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )
            fig_nt = plot_trf_roi(roi_tfr_nt, freqs_nt, times_nt, n_nt, epoch_type='random', vmin=None, vmax=None)

            # save the ouputs 
            out_dir_nt = PREPROC_DIR / sub / "tfr" / "no_tic" / phase
            out_dir_nt.mkdir(parents=True, exist_ok=True)

            fname_nt = f"{sub}_baseline_tfr.png"
            fig_nt.savefig(out_dir_nt / fname_nt, dpi=150)
            plt.close(fig_nt)

    # average across all patients 
    out_dir_grand = PREPROC_DIR / "grand_average" / "tfr"
    out_dir_grand.mkdir(parents=True, exist_ok=True)
    if grand_avg_tfrs:
        grand_avg = {}
        print(f"\n{'='*60}")
        print(f"[4/4] Plotting grand average across patients  ...")
        print(f"{'='*60}")
        print(f"[INFO] Computing grand average across {len(grand_avg_tfrs)} patients for PHASE_FREE | expressed condition.")
        for roi in grand_avg_tfrs[0].keys():
            grand_avg[roi] = np.mean([tfr[roi] for tfr in grand_avg_tfrs], axis=0)
        fig_grand_avg = plot_trf_roi(grand_avg, grand_avg_freqs, grand_avg_times, n = len(grand_avg_tfrs), epoch_type='pre_tic', vmin=None, vmax=None)

        fname_grand = f"grand_average_pre_tic_tfr.png"
        fig_grand_avg.savefig(out_dir_grand / fname_grand, dpi=150)
        plt.close(fig_grand_avg)

    # average across all patients and epochs (concat)
    if grand_concat_tfrs:
        grand_concat = {}
        total_epochs = 0
        for roi in grand_concat_tfrs[0].keys():
            all_epochs = np.concatenate(
                [X[roi] for X in grand_concat_tfrs if roi in X], axis=0
            )  # (total_epochs, n_freqs, n_times)
            grand_concat[roi] = all_epochs.mean(axis=0)  # (n_freqs, n_times)
            total_epochs = all_epochs.shape[0]
            print(f"[INFO] ROI {roi}: {all_epochs.shape[0]} total epochs across subjects")
    
        fig_concat = plot_trf_roi(
            grand_concat, grand_avg_freqs, grand_avg_times,
            n=total_epochs, epoch_type='pre_tic', vmin=None, vmax=None
        )
        fname_concat = "grand_average_PHASE_FREE_expressed_concat_tfr.png"
        fig_concat.savefig(out_dir_grand / fname_concat, dpi=150)
        plt.close(fig_concat)
        print(f"[OK] Grand average (concat, n={total_epochs} epochs) saved → {fname_concat}")
            



        print(f"[OK] Saved → {fname}")
    print(f"[OK] Saved → {fname_nt}")


    print("\n[DONE] All data saved.")

            


if __name__ == "__main__":
    main()
       
