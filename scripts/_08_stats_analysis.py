# ---------------------------------------------- #
# Script: _08_stats_analysis.py
# Function: Statistical analysis of TFR results
# Author: Martyna Siatka
# -------------------------------------------- #
""""
This script performs statistical analysis on time-frequency representations (TFRs)
of tic-related vs. baseline EEG epochs, per subject, phase, ROI, and condition.
For each subject it:
1. Loads preprocessed epochs for tic and no-tic intervals.
2. Computes TFRs for tic epochs normalized by no-tic epochs, per ROI.
3. Performs permutation cluster tests against zero for each phase/condition.
4. Performs between-condition cluster tests (e.g. mimicked vs expressed).
5. Performs power spectrum analysis for predefined frequency bands with multiple comparison correction.
6. Plots significant clusters on TFRs and saves results.

Finally, it aggregates data across subjects for grand average stats and plots.
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
    TFR_PARAMS, STAT_PARAMS,
)
from src.time_frequency_analysis import tfr_per_ROI_normalized
from src.stats_analysis import cluster_stats, plot_cluster_results, between_cluster_stats, plot_power_spectrum_per_roi, save_cluster_results_to_csv

from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon
from mne.stats import fdr_correction

TASK   = "tictrack"
SES    = "01"
RUN = "01"
PHASES = ["PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis of each phase and between different conditions"
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

    grand_X = {}
    grand_avg_tfrs = []
    spectra_per_roi = {}   
    spectra_sub_ids = []

    for sub, cfg in subjects.items():
        tfr_data = {}
        grand_avg_tfrs = []

        # --- 1. Load preprocessed epochs data 
        print(f"\n{'='*60}")
        print(f"[1/5] Loading epochs data for {sub} ...")
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
        print(f"[2/5] Time-Frequency Analysis ...")
        print(f"{'='*60}")

        for phase in phases:
            tfr_data[phase] = {}
            sel = no_tic_epochs.metadata["phase"] == phase
            random_epochs = no_tic_epochs[sel]
            print(f"[INFO] Baseline: {phase} ({len(random_epochs)} no-tic epochs)")
            if len(random_epochs) < 5:
                print(f"[WARN] Only {len(random_epochs)} no-tic epochs for {phase}, "
                    f"falling back to PHASE_FREE baseline.")
                sel = no_tic_epochs.metadata["phase"] == "PHASE_FREE"
                random_epochs = no_tic_epochs[sel]
                print(f"[INFO] Fallback baseline: PHASE_FREE ({len(random_epochs)} no-tic epochs)")

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
                # print(f"  {phase} | {tic}: tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}, n={len(tic_epochs_sel)}")
                # print(f"  pre_tic tmin={tic_epochs_sel.tmin}, tmax={tic_epochs_sel.tmax}")
                # print(f"  random  tmin={random_epochs.tmin},  tmax={random_epochs.tmax}")

                # Compute TFR
                roi_tfr, freqs, times, n , X = tfr_per_ROI_normalized(
                    patient = cfg, pre_tic_epochs = tic_epochs_sel, 
                    random_epochs = random_epochs, epoch_type='pre_tic', freqs=TFR_PARAMS["freqs"], normalization =TFR_PARAMS["normalization"] )
                """
                roi_tfr : dictionary of each ROI with (freqs, times) 
                X : dictionary of each ROI with (epochs, freqs, times)
                """
                # === Add the stats of the grand average ==== #
                print(f"\n{'='*60}")
                print(f"[3/5] Calculating stats for the grand average  ...")
                print(f"{'='*60}")
                SPECTRUM_TMIN = -1.5
                SPECTRUM_TMAX = 0.0

                time_mask = (times >= SPECTRUM_TMIN) & (times <= SPECTRUM_TMAX)
                if phase == "PHASE_FREE" and tic == "expressed":
                    for roi, X_roi in X.items():
                        if roi not in grand_X:
                            grand_X[roi] = []
                        grand_X[roi].append(X_roi)
                        spectrum = X_roi[:, :, time_mask].mean(axis=0).mean(axis=-1)
                        spectra_per_roi.setdefault(roi, []).append(spectrum)
                    if sub not in spectra_sub_ids:
                        spectra_sub_ids.append(sub)
                    grand_avg_tfrs.append(roi_tfr)
                    grand_freqs = freqs
                    grand_times = times


                # -------- Significance of the power --------- #
                print(f"\n{'='*60}")
                print(f"[4/5] Calculating stats for the power spectrum for each frequency band  ...")

                sig_bands = {}  # {roi: {sub_id: {band_name: bool}}}

                BANDS = {
                    "delta": (1, 4),
                    "theta": (4, 8),
                    "alpha": (8, 13),
                    "beta":  (13, 30),
                    "gamma": (30, 40),
                }
                # not sure about the correction 
                #alpha_corrected = 0.05 / 5 

                for roi, spectra_list in spectra_per_roi.items():
                    sig_bands[roi] = {}
                    for j, sub_id in enumerate(spectra_sub_ids):
                        sig_bands[roi][sub_id] = {}
                        # Get all epochs for this subject/roi: (n_epochs, n_freqs)
                        X_sub = grand_X[roi][j]  # (n_epochs, n_freqs, n_times)

                        t_stats = []
                        p_vals  = []
                        for band_name, (fmin, fmax) in BANDS.items():
                            freq_mask = (grand_freqs >= fmin) & (grand_freqs < fmax)
                            # Mean power per epoch in this band → (n_epochs,)
                            time_mask = (grand_times >= SPECTRUM_TMIN) & (grand_times <= SPECTRUM_TMAX)
                            band_power = X_sub[:, freq_mask, :][:, :, time_mask].mean(axis=(1, 2))

                            try:
                                stat, p_val = wilcoxon(band_power)
                                t_stat = np.mean(band_power) 
                            except ValueError:
                                stat, p_val, t_stat = 0.0, 1.0, 0.0
                            t_stats.append(t_stat)
                            p_vals.append(p_val)

                        # ----- multiple comparison correction ----- #
                        reject, pvals_corrected = fdr_correction(np.array(p_vals), alpha=0.05, method='indep')
                                
                            #t_stat, p_val = ttest_1samp(band_power, popmean=0) # one-sample t-test against zero
                        band_names = list(BANDS.keys())
                        for k, band_name in enumerate(band_names):  
                            sig_bands[roi][sub_id][band_name] = {
                                "significant": reject[k],
                                "t_stat": t_stats[k],
                                "p_val": pvals_corrected[k]
                            }
                            if reject[k]:
                                direction = "↑" if t_stats[k] > 0 else "↓"
                                print(f"  [{roi}] {sub_id} | {band_name}: p={pvals_corrected[k]:.4f} {direction}")

                            # if p_val < alpha_corrected:
                            #     direction = "↑" if t_stat > 0 else "↓"
                            #     print(f"  [{roi}] {sub_id} | {band_name}: p={p_val:.4f} {direction}")


                # ====== PLOTTING ERP ======== #
                #roi_erp = {roi_name: X_roi.mean(axis=1) for roi_name, X_roi in X.items()}

                tfr_data[phase][tic] = {
                    "roi_tfr": roi_tfr,
                    "X": X,
                    "freqs": freqs,
                    "times": times,
                    "n": n,
                    #"roi_erp": roi_erp,
                }
                
                # ---3. Permutation Cluster 1 Sample Test
                
                print(f"\n{'='*60}")
                print(f"[3/5] Permutation Cluster 1 Sample Test and Plotting for {phase} & {tic}..")
                print(f"{'='*60}")

                cluster_results = cluster_stats(X, n_permutations=STAT_PARAMS['n_permutations'], tail= STAT_PARAMS['tail'], threshold = STAT_PARAMS['threshold'], correction =STAT_PARAMS['correction'])

                print(f"[INFO] Found {len(cluster_results)} significant clusters for {phase} | {tic}")


                fig = plot_cluster_results(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub)


                # ----------- Output folder ---------
                out_dir = PREPROC_DIR / sub / "stats" / 'cluster_1sample'/ phase
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{sub}_{phase}_{tic}tfr.png"
                fig.savefig(out_dir / fname, dpi=150)
                plt.close(fig)

                save_cluster_results_to_csv(
                    cluster_results=cluster_results,
                    freqs=freqs, times=times,
                    out_dir=out_dir,
                    label=f"{sub}_{phase}_{tic}",
                )
                print(f"[OK] Cluster results saved to CSV for {phase} | {tic}")

            print(f"[OK] Saved → {fname}")
            
            # --- 4. Permutation cluster test ---
            '''
            print(f"\n{'='*60}")
            print(f"[4/5] Permutation Cluster Test and Plotting for {phase}...")
            print(f"{'='*60}")
            if phase == "PHASE_MIM":
                cond1, cond2 = "mimicked", "expressed"
            elif phase == "PHASE_SUP":
                cond1, cond2 = "suppressed", "expressed"
            else:
                continue

            if cond1 not in tfr_data[phase] or cond2 not in tfr_data[phase]:
                continue

            X1 = tfr_data[phase][cond1]["X"]
            X2 = tfr_data[phase][cond2]["X"]
            comparison =  f"{cond1} vs {cond2}"
            print(f"[INFO] Comparing {cond1} vs {cond2}")


            between_cluster_results = between_cluster_stats(X1, X2, n_permutations=STAT_PARAMS['n_permutations'], tail= STAT_PARAMS['tail'], threshold = STAT_PARAMS['threshold'], correction =STAT_PARAMS['correction'])
            
            
            fig2 = plot_cluster_results(roi_tfr, between_cluster_results, freqs, times, n, phase, comparison, sub)

            # ----- output folder -----
            out_dir2 = PREPROC_DIR / sub / "stats" / 'between_cluster'/ phase
            out_dir2.mkdir(parents=True, exist_ok=True)
            fname2 = f"{sub}_{phase}_{comparison}tfr.png"
            fig2.savefig(out_dir2 / fname2, dpi=150)
            plt.close(fig2)
            '''

    
    if grand_X:
        print(f"\n{'='*60}")
        print(f"[5/5] Grand average stats: PHASE_FREE | expressed ...")
        print(f"{'='*60}")

        # Concatenate all epochs across subjects per ROI
        X_grand = {
            roi: np.concatenate(epochs_list, axis=0)
            for roi, epochs_list in grand_X.items()
        }
        for roi, arr in X_grand.items():
            print(f"[INFO] ROI {roi}: {arr.shape[0]} total epochs")

        # Grand average TFR for plotting (mean across all epochs)
        roi_tfr_grand = {roi: arr.mean(axis=0) for roi, arr in X_grand.items()}
        #roi_erp_grand = {roi: arr.mean(axis=1) for roi, arr in X_grand.items()}
        n_total = next(iter(X_grand.values())).shape[0]

        # 1-sample cluster permutation test against zero
        cluster_results_grand = cluster_stats(
            X_grand,
            n_permutations=STAT_PARAMS['n_permutations'],
            tail=STAT_PARAMS['tail'],
            threshold=STAT_PARAMS['threshold'],
            correction=STAT_PARAMS['correction']
        )
        print(f"[INFO] Found {len(cluster_results_grand)} significant clusters in grand average")


        fig_grand = plot_cluster_results(
            roi_tfr_grand, cluster_results_grand,
            grand_freqs, grand_times,
            n=n_total,
            phase="PHASE_FREE", tic="expressed", sub="grand_average",
            #roi_erp=roi_erp_grand
        )

        out_dir_grand = PREPROC_DIR / "grand_average" / "stats"
        out_dir_grand.mkdir(parents=True, exist_ok=True)
        fname_grand = "grand_average_PHASE_FREE_concat.png"
        fig_grand.savefig(out_dir_grand / fname_grand, dpi=150)
        plt.close(fig_grand)
        print(f"[OK] Grand average stats saved → {out_dir_grand / fname_grand}")

        save_cluster_results_to_csv(
            cluster_results=cluster_results_grand,
            freqs=grand_freqs, times=grand_times,
            out_dir=out_dir_grand,
            label="grand_average_PHASE_FREE_expressed",
            csv_name="grand_average_clusters.csv",   # single combined file
        )


    # ──  average of per-subject averages  ──
    if grand_avg_tfrs:
        print(f"\n[5b/5] Grand average stats (subject-level): PHASE_FREE | expressed ...")

        # (n_subjects, n_freqs, n_times) per ROI
        X_subject_level = {
            roi: np.stack([tfr[roi] for tfr in grand_avg_tfrs if roi in tfr], axis=0)
            for roi in grand_avg_tfrs[0].keys()
        }
        for roi, arr in X_subject_level.items():
            print(f"[INFO] ROI {roi}: {arr.shape[0]} subjects")

        n_subjects = len(grand_avg_tfrs)
        roi_tfr_avg  = {roi: arr.mean(axis=0) for roi, arr in X_subject_level.items()}
        #roi_erp_avg  = {roi: arr.mean(axis=1) for roi, arr in X_subject_level.items()}

        cluster_results_avg = cluster_stats(
            X_subject_level,
            n_permutations=STAT_PARAMS['n_permutations'],
            tail=STAT_PARAMS['tail'],
            threshold=STAT_PARAMS['threshold'],
            correction=STAT_PARAMS['correction']
        )

        fig_avg = plot_cluster_results(
            roi_tfr_avg, cluster_results_avg,
            grand_freqs, grand_times,
            n=n_subjects,
            phase="PHASE_FREE", tic="expressed", sub="grand_average_subjects",
            #roi_erp=roi_erp_avg
        )

        out_dir_grand = PREPROC_DIR / "grand_average" / "stats"
        out_dir_grand.mkdir(parents=True, exist_ok=True)
        fname_avg = "grand_average_PHASE_FREE_expressed.png"
        fig_avg.savefig(out_dir_grand / fname_avg, dpi=150)
        plt.close(fig_avg)
        print(f"[OK] Subject-level grand average stats saved → {out_dir_grand / fname_avg}")
        
    if spectra_per_roi:

        # Stack into (n_subjects, n_freqs) per ROI
        spectra_stacked = {
            roi: np.stack(spectra_list, axis=0)
            for roi, spectra_list in spectra_per_roi.items()
        }

        fig_spec = plot_power_spectrum_per_roi(
                spectra_stacked, grand_freqs, spectra_sub_ids,
                sig_bands=sig_bands
            )

        out_dir_grand = PREPROC_DIR / "grand_average" / "stats"
        out_dir_grand.mkdir(parents=True, exist_ok=True)
        fname_spec = "power_spectrum_per_roi_PHASE_FREE_expressed.png"
        fig_spec.savefig(out_dir_grand / fname_spec, dpi=150)
        plt.close(fig_spec)
        print(f"[OK] Power spectrum plot saved → {out_dir_grand / fname_spec}")

if __name__ == "__main__":
    main()
       
