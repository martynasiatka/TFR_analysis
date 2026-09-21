# --------------------------------------------- #
# Function: Exctract events for tic epochs
# Author: Martyna
# ---------------------------------------------- #
"""
After the inspection of the events TTLs & excel annotations the start and end of each tic/urge was manually annotated. We will use the start of the tic/urge as the center of the epoch. However, we will study the signal precedeing the tic/urge onset. 
For the ICA we will use the gaps in between the tics to make sure we do not reject muscle or eye signal present during the tic. We find artefacts in the signal that does not capture the tic signal.
To control for the posterior alpha power we will compare the power spectrum of the no tic epochs during eyes open and eyes closed phases, separately for each ROI. We expect to see stronger alpha power during eyes closed than eyes open, especially in posterior ROIs. This will be a sanity check that our no tic epochs capture expected physiological differences between EO and EC states.
"""
import numpy as np
import mne
from config.config import PHASES_TTL, EPOCH_EXT_PARAMS
from config.config import (
    EPOCH_EXT_PARAMS, PREPROC_DIR, PATIENTS, PHASES_TTL,
    CHANNELS_32, CHANNELS_64,
    ROI_LIST_32, ROI_LIST_64,
    ROI_COLORS, ANNOTATION_COLORS, ANNOTATION_COLORS_DEFAULT
)

import argparse
from pathlib import Path
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys
import os
import numpy as np


def no_tic_gaps(raw, tics_df, phase_boundaries  = None, epoch_duration = 2.0, min_gap = 2.0, urge_dur = 2.0, used_phases = None):
    """
    We only extract the in-between tics gaps from EYES_OPEN, EYES_CLOSED, and PHASE_SUP phases.That way we will be able to capture eye blink artefacts and muscle artefacts.  
    """
    if used_phases ==None:
        used_phases = ["PHASE_EC","PHASE_EO","PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]
    else:
        used_phases = used_phases

    epochs_onsets = []
    epochs_phase = []
    for phase in used_phases:
        if phase not in PHASES_TTL:
            print(f"Phase {phase} not found in phases_ttl, skipping.")
            continue

            
        print(f"  phase_boundaries keys: {list(phase_boundaries.keys())}")
        print(f"  used_phases: {used_phases}")

        phase_start = phase_boundaries[phase]["start"].astype(float)
        phase_end = phase_boundaries[phase]["end"].astype(float)

        # get tics within this phase, sorted by start time
        phase_tics = tics_df[tics_df["phase"] == phase]
        start_tic = phase_tics["start"].astype(float) - urge_dur
        end_tic = phase_tics["end"].astype(float)   
        tic_intervals = list(zip(start_tic,end_tic)) # (start - urge_dur, end)
    

        boundaries = [(phase_start, phase_start)] + tic_intervals + [(phase_end, phase_end)] 

        for i in range(len(boundaries) - 1):
            gap_start = boundaries[i][1] # end of the current tic 
            gap_end = boundaries[i+1][0] # start of the next tic
            gap_duration = gap_end - gap_start 

            # do not consider epochs shorter than 2s
            if gap_duration < min_gap:
                print(f"  Skipping gap of {gap_duration:.2f}s in {phase} (too short)")
                continue

            # split the long gaps into epoch_duration interval 
            n_epochs = int(gap_duration // epoch_duration) 
            
            if n_epochs==0:
                epochs_onsets.append(gap_start)
                epochs_phase.append(phase)
            else:
                for j in range(n_epochs):
                    epoch_onset = gap_start +j*epoch_duration
                    epochs_onsets.append(epoch_onset)
                    epochs_phase.append(phase)
                    
    # build an events array
    events = []
    for t in epochs_onsets:
        sample_idx = int(t * raw.info['sfreq'])
        events.append([sample_idx, 0, 2])
    events = np.array(events, dtype=int)

    # create epochs 
    epochs = mne.Epochs(
    raw, 
    events,
    event_id={"gap": 2},
    tmin=0, 
    tmax=epoch_duration,
    baseline=None, 
    preload=True
    )
    return epochs, epochs_phase

def create_tic_epochs(raw, tics_df, phase_boundaries  = None):
    """
    From the excel file with manually annotated tics we create epochs surrounded around the strat of the tic. 
    """
    used_phases = ["PHASE_EC", "PHASE_EO", "PHASE_FREE", "PHASE_MIM", "PHASE_SUP"]
    epochs_onsets = []
    epochs_phase = []
    epochs_type = []
    epochs_annot_type = []

    for phase in used_phases:
        if phase not in phase_boundaries:
            print(f"  [SKIP] {phase} not found in phase_boundaries.")
            continue

        phase_tics = tics_df[tics_df["phase"]== phase]
        print(f"phase={phase}, n_tics={len(phase_tics)}")  
        
        for _, column in phase_tics.iterrows():
            epochs_onsets.append(column["start"])
            epochs_phase.append(phase)
            epochs_type.append(column['tic_type'])
            epochs_annot_type.append(column["annot_type"])

        
        print(f"[OK] {len(phase_tics)} tics found in the {phase} ")
    
    # build events array 
    sfreq  = raw.info["sfreq"]
    events = np.array(
        [[int(t * sfreq), 0, 1] for t in epochs_onsets],
        dtype=int
    )

    # create epochs from the events 
    epochs = mne.Epochs(
        raw,
        events, 
        event_id={"tic":1},
        tmin = -EPOCH_EXT_PARAMS["pre_seconds"],
        tmax = EPOCH_EXT_PARAMS["post_seconds"],
        baseline = None, 
        preload = True
    )
    return epochs, epochs_phase, epochs_type,  epochs_annot_type


# ========= Alpha power spectrum for no-tic epochs, by ROI and EO vs EC =========== #

def plot_eo_ec_spectrum_no_tic_by_roi(
    no_tic_epochs: mne.Epochs,
    patient,
    patient_id: str,
    out_dir: Path,
    fmin: float = 2.0,
    fmax: float = 30.0,
    alpha_band: tuple[float, float] = (8.0, 12.0),
):
    """
    Plot EO vs EC power spectrum for no-tic epochs, separately for each ROI.

    This is mainly useful as a sanity check:
    posterior alpha power should be stronger during eyes closed than eyes open.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if no_tic_epochs.metadata is None:
        print("[SKIP] no_tic_epochs has no metadata.")
        return

    if "phase" not in no_tic_epochs.metadata.columns:
        print("[SKIP] no_tic_epochs metadata does not contain a 'phase' column.")
        return

    # Select ROI definition depending on montage
    if patient.montage == "standard_1020":
        roi_lists = ROI_LIST_32
    elif patient.montage == "standard_1005":
        roi_lists = ROI_LIST_64
    else:
        raise ValueError(f"Unknown montage: {patient.montage}")

    # Select EO and EC no-tic epochs
    eo_epochs_all = no_tic_epochs[
        no_tic_epochs.metadata["phase"].values == "PHASE_EO"
    ]

    ec_epochs_all = no_tic_epochs[
        no_tic_epochs.metadata["phase"].values == "PHASE_EC"
    ]

    if len(eo_epochs_all) == 0:
        print(f"[SKIP] {patient_id}: no PHASE_EO no-tic epochs.")
        return

    if len(ec_epochs_all) == 0:
        print(f"[SKIP] {patient_id}: no PHASE_EC no-tic epochs.")
        return

    print(f"\n[INFO] {patient_id}")
    print(f"[INFO] EO no-tic epochs: {len(eo_epochs_all)}")
    print(f"[INFO] EC no-tic epochs: {len(ec_epochs_all)}")

    results = []

    for roi_name, roi_channels in roi_lists.items():

        available_channels = [
            ch for ch in roi_channels
            if ch in no_tic_epochs.ch_names
        ]

        if len(available_channels) == 0:
            print(f"[SKIP] {patient_id} | {roi_name}: no available channels.")
            continue

        print(f"\n[INFO] ROI: {roi_name}")
        print(f"[INFO] Channels: {available_channels}")

        eo_epochs = eo_epochs_all.copy().pick(available_channels)
        ec_epochs = ec_epochs_all.copy().pick(available_channels)

        # Compute PSD
        psd_eo = eo_epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=available_channels,
            verbose=False,
        )

        psd_ec = ec_epochs.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=available_channels,
            verbose=False,
        )

        freqs = psd_eo.freqs

        # Shape: n_epochs x n_channels x n_freqs
        power_eo = psd_eo.get_data()
        power_ec = psd_ec.get_data()

        # Average across epochs and channels
        mean_eo = power_eo.mean(axis=(0, 1))
        mean_ec = power_ec.mean(axis=(0, 1))

        # Log-transform for easier visualization
        mean_eo_log = np.log10(mean_eo) #np.log10
        mean_ec_log = np.log10(mean_ec) #np.log10

        # Alpha summary
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

        alpha_eo = mean_eo_log[alpha_mask].mean()
        alpha_ec = mean_ec_log[alpha_mask].mean()
        alpha_diff = alpha_ec - alpha_eo

        print(f"[RESULT] Alpha EO: {alpha_eo:.4f}")
        print(f"[RESULT] Alpha EC: {alpha_ec:.4f}")
        print(f"[RESULT] Alpha EC - EO: {alpha_diff:.4f}")

        results.append({
            "subject": patient_id,
            "roi": roi_name,
            "channels": ", ".join(available_channels),
            "n_eo_epochs": len(eo_epochs),
            "n_ec_epochs": len(ec_epochs),
            "alpha_eo_log10": alpha_eo,
            "alpha_ec_log10": alpha_ec,
            "alpha_ec_minus_eo_log10": alpha_diff,
        })

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            freqs,
            mean_eo_log,
            label=f"Eyes open, n={len(eo_epochs)}"
        )

        ax.plot(
            freqs,
            mean_ec_log,
            label=f"Eyes closed, n={len(ec_epochs)}"
        )

        ax.axvspan(
            alpha_band[0],
            alpha_band[1],
            alpha=0.2,
            label="Alpha band 8-12 Hz"
        )

        ax.set_title(f"{patient_id} | {roi_name} | no-tic EO vs EC spectrum")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Log10 power")
        ax.legend()
        ax.grid(True, alpha=0.3)

        safe_roi_name = roi_name.replace(" ", "_").replace("/", "-")

        fname = (
            f"{patient_id}_ses_task"
            f"_no_tic_EO_vs_EC_{safe_roi_name}_spectrum.png"
        )

        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] Saved: {fname}")

    # Save CSV summary
    if results:
        results_df = pd.DataFrame(results)

        csv_name = (
            f"{patient_id}_ses_task"
            f"_no_tic_EO_vs_EC_alpha_by_roi.csv"
        )

        results_df.to_csv(out_dir / csv_name, index=False)
        print(f"\n[OK] Saved summary CSV: {csv_name}")






