
# ---------------------------------------------------- 
# Function: Pre-processing of EEG data
# Author: Martyna  
# Goal: Functions for the full pre-processing pipeline, each result is saved in the plots directory for visual inspection.
# ----------------------------------------------------
import matplotlib
matplotlib.use('Agg')  # non-interactive, saves to file without displaying
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pandas as pd
import mne
import sys
import os
import numpy as np
from autoreject import Ransac  
from autoreject.utils import interpolate_bads  
from mne.preprocessing import ICA
import autoreject
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt
from config.config import PATIENTS,  PREPROC_DIR, ICA_EXCLUSIONS, PREPROC_PARAMS


def preprocess_raw(raw: mne.io.BaseRaw, subject_name:str, montage_name:str, plots_dir:Path ):
    '''
    Filtering of raw data
    - Plot raw data for inspection
    - Setting montage
    - Band-pass filter (0.1-45 Hz)
    - Notch filter (50 Hz)
    Special crop for BB28 due to last-second artifact.
    '''

    raw.set_montage(montage_name)

    print(f"Total channels: {len(raw.ch_names)}")
    print("Channel names:", raw.ch_names)

    if subject_name == 'sub-BB28':
        """
        Based on the visual inspection of the data the last 10s of the recording of patient BB28 was incorrect. Probable reason is EEG stopped woking. That is why, we crop the last second (meanigless for the analysis).
        """
        raw.crop(tmax=1646)
    
    fig = raw.plot(n_channels=64, title=f"Raw data - {subject_name}", show=True)
    fig.savefig(plots_dir / f"{subject_name}_raw.png")
    plt.close(fig)

    raw = raw.filter(l_freq=PREPROC_PARAMS["low_freq"] , h_freq=PREPROC_PARAMS["high_freq"])
    raw = raw.notch_filter(freqs= PREPROC_PARAMS["notch_filt"], picks="data", method="spectrum_fit")
    return raw


def Ransac_bad_channel_detection(raw, subject_name, plots_dir: Path):
    '''
    Iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers do not affect the values of the estimates. 
    - First average reference is applied to the raw data, as RANSAC requires it.
        - Then, fixed-length epochs are created from the raw data to be used as input for RANSAC.
        - Finally, RANSAC is applied to identify bad channels, which are added to the raw.info['bads'] list.
        '''
    eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True)
    raw_avg = raw.copy().set_eeg_reference('average') 

    # epochs needed to fit the Ransac method - only used for this purpose
    temporary_epochs = mne.make_fixed_length_epochs(raw_avg, duration=3.0, overlap=0.0, preload=True)

    # Plot butterfly to visually inspect channels
    fig = temporary_epochs.average().plot(spatial_colors=True, titles=f'Butterfly plot - {subject_name}')
    fig.savefig(plots_dir / f"{subject_name}_butterfly_before_RANSAC.png")
    plt.close(fig)

    ransac = Ransac(n_jobs=1, verbose = True)
    ransac.fit_transform(temporary_epochs)
    raw.info['bads'].extend(ransac.bad_chs_)
    if subject_name == 'sub-MM30':
        """
        Based on the visual inspection 1 bad channels were identified for the patient MM30.
        """
        raw.info['bads'].append('AFz')

    if subject_name == 'sub-CM33':
        """
        Based on the visual inspection 1 bad channels were identified for the patient CM33.
        """
        #raw.info['bads'].append('TP10')
        raw.info['bads'].append('PO7')
        raw.info['bads'].append('TP9')
        raw.info['bads'].append('FT9')
        raw.info['bads'].append('O1')
    
    if subject_name == 'sub-NA32':
        # raw.info['bads'].append('Pz')
        # raw.info['bads'].append('PO7')
        # raw.info['bads'].append('FC5')
        raw.info['bads'].append('AF4') # important
        # raw.info['bads'].append('TP9')
        # raw.info['bads'].append('FT9')
        # raw.info['bads'].append('F5')
        # raw.info['bads'].append('AFz')
        raw.info['bads'].append('Fp1') #important exclusion based on the butterfly plot
        raw.info['bads'].append('P7') # important
        raw.info['bads'].append('F5') # important 
        raw.info['bads'].append('FT8') #important
        #raw.info['bads'].append('F6') 
        raw.info['bads'].append('FCz')
    
    bad_channels = raw.info['bads']
    good_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    data = raw.get_data(picks=good_picks)
    ptp = np.ptp(data, axis=1)
    max_idx = np.argmax(ptp)
    print(f"Channel with highest amplitude: {raw.ch_names[good_picks[max_idx]]} ({ptp[max_idx] * 1e6:.2f} µV)")
    print("Ransac detected bad channels:", ransac.bad_chs_)
    print("All bad channels detected:", bad_channels)

    return raw, bad_channels


def rejection_threshold_std(raw, subject_name, plots_dir: Path, threshold=PREPROC_PARAMS["threshold"],epochs=None):
    """
    Computes rejection threshold for bad epochs based on the FASTER algorithm.
    - The raw data is first re-referenced to the average reference.
    - Then, fixed-length epochs are created from the raw data to be used as input for the FASTER algorithm.
    """
    from mne.preprocessing.bads import _find_outliers

    raw_avg = raw.copy().set_eeg_reference('average')

    if epochs==None:
        events = mne.make_fixed_length_events(raw_avg, duration=2)
        epochs_temp = mne.Epochs(raw_avg, events, tmin=0.0, tmax=2, baseline=None, preload=True)
    else:
        epochs_temp = epochs

    def _deviation(data):
        """Computes the deviation from mean for each channel."""
        channels_mean = np.mean(data, axis=2)
        return channels_mean - np.mean(channels_mean, axis=0)

    metrics = {
        "amplitude": lambda x: np.mean(np.ptp(x, axis=2), axis=1),
        "deviation": lambda x: np.mean(_deviation(x), axis=1),
        "variance":  lambda x: np.mean(np.var(x, axis=2), axis=1),
    }

    epochs_data = epochs_temp.get_data()
    bad_epochs = []

    print(f"\nFASTER epoch rejection for {subject_name}:")
    for metric_name, metric_func in metrics.items():
        scores = metric_func(epochs_data)
        outliers = _find_outliers(scores, threshold=threshold)
        print(f"  Bad epochs by {metric_name}: {outliers}")
        for idx in outliers:
            print(f"Epoch {idx} (t={idx*2:.1f}s): score={scores[idx]:.4f}")
        bad_epochs.extend(outliers)

    bad_epochs = list(set(bad_epochs))
    print(f"\n  Total bad epochs (union): {len(bad_epochs)} → {bad_epochs}")

    epochs_temp.drop(bad_epochs, reason='FASTER')
    print(f"  Remaining epochs: {len(epochs_temp)}")

    fig = epochs_temp.average().plot(titles=f'Butterfly plot after FASTER rejection - {subject_name}')
    fig.savefig(plots_dir / f"{subject_name}_butterfly_after_FASTER.png")
    plt.close(fig)
    fig = epochs_temp.plot_drop_log()
    fig.savefig(plots_dir / f"{subject_name}_FASTER_drop_log.png")
    plt.close(fig)

    return epochs_temp


def apply_ICA(epochs_ica, raw, subject_name:str, ica_exclusions: dict, plots_dir: Path):
    """
    Applies Independent Component Analysis (ICA) to the EEG data to identify and remove artifacts.
    Uses manual exclusions defined in ICA_EXCLUSIONS.
    - Average reference is applied to the raw data before ICA fitting, as recommended for ICA (already in passed cleaned epochs)
    - Band-pass filter is applied to the raw data before ICA fitting. Removes slow drifts. 
    - Fitted on averaged and filtered data by applied on raw data 
    """
    # filtering needed to remove slow drifts 
    filt_raw = epochs_ica.copy().filter(l_freq=1.0, h_freq = None)
    picks_eeg = mne.pick_types(filt_raw.info, meg=False, eeg=True, eog=False,
                                exclude='bads')
    data = filt_raw.get_data()

    ica = mne.preprocessing.ICA(
    n_components=PREPROC_PARAMS["n_components"],  method="picard", max_iter="auto", random_state=97)
    ica.fit(filt_raw, picks = picks_eeg) # reject=reject epochs that exceed the threshold are not considered in ICA fitting 
 
    fig = ica.plot_components(show=False) # topomaps showing each ICA component and its spatial distribution
    fig.savefig(plots_dir / f"{subject_name}_ica_components.png")
    plt.close(fig)

    ica.exclude = []

    ica.exclude = ica_exclusions.get(subject_name, [])
    print(f"[{subject_name}] Excluding ICA components: {ica.exclude}")

    figs = ica.plot_properties(epochs_ica[:100], picks=range(20), psd_args={'fmax': 45}) # plot the properties of the ICA components to help identify which ones to exclude (topomap, time series, power spectrum)
    for i, f in enumerate(figs):
        f.savefig(plots_dir / f"{subject_name}_ica_property_{i}.png")
        plt.close(f)
        
    fig = ica.plot_sources(epochs_ica, picks = range(20) , show=True) # plot the time series of the ICA components 
    fig.savefig(plots_dir / f"{subject_name}_ica_sources.png")
    plt.close(fig)

    fig = ica.plot_overlay(raw, exclude=ica.exclude) # plot the raw signal before and after ICA correction to visualize the effect of artifact removal (overlaid)
    fig.savefig(plots_dir / f"{subject_name}_ica_overlay.png")
    plt.close(fig)

    print("N components fitted:", ica.n_components_)
    print("Explained variance:", ica.pca_explained_variance_)

    raw = ica.apply(raw.copy())

    return raw

"""
def local_Autoreject(epochs, subject_name, plots_dir: Path):
    """
    Applies the Autoreject algorithm to the epochs to identify and correct bad epochs and channels.
    """
    n_epochs = len(epochs)
    
    if n_epochs < 5:
        print(f"WARNING: Too few epochs ({n_epochs}) for autoreject, skipping")
        return epochs
    
    n_splits = min(10, n_epochs)
    ar = autoreject.AutoReject(n_interpolate=PREPROC_PARAMS["n_interpolate"], random_state=11,
                           n_jobs=1, verbose=True, cv=n_splits)
    ar.fit(epochs)  
    epochs_clean, reject_log = ar.transform(epochs, return_log=True)

    fig = reject_log.plot('horizontal')
    fig.savefig(plots_dir / f"{subject_name}_autoreject_reject_log.png")
    plt.close(fig)

    fig = epochs[reject_log.bad_epochs].plot( title=f"Rejected epochs - {subject_name}")
    fig.savefig(plots_dir / f"{subject_name}_rejected_epochs.png")
    plt.close(fig)

    fig = epochs_clean.plot(show=True)
    fig.savefig(plots_dir / f"{subject_name}_clean_epochs.png")
    plt.close(fig)

    return epochs_clean
"""

def apply_rest_reference(raw, subject_name, plots_dir: Path):
    """
    Applies the REST (Reference Electrode Standardization Technique) reference to the raw EEG data.
    - First, an anatomical sphere model is built for the REST computation.
    - Then, a volume source space is created inside this sphere.
    - The forward model is computed for referencing.
    """  
    Sphere = mne.make_sphere_model('auto', 'auto', raw.info)
    Source = mne.setup_volume_source_space(sphere=Sphere, exclude=30.0, pos=5.0, verbose=False)
    Forward = mne.make_forward_solution(raw.info, trans=None, src=Source, bem=Sphere, verbose=False)

    # raw = raw.copy().set_eeg_reference('REST', forward=Forward)
    fig = raw.plot(n_channels=64, title=f"Raw data with REST reference - {subject_name}", show=True)
    fig.savefig(plots_dir / f"{subject_name}_raw_rest_reference.png")
    plt.close(fig)

    return raw