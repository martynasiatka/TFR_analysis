### TicTrack EEG Analysis 

This repository contains the analysis pipeline for Movidoc project, that explores the neural dynamics associated with premonitory urges in Tourette syndrome patients. 
The project investigates whether EEG time-frequency activity can capture neural dynamics associated with premonitory urges. The analysis combines EEG recordings, behavioral key presses, and video-based tic annotations to extract event-locked epochs and compare oscillatory activity across experimental conditions.

### Goal of the analysis
1. Synchronization of the EEG signal, behavioral data, and video-annotated data 
2. Preprocessing of the EEG data 
3. Time-frequency analysis 
4. Statistical analysis 

## Repository structure

The project is organized as follows:

tictrack_eeg_analysis_Lizbeth/
├── dataset/
│   ├── sub-XX/
│   │   └── ses-01/
│   │       ├── excel/
│   │       └── eeg/
│   ├── sub-YY/
│   │   └── ses-01/
│   │       ├── excel/
│   │       └── eeg/
│   └── ...
│
├── config/
│   └── config.py
│
├── scripts/
│   ├── _01_extract_events_tsv.py
│   ├── _02_realign_excel_eeg.py
│   ├── _03_merge_events.py
│   ├── _04_manual_epochs.py
│   ├── _05_preprocess.py
│   ├── _06_plot_epochs.py
│   ├── _07_TFR_analysis.py
│   └── _08_stats_analysis.py
│
└── src/
    ├── align_eeg_excel.py
    ├── excel_tic_extraction.py
    ├── helper_functions.py
    ├── preproc.py
    ├── manual_tic_epochs.py
    ├── stats_analysis.py
    └── time_frequency_analysis.py

### Detailed description of each script
1. _01_extract_events_tsv
Goal: Summary of TTLs and their times 
Output: tsv with the TTLs and their times
Functions called: extract_ttl_events - to convert MNE annotations to a tidy TTL events table. Further add phase labels to each event

2. _02_realign_excel_egg
Goal: Recalibrated EXCEL files to EEG files to the green led
Output: tsv with the recalibrated TTLs and their times
Functions called: extract_ttl_events, recalibrate_from_first_event - to recalibrate from the green led stimulus

3. _03_merge_events
Goal: Merge EEG TTL events and Excel tic annotations into one tidy .tsv file with event, time, phase
Output: tsv with the TTLs and their times from both EXCEL and EEG
Functions called: extract_tics_from_excel, plot_raw, compute_key_d_to_start_delays - to plot the annotated data and calculate the difference between tic start and D key press

4. _04_manual_epochs
Goal: Create epochs of the manually annotated tic events
Output: tsv file with summary of the start of each tic, their tic type, and phase
Functions called: no_tic_gaps, create_tic_epochs  - to create epochs for tics and gaps with 'no-tic' period

5. _05_preprocess
Goal: Run full preprocessing pipeline for all patients
Output: ica_components to exclude artefacts, saving preprocessed data as fif. and plotting the raw data
Functions called: preprocess_raw, Ransac_bad_channel_detection rejection_threshold_std, apply_ICA, apply_rest_reference

6. _06_plot_epochs
Goal: Create the epochs again, however, now based on the preprocessed raw data and the manually annotated tic events. Plot the epochs for each phase. 
Output: Saving preprocessed epochs as fif. and plots of each epoch seperately 
Functions called: create_tic_epochs, no_tic_gaps, plot_eo_ec_spectrum_no_tic_by_roi 

7. _07_TFR_analysis 
Goal: This runs a time-frequency (TFR) analysis pipeline for tic-related EEG data, per subject, phase, and tic type.
Output: Time-frequency representation for each phase
Functions called: tfr_per_ROI_normalized, plot_trf_roi

8. _08_stats_analysis
Goal: This script performs statistical analysis on time-frequency representations (TFRs) of tic-related vs. baseline EEG epochs, per subject, phase, ROI, and condition.
Ouput: time-frequency representations with significant clusters, csv file with quantitative results of the statistical analysis 
Functions called: cluster_stats, plot_cluster_results, between_cluster_stats, plot_power_spectrum_per_roi, save_cluster_results_to_csv

