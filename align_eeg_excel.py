# ------------------------------------------------------------
# Function: Align excel with eeg 
# Author: Martyna (structure) & Indira (functions)
# Goal: realign the times from the Excel file with the times from the EEG recording (TTLs)
# ------------------------------------------------------------


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
import mne


def recalibrate_from_first_event(raw, target_stim="Stimulus/S  2"):
    """
    Stimulus/S 2 is the first TTL event that marks the start of the tic track task (green led)
    Excel file starts from the green led manually annotated based on the video recordings (green led = 0ms). 
    EEG recording starts before the green led, so we need to realign the EEG signal to the first occurrence of the target stimulus (green led) and adjust the annotations accordingly.
    """
    # Verify if the annotations exist
    if raw.annotations is None or len(raw.annotations) == 0:
        print("No annotations found — cannot realign to target stimulus.")
        return raw
    
    # search the 1st occurrence of the target stimulus
    target_onsets = [onset for onset, desc in zip(raw.annotations.onset, raw.annotations.description) if desc == target_stim]

    # check if the list of event times is empty, if no events found → cannot crop
    if not target_onsets:
        print(f"Target stimulus '{target_stim}' not found — cannot realign.")
        return raw
    first_stimulus_time = target_onsets[0]
    print(f"Cropping EEG at stimulus '{target_stim}' = {first_stimulus_time:.3f} s")

    anns = raw.annotations.copy()

    raw_cropped = raw.copy().crop(tmin=first_stimulus_time, reset_first_samp=True)

    new_annotations = mne.Annotations(
        onset=anns.onset - first_stimulus_time,
        duration=anns.duration,
        description=anns.description,
        orig_time=None
    )

    raw_cropped.set_annotations(new_annotations)


    # First adjust annotations on the ORIGINAL raw
    """
    - Before: cropping and the adjustement of annotations [ERROR]
    - After: adjustement of annotations on the original raw, then crop [CORRECT]
    """

    # # Find the exact sample index of the green LED — avoids sub-sample float errors
    # first_sample_idx = raw.time_as_index(first_stimulus_time)[0]
    # exact_crop_time  = raw.times[first_sample_idx]
    # print(f"Exact crop time (sample-aligned): {exact_crop_time:.6f} s")


    # new_onsets = raw.annotations.onset - exact_crop_time
    # new_durations = raw.annotations.duration
    # new_descriptions = list(raw.annotations.description)

    # raw.set_annotations(
    #     mne.Annotations(onset=new_onsets, duration=new_durations, description=new_descriptions)
    # )
    # # Then crop
    # print(f"[DEBUG] First 0 annotation time {new_onsets[0]}")
    # raw_cropped = raw.copy().crop(tmin=0.0)

    # Now check MNE's actual crop start
    print("Cropped first time:", raw_cropped.times[0])

    print(f"[DEBUG] First 3 onsets: {raw_cropped.annotations.onset[:3]}")

    # Should show: [0.0, 9.886, ...]
    return raw_cropped


