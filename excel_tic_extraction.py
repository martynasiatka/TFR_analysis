# ------------------------------------------------------------- #
# Function: Pre-tic and random epochs extraction 
# Author : Martyna
# Goal: Exctract pre-tic and random epochs from the EXCEL data 
#------------------------------------------------------------- #
"""
EXCEL file, which is based on maual annotations of tics, contains columns of "Absence" (0 - tic present, 1 - absence of tic) and columns for each defined body part (1 - tic present on this body part, 0 - no tic on this body part). 
Tic is defined by 3 conditions:
1. At least 30 frames of "Absence = 1" before the tic (no tic present for at least 1 second before the tic).
2. At least 1 frame of "Absence = 0" and at least 1 body part column with value 1 (tic present) at the tic moment.
3. At least 30 frames of "Absence = 1" after the tic (no tic present for at least 1 second after the tic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def extract_tics_from_excel(excel_file: Path, fps: int, min_absence_frames: int = 30,) -> list[tuple[float, float]]:
    """
    check if 30 frames before and after
    """
    df = pd.read_excel(excel_file)

    # --- 2. Select time column based on fps ---
    if fps == 30:
        time_col = "Time (30 fps) s"
        min_absence_frames = 30
    elif fps == 25:
        time_col = "Time (25 fps) s"
        min_absence_frames = 25
    else:
        raise ValueError(f"fps must be 30 or 25, got {fps}")

    if time_col not in df.columns:
        raise ValueError(f"[ERROR] Missing required column: {time_col}")
    if "Absence" not in df.columns:
        raise ValueError("[ERROR] Missing required column: 'Absence'")

    times   = df[time_col].values
    absence = df["Absence"].values

    # --- 3. Sum movement across all body part columns ---
    movement_cols = [
        "Epaules", "Main - Bras", "Tête", "Yeux", "Visage",
        "Bassin - Tronc", "Jambes - Pieds", "Phonique", "Vocaux"
    ]
    movement_sum = df[movement_cols].values.sum(axis=1)

    # --- 4. Validate annotations: Absence=1 cannot have movement ---
    invalid_mask = (absence == 1) & (movement_sum > 0)
    if np.any(invalid_mask):
        bad_indices = np.where(invalid_mask)[0]
        raise ValueError(
            f"[ERROR] Annotation error: Absence=1 but movement>0 at rows: {bad_indices.tolist()}"
        )

    # --- 5. Main tic detection loop ---
    tics = []
    n    = len(df)
    i    = 0

    while i < n:

        # --- Condition 1: min_absence_frames of pure absence BEFORE ---
        if len(tics) == 0:
            # first tic — no need for absence before
            before_ok = True
        else:
            before_ok = (
                i >= min_absence_frames and
                np.all(absence[i - min_absence_frames:i] == 1) and
                np.all(movement_sum[i - min_absence_frames:i] == 0)
            )

        if before_ok and absence[i] == 0 and movement_sum[i] > 0:

            start_time = times[i]
            j          = i
            after_len  = 0

            # --- Search for tic end ---
            while after_len < min_absence_frames:

                # move forward while movement is active
                while j < n and (absence[j] == 0 and movement_sum[j] > 0):
                    j += 1

                end_time_candidate = times[j - 1]

                # --- Condition 3: min_absence_frames of pure absence AFTER ---
                k = j
                while k < n and absence[k] == 1 and movement_sum[k] == 0:
                    k += 1
                after_len = k - j

                if after_len >= min_absence_frames:
                    # valid tic — save and jump forward
                    tics.append((start_time, end_time_candidate))
                    i = k
                    break
                else:
                    # tic not finished yet — keep looking
                    j = k
        else:
            i += 1

    return tics



  

    
