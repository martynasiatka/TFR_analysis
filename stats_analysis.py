# ------------------------------------------ #
# Function: Statistical analysis 
# Author: Martyna
# ----------------------------------------- #

import numpy as np
import mne 
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
from mne.stats import fdr_correction


def cluster_stats(X, threshold = None, n_permutations=1024, tail=0, correction = "FDR"):
    """
    Run permutation cluster 1-sample test for each ROI.
    X: dict of {roi_name: (n_epochs, n_freqs, n_times)}
    Returns dict of {roi_name: (T_obs, clusters, cluster_pv, H0)}
    """
    cluster_results = {}
    n_rois = len(X) 
    alpha = 0.05 
    print(f"Length of n_rois {n_rois}")
    #alpha_corrected = 0.05 / n_rois  # Bonferroni correction
    for roi_name, X_roi in X.items():
        print(f"  Running cluster test for {roi_name}...")
        T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
            X_roi,
            threshold=threshold,
            n_permutations=n_permutations,
            tail=tail,
            verbose=False
        )
        
        # ------ Correcting for multiple comparisons -----
        if correction == 'FDR':
            if len(cluster_pv) > 0:
                reject, pvals_corrected = fdr_correction(cluster_pv, alpha = alpha, method = 'indep')
            else:
                reject, pvals_corrected = [], []

        if correction == None:
            reject = cluster_pv < alpha
            pvals_corrected = cluster_pv
                
        cluster_results[roi_name] = {
            "T_obs": T_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "H0": H0,
            "reject": reject,
            "pvals_corrected" : pvals_corrected,
        }


        # Print significant clusters
        for cluster, pval_corr, signif in zip(clusters, pvals_corrected, reject ):
            if signif:
                print(f"  [{roi_name}] Significant cluster: p={pval_corr:.4f} < alpha")

    return cluster_results


def between_cluster_stats(X1, X2, n_permutations=1024, threshold = None, tail=0, correction = 'FDR'):
    """
    """
    alpha = 0.05
    cluster_results = {}
    for roi in X1.keys():
        if roi not in X2:
            continue 

        X1_roi = X1[roi]
        X2_roi = X2[roi]

        F_obs, clusters, cluster_pv, H0 = permutation_cluster_test(X = [X1_roi, X2_roi], threshold = threshold, n_permutations = n_permutations, tail = tail)

        # ------ Correcting for multiple comparisons -----
        
        if correction == 'FDR':
            if len(cluster_pv) > 0:
                reject, pvals_corrected = fdr_correction(cluster_pv, alpha = alpha, method = 'indep')
            else:
                reject, pvals_corrected = [], []
        if correction == None:
            reject = cluster_pv < alpha
            pvals_corrected = cluster_pv
                
        cluster_results[roi] = {
            "T_obs": F_obs,
            "clusters": clusters,
            "cluster_pv": cluster_pv,
            "H0": H0,
            "reject": reject,
            "pvals_corrected" : pvals_corrected,
        }

         # Print significant clusters
        for cluster, pval_corr, signif in zip(clusters, pvals_corrected, reject ):
            if signif:
                print(f"  [{roi}] Significant cluster: p={pval_corr:.4f} < alpha")

    return cluster_results


def plot_cluster_results(roi_tfr, cluster_results, freqs, times, n, phase, tic, sub,
                         roi_erp=None):
    """
    Plot actual TFR signal with significant clusters contoured.
    If roi_erp is provided, an ERP (mean ± SEM) is plotted beneath each TFR panel.

    Color scale is shared across ROIs within the same figure.
    """

    roi_names = list(roi_tfr.keys())
    n_rois = len(roi_names)
    n_cols = 3
    n_tfr_rows = 2

    # -------------------------------------------------
    # Prepare all TFRs first
    # -------------------------------------------------
    signal_plots = {}

    for roi_name in roi_names:
        signal = roi_tfr[roi_name]

        if signal.ndim == 3:
            signal_plot = signal.mean(axis=0)
        elif signal.ndim == 2:
            signal_plot = signal
        else:
            raise ValueError(
                f"{roi_name}: expected 2D or 3D array, got shape {signal.shape}"
            )

        if signal_plot.shape != (len(freqs), len(times)):
            raise ValueError(
                f"{roi_name}: signal shape {signal_plot.shape} does not match "
                f"(n_freqs, n_times)=({len(freqs)}, {len(times)})"
            )

        signal_plots[roi_name] = signal_plot

    # -------------------------------------------------
    # Shared symmetric color scale across all ROIs
    # -------------------------------------------------
    global_vmax = max(
        np.nanmax(np.abs(signal_plot))
        for signal_plot in signal_plots.values()
    )

    vmin = -global_vmax
    vmax = global_vmax

    # -------------------------------------------------
    # Create figure
    # -------------------------------------------------
    if roi_erp is not None:
        fig, axs = plt.subplots(
            n_tfr_rows * 2, n_cols,
            figsize=(18, 14),
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3, 1, 3, 1]}
        )
    else:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    fig.suptitle(f"{sub} | {phase} | {tic} | n={n}", fontsize=18, fontweight="bold")

    im = None
    extent = [times[0], times[-1], freqs[0], freqs[-1]]

    for i, roi_name in enumerate(roi_names):
        col = i % n_cols
        tfr_row_block = (i // n_cols) * (2 if roi_erp is not None else 1)

        ax_tfr = axs[tfr_row_block, col]

        signal_plot = signal_plots[roi_name]

        # -------------------------------------------------
        # Plot TFR using shared scale
        # -------------------------------------------------
        im = ax_tfr.imshow(
            signal_plot,
            cmap="RdBu_r",
            extent=extent,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )

        # -------------------------------------------------
        # Significant cluster contours
        # -------------------------------------------------
        clusters = cluster_results[roi_name]["clusters"]
        reject = cluster_results[roi_name]["reject"]

        sig_mask = np.zeros_like(signal_plot, dtype=bool)

        for cluster, signif in zip(clusters, reject):
            if signif:
                sig_mask[cluster] = True

        if sig_mask.any():
            ax_tfr.contour(
                times,
                freqs,
                sig_mask,
                levels=[0.5],
                colors="black",
                linewidths=2
            )

        n_sig = np.sum(reject)
        ax_tfr.set_title(f"{roi_name} | sig clusters: {n_sig}")
        ax_tfr.set_xlabel("Time (s)")
        ax_tfr.set_ylabel("Frequency (Hz)")
        ax_tfr.axvline(0, color="k", linestyle="--", linewidth=0.8)

        # -------------------------------------------------
        # ERP panel
        # -------------------------------------------------
        if roi_erp is not None and roi_name in roi_erp:
            ax_erp = axs[tfr_row_block + 1, col]
            _plot_erp_panel(ax_erp, roi_erp[roi_name], times, roi_name)

    # -------------------------------------------------
    # Hide unused axes
    # -------------------------------------------------
    all_axes = axs.flatten()
    n_axes_per_roi = 2 if roi_erp is not None else 1

    used_indices = set()
    for i in range(n_rois):
        col = i % n_cols
        tfr_row_block = (i // n_cols) * n_axes_per_roi
        used_indices.add(tfr_row_block * n_cols + col)

        if roi_erp is not None:
            used_indices.add((tfr_row_block + 1) * n_cols + col)

    for j, ax in enumerate(all_axes):
        if j not in used_indices:
            fig.delaxes(ax)

    # -------------------------------------------------
    # Shared colorbar
    # -------------------------------------------------
    if im is not None:
        fig.colorbar(
            im,
            ax=axs,
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
            label="TFR signal"
        )

    return fig

def _plot_erp_panel(ax, erp_data, times, roi_name):
    """
    Plot mean ERP ± SEM on *ax*.

    Parameters
    ----------
    ax       : matplotlib Axes
    erp_data : ndarray, shape (n_epochs, n_times) or (n_times,)
    times    : array-like
    roi_name : str
    """
    if erp_data.ndim == 2:
        erp_mean = erp_data.mean(axis=0)
        erp_sem  = erp_data.std(axis=0) / np.sqrt(erp_data.shape[0])
        ax.fill_between(times, erp_mean - erp_sem, erp_mean + erp_sem,
                        alpha=0.25, color="steelblue", label="±SEM")
    elif erp_data.ndim == 1:
        erp_mean = erp_data
    else:
        raise ValueError(
            f"{roi_name} ERP: expected 1D or 2D array, got shape {erp_data.shape}"
        )

    ax.plot(times, erp_mean, color="steelblue", linewidth=1.2)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlim(times[0], times[-1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"{roi_name} – ERP", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

def plot_power_spectrum_per_roi(spectra, freqs, sub_ids, sig_bands=None):
    """
    Plot normalized power spectrum (1-40 Hz) for each ROI on one figure.
    Each line = one subject. Grand average line added on top.

    Parameters
    ----------
    spectra  : dict {roi_name: (n_subjects, n_freqs)}
    freqs    : array-like (n_freqs,)
    sub_ids  : list of str — subject labels for the legend
    """
    BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 40),
    }
    BAND_COLORS = {
        "delta": "#d0e8ff",
        "theta": "#d0ffe8",
        "alpha": "#fff5d0",
        "beta":  "#ffd0d0",
        "gamma": "#ead0ff",
    }

    roi_names = list(spectra.keys())
    n_rois    = len(roi_names)
    n_cols    = 3
    n_rows    = int(np.ceil(n_rois / n_cols))

    # One color per subject
    cmap      = plt.cm.get_cmap("tab10", len(sub_ids))
    sub_colors = {sub: cmap(i) for i, sub in enumerate(sub_ids)}

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(6 * n_cols, 5 * n_rows),
                            constrained_layout=True)
    fig.suptitle("Pre-tic power spectrum | PHASE_FREE | expressed",
                 fontsize=16, fontweight="bold")

    axs_flat = axs.flatten() if n_rois > 1 else [axs]

    for i, roi_name in enumerate(roi_names):
        ax = axs_flat[i]
        data = spectra[roi_name]  # (n_subjects, n_freqs)

        # Shade frequency bands
        for band_name, (fmin, fmax) in BANDS.items():
            ax.axvspan(fmin, fmax, color=BAND_COLORS[band_name],
                       alpha=0.4, label=band_name)

        # One line per subject
        for j, sub_id in enumerate(sub_ids):
            ax.plot(freqs, data[j], color=sub_colors[sub_id],
                    linewidth=1.2, alpha=0.8, label=sub_id)


        if sig_bands is not None and roi_name in sig_bands:
            for j, sub_id in enumerate(sub_ids):
                if sub_id not in sig_bands[roi_name]:
                    continue
                for band_name, (fmin, fmax) in BANDS.items():
                    band_result = sig_bands[roi_name][sub_id].get(band_name, {})
                    if band_result.get("significant", False):
                        band_center = (fmin + fmax) / 2
                        t_stat = band_result["t_stat"]

                        if t_stat > 0:
                            # Star above the plot — staggered per subject
                            y_pos = 0.97 - (j * 0.07)
                            marker = "^"  # or "*"
                        else:
                            # Star below the plot — staggered per subject
                            y_pos = 0.03 + (j * 0.07)
                            marker = "v"  # or "*"

                        ax.plot(
                            band_center, y_pos,
                            marker="*",
                            color=sub_colors[sub_id],
                            markersize=10,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                            zorder=10,
                        )
                        # Small arrow indicating direction
                        ax.annotate(
                            "↑" if t_stat > 0 else "↓",
                            xy=(band_center, y_pos),
                            xycoords=ax.get_xaxis_transform(),
                            fontsize=7,
                            color=sub_colors[sub_id],
                            ha="center",
                            va="bottom" if t_stat > 0 else "top",
                            clip_on=False,
                        )

        # Grand average line
        grand_mean = data.mean(axis=0)
        grand_sem  = data.std(axis=0) / np.sqrt(data.shape[0])
        ax.plot(freqs, grand_mean, color="black",
                linewidth=2.5, label="grand avg", zorder=5)
        ax.fill_between(freqs, grand_mean - grand_sem, grand_mean + grand_sem,
                        color="black", alpha=0.15)

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(roi_name, fontsize=12)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Normalized power (z-score)")
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(-1, 1)
        ax.spines[["top", "right"]].set_visible(False)

    # Single legend for subjects + bands on last axis
    handles, labels = axs_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               fontsize=8, ncol=2, framealpha=0.7)

    # Hide unused axes
    for j in range(n_rois, len(axs_flat)):
        fig.delaxes(axs_flat[j])

    return fig

"""
save_cluster_results.py
-----------------------
Helper to log significant cluster details (p-value, freq range, time range)
to a CSV file. Drop this function into src/stats_analysis.py or import it
directly in your main script.

Usage in main script:
    from save_cluster_results import save_cluster_results_to_csv

    save_cluster_results_to_csv(
        cluster_results=cluster_results,
        freqs=freqs,
        times=times,
        out_dir=out_dir,          # Path object or string
        label=f"{sub}_{phase}_{tic}",
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path

"""
save_cluster_results.py
-----------------------
Helper to log significant cluster details (p-value, freq range, time range)
to a CSV file. Drop this function into src/stats_analysis.py or import it
directly in your main script.

Usage in main script:
    from save_cluster_results import save_cluster_results_to_csv

    save_cluster_results_to_csv(
        cluster_results=cluster_results,
        freqs=freqs,
        times=times,
        out_dir=out_dir,
        label=f"{sub}_{phase}_{tic}",
    )
"""
"""
save_cluster_results.py
-----------------------
Helper to log significant cluster details (p-value, freq range, time range)
to a CSV file.

MNE's permutation_cluster_1samp_test returns clusters as tuples of index
arrays, e.g. (freq_indices_array, time_indices_array). This function handles
that format correctly.

Usage in main script:
    from save_cluster_results import save_cluster_results_to_csv

    save_cluster_results_to_csv(
        cluster_results=cluster_results,
        freqs=freqs,
        times=times,
        out_dir=out_dir,
        label=f"{sub}_{phase}_{tic}",
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _cluster_to_mask(clust, shape):
    """
    Convert an MNE cluster (tuple of index arrays) to a boolean mask.

    MNE returns clusters as a tuple of arrays, one per dimension, e.g.:
        (array([0, 0, 1, 1, ...]), array([4, 5, 4, 5, ...]))
    which means: mask[freq_idx, time_idx] = True for each paired entry.

    Parameters
    ----------
    clust : tuple of np.ndarray
        MNE-style cluster index arrays.
    shape : tuple
        Shape of the output mask, e.g. (n_freqs, n_times).

    Returns
    -------
    np.ndarray of bool, shape == shape
    """
    mask = np.zeros(shape, dtype=bool)
    mask[clust] = True
    return mask


def save_cluster_results_to_csv(
    cluster_results: dict,
    freqs: np.ndarray,
    times: np.ndarray,
    out_dir,
    label: str = "",
    csv_name: str | None = None,
) -> Path:
    """
    Save cluster details (all clusters, flagged by significance) to a CSV.

    Parameters
    ----------
    cluster_results : dict
        Output of cluster_stats(). Expected structure:
            {
              roi_name: {
                  "T_obs":            np.ndarray,          # (n_freqs, n_times)
                  "clusters":         [tuple, ...],        # MNE index-array tuples
                  "cluster_pv":       np.ndarray,          # raw p-values
                  "pvals_corrected":  np.ndarray,          # FDR-corrected p-values
                  "reject":           np.ndarray of bool,
                  "H0":               np.ndarray,
              },
              ...
            }
    freqs : np.ndarray
    times : np.ndarray
    out_dir : str | Path
    label : str
        Added to every row, e.g. "sub-028_PHASE_FREE_expressed".
    csv_name : str | None
        Overrides default filename f"clusters_{label}.csv".

    Returns
    -------
    Path  – full path of the written CSV.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfr_shape = (len(freqs), len(times))
    rows = []

    for roi, result in cluster_results.items():
        clusters        = result.get("clusters", [])
        pvals_corrected = result.get("pvals_corrected", [])
        reject          = result.get("reject", [])
        T_obs           = result.get("T_obs", None)   # (n_freqs, n_times)

        if len(clusters) == 0:
            rows.append({
                "label":        label,
                "roi":          roi,
                "cluster_idx":  None,
                "significant":  None,
                "p_value_corr": None,
                "direction":    None,
                "freq_min_hz":  None,
                "freq_max_hz":  None,
                "time_min_s":   None,
                "time_max_s":   None,
                "n_freq_bins":  None,
                "n_time_bins":  None,
                "mean_T":       None,
                "peak_T":       None,
            })
            continue

        for i, (clust, p_corr, sig) in enumerate(zip(clusters, pvals_corrected, reject)):

            # --- convert MNE tuple → boolean mask ---
            mask = _cluster_to_mask(clust, tfr_shape)

            freq_indices = np.where(mask.any(axis=1))[0]
            time_indices = np.where(mask.any(axis=0))[0]

            freq_min = float(freqs[freq_indices[0]])
            freq_max = float(freqs[freq_indices[-1]])
            time_min = float(times[time_indices[0]])
            time_max = float(times[time_indices[-1]])

            if T_obs is not None:
                cluster_T = T_obs[mask]
                mean_T    = float(cluster_T.mean())
                peak_T    = float(cluster_T[np.argmax(np.abs(cluster_T))])
                direction = "positive" if mean_T > 0 else "negative"
            else:
                mean_T, peak_T, direction = None, None, None

            rows.append({
                "label":        label,
                "roi":          roi,
                "cluster_idx":  i + 1,
                "significant":  bool(sig),
                "p_value_corr": round(float(p_corr), 6),
                "direction":    direction,
                "freq_min_hz":  round(freq_min, 2),
                "freq_max_hz":  round(freq_max, 2),
                "time_min_s":   round(time_min, 4),
                "time_max_s":   round(time_max, 4),
                "n_freq_bins":  int(len(freq_indices)),
                "n_time_bins":  int(len(time_indices)),
                "mean_T":       round(mean_T, 4) if mean_T is not None else None,
                "peak_T":       round(peak_T, 4) if peak_T is not None else None,
            })

    df = pd.DataFrame(rows, columns=[
        "label", "roi", "cluster_idx", "significant", "p_value_corr", "direction",
        "freq_min_hz", "freq_max_hz", "time_min_s", "time_max_s",
        "n_freq_bins", "n_time_bins", "mean_T", "peak_T",
    ])

    fname = csv_name if csv_name else f"clusters_{label}.csv"
    fpath = out_dir / fname
    df.to_csv(fpath, index=False)
    n_sig = int(df["significant"].sum()) if df["significant"].notna().any() else 0
    print(f"[OK] Cluster results saved → {fpath}  ({len(df)} rows, {n_sig} significant)")
    return fpath