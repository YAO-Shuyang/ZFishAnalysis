"""Position and head-direction tuning from Suite2p and 28-channel behavior.

Run: python process_openfield.py SUITE2P_DIR BEHAV_FILE --speed-threshold 1
Or call run_OpenField2D(i, sheet_file, ...), using the original spreadsheet IO.
Times are seconds; coordinates are task units, so speed is task units/second.
The two recording clocks must be synchronized. Supply frame_times for measured
imaging timestamps, or imaging_time_offset when using a regular sampling clock.

Signed dF/F is retained. Skaggs-form information uses max(dF/F, 0), separately
from signed tuning, and is a descriptive calcium-response score. Optional
circular-shift tests report per-cell p-values and 95th-percentile flags, plus
Benjamini-Hochberg q-values across position and HD.

Large outputs are backed by .npy files in openfield_arrays/. trace.pkl embeds
ordinary NumPy arrays (not external-path references) and is self-contained.
Loading it consequently requires RAM for those arrays. Import-time memory use
is determined by the user's import_suite2p/import16chFlt implementations.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time as ttt
from tqdm import tqdm
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.ndimage import (gaussian_filter, gaussian_filter1d, maximum_filter1d,
                           minimum_filter1d, percentile_filter, uniform_filter1d)

from zfish._io import import_suite2p, import16chFlt


def compute_dff(raw_f, fneu, fs, output_path, *, start_frame=0,
                neuropil_coeff=0.7, baseline_method="maximin",
                baseline_window_seconds=60.0, baseline_percentile=20.0,
                baseline_sigma_seconds=1.0, baseline_min=1e-6, chunk_size=256):
    """Compute signed dF/F in neuron blocks, writing a disk-backed array.

    Parameters
    ----------
    raw_f, fneu : array_like, shape (n_neuron, n_frame)
        ROI and neuropil fluorescence from the user's Suite2p importer.
    fs : float
        Imaging sampling rate per plane, in Hz.
    output_path : str or pathlib.Path
        Destination .npy file for dF/F.
    start_frame : int, optional
        Discard earlier frames before baseline estimation.
    neuropil_coeff : float, optional
        Coefficient in Fc = F - neuropil_coeff * Fneu.
    baseline_method : {'maximin', 'percentile', 'rolling_percentile'}, optional
        Maximin applies Gaussian smoothing, running minimum, then maximum.
        Percentile is constant across the retained session. Rolling percentile
        is slower and intended for explicit sensitivity checks.
    baseline_window_seconds, baseline_sigma_seconds : float, optional
        Baseline window and Gaussian standard deviation, in seconds.
    baseline_percentile : float, optional
        Percentile used by either percentile method.
    baseline_min : float, optional
        Baselines at or below this value invalidate the ROI, without flooring.
    chunk_size : int, optional
        Maximum neurons processed simultaneously.

    Returns
    -------
    dff : numpy.memmap, shape (n_neuron, n_retained_frame)
        Signed dF/F; invalid ROIs contain NaN.
    valid_roi : ndarray of bool, shape (n_neuron,)
        True when all retained fluorescence and baseline values are valid.
    baseline_summary : ndarray, shape (n_neuron, 3)
        Minimum, median, and maximum F0; invalid ROIs contain NaN.

    Notes
    -----
    A conservative whole-ROI exclusion avoids silently imputing missing data
    or making occupancy probabilities depend on a neuron's missing samples.
    """
    if raw_f.ndim != 2 or raw_f.shape != fneu.shape:
        raise ValueError("RawTraces and Fneu must have the same 2-D shape.")
    if baseline_method not in {"maximin", "percentile", "rolling_percentile"}:
        raise ValueError("Unknown baseline_method.")
    if (not np.isfinite(fs) or fs <= 0 or chunk_size < 1 or
            not 0 <= start_frame < raw_f.shape[1] or
            baseline_window_seconds <= 0 or baseline_sigma_seconds < 0 or
            not 0 <= baseline_percentile <= 100 or baseline_min <= 0 or
            not np.isfinite(neuropil_coeff) or neuropil_coeff < 0):
        raise ValueError("Invalid dF/F parameters.")
    shape = (raw_f.shape[0], raw_f.shape[1] - start_frame)
    dff = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=shape)
    valid_roi = np.zeros(shape[0], dtype=bool)
    summary = np.full((shape[0], 3), np.nan, dtype=np.float32)
    window = max(1, round(baseline_window_seconds * fs))
    sigma = baseline_sigma_seconds * fs
    for start in tqdm(range(0, shape[0], chunk_size), desc="dF/F computation"):
        stop = min(start + chunk_size, shape[0])
        fc = np.array(raw_f[start:stop, start_frame:], dtype=np.float32, copy=True)
        fc -= neuropil_coeff * np.asarray(fneu[start:stop, start_frame:], dtype=np.float32)
        finite = np.isfinite(fc).all(axis=1)
        fc[~finite] = 0  # Excluded rows must not contaminate filtering operations.
        if baseline_method == "percentile":
            f0 = np.percentile(fc, baseline_percentile, axis=1)[:, None]
        elif baseline_method == "rolling_percentile":
            f0 = percentile_filter(fc, baseline_percentile, size=(1, window), mode="reflect")
        else:
            f0 = gaussian_filter1d(fc, sigma, axis=1, mode="reflect") if sigma else fc.copy()
            f0 = minimum_filter1d(f0, window, axis=1, mode="reflect")
            f0 = maximum_filter1d(f0, window, axis=1, mode="reflect")
        valid = finite & np.isfinite(f0).all(axis=1) & (f0 > baseline_min).all(axis=1)
        block = np.full_like(fc, np.nan)
        with np.errstate(over="ignore", invalid="ignore"):
            block[valid] = (fc[valid] - f0[valid]) / f0[valid]
        valid &= np.isfinite(block).all(axis=1)
        block[~valid] = np.nan
        dff[start:stop] = block
        valid_roi[start:stop] = valid
        summary[start:stop][valid] = np.column_stack(
            (f0[valid].min(axis=1), np.median(f0[valid], axis=1), f0[valid].max(axis=1)))
    dff.flush()
    return dff, valid_roi, summary


def tuning_metrics(first, second, positive_map, occupancy):
    """Compute vectorized Pearson stability and Skaggs-form information.

    Parameters
    ----------
    first, second : ndarray, shape (n_neuron, n_bin)
        Signed tuning curves in chronological session halves. NaN marks bins
        below the occupancy threshold; correlation uses common finite bins.
    positive_map : ndarray, shape (n_neuron, n_bin)
        Means of framewise rectified dF/F, not rectified signed bin means.
    occupancy : ndarray, shape (n_bin,)
        Time in seconds in each bin for the complete analyzed session.

    Returns
    -------
    correlation, information : ndarray, shape (n_neuron,)
        Pearson r and sum(p * q * log2(q)), q = response/mean_response.
        Zero variance, fewer than three common bins, or zero total response
        yields NaN. Occupancy is renormalized over eligible bins.
    """
    common = np.isfinite(first) & np.isfinite(second)
    count = common.sum(axis=1, keepdims=True)
    a = np.where(common, first, 0).astype(np.float64)
    b = np.where(common, second, 0).astype(np.float64)
    a -= a.sum(axis=1, keepdims=True) / np.maximum(count, 1)
    b -= b.sum(axis=1, keepdims=True) / np.maximum(count, 1)
    a[~common] = 0
    b[~common] = 0
    denominator = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    correlation = np.full(first.shape[0], np.nan)
    np.divide(np.sum(a*b, axis=1), denominator, out=correlation,
              where=(count[:, 0] >= 3) & (denominator > 0))
    correlation = np.clip(correlation, -1, 1)
    return correlation.astype(np.float32), information_score(positive_map, occupancy)


def information_score(response, occupancy, *, zero_value=np.nan):
    """Compute occupancy-weighted Skaggs-form information for positive activity.

    Parameters
    ----------
    response : ndarray, shape (n_neuron, n_bin)
        Nonnegative bin means; NaN bins are excluded.
    occupancy : ndarray, shape (n_bin,)
        Sampled duration in seconds, renormalized over finite bins per neuron.
    zero_value : float, optional
        Score assigned when total response is zero. Observed silent neurons
        are untestable (NaN); silent shuffled samples receive zero information
        so they remain in the null distribution instead of being discarded.

    Returns
    -------
    information : ndarray, shape (n_neuron,)
        Sum p_i * (r_i/r_mean) * log2(r_i/r_mean), using 0*log(0) = 0.
    """
    eligible = np.isfinite(response) & (occupancy[None, :] > 0)
    p = np.where(eligible, occupancy[None, :], 0).astype(np.float64)
    np.divide(p, p.sum(axis=1, keepdims=True), out=p,
              where=p.sum(axis=1, keepdims=True) > 0)
    response = np.where(eligible, response, 0).astype(np.float64)
    if np.any(response < 0):
        raise ValueError("Information requires nonnegative responses.")
    mean = np.sum(p * response, axis=1, keepdims=True)
    ratio = np.zeros_like(response)
    np.divide(response, mean, out=ratio, where=mean > 0)
    log_ratio = np.zeros_like(ratio)
    np.log2(ratio, out=log_ratio, where=ratio > 0)
    information = np.maximum(0, np.sum(p * ratio * log_ratio, axis=1))
    information[mean[:, 0] <= 0] = zero_value
    return information.astype(np.float32)


def compute_tuning(dff, frame_idx, bins, frame_seconds, first_half, valid_roi,
                   bin_shape, output_dir, label, *, chunk_size=256,
                   min_occupancy_seconds=1.0, smoothing_sigma_bins=1.0,
                   save_half_maps=False):
    """Aggregate spatial or circular HD tuning with sparse matrix products.

    Parameters
    ----------
    dff : ndarray, shape (n_neuron, n_frame)
        Signed fluorescence response.
    frame_idx, bins : ndarray, shape (n_selected_frame,)
        Indices into dff and zero-based bin assignments.
    frame_seconds : ndarray, shape (n_selected_frame,)
        Imaging-frame duration weights; removed intervals add no occupancy.
    first_half : ndarray of bool, shape (n_selected_frame,)
        Membership in the first chronological half before speed filtering.
    valid_roi : ndarray of bool, shape (n_neuron,)
        ROIs eligible for analysis.
    bin_shape : tuple of int
        (n_y, n_x) for space or (n_hd,) for circular heading.
    output_dir : pathlib.Path
        Destination directory for disk-backed maps.
    label : str
        Prefix for output array filenames.
    chunk_size : int, optional
        Neurons per block; avoids a full filtered neural-array copy.
    min_occupancy_seconds : float, optional
        Minimum sampled time required per bin, independently in each half.
    smoothing_sigma_bins : float, optional
        Gaussian width. Zero disables smoothing. Sums and occupancy are
        smoothed separately; originally ineligible bins remain NaN.
    save_half_maps : bool, optional
        Also retain raw/smoothed half maps; otherwise save only half metrics.

    Returns
    -------
    result : dict
        Raw/smoothed signed and positive session tuning, occupancy seconds and
        counts (session and halves), and raw/smoothed split-half correlations.
        All tuning arrays have shape (n_neuron, n_bin). Spatial bin order is
        y * n_x + x. Information uses unsmoothed positive tuning.
    """
    n_neuron, n_bin = dff.shape[0], int(np.prod(bin_shape))
    selectors = [np.ones(bins.size, bool), first_half, ~first_half]
    designs, occupancy, counts = [], [], []
    for select in selectors:
        weights = np.where(select, frame_seconds, 0)
        designs.append(sparse.csr_matrix((weights, (bins, np.arange(bins.size))),
                                        shape=(n_bin, bins.size)))
        occupancy.append(np.bincount(bins, weights=weights, minlength=n_bin))
        counts.append(np.bincount(bins[select], minlength=n_bin))
    result = {"occupancy_seconds": np.array(occupancy), "occupancy_samples": np.array(counts)}
    keys = ["tuning_raw", "tuning_smooth", "positive_tuning_raw"]
    if save_half_maps:
        keys += ["first_raw", "second_raw", "first_smooth", "second_smooth"]
    for key in keys:
        result[key] = np.lib.format.open_memmap(output_dir / f"{label}_{key}.npy",
            mode="w+", dtype="float32", shape=(n_neuron, n_bin))
    for key in ["split_half_raw", "split_half_smooth", "information"]:
        result[key] = np.full(n_neuron, np.nan, dtype=np.float32)
    # Include occupancy only from bins meeting the original sampling threshold.
    masks = [(o > 0) & (o >= min_occupancy_seconds) for o in occupancy]
    smoothed_occ = []
    for occ, mask in zip(occupancy, masks):
        grid = np.where(mask, occ, 0).reshape(bin_shape)
        smoothed_occ.append(gaussian_filter(grid, smoothing_sigma_bins,
            mode="wrap" if len(bin_shape) == 1 else "constant").ravel())
    for start in tqdm(range(0, n_neuron, chunk_size), desc=f"Tuning Curve [{label}]"):
        stop = min(start + chunk_size, n_neuron)
        block = np.asarray(dff[start:stop, frame_idx], dtype=np.float32)
        valid = valid_roi[start:stop]
        block[~valid] = 0
        raw, smooth = [], []
        for design, occ, mask, smooth_occ in zip(designs, occupancy, masks, smoothed_occ):
            sums = np.asarray(design @ block.T).T
            curve = np.full_like(sums, np.nan)
            np.divide(sums, occ, out=curve, where=mask[None, :])
            numerator = np.where(mask, sums, 0).reshape((stop-start, *bin_shape))
            sigma = (0, *([smoothing_sigma_bins] * len(bin_shape)))
            numerator = gaussian_filter(numerator, sigma,
                mode="wrap" if len(bin_shape) == 1 else "constant").reshape(sums.shape)
            smoothed = np.full_like(sums, np.nan)
            np.divide(numerator, smooth_occ, out=smoothed,
                      where=mask[None, :] & (smooth_occ[None, :] > 0))
            curve[~valid] = np.nan
            smoothed[~valid] = np.nan
            raw.append(curve)
            smooth.append(smoothed)
        np.maximum(block, 0, out=block)
        positive_sums = np.asarray(designs[0] @ block.T).T
        positive_map = np.full_like(positive_sums, np.nan)
        np.divide(positive_sums, occupancy[0], out=positive_map, where=masks[0][None, :])
        positive_map[~valid] = np.nan
        result["tuning_raw"][start:stop] = raw[0]
        result["tuning_smooth"][start:stop] = smooth[0]
        result["positive_tuning_raw"][start:stop] = positive_map
        r_raw, info = tuning_metrics(raw[1], raw[2], positive_map, occupancy[0])
        r_smooth, _ = tuning_metrics(smooth[1], smooth[2], positive_map, occupancy[0])
        result["split_half_raw"][start:stop] = r_raw
        result["split_half_smooth"][start:stop] = r_smooth
        result["information"][start:stop] = info
        if save_half_maps:
            for key, array in zip(keys[3:], [raw[1], raw[2], smooth[1], smooth[2]]):
                result[key][start:stop] = array
    for key in keys:
        result[key].flush()
    return result


def shuffle_test(trace, *, n_shuffles=1000, min_shift_seconds=20.0,
                 seed=42, chunk_size=256, save_null=False):
    """Add circular-shift tests for position and HD information to trace.

    Parameters
    ----------
    trace : dict
        Output of process_openfield, including continuous post-5-second dFF,
        timestamps, analysis indices, occupancy, and observed information.
        This function updates the dictionary in place; the caller saves it.
    n_shuffles : int, optional
        Requested number of unique offsets, default 1000. If fewer offsets
        exist, use all eligible offsets and report the actual count.
    min_shift_seconds : float, optional
        Exclude offsets closer than this to zero in either circular direction.
        Default 20 s is illustrative; choose beyond relevant autocorrelation.
    seed : int, optional
        NumPy random generator seed. The same offsets are used for all cells
        and both coding types, preserving population synchrony per surrogate.
    chunk_size : int, optional
        Neurons per block. Null scores occupy O(chunk_size * n_shuffles) RAM;
        work scales with neurons * shuffles * selected frames.
    save_null : bool, optional
        Retain neuron-by-shuffle arrays as disk-backed files and in trace.
        False stores summary statistics and offsets only.

    Returns
    -------
    trace : dict
        Each position/hd dictionary gains shuffle_threshold95, shuffle_p,
        above_shuffle_95, significant_uncorrected, shuffle_q, significant_fdr,
        and shuffle_tested. Optional shuffle_information stores null scores.
        Untestable neurons have NaN statistics and False flags. The top-level
        shuffle dictionary records offsets, seed, count, and time boundaries.

    Notes
    -----
    Shift the complete continuous neural trace within behavioral coverage,
    THEN apply the original speed/position/heading selections. Shifting an
    already speed-filtered trace would join nonadjacent times. Occupancy,
    eligible bins, rectification, and the unsmoothed statistic stay fixed.
    Regular imaging timestamps are required (1% interval tolerance).

    One-sided p = (1 + count(null >= observed)) / (1 + n_shuffles); ties count
    against significance. The strict observed > 95th-percentile flag and
    p <= .05 are saved separately because finite samples/ties can differ.
    Benjamini-Hochberg correction pools all tested position and HD hypotheses.
    These tests assume approximate temporal stationarity and do not isolate
    position/HD effects from correlated variables such as speed.
    """
    if (not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 1 or
            not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1 or
            not np.isfinite(min_shift_seconds) or min_shift_seconds <= 0):
        raise ValueError("Use positive shuffle count, chunk size, and minimum shift.")
    covered = np.flatnonzero(trace["alignment_valid"])
    if len(covered) < 2:
        raise ValueError("Too few aligned frames for a circular-shift test.")
    first, stop = int(covered[0]), int(covered[-1])+1
    intervals = np.diff(trace["frame_time"][first:stop])
    dt = float(np.median(intervals))
    if dt <= 0 or not np.allclose(intervals, dt, rtol=.01, atol=1e-8):
        raise ValueError("Shuffle needs regularly sampled imaging; resolve dropped frames first.")
    n_time = stop-first
    min_shift = max(1, int(np.ceil(min_shift_seconds/dt)))
    allowed = np.arange(min_shift, n_time-min_shift+1, dtype=np.int64)
    if not allowed.size:
        raise ValueError("Recording is too short for min_shift_seconds in both directions.")
    actual = min(n_shuffles, len(allowed))
    rng = np.random.default_rng(seed)
    shifts = rng.choice(allowed, size=actual, replace=False)
    if actual < n_shuffles:
        print(f"Only {actual} unique eligible shifts; using all of them.", flush=True)
    n_neuron = trace["dFF"].shape[0]
    directory = Path(trace["suite2p_dir"])/"openfield_arrays"
    if save_null:
        directory.mkdir(exist_ok=True)
    setups = []
    for label, index_key, bin_key in [("position", "analysis_frame_idx", "spatial_bins"),
                                      ("hd", "hd_frame_idx", "hd_bins")]:
        result = trace[label]
        occupancy = result["occupancy_seconds"][0]
        eligible = (occupancy > 0) & (occupancy >= trace["parameters"]["min_occupancy_seconds"])
        indices, bins = trace[index_key], trace[bin_key]
        select = eligible[bins]
        chosen = indices[select]
        if np.any((chosen < first) | (chosen >= stop)):
            raise ValueError("Analysis frames lie outside the shuffle interval.")
        eligible_ids = np.flatnonzero(eligible)
        rows = np.searchsorted(eligible_ids, bins[select])
        design = sparse.csr_matrix((trace["frame_duration_seconds"][chosen],
                                   (rows, np.arange(len(chosen)))),
                                  shape=(len(eligible_ids), len(chosen)))
        tested = trace["valid_roi"] & np.isfinite(result["information"])
        # One occupied bin cannot establish selectivity across bins.
        tested &= len(eligible_ids) >= 2
        result["shuffle_tested"] = tested
        for key in ["shuffle_threshold95", "shuffle_p", "shuffle_q"]:
            result[key] = np.full(n_neuron, np.nan, dtype=np.float64)
        for key in ["above_shuffle_95", "significant_uncorrected", "significant_fdr"]:
            result[key] = np.zeros(n_neuron, dtype=bool)
        result.pop("shuffle_information", None)
        if save_null:
            result["shuffle_information"] = np.lib.format.open_memmap(
                directory/f"{label}_shuffle_information.npy", mode="w+",
                dtype="float32", shape=(n_neuron, actual))
        setups.append((label, chosen-first, design, occupancy[eligible], tested))
    print(f"Circular shuffles: {actual} shifts, {n_neuron} neurons; p resolution {1/(actual+1):.6g}", flush=True)
    shuffle_started = ttt.monotonic()
    for start in range(0, n_neuron, chunk_size):
        end = min(start+chunk_size, n_neuron)
        print(f"Shuffle neurons {start}:{end}/{n_neuron}", flush=True)
        block = np.maximum(np.asarray(trace["dFF"][start:end, first:stop]), 0)
        block[~trace["valid_roi"][start:end]] = 0
        null = [np.full((end-start, actual), np.nan, dtype=np.float32) for _ in setups]
        last_update = ttt.monotonic()
        for j, shift in enumerate(shifts):
            for k, (_, indices, design, occupancy, tested) in enumerate(setups):
                if not tested[start:end].any():
                    continue
                # Equivalent to np.roll(block, shift, axis=1)[:, indices].
                samples = block[:, (indices-shift) % n_time]
                response = np.asarray(design @ samples.T).T / occupancy
                null[k][:, j] = information_score(response, occupancy, zero_value=0.0)
            if ttt.monotonic()-last_update >= 5:
                print(f"  Shifts {j+1}/{actual}", flush=True)
                last_update = ttt.monotonic()
        for scores, (label, _, _, _, tested) in zip(null, setups):
            result = trace[label]
            valid = tested[start:end]
            scores[~valid] = np.nan
            ids = np.flatnonzero(valid)+start
            if len(ids):
                result["shuffle_threshold95"][ids] = np.percentile(scores[valid], 95, axis=1)
                result["shuffle_p"][ids] = (1 + np.sum(
                    scores[valid] >= result["information"][ids, None], axis=1)) / (actual+1)
            if save_null:
                result["shuffle_information"][start:end] = scores
        elapsed = ttt.monotonic()-shuffle_started
        remaining = elapsed*(n_neuron-end)/end
        print(f"  Elapsed {elapsed/60:.1f} min; estimated remaining {remaining/60:.1f} min", flush=True)
    # Joint correction over all tested neuron x coding-type hypotheses.
    pooled = np.concatenate([trace[label]["shuffle_p"] for label in ["position", "hd"]])
    finite = np.flatnonzero(np.isfinite(pooled))
    order = finite[np.argsort(pooled[finite])]
    qvalues = np.full_like(pooled, np.nan)
    adjusted = pooled[order]*len(order)/np.arange(1, len(order)+1)
    qvalues[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    for k, label in enumerate(["position", "hd"]):
        result = trace[label]
        result["shuffle_q"] = qvalues[k*n_neuron:(k+1)*n_neuron]
        result["above_shuffle_95"] = result["shuffle_tested"] & (result["information"] > result["shuffle_threshold95"])
        result["significant_uncorrected"] = result["shuffle_p"] <= .05
        result["significant_fdr"] = result["shuffle_q"] <= .05
        if save_null:
            result["shuffle_information"].flush()
            result["shuffle_information"] = np.asarray(result["shuffle_information"])
        print(f"{label}: {result['above_shuffle_95'].sum()} above null 95th percentile; "
              f"{result['significant_fdr'].sum()} pass joint BH q <= .05", flush=True)
    trace["shuffle"] = dict(method="circular shift before behavioral filtering",
        requested_n_shuffles=n_shuffles, n_shuffles=actual, seed=seed,
        min_shift_seconds=min_shift_seconds, shift_frames=shifts,
        shift_seconds=shifts*dt, first_frame=first, stop_frame=stop,
        frame_interval_seconds=dt, unique_allowed_shifts=len(allowed),
        minimum_p=1/(actual+1), fdr_family="all tested position and HD hypotheses",
        statistic="unsmoothed information from framewise positive dF/F", save_null=save_null)
    return trace


def plot_qc(trace, output_path):
    """Save a compact diagnostic figure without plotting every neuron.

    Parameters
    ----------
    trace : dict
        Output from process_openfield.
    output_path : str or pathlib.Path
        Destination PNG file.

    Returns
    -------
    None
        Saves occupancy, speed, information, and stability distributions.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    ny, nx = trace["parameters"]["n_bins_y"], trace["parameters"]["n_bins_x"]
    image = axes[0, 0].imshow(trace["position"]["occupancy_seconds"][0].reshape(ny, nx),
                            origin="lower", extent=(35, 65, 35, 65))
    axes[0, 0].set(title="Position occupancy", xlabel="x", ylabel="y")
    fig.colorbar(image, ax=axes[0, 0], label="seconds")
    centers = trace["hd_bin_centers"]
    axes[0, 1].plot(centers, trace["hd"]["occupancy_seconds"][0])
    axes[0, 1].set(title="HD occupancy", xlabel="heading (radians)", ylabel="seconds")
    speed = trace["frame_speed"]
    axes[0, 2].hist(speed[np.isfinite(speed)], bins=60)
    axes[0, 2].axvline(trace["parameters"]["speed_threshold"], color="r")
    axes[0, 2].set(title="Aligned speed", xlabel="task units / second")
    for label in ["position", "hd"]:
        values = trace[label]["information"]
        axes[1, 0].hist(values[np.isfinite(values)], bins=60, alpha=.5, label=label)
        values = trace[label]["split_half_smooth"]
        axes[1, 1].hist(values[np.isfinite(values)], bins=60, alpha=.5, label=label)
    axes[1, 0].set(title="Positive dF/F information", xlabel="Skaggs-form score")
    axes[1, 1].set(title="Split-half stability", xlabel="Pearson r")
    axes[1, 0].legend()
    axes[1, 1].legend()
    stride = max(1, len(trace["frame_time"]) // 10000)
    axes[1, 2].plot(trace["frame_time"][::stride], trace["frame_speed"][::stride], lw=.5)
    axes[1, 2].set(title="Speed over time", xlabel="seconds", ylabel="task units / second")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_openfield(suite2p_dir, behav_file, *, speed_threshold=0.0,
                      speed_window_samples=200, n_bins_x=30, n_bins_y=30,
                      n_hd_bins=36, min_occupancy_seconds=1.0,
                      smoothing_sigma_bins=1.0, neuropil_coeff=0.7,
                      baseline_method="maximin", baseline_window_seconds=60.0,
                      baseline_percentile=20.0, baseline_sigma_seconds=1.0,
                      baseline_min=1e-6, chunk_size=256, frame_times=None,
                      imaging_time_offset=0.0, max_alignment_gap_seconds=None,
                      save_half_maps=False, make_plots=True, session_metadata=None,
                      n_shuffles=0, min_shuffle_shift_seconds=20.0,
                      shuffle_seed=42, save_shuffle_null=False):
    """Process one recording and save trace.pkl beside the neural data.

    Parameters
    ----------
    suite2p_dir, behav_file : str or pathlib.Path
        Paths passed unchanged to import_suite2p and import16chFlt. Importer
        output requires RawTraces, Fneu, fs, and behav_time/pos_x/pos_y/orient.
        All imported ROIs are retained; invalid baselines are flagged.
    speed_threshold : float, optional
        Keep frames with speed strictly greater than this task-unit/s value.
        Default zero excludes stationary frames without assuming a physical
        calibration or a biological locomotion cutoff.
    speed_window_samples : int, optional
        Centered behavior-sample boxcar length, default 200 (~33 ms at 6 kHz).
    n_bins_x, n_bins_y, n_hd_bins : int, optional
        Spatial bins across [35, 65] and HD bins across [0, 2*pi).
    min_occupancy_seconds, smoothing_sigma_bins : float, optional
        Per-bin sampled-time threshold and occupancy-normalized smoothing.
    neuropil_coeff, baseline_method, baseline_window_seconds : optional
        See compute_dff. Baseline uses all imaging frames at t >= 5, before
        speed filtering. Default is a running maximin baseline.
    baseline_percentile, baseline_sigma_seconds, baseline_min : float, optional
        See compute_dff. Baseline parameters should be checked for sustained
        activity, which a short running baseline may attenuate.
    chunk_size : int, optional
        Neuron block size for dF/F and sparse tuning aggregation.
    frame_times : array_like, shape (n_frame,), optional
        Measured timestamps in seconds on the behavior clock, shared by ROIs.
        If omitted, use arange(n_frame)/fs + imaging_time_offset. For combined
        planes with materially different clocks, analyze planes separately.
    imaging_time_offset : float, optional
        Offset in seconds for the regular imaging clock; not added to supplied
        frame_times. An unverified offset is not automatic synchronization.
    max_alignment_gap_seconds : float, optional
        Maximum nearest-behavior distance. Default 2/6000 seconds. Frames
        outside behavioral coverage or inside long behavioral gaps are dropped.
    save_half_maps, make_plots : bool, optional
        Save extra half tuning arrays and make a diagnostic PNG, respectively.
    session_metadata : dict, optional
        FishID/session or other small identifiers, stored under session_metadata.
    n_shuffles : int, optional
        Circular shifts for each information test. Zero disables this expensive
        step; 1000 is a starting choice. See shuffle_test for assumptions.
    min_shuffle_shift_seconds : float, optional
        Minimum offset from zero in either circular direction, default 20 s.
    shuffle_seed : int, optional
        Seed for reproducible unique offsets.
    save_shuffle_null : bool, optional
        Also store neuron-by-shuffle null scores; default False saves summaries.

    Returns
    -------
    trace : dict
        Self-contained arrays and parameters. dFF includes all imaging frames
        at t >= 5; analysis_frame_idx selects speed/position-valid frames.
        HD additionally requires finite heading. Position/HD tuning arrays are
        (n_neuron, n_bin). Occupancy arrays are (3, n_bin), ordered session,
        first half, second half. Frame indices refer to retained dFF columns;
        original_frame_idx maps those columns to imported neural frames.

    Notes
    -----
    Occupancy is sampled at imaging times (not dense behavioral occupancy),
    weighted by imaging-frame intervals capped at 1/fs (no gap bridging).
    Half membership uses the temporal
    midpoint of the overlapping recording before speed filtering. Circular
    projection preserves direction: (dx,dy) *= min(1,15/hypot(dx,dy)).
    """
    if not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 0:
        raise ValueError("n_shuffles must be a nonnegative integer.")
    if (any(not isinstance(n, (int, np.integer)) or n < 1 for n in
            [speed_window_samples, n_bins_x, n_bins_y, n_hd_bins, chunk_size]) or
            not np.isfinite(speed_threshold) or speed_threshold < 0 or
            not np.isfinite(min_occupancy_seconds) or min_occupancy_seconds < 0 or
            not np.isfinite(smoothing_sigma_bins) or smoothing_sigma_bins < 0):
        raise ValueError("Invalid bin, speed, chunk, occupancy, or smoothing parameters.")
    save_dir = Path(suite2p_dir)
    if not save_dir.is_dir():
        raise ValueError("suite2p_dir must be an existing neural-data directory.")
    output_dir = save_dir / "openfield_arrays"
    output_dir.mkdir(exist_ok=True)
    print(ttt.strftime("%Y-%m-%d %H:%M:%S", ttt.localtime()))
    print("Importing neural and 28-channel behavioral data", flush=True)
    neural = import_suite2p(suite2p_dir)
    res = import16chFlt(behav_file, nchannel=28)
    raw_f, fneu = neural["RawTraces"], neural["Fneu"]
    fs = float(neural["fs"])
    if not np.isfinite(fs) or fs <= 0 or raw_f.ndim != 2 or raw_f.shape != fneu.shape:
        raise ValueError("Require matching 2-D RawTraces/Fneu and positive per-plane fs.")
    time = np.asarray(res["behav_time"], dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Behavior timestamps must be finite and strictly increasing seconds.")
    first = np.searchsorted(time, 5.0)
    if len(time)-first < 2:
        raise ValueError("Too few behavioral samples at t >= 5 seconds.")
    bt = time[first:]
    
    position = []
    for key in ["behav_pos_x", "behav_pos_y", "behav_orient"]:
        array = np.asarray(res[key])
        if array.shape != time.shape:
            raise ValueError(f"{key} must match behav_time.")
        position.append(array[first:].astype(np.float64))
    x, y, heading = position
    valid_position = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < 100) & (y >= 0) & (y < 100)
    dx, dy = x-50, y-50
    radius = np.hypot(dx, dy)
    scale = np.ones_like(radius)
    np.divide(15.0, radius, out=scale, where=radius > 15)
    x, y = 50+dx*scale, 50+dy*scale
    x[~valid_position] = np.nan
    y[~valid_position] = np.nan
    dt = np.diff(bt)
    dl = np.hypot(np.diff(x), np.diff(y))
    # A boxcar mean ratio equals the requested sum(dl)/sum(dt), in O(T) time.
    # Constant padding reproduces convolution edges; invalid intervals exclude
    # every speed window that touches them, rather than bridging missing data.
    interval_ok = np.isfinite(dl) & (dt <= 2/6000)
    win = speed_window_samples
    sum_dl = uniform_filter1d(np.where(interval_ok, dl, 0), win, mode="constant")
    sum_dt = uniform_filter1d(dt, win, mode="constant")
    bad = uniform_filter1d((~interval_ok).astype(float), win, mode="constant")
    interval_speed = np.full_like(dl, np.nan)
    np.divide(sum_dl, sum_dt, out=interval_speed, where=(sum_dt > 0) & (bad < 1e-12))
    speed = np.r_[interval_speed[0], interval_speed]
    ft = (np.arange(raw_f.shape[1], dtype=float)/fs + imaging_time_offset
          if frame_times is None else np.asarray(frame_times, dtype=float))
    if (ft.shape != (raw_f.shape[1],) or not np.isfinite(ft).all() or
            len(ft) < 2 or np.any(np.diff(ft) <= 0)):
        raise ValueError("Frame times must match neural columns and increase strictly.")
    # Frame weights come from the original clock, never from speed-filtered gaps.
    # Missing imaging frames must not assign the entire gap to one position.
    frame_duration = np.minimum(np.r_[np.diff(ft), 1/fs], 1/fs)
    start_frame = int(np.searchsorted(ft, 5.0))
    if len(ft)-start_frame < 2:
        raise ValueError("Too few imaging frames at t >= 5 seconds.")
    original_idx = np.arange(start_frame, len(ft))
    frame_duration = frame_duration[start_frame:]
    ft = ft[start_frame:]
    right = np.clip(np.searchsorted(bt, ft), 0, len(bt)-1)
    left = np.maximum(right-1, 0)
    idx = np.where(np.abs(bt[left]-ft) <= np.abs(bt[right]-ft), left, right)
    gap_limit = 2/6000 if max_alignment_gap_seconds is None else max_alignment_gap_seconds
    if not np.isfinite(gap_limit) or gap_limit <= 0:
        raise ValueError("max_alignment_gap_seconds must be positive.")
    aligned = (ft >= bt[0]) & (ft <= bt[-1]) & (np.abs(bt[idx]-ft) <= gap_limit)
    frame_xy = np.column_stack([x[idx], y[idx]]).astype(np.float32)
    print(frame_xy.shape)
    frame_heading = np.mod(heading[idx], 2*np.pi).astype(np.float64)
    frame_speed = speed[idx].astype(np.float32)
    frame_xy[~aligned] = np.nan
    frame_heading[~aligned] = np.nan
    frame_speed[~aligned] = np.nan
    analysis_mask = aligned & np.isfinite(frame_xy).all(axis=1) & (frame_speed > speed_threshold)
    analysis_idx = np.flatnonzero(analysis_mask)
    if len(analysis_idx) < 2:
        raise ValueError("Fewer than two imaging frames survive alignment/position/speed filtering.")
    midpoint = (max(5.0, bt[0], ft[0]) + min(bt[-1], ft[-1]))/2
    first_half = ft < midpoint
    xy_bins = np.floor((frame_xy[analysis_idx]-35)/30 * [n_bins_x, n_bins_y]).astype(np.int32)
    xy_bins = np.clip(xy_bins, 0, [n_bins_x-1, n_bins_y-1])
    spatial_bins = xy_bins[:, 1]*n_bins_x + xy_bins[:, 0]
    hd_idx = analysis_idx[np.isfinite(frame_heading[analysis_idx])]
    hd_bins = np.floor(frame_heading[hd_idx]/(2*np.pi)*n_hd_bins).astype(np.int32)
    hd_bins = np.clip(hd_bins, 0, n_hd_bins-1)
    print(f"Retained {len(analysis_idx)}/{len(ft)} position frames; {len(hd_idx)} HD frames", flush=True)
    
    parameters = dict(
        speed_threshold=speed_threshold, 
        speed_window_samples=win,
        behavioral_fs=6000, 
        discard_before_seconds=5.0, 
        circle_center=(50,50), 
        circle_radius=15,
        n_bins_x=n_bins_x, 
        n_bins_y=n_bins_y, 
        n_hd_bins=n_hd_bins,
        min_occupancy_seconds=min_occupancy_seconds, 
        smoothing_sigma_bins=smoothing_sigma_bins,
        neuropil_coeff=neuropil_coeff, 
        baseline_method=baseline_method,
        baseline_window_seconds=baseline_window_seconds, 
        baseline_percentile=baseline_percentile,
        baseline_sigma_seconds=baseline_sigma_seconds, 
        baseline_min=baseline_min,
        chunk_size=chunk_size, 
        imaging_time_offset=imaging_time_offset,
        measured_frame_times=frame_times is not None, 
        max_alignment_gap_seconds=gap_limit,
        save_half_maps=save_half_maps, 
        information_signal="framewise max(signed dF/F, 0)",
        spatial_bin_order="y * n_bins_x + x", 
        occupancy_order=("session","first","second"),
    )
    
    neural_data = {
        'roi_coord': neural['roi_coord'].copy(),
        'snr': neural['snr'].copy(),
        'suite2p_version': neural['suite2p_version'],
        'nplanes': neural['nplanes'],
        'nchannels': neural['nchannels'],
        'tau': neural['tau'],
        'fs': neural['fs'],
        'pretrained_model': neural['pretrained_model'],
        'lenX': neural['lenX'],
        'lenY': neural['lenY'],
        'lenZ': neural['lenZ'],
        'meanImg': neural['meanImg'].copy(),
        'nframes': neural['nframes']
    }
    # Release high-rate imports and unused neural arrays before large computations.
    metadata = {key: neural[key] for key in ["stat", "iscell"] if key in neural}
    del res, neural, time, bt, position, x, y, heading, dx, dy, radius, scale, dt, dl
    del sum_dl, sum_dt, bad, interval_speed, speed, interval_ok, valid_position
    dff, valid_roi, baseline_summary = compute_dff(raw_f, fneu, fs, output_dir/"dff.npy",
        start_frame=start_frame, neuropil_coeff=neuropil_coeff, baseline_method=baseline_method,
        baseline_window_seconds=baseline_window_seconds, baseline_percentile=baseline_percentile,
        baseline_sigma_seconds=baseline_sigma_seconds, baseline_min=baseline_min, chunk_size=chunk_size)
    del raw_f, fneu
    
    print(f"Valid fluorescence baselines: {valid_roi.sum()}/{len(valid_roi)} ROIs", flush=True)
    trace = dict(
        schema_version=3, 
        suite2p_dir=str(suite2p_dir), 
        behav_file=str(behav_file),
        n_neuron=dff.shape[0], 
        parameters=parameters, 
        dFF=np.asarray(dff),
        valid_roi=valid_roi, 
        baseline_summary=baseline_summary, 
        frame_time=ft,
        session_metadata=dict(session_metadata or {}),
        original_frame_idx=original_idx, 
        frame_duration_seconds=frame_duration,
        frame_xy=frame_xy, 
        frame_heading=frame_heading, 
        frame_speed=frame_speed,
        alignment_valid=aligned, 
        behavior_sample_idx=idx+first, 
        analysis_mask=analysis_mask,
        analysis_frame_idx=analysis_idx, 
        spatial_bins=spatial_bins, 
        hd_frame_idx=hd_idx,
        hd_bins=hd_bins, 
        split_midpoint_seconds=midpoint,
        position_x_edges=np.linspace(35,65,n_bins_x+1),
        position_y_edges=np.linspace(35,65,n_bins_y+1),
        hd_bin_centers=(np.arange(n_hd_bins)+.5)*2*np.pi/n_hd_bins, 
        **metadata, 
        **neural_data
    )
    
    for label, indices, bins, shape in [("position", analysis_idx, spatial_bins, (n_bins_y,n_bins_x)),
                                      ("hd", hd_idx, hd_bins, (n_hd_bins,))]:
        result = compute_tuning(
            dff, 
            indices, 
            bins, 
            frame_duration[indices], 
            first_half[indices],
            valid_roi, 
            shape, 
            output_dir, 
            label, 
            chunk_size=chunk_size,
            min_occupancy_seconds=min_occupancy_seconds, 
            smoothing_sigma_bins=smoothing_sigma_bins,
            save_half_maps=save_half_maps
        )
        # ndarray views keep disk-backed storage while making the pickle portable.
        trace[label] = {key: np.asarray(value) for key, value in result.items()}
        
    if n_shuffles:
        shuffle_test(trace, n_shuffles=n_shuffles, min_shift_seconds=min_shuffle_shift_seconds,
                     seed=shuffle_seed, chunk_size=chunk_size, save_null=save_shuffle_null)
        
    if make_plots:
        plot_qc(trace, save_dir/"openfield_qc.png")
        
    print("Saving self-contained trace.pkl (large recordings may take several minutes)", flush=True)
    path = save_dir/"trace.pkl"
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as stream:
        pickle.dump(trace, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    print(f"Done: {path}", flush=True)
    print(ttt.strftime("%Y-%m-%d %H:%M:%S", ttt.localtime()))
    return trace


def run_OpenField2D(i, sheet_file, **kwargs):
    """Process a spreadsheet row using the original calling interface.

    Parameters
    ----------
    i : int or hashable
        DataFrame row label.
    sheet_file : pandas.DataFrame
        Must contain suite2p_dir and behav_dir; FishID/session are optional.
    **kwargs
        Keyword arguments forwarded to process_openfield.

    Returns
    -------
    trace : dict
        Saved analysis dictionary, including available FishID/session values.
    """
    row = sheet_file.loc[i]
    print(f"Row {i}: FishID={row.get('FishID', 'unknown')}, session={row.get('session', 'unknown')}")
    metadata = dict(kwargs.pop("session_metadata", {}) or {})
    metadata.update({key: row[key] for key in ["FishID", "session"] if key in row})
    return process_openfield(row["suite2p_dir"], row["behav_dir"],
                             session_metadata=metadata, **kwargs)


if __name__ == "__main__":
    trace = process_openfield(
        suite2p_dir=r"D:\EnData\Light-sheet\open field\10258\S4-OF-Post\snr filtered",
        behav_file=r"D:\EnData\Light-sheet\open field\10258\S4-OF-Post\res.16chFlt",
        speed_threshold=0.0,
        speed_window_samples=200,
        chunk_size=1024,
        baseline_method="maximin",
        baseline_window_seconds=60.0,        
        n_shuffles=1000,
        min_shuffle_shift_seconds=20.0,
        shuffle_seed=42,
        save_shuffle_null=False,
    )