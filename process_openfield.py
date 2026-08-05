"""Open-field spatial tuning analysis for Suite2p + .16chFlt recordings.

The pipeline is intentionally staged:
1. import and align data;
2. compute and save 2-D tuning maps immediately;
3. compute and save spatial metrics;
4. optionally compute temporal-shuffle null distributions.

Designed for paradigm 20251102 and meaningful x/y coordinates in [35, 65].
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import sparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from zfish._io import import_suite2p, import16chFlt
from mazepy.basic.conversion import coordinate_recording_time


EXPECTED_PARADIGM = 20251102
POSITION_RANGE = (35.0, 65.0)
N_BINS_X = 15
N_BINS_Y = 15


def _atomic_pickle_dump(obj: object, path: str | Path) -> None:
    """Write a pickle atomically to reduce the chance of a corrupted checkpoint."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _save_trace(trace: dict, save_dir: str | Path, stage: str) -> Path:
    """Checkpoint trace.pkl and record the completed stage."""
    save_dir = Path(save_dir)
    trace["processing_stage"] = stage
    trace["last_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = save_dir / "trace.pkl"
    _atomic_pickle_dump(trace, path)
    print(f"Saved checkpoint ({stage}): {path}")
    return path


def compute_nonnegative_dff(
    raw_f: np.ndarray,
    fneu: np.ndarray,
    neuropil_coeff: float = 0.7,
    baseline_percentile: float = 20.0,
    cell_chunk_size: int = 2048,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute nonnegative dF/F from Suite2p fluorescence.

    F_corrected = F - neuropil_coeff * Fneu
    F0 is the temporal percentile of F_corrected for each ROI.
    dF/F values below zero are rectified to zero.

    Notes
    -----
    Rectification is used because downstream spatial-information calculations
    require nonnegative responses. The unrectified corrected fluorescence and
    baseline are not retained to avoid duplicating very large arrays.
    """
    raw_f = np.asarray(raw_f, dtype=np.float32)
    fneu = np.asarray(fneu, dtype=np.float32)
    if raw_f.shape != fneu.shape:
        raise ValueError(f"F and Fneu shape mismatch: {raw_f.shape} vs {fneu.shape}")

    dff = np.empty_like(raw_f, dtype=np.float32)
    n_cells = raw_f.shape[0]

    for start in tqdm(range(0, n_cells, cell_chunk_size), desc="Computing dF/F"):
        stop = min(start + cell_chunk_size, n_cells)
        corrected = raw_f[start:stop] - neuropil_coeff * fneu[start:stop]
        f0 = np.nanpercentile(corrected, baseline_percentile, axis=1).astype(np.float32)

        # A nonpositive F0 makes a fluorescence ratio undefined. Use a small,
        # data-scaled positive floor rather than dividing by zero/negative values.
        positive = corrected[corrected > 0]
        data_floor = np.nanpercentile(positive, 1) if positive.size else eps
        f0 = np.maximum(f0, max(float(data_floor), eps))

        block = (corrected - f0[:, None]) / f0[:, None]
        np.maximum(block, 0.0, out=block)
        block[~np.isfinite(block)] = 0.0
        dff[start:stop] = block.astype(np.float32, copy=False)

    return dff


def positions_to_nodes(
    xpos: np.ndarray,
    ypos: np.ndarray,
    position_range: tuple[float, float] = POSITION_RANGE,
    n_bins_x: int = N_BINS_X,
    n_bins_y: int = N_BINS_Y,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select valid open-field samples and convert x/y positions to 2-D bins."""
    xpos = np.asarray(xpos, dtype=np.float64)
    ypos = np.asarray(ypos, dtype=np.float64)
    lo, hi = position_range

    valid = np.where(
        np.isfinite(xpos)
        & np.isfinite(ypos)
        & (xpos >= lo)
        & (xpos <= hi)
        & (ypos >= lo)
        & (ypos <= hi)
    )[0]
    if valid.size == 0:
        raise ValueError(f"No x/y samples fall inside [{lo}, {hi}] on both axes.")

    # Coordinates 35..65 span 30 units. Values exactly at 65 are assigned to
    # the last bin rather than producing index 30.
    xbin = np.floor((xpos[valid] - lo) / (hi - lo) * n_bins_x).astype(np.int32)
    ybin = np.floor((ypos[valid] - lo) / (hi - lo) * n_bins_y).astype(np.int32)
    xbin = np.clip(xbin, 0, n_bins_x - 1)
    ybin = np.clip(ybin, 0, n_bins_y - 1)
    nodes = ybin * n_bins_x + xbin
    return valid, nodes.astype(np.int32), np.column_stack([xbin, ybin]).astype(np.int16)


def _node_design_matrix(nodes: np.ndarray, n_spatial_bins: int) -> sparse.csr_matrix:
    rows = np.arange(nodes.size, dtype=np.int64)
    data = np.ones(nodes.size, dtype=np.float32)
    return sparse.csr_matrix((data, (rows, nodes)), shape=(nodes.size, n_spatial_bins))


def compute_2d_response_map(
    activity: np.ndarray,
    nodes: np.ndarray,
    n_bins_x: int = N_BINS_X,
    n_bins_y: int = N_BINS_Y,
    min_occupancy_samples: int = 1,
    cell_chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean activity in each spatial bin using sparse matrix products."""
    activity = np.asarray(activity, dtype=np.float32)
    nodes = np.asarray(nodes, dtype=np.int32)
    if activity.ndim != 2 or activity.shape[1] != nodes.size:
        raise ValueError("activity must be (n_cells, n_time) and match nodes length")

    n_spatial_bins = n_bins_x * n_bins_y
    design = _node_design_matrix(nodes, n_spatial_bins)
    occupancy = np.asarray(design.sum(axis=0)).ravel().astype(np.int64)

    maps = np.full((activity.shape[0], n_spatial_bins), np.nan, dtype=np.float32)
    valid_bins = occupancy >= min_occupancy_samples

    for start in tqdm(range(0, activity.shape[0], cell_chunk_size), desc="2-D tuning maps"):
        stop = min(start + cell_chunk_size, activity.shape[0])
        sums = np.asarray(design.T @ activity[start:stop].T, dtype=np.float32).T
        block = np.full_like(sums, np.nan, dtype=np.float32)
        block[:, valid_bins] = sums[:, valid_bins] / occupancy[valid_bins][None, :]
        maps[start:stop] = block

    return maps.reshape(activity.shape[0], n_bins_y, n_bins_x), occupancy.reshape(n_bins_y, n_bins_x)


def smooth_2d_maps(
    maps: np.ndarray,
    sigma_bins: float = 1.0
) -> np.ndarray:
    """Occupancy-mask-normalized Gaussian smoothing without filling unvisited bins."""
    maps = np.asarray(maps, dtype=np.float32).reshape(maps.shape[0], N_BINS_Y * N_BINS_X)
    maps[np.isnan(maps)] = 0
    
    y_bin, x_bin = np.meshgrid(
        np.arange(N_BINS_Y),
        np.arange(N_BINS_X),
        indexing="ij",
    )

    coords = np.column_stack([
        x_bin.ravel(),
        y_bin.ravel(),
    ]).astype(np.float32)
    displacement = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt(
        np.sum(displacement**2, axis=2)
    )
    
    to_center_displacement = (coords - np.array([[ (N_BINS_X-1) / 2, (N_BINS_Y-1) / 2 ]]))
    to_center_dist = np.sqrt(np.sum(to_center_displacement**2, axis=1))
    mask = to_center_dist > (N_BINS_X+1)/2
    
    kernel = np.exp(-0.5 * (dist_mat / sigma_bins) ** 2)
    for j in range(kernel.shape[1]):
        kernel[:, j] /= np.sum(kernel[:, j])
        
    smoothed = maps @ kernel
    smoothed[:, mask] = np.nan
    
    return smoothed

def smooth_1dloop_maps(
    maps: np.ndarray,
    sigma_bins: float = 1.0
) -> np.ndarray:
    """Occupancy-mask-normalized Gaussian smoothing for 1-D head-direction maps."""
    maps = np.asarray(maps, dtype=np.float32)
    maps[np.isnan(maps)] = 0
    n_bins = maps.shape[1]
    dist = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(n_bins):
        for j in range(n_bins):
            d = abs(i - j)
            dist[i, j] = min(d, n_bins - d)
    kernel = np.exp(-0.5 * (dist / sigma_bins) ** 2)
    for j in range(kernel.shape[1]):
        kernel[:, j] /= np.sum(kernel[:, j])

    smoothed = maps @ kernel
    return smoothed


def split_half_correlation(first_maps: np.ndarray, second_maps: np.ndarray) -> np.ndarray:
    """Per-cell Pearson correlation across bins visited in both halves."""
    n_cells = first_maps.shape[0]
    corr = np.full(n_cells, np.nan, dtype=np.float32)
    a = first_maps.reshape(n_cells, -1)
    b = second_maps.reshape(n_cells, -1)

    for i in tqdm(range(n_cells), desc="Split-half reliability"):
        valid = np.isfinite(a[i]) & np.isfinite(b[i])
        if valid.sum() < 3:
            continue
        av = a[i, valid]
        bv = b[i, valid]
        if np.std(av) == 0 or np.std(bv) == 0:
            continue
        corr[i] = np.corrcoef(av, bv)[0, 1]
    return corr


def spatial_information(
    rate_maps: np.ndarray,
    occupancy: np.ndarray,
) -> np.ndarray:
    """Compute occupancy-weighted spatial information for nonnegative responses.

    This uses the standard information form sum_i p_i * (r_i/r_bar) *
    log2(r_i/r_bar). For calcium fluorescence, the result is best interpreted
    as bits per unit mean fluorescence response rather than bits per spike.
    """
    flat_maps = np.asarray(rate_maps, dtype=np.float64).reshape(rate_maps.shape[0], -1)
    occ = np.asarray(occupancy, dtype=np.float64).ravel()
    visited = (occ > 0) & np.any(np.isfinite(flat_maps), axis=0)
    p = occ[visited] / np.sum(occ[visited])
    response = flat_maps[:, visited]

    if np.nanmin(response) < 0:
        raise ValueError("Spatial information requires nonnegative response maps.")

    mean_response = np.nansum(response * p[None, :], axis=1)
    ratio = np.divide(
        response,
        mean_response[:, None],
        out=np.zeros_like(response),
        where=mean_response[:, None] > 0,
    )
    term = np.zeros_like(ratio)
    positive = ratio > 0
    term[positive] = ratio[positive] * np.log2(ratio[positive])
    return np.nansum(term * p[None, :], axis=1).astype(np.float32)


def compute_temporal_shuffle_distributions(
    activity: np.ndarray,
    nodes: np.ndarray,
    occupancy: np.ndarray,
    output_dir: str | Path,
    n_shuffles: int = 1000,
    min_shift_seconds: float = 10.0,
    fs: float = 5.0,
    cell_chunk_size: int = 512,
    seed: int = 42,
) -> dict:
    """Generate temporal-shuffle null distributions for PTP and spatial information.

    A single random circular displacement is applied to the continuous neural
    activity relative to the behavioral trajectory for each shuffle. Raw null
    arrays are stored as .npy memmaps to avoid inflating trace.pkl.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_cells, n_time = activity.shape
    n_bins_y, n_bins_x = occupancy.shape
    n_spatial_bins = n_bins_x * n_bins_y
    design = _node_design_matrix(nodes, n_spatial_bins)
    occ_flat = occupancy.ravel().astype(np.int64)
    valid_bins = occ_flat > 0
    p = occ_flat[valid_bins] / occ_flat[valid_bins].sum()

    min_shift = max(1, int(round(min_shift_seconds * fs)))
    if n_time <= 2 * min_shift:
        raise ValueError("Recording is too short for the requested minimum shuffle displacement.")

    ptp_path = output_dir / "shuffle_ptp.npy"
    si_path = output_dir / "shuffle_spatial_information.npy"
    shifts_path = output_dir / "shuffle_temporal_offsets.npy"
    ptp_null = np.lib.format.open_memmap(ptp_path, mode="w+", dtype=np.float32, shape=(n_cells, n_shuffles))
    si_null = np.lib.format.open_memmap(si_path, mode="w+", dtype=np.float32, shape=(n_cells, n_shuffles))

    rng = np.random.default_rng(seed)
    allowed = np.arange(min_shift, n_time - min_shift + 1, dtype=np.int64)
    shifts = rng.choice(allowed, size=n_shuffles, replace=True)
    np.save(shifts_path, shifts)

    for start in tqdm(range(0, n_cells, cell_chunk_size), desc="Shuffle cell chunks"):
        stop = min(start + cell_chunk_size, n_cells)
        block = np.asarray(activity[start:stop], dtype=np.float32)

        for s, shift in enumerate(shifts):
            shifted = np.roll(block, int(shift), axis=1)
            sums = np.asarray(design.T @ shifted.T, dtype=np.float32).T
            maps = sums[:, valid_bins] / occ_flat[valid_bins][None, :]
            ptp_null[start:stop, s] = np.nanmax(maps, axis=1) - np.nanmin(maps, axis=1)

            mean_response = np.sum(maps * p[None, :], axis=1)
            ratio = np.divide(
                maps,
                mean_response[:, None],
                out=np.zeros_like(maps),
                where=mean_response[:, None] > 0,
            )
            term = np.zeros_like(ratio)
            positive = ratio > 0
            term[positive] = ratio[positive] * np.log2(ratio[positive])
            si_null[start:stop, s] = np.sum(term * p[None, :], axis=1)

        ptp_null.flush()
        si_null.flush()

    return {
        "shuffle_ptp_path": str(ptp_path),
        "shuffle_spatial_information_path": str(si_path),
        "shuffle_offsets_path": str(shifts_path),
        "n_shuffles": int(n_shuffles),
        "min_shift_seconds": float(min_shift_seconds),
        "shuffle_seed": int(seed),
    }


def _save_figure(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_processing_qc(trace: dict, figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(trace["ms_pos_x"], trace["ms_pos_y"], linewidth=0.35)
    ax.set_xlim(*trace["position_range"])
    ax.set_ylim(*trace["position_range"])
    ax.set_aspect("equal")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_title("Valid open-field trajectory")
    _save_figure(fig, figure_dir / "trajectory_qc")

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(trace["occupancy_seconds"], origin="lower", interpolation="nearest")
    ax.set_xlabel("x bin")
    ax.set_ylabel("y bin")
    ax.set_title("Occupancy (s)")
    fig.colorbar(image, ax=ax, label="seconds")
    _save_figure(fig, figure_dir / "occupancy_map")


def plot_example_cells(
    trace: dict,
    figure_dir: str | Path,
    n_examples: int = 20,
) -> None:
    """Plot cells ranked by split-half reliability; no spatial-cell cutoff is imposed."""
    figure_dir = Path(figure_dir) / "example_cells"
    figure_dir.mkdir(parents=True, exist_ok=True)
    reliability = np.asarray(trace["split_half_correlation"])
    order = np.argsort(np.nan_to_num(reliability, nan=-np.inf))[::-1][:n_examples]

    for cell in order:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        titles = ["All", "First half", "Second half"]
        maps = [trace["smooth_map_all"][cell], trace["smooth_map_first"][cell], trace["smooth_map_second"][cell]]
        vmax = np.nanmax(maps[0])
        for ax, title, rate_map in zip(axes, titles, maps):
            image = ax.imshow(rate_map.reshape((N_BINS_X, N_BINS_Y)), origin="lower", interpolation="nearest", vmin=0, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel("x bin")
            ax.set_ylabel("y bin")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"ROI {cell} | split-half r={trace['split_half_correlation'][cell]:.3f} | "
            f"SI={trace['spatial_information_smooth'][cell]:.3f}"
        )
        _save_figure(fig, figure_dir / f"roi_{cell:05d}")


def run_OpenField2D(
    i: int,
    sheet_file: pd.DataFrame,
    ds_behav_to: int = 50,
    position_range: tuple[float, float] = POSITION_RANGE,
    n_bins_x: int = N_BINS_X,
    n_bins_y: int = N_BINS_Y,
    min_occupancy_seconds: float = 0.2,
    smoothing_sigma_bins: float = 0.5,
    baseline_percentile: float = 20.0,
    neuropil_coeff: float = 0.7,
    run_shuffle: bool = False,
    n_shuffles: int = 1000,
    min_shuffle_shift_seconds: float = 10.0,
    n_example_cells: int = 20,
    n_direction_bins = 36
) -> None:
    """Process one continuous 2-D open-field recording."""
    suite2p_dir = sheet_file.loc[i, "suite2p_dir"]
    behav_file = sheet_file.loc[i, "behav_dir"]
    save_dir = Path(os.path.dirname(behav_file))
    figure_dir = save_dir / "process" / "figures"
    shuffle_dir = save_dir / "process" / "shuffle"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"{i}, Fish ID: {sheet_file.loc[i, 'FishID']}, session: {sheet_file.loc[i, 'session']}")
    print("1. Import Suite2p and behavioral data")
    trace = import_suite2p(suite2p_dir)
    res = import16chFlt(behav_file, nchannel=21)

    paradigm_values = np.unique(res["Paradigm"])
    if paradigm_values.size != 1 or int(paradigm_values[0]) != EXPECTED_PARADIGM:
        raise ValueError(
            f"Expected paradigm {EXPECTED_PARADIGM}, but found {paradigm_values.tolist()}. "
            "This file should not be processed as the 2-D open-field assay."
        )

    trace.update(
        {
            "FishID": sheet_file.loc[i, "FishID"],
            "session": sheet_file.loc[i, "session"],
            "save_dir": str(save_dir),
            "paradigm": int(paradigm_values[0]),
            "position_range": tuple(map(float, position_range)),
            "n_bins_x": int(n_bins_x),
            "n_bins_y": int(n_bins_y),
            "dff_method": {
                "source": "F - neuropil_coeff * Fneu",
                "neuropil_coeff": float(neuropil_coeff),
                "baseline": f"per-ROI temporal {baseline_percentile}th percentile",
                "rectified_nonnegative": True,
            },
        }
    )
    tiff.imwrite(save_dir / "mean_image.tif", trace["meanImg"])

    print("2. Compute nonnegative dF/F")
    trace["dFF"] = compute_nonnegative_dff(
        trace["RawTraces"],
        trace["Fneu"],
        neuropil_coeff=neuropil_coeff,
        baseline_percentile=baseline_percentile,
    )
    # Preserve Suite2p deconvolution as imported, but do not use it for dF/F.
    trace["DeconvSignal"] = np.maximum(trace["DeconvSignal"], 0).astype(np.float32)
    _save_trace(trace, save_dir, "nonnegative_dff")

    print("3. Downsample behavior and retain only x/y in [35, 65]")
    if 6000 % ds_behav_to != 0:
        raise ValueError("ds_behav_to must divide the 6000-Hz behavioral sampling rate exactly.")
    downsample_factor = 6000 // ds_behav_to
    for key in list(res.keys()):
        res[key] = res[key][::downsample_factor]

    behav_time_ms = (res["behav_time"] * 1000).astype(np.int64)

    # First align every imaging frame to the full behavioral stream. Only then
    # remove frames outside the open field; otherwise excluded intervals would
    # be assigned to the nearest valid position.
    ms_time_full = np.arange(trace["dFF"].shape[1], dtype=np.float64) / float(trace["fs"]) * 1000.0
    ms_time_full = ms_time_full.astype(np.int64)
    coord_idx_full = coordinate_recording_time(ms_time_full, behav_time_ms)
    aligned_x_full = res["behav_pos_x"][coord_idx_full]
    aligned_y_full = res["behav_pos_y"][coord_idx_full]
    aligned_orient_full = res["behav_orient"][coord_idx_full]
    
    lo, hi = position_range
    analysis_frame_idx = np.where(
        np.isfinite(aligned_x_full)
        & np.isfinite(aligned_y_full)
        & (aligned_x_full >= lo)
        & (aligned_x_full <= hi)
        & (aligned_y_full >= lo)
        & (aligned_y_full <= hi)
    )[0]
    if analysis_frame_idx.size == 0:
        raise ValueError("No imaging frames align to positions inside the open field.")

    _, analysis_nodes, analysis_xy_bins = positions_to_nodes(
        aligned_x_full[analysis_frame_idx],
        aligned_y_full[analysis_frame_idx],
        position_range=position_range,
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
    )
    analysis_activity = trace["dFF"][:, analysis_frame_idx]

    trace["analysis_frame_idx"] = analysis_frame_idx.astype(np.int64)
    trace["ms_time"] = ms_time_full[analysis_frame_idx]
    trace["ms_pos_x"] = aligned_x_full[analysis_frame_idx].astype(np.float32)
    trace["ms_pos_y"] = aligned_y_full[analysis_frame_idx].astype(np.float32)
    trace["ms_orient"] = ms_orient = aligned_orient_full[analysis_frame_idx]
    trace['ms_map'] = res["map"][coord_idx_full][analysis_frame_idx]
    trace['behav_to_ms_idx'] = coord_idx_full[analysis_frame_idx]
    trace["spike_nodes"] = analysis_nodes.astype(np.int32)
    trace["ms_xy_bins"] = analysis_xy_bins
    trace["n_neuron"] = int(trace["dFF"].shape[0])

    print("4. Compute 30 x 30 tuning maps and save immediately")
    min_occ_samples = max(1, int(np.ceil(min_occupancy_seconds * trace["fs"])))
    rate_map_all, occupancy_samples = compute_2d_response_map(
        analysis_activity,
        trace["spike_nodes"],
        n_bins_x=n_bins_x,
        n_bins_y=n_bins_y,
        min_occupancy_samples=min_occ_samples,
    )
    trace["rate_map_all"] = rate_map_all
    trace["occupancy_samples"] = occupancy_samples
    trace["occupancy_seconds"] = occupancy_samples.astype(np.float32) / float(trace["fs"])
    trace["min_occupancy_seconds"] = float(min_occupancy_seconds)
    trace["smooth_map_all"] = smooth_2d_maps(rate_map_all, sigma_bins=smoothing_sigma_bins)

    midpoint = analysis_activity.shape[1] // 2
    first_map, first_occ = compute_2d_response_map(
        analysis_activity[:, :midpoint], trace["spike_nodes"][:midpoint], n_bins_x, n_bins_y, min_occ_samples
    )
    second_map, second_occ = compute_2d_response_map(
        analysis_activity[:, midpoint:], trace["spike_nodes"][midpoint:], n_bins_x, n_bins_y, min_occ_samples
    )
    trace["rate_map_first"] = first_map
    trace["rate_map_second"] = second_map
    trace["occupancy_first_samples"] = first_occ
    trace["occupancy_second_samples"] = second_occ
    trace["smooth_map_first"] = smooth_2d_maps(first_map, sigma_bins=smoothing_sigma_bins)
    trace["smooth_map_second"] = smooth_2d_maps(second_map, sigma_bins=smoothing_sigma_bins)
    _save_trace(trace, save_dir, "tuning_maps_complete")

    print("5. Compute spatial metrics and save")
    trace["split_half_correlation"] = split_half_correlation(
        trace["smooth_map_first"], trace["smooth_map_second"]
    )
    trace["spatial_information_raw"] = spatial_information(
        trace["rate_map_all"], trace["occupancy_samples"]
    )
    trace["spatial_information_smooth"] = spatial_information(
        trace["smooth_map_all"], trace["occupancy_samples"]
    )
    flat_map = trace["rate_map_all"].reshape(trace["n_neuron"], -1)
    trace["peak_response"] = np.nanmax(flat_map, axis=1).astype(np.float32)
    trace["ptp_response"] = (np.nanmax(flat_map, axis=1) - np.nanmin(flat_map, axis=1)).astype(np.float32)
    trace["preferred_bin"] = np.nanargmax(np.nan_to_num(flat_map, nan=-np.inf), axis=1).astype(np.int32)
    trace["preferred_y_bin"] = (trace["preferred_bin"] // n_bins_x).astype(np.int16)
    trace["preferred_x_bin"] = (trace["preferred_bin"] % n_bins_x).astype(np.int16)
    _save_trace(trace, save_dir, "spatial_metrics_complete")

    print("6. Save QC figures as PNG and SVG")
    plot_processing_qc(trace, figure_dir)
    plot_example_cells(trace, figure_dir, n_examples=n_example_cells)
    trace["figure_dir"] = str(figure_dir)
    _save_trace(trace, save_dir, "figures_complete")

    if run_shuffle:
        print("7. Compute temporal-shuffle null distributions")
        shuffle_metadata = compute_temporal_shuffle_distributions(
            analysis_activity,
            trace["spike_nodes"],
            trace["occupancy_samples"],
            output_dir=shuffle_dir,
            n_shuffles=n_shuffles,
            min_shift_seconds=min_shuffle_shift_seconds,
            fs=float(trace["fs"]),
        )
        trace.update(shuffle_metadata)

        # Calculate post-hoc-ready empirical p-values while retaining all raw null values.
        ptp_null = np.load(trace["shuffle_ptp_path"], mmap_mode="r")
        si_null = np.load(trace["shuffle_spatial_information_path"], mmap_mode="r")
        trace["ptp_shuffle_p"] = (
            (np.sum(ptp_null >= trace["ptp_response"][:, None], axis=1) + 1) / (n_shuffles + 1)
        ).astype(np.float32)
        trace["spatial_information_shuffle_p"] = (
            (np.sum(si_null >= trace["spatial_information_raw"][:, None], axis=1) + 1) / (n_shuffles + 1)
        ).astype(np.float32)
        trace["ptp_shuffle_95"] = np.percentile(ptp_null, 95, axis=1).astype(np.float32)
        trace["spatial_information_shuffle_95"] = np.percentile(si_null, 95, axis=1).astype(np.float32)
        _save_trace(trace, save_dir, "shuffle_complete")

    print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Head-direction analysis
    theta = np.mod(ms_orient, 2 * np.pi)

    direction_bin = np.floor(
        theta / (2 * np.pi) * n_direction_bins
    ).astype(np.int32)

    direction_bin = np.clip(
        direction_bin,
        0,
        n_direction_bins - 1,
    )
    n_neuron = trace["n_neuron"]
    direction_map = np.zeros((n_neuron, n_direction_bins))
    for j in tqdm(range(n_direction_bins)):
        binidx = direction_bin == j
        direction_map[:, j] = np.nanmean(
            trace["dFF"][:, binidx], axis=1
        )
    trace["direction_map_raw"] = direction_map
    trace["direction_map_smooth"] = smooth_2d_maps(direction_map, sigma_bins=smoothing_sigma_bins)
    _save_trace(trace, save_dir, "direction_maps_complete")


if __name__ == "__main__":
    info = {
        "FishID": ["10220"],
        "session": [2],
        "suite2p_dir": [r"D:\EnData\Light-sheet\10220\snr filtered"],
        "behav_dir": [r"D:\EnData\Light-sheet\10220\S2\res.16chFlt"],
    }
    sheet_file = pd.DataFrame(info)
    for session_idx in range(len(sheet_file)):
        run_OpenField2D(
            session_idx,
            sheet_file,
            run_shuffle=True,  # Run tuning maps first; enable later for costly null tests.
            n_shuffles=1000,
        )