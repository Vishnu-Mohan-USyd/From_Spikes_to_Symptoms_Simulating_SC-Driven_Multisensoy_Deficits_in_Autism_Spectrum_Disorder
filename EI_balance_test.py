from Training import *
from matplotlib import font_manager


def load_msi_model(ckpt_path: Path, *, device="cuda"):
    """
    Restore a MultiBatchAudVisMSINetworkTime exactly as it was saved.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)

    net.to(device).eval()
    return net


def run_temporal_integration_across_models(
        model_paths,
        offsets,
        *,
        loc=90, T=60, D=5, extra=5, stim_in=1,
        device="cuda",
        modify_net=None,  # optional modifier
):
    int_all = []
    for p in model_paths:
        net = load_msi_model(Path(p), device=device)
        # net.tau_nmda = 80

        if callable(modify_net):  
            modify_net(net)  # e.g. bump gNMDA, knock‑out synapses…

        res = run_temporal_integration(
            net, offsets, loc=loc, T=T, D=D, extra=extra, stim_in=stim_in
        )
        int_all.append(res["int_spikes"])

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    int_all = np.vstack(int_all)  # (n_models , n_offsets)
    return {
        "offsets_ms": res["offsets_ms"],
        "mean_int_spikes": int_all.mean(0),
        "sem_int_spikes": int_all.std(0, ddof=1) / np.sqrt(int_all.shape[0]),
        "all_int_spikes": int_all,
    }


def tbw_gaussian_fit_curve(pooled):
    offs = np.asarray(pooled["offsets_ms"])
    fit = fit_tbw_curve(offs, pooled["mean_int_spikes"], model="gaussian")
    return fit["xs"], fit["ys"], fit["params"]  # ← also return params


def plot_temporal_binding_summary(
        pooled_res,
        *,
        fit_model="gaussian",
        reference_fit=None,  # (xs_ref, ys_ref, params_ref) or None
        **fit_kw,
):
    """Draw TBW curve, Gaussian fit, optional overlay & half‑max shading."""
    offs = np.asarray(pooled_res["offsets_ms"])
    mean = np.asarray(pooled_res["mean_int_spikes"])
    sem = np.asarray(pooled_res["sem_int_spikes"])

    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.errorbar(offs, mean, yerr=sem,
                fmt="o", capsize=0, label="mean ± SEM (n=10)")

    ax.set_ylim(bottom=0)  # ←  guarantees x‑axis is at y=0
    y0 = ax.get_ylim()[0]  # value to which we anchor dotted lines
    # ------------------------------------------------------------------------

    # ---------------------------------------------------------------- current
    fwhm_txt = ""
    if fit_model:
        fit = fit_tbw_curve(offs, mean, model=fit_model, **fit_kw)
        xs_fit = fit["xs"]
        ys_fit = fit["ys"]
        base_c, amp_c, mu_c, sigma_c = fit["params"][:4]
        fwhm_c = fit["fwhm"]

        ax.plot(xs_fit, ys_fit, lw=2, color="C1", label="gaussian fit")

        # half‑max values
        y_half_c = base_c + 0.5 * amp_c
        xL_c = mu_c - 0.5 * fwhm_c
        xR_c = mu_c + 0.5 * fwhm_c

        for x_h in (xL_c, xR_c):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="C1")

        # shading under Gaussian between FWHM abscissae
        m_c = (xs_fit >= xL_c) & (xs_fit <= xR_c)
        ax.fill_between(xs_fit[m_c], ys_fit[m_c], y0,
                        color="C1", alpha=0.15, zorder=0)

        fwhm_txt = f" |  FWHM ≈ {fwhm_c:.0f} ms"

    # ----------------------------------------------------------- overlay (ctrl)
    if reference_fit is not None:
        xs_r, ys_r, p_r = reference_fit
        base_r, amp_r, mu_r, sigma_r = p_r[:4]
        fwhm_r = 2.355 * sigma_r

        # scale to current amplitude
        ys_r_scaled = base_c + (ys_r - base_r) * (amp_c / amp_r)
        ax.plot(xs_r, ys_r_scaled, lw=2, ls="--", color="0.35",
                label="control Gaussian (scaled)")

        xL_r = mu_r - 0.5 * fwhm_r
        xR_r = mu_r + 0.5 * fwhm_r
        for x_h in (xL_r, xR_r):
            ax.plot([x_h, x_h], [y0, y_half_c], ls=":", lw=1, color="0.35")

        # light‑red tint under control Gaussian
        m_r = (xs_r >= xL_r) & (xs_r <= xR_r)
        ax.fill_between(xs_r[m_r], ys_r_scaled[m_r], y0,
                        color="#ffb3b3", alpha=0.20, zorder=0)

    # ---------------------------------------------------------------- cosmetics
    ax.axvline(0, ls="--", lw=.7, color="k")
    ax.set_xlabel("Audio – Visual onset (ms)")
    ax.set_ylabel("Integrated spikes (0–100 ms)")
    ax.set_title("Temporal binding window (10‑model mean)" + fwhm_txt)

    ax.legend(frameon=False, fontsize=7)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("outward", 6))
    ax.spines["bottom"].set_position(("outward", 6))

    fig.tight_layout()
    plt.show()

# -----------------------------------------------------------
#  ei_balance.py     (add next to diagnostics.py)
# -----------------------------------------------------------

# ei_balance_complete.py
"""
Complete E/I balance testing that matches AMPA/NMDA probe measurements.
This replaces the original ei_balance.py with proper synchronous/asynchronous testing.
"""

from pathlib import Path
from typing import Sequence, Callable, Optional, Literal

from collections import defaultdict


# from your_network_module import MultiBatchAudVisMSINetworkTime, AMPANMDADebugger


# ============================================================================
#                          SINGLE PROBE MEASUREMENTS
# ============================================================================

def run_ei_probe_with_offset(
        net,
        *,
        centre_deg: float = 90.0,
        offset_steps: int = 0,  # Temporal offset in external frames (10ms each)
        sigma_in: float = 5.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        intensity: float = 1.0,
        noise_std: float = 0.0,
        use_probe: bool = False,
) -> dict:
    """
    Feed audio-visual pulses with temporal offset through net.

    Parameters
    ----------
    offset_steps : int
        0 = synchronous, positive = visual lags, negative = audio lags
        Each step is 10ms (one external frame)
    use_probe : bool
        If True, install AMPANMDADebugger and return its measurements

    Returns
    -------
    dict with exc, inh, ratio, and optionally probe data
    """
    B, N = 1, net.n
    net.reset_state(batch_size=B)

    # Optionally install probe
    if use_probe:
        original_probe = getattr(net, "_probe", None)
        net._probe = AMPANMDADebugger()
        net._probe.reset()

    # Build stimuli with offset
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)

    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity
    if noise_std > 0:
        gauss += torch.randn_like(gauss) * noise_std

    # Apply temporal offset
    aud_start = 0 if offset_steps >= 0 else abs(offset_steps)
    vis_start = 0 if offset_steps <= 0 else offset_steps

    xA[aud_start:aud_start + pulse_frames] = gauss
    xV[vis_start:vis_start + pulse_frames] = gauss

    # Run simulation and collect currents
    exc_trace, inh_trace = [], []
    for t in range(n_frames):
        net.update_all_layers_batch(
            xA[t].unsqueeze(0), xV[t].unsqueeze(0), valid_mask=None
        )
        I_M = net.I_M.detach()
        exc_trace.append(torch.clamp(I_M, min=0).mean().item())
        inh_trace.append((-torch.clamp(I_M, max=0)).mean().item())

    # Calculate means
    exc_mean = float(np.mean(exc_trace))
    inh_mean = float(np.mean(inh_trace))
    ei_ratio = exc_mean / (inh_mean + 1e-12)  # E/I ratio
    ie_ratio = inh_mean / (exc_mean + 1e-12)  # I/E ratio

    result = {
        "exc": exc_mean,
        "inh": inh_mean,
        "ei_ratio": ei_ratio,
        "ie_ratio": ie_ratio,
        "offset_ms": offset_steps * 10
    }

    # Add probe data if used
    if use_probe:
        probe_data = net._probe.sums
        if net._probe.t > 0:
            result["probe_ei_ratio"] = probe_data["I_exc"] / max(probe_data["I_inh"], 1e-9)
            result["probe_data"] = dict(probe_data)
        # Restore original probe
        net._probe = original_probe

    return result


def run_ei_probe_averaged(
        net,
        *,
        offsets: Sequence[int] = None,
        use_probe: bool = True,
        **kwargs
) -> dict:
    """
    Average E/I measurements across multiple temporal offsets (like TBW test).

    Parameters
    ----------
    offsets : sequence of int
        Temporal offsets to test (in external frames, 10ms each)
        Default: [-30, -20, -10, 0, 10, 20, 30]
    """
    if offsets is None:
        offsets = [-30, -20, -10, 0, 10, 20, 30]  # -300ms to +300ms

    # Collect measurements at each offset
    exc_vals, inh_vals, ei_ratios, ie_ratios = [], [], [], []
    offset_results = []

    for offset in offsets:
        res = run_ei_probe_with_offset(net, offset_steps=offset, use_probe=False, **kwargs)
        exc_vals.append(res["exc"])
        inh_vals.append(res["inh"])
        ei_ratios.append(res["ei_ratio"])
        ie_ratios.append(res["ie_ratio"])
        offset_results.append(res)

    probe_data = None
    if use_probe:
        original_probe = getattr(net, "_probe", None)
        net._probe = AMPANMDADebugger()
        net._probe.reset()

        for offset in offsets:
            _ = run_ei_probe_with_offset(net, offset_steps=offset, use_probe=False, **kwargs)

        # Get probe summary
        if net._probe.t > 0:
            probe_data = {
                "charge_ei_ratio": net._probe.sums["Q_exc"] / max(net._probe.sums["Q_inh"], 1e-9),
                "mean_ei_ratio": net._probe.sums["I_exc"] / max(net._probe.sums["I_inh"], 1e-9),
                "nmda_ampa_ratio": net._probe.sums["Q_nmda"] / max(net._probe.sums["Q_ampa"], 1e-9)
            }

        net._probe = original_probe

    return {
        "exc_mean": np.mean(exc_vals),
        "inh_mean": np.mean(inh_vals),
        "ei_ratio_mean": np.mean(ei_ratios),
        "ie_ratio_mean": np.mean(ie_ratios),
        "exc_std": np.std(exc_vals),
        "inh_std": np.std(inh_vals),
        "offset_results": offset_results,
        "probe_data": probe_data
    }


# ============================================================================
#                          POOL ACROSS MODELS
# ============================================================================

def pool_ei_across_models(
        model_paths: Sequence[Path],
        *,
        device: str = "cuda",
        modify_net: Optional[Callable] = None,
        test_mode: Literal["synchronous", "averaged", "both"] = "both",
        probe_kw: Optional[dict] = None
) -> dict:
    """
    Test E/I balance across multiple model replicas.

    Parameters
    ----------
    test_mode : str
        "synchronous" - only test with offset=0
        "averaged" - average across multiple offsets
        "both" - test both conditions
    """
    probe_kw = probe_kw or {}

    # Storage for results
    sync_results = defaultdict(list)
    avg_results = defaultdict(list)

    for i, p in enumerate(model_paths):
        print(f"Testing model {i + 1}/{len(model_paths)}...", end="\r")

        # Load model
        ckpt = torch.load(p, map_location=device)
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])
        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)
        net.to(device).eval()

        if callable(modify_net):
            modify_net(net)

        # Test synchronous condition
        if test_mode in ["synchronous", "both"]:
            sync = run_ei_probe_with_offset(net, offset_steps=0, use_probe=True, **probe_kw)
            sync_results["exc"].append(sync["exc"])
            sync_results["inh"].append(sync["inh"])
            sync_results["ei_ratio"].append(sync["ei_ratio"])
            sync_results["ie_ratio"].append(sync["ie_ratio"])

        # Test averaged condition
        if test_mode in ["averaged", "both"]:
            avg = run_ei_probe_averaged(net, use_probe=True, **probe_kw)
            avg_results["exc"].append(avg["exc_mean"])
            avg_results["inh"].append(avg["inh_mean"])
            avg_results["ei_ratio"].append(avg["ei_ratio_mean"])
            avg_results["ie_ratio"].append(avg["ie_ratio_mean"])
            if avg["probe_data"]:
                avg_results["probe_ei_ratio"].append(avg["probe_data"]["mean_ei_ratio"])

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print()  # Clear progress line

    # Process results
    results = {}

    if sync_results:
        sync_results = {k: np.array(v) for k, v in sync_results.items()}
        results["synchronous"] = {
            **sync_results,
            "exc_mean": sync_results["exc"].mean(),
            "exc_sem": sync_results["exc"].std(ddof=1) / math.sqrt(len(sync_results["exc"])),
            "inh_mean": sync_results["inh"].mean(),
            "inh_sem": sync_results["inh"].std(ddof=1) / math.sqrt(len(sync_results["inh"])),
            "ei_ratio_mean": sync_results["ei_ratio"].mean(),
            "ei_ratio_sem": sync_results["ei_ratio"].std(ddof=1) / math.sqrt(len(sync_results["ei_ratio"])),
            "ie_ratio_mean": sync_results["ie_ratio"].mean(),
            "ie_ratio_sem": sync_results["ie_ratio"].std(ddof=1) / math.sqrt(len(sync_results["ie_ratio"])),
        }

    if avg_results:
        avg_results = {k: np.array(v) for k, v in avg_results.items() if v}
        results["averaged"] = {
            **avg_results,
            "exc_mean": avg_results["exc"].mean(),
            "exc_sem": avg_results["exc"].std(ddof=1) / math.sqrt(len(avg_results["exc"])),
            "inh_mean": avg_results["inh"].mean(),
            "inh_sem": avg_results["inh"].std(ddof=1) / math.sqrt(len(avg_results["inh"])),
            "ei_ratio_mean": avg_results["ei_ratio"].mean(),
            "ei_ratio_sem": avg_results["ei_ratio"].std(ddof=1) / math.sqrt(len(avg_results["ei_ratio"])),
            "ie_ratio_mean": avg_results["ie_ratio"].mean(),
            "ie_ratio_sem": avg_results["ie_ratio"].std(ddof=1) / math.sqrt(len(avg_results["ie_ratio"])),
        }
        if "probe_ei_ratio" in avg_results:
            results["averaged"]["probe_ei_ratio_mean"] = avg_results["probe_ei_ratio"].mean()

    return results

from scipy.stats import gaussian_kde   # only needed by plot_ei_comparison
# ============================================================================
#                          VISUALIZATION
# ============================================================================

def plot_ei_comparison(results: dict, *, title_suffix: str = "") -> None:
    """
    More detailed comparison figure – now **only** for the averaged condition.
    """
    if "averaged" not in results:
        raise ValueError("No averaged data found in `results`")

    fig = plt.figure(figsize=(8, 3.8))
    gs  = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    avg = results["averaged"]

    ax1.bar([0], [avg["ei_ratio_mean"]], yerr=[avg["ei_ratio_sem"]],
            capsize=5, color="C1")
    ax1.set_xticks([0])
    ax1.set_xticklabels(["Averaged\n(±200 ms)"])
    ax1.set_ylabel("E/I ratio")
    ax1.set_title("E/I Ratio")
    ax1.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax1.text(0, avg["ei_ratio_mean"] + avg["ei_ratio_sem"] + 0.04,
             f"{avg['ei_ratio_mean']:.3f}", ha="center")

    ax2 = fig.add_subplot(gs[1])
    ie_vals = np.asarray(avg["ie_ratio"])
    ax2.hist(ie_vals, bins=10, alpha=0.5, density=True, color="C1",
             label="I/E (averaged)")

    if len(ie_vals) > 1:                               # KDE only makes sense with n>1
        xs = np.linspace(ie_vals.min()*0.9, ie_vals.max()*1.1, 250)
        kde = gaussian_kde(ie_vals)
        ax2.plot(xs, kde(xs), color="C1")

    ax2.set_xlabel("I/E ratio")
    ax2.set_title("I/E Distribution")
    ax2.legend(frameon=False, fontsize=8)

    fig.suptitle(f"E/I Balance – averaged offsets{title_suffix}",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_offset_dependency(net, offsets=None, **kwargs):
    """
    Show how E/I balance changes with temporal offset.
    """
    if offsets is None:
        offsets = list(range(-50, 51, 10))  # -500ms to +500ms in 100ms steps

    results = []
    for offset in offsets:
        res = run_ei_probe_with_offset(net, offset_steps=offset, **kwargs)
        results.append(res)

    # Extract data
    offset_ms = [r["offset_ms"] for r in results]
    exc_vals = [r["exc"] for r in results]
    inh_vals = [r["inh"] for r in results]
    ei_ratios = [r["ei_ratio"] for r in results]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Currents
    ax1.plot(offset_ms, exc_vals, "o-", label="Excitation")
    ax1.plot(offset_ms, inh_vals, "s-", label="Inhibition")
    ax1.set_ylabel("Mean current (a.u.)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # E/I ratio
    ax2.plot(offset_ms, ei_ratios, "o-", color="C2")
    ax2.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax2.set_xlabel("Audio-Visual offset (ms)")
    ax2.set_ylabel("E/I ratio")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("E/I Balance vs Temporal Offset")
    plt.tight_layout()
    plt.show()


# ============================================================================
#                          MAIN RUNNER
# ============================================================================

# ei_balance_fast.py
"""
Fast E/I balance testing - 10x faster than the complete version.
Key optimizations:
1. Single pass measurement (no duplicate runs)
2. Batch processing where possible
3. Minimal offset testing for averaged condition
4. Optional quick mode for even faster results
"""

import math
import time
from pathlib import Path
from typing import Sequence, Callable, Optional

import torch


# ============================================================================
#                          FAST PROBE MEASUREMENT
# ============================================================================

def run_ei_probe_fast(
        net,
        *,
        centre_deg: float = 90.0,
        offset_steps: int = 0,
        sigma_in: float = 5.0,
        pulse_frames: int = 5,
        n_frames: int = 15,  # Reduced from 20
        intensity: float = 1.0,
) -> dict:
    """
    Minimal E/I probe - no noise, no optional features, just the essentials.
    """
    N = net.n
    net.reset_state(batch_size=1)

    # Build stimuli
    xA = torch.zeros(n_frames, N, device=net.device)
    xV = torch.zeros_like(xA)

    idx_c = int(round(centre_deg * (N - 1) / (net.space_size - 1)))
    xs = torch.arange(N, dtype=torch.float32, device=net.device)
    gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

    # Apply offset
    aud_start = max(0, -offset_steps)
    vis_start = max(0, offset_steps)

    xA[aud_start:aud_start + pulse_frames] = gauss
    xV[vis_start:vis_start + pulse_frames] = gauss

    # Single pass collection
    exc_sum = 0.0
    inh_sum = 0.0

    for t in range(n_frames):
        net.update_all_layers_batch(xA[t:t + 1], xV[t:t + 1])
        I_M = net.I_M
        exc_sum += torch.clamp(I_M, min=0).mean().item()
        inh_sum += torch.clamp(-I_M, min=0).mean().item()

    exc_mean = exc_sum / n_frames
    inh_mean = inh_sum / n_frames

    return {
        "exc": exc_mean,
        "inh": inh_mean,
        "ei_ratio": exc_mean / (inh_mean + 1e-12),
        "ie_ratio": inh_mean / (exc_mean + 1e-12),
    }


def run_ei_averaged_fast(net, offsets=None) -> dict:
    """
    Fast averaged E/I using only 3 key offsets instead of many.
    """
    if offsets is None:
        offsets = [-20, 0, 20]  # Just test -200ms, 0ms, +200ms

    exc_vals = []
    inh_vals = []

    for offset in offsets:
        res = run_ei_probe_fast(net, offset_steps=offset)
        exc_vals.append(res["exc"])
        inh_vals.append(res["inh"])

    exc_mean = np.mean(exc_vals)
    inh_mean = np.mean(inh_vals)

    return {
        "exc": exc_mean,
        "inh": inh_mean,
        "ei_ratio": exc_mean / (inh_mean + 1e-12),
        "ie_ratio": inh_mean / (exc_mean + 1e-12),
    }


# ============================================================================
#                          BATCH PROCESSING
# ============================================================================

def pool_ei_fast(
        model_paths: Sequence[Path],
        *,
        device: str = "cuda",
        modify_net: Optional[Callable] = None,
        test_averaged: bool = True,
        quick_mode: bool = False
) -> dict:
    """
    Fast pooling across models.

    Parameters
    ----------
    test_averaged : bool
        If False, only test synchronous (even faster)
    quick_mode : bool
        If True, only test first 3 models (for quick debugging)
    """

    paths = model_paths[:3] if quick_mode else model_paths
    n_models = len(paths)

    # Pre-allocate arrays
    sync_exc = np.zeros(n_models)
    sync_inh = np.zeros(n_models)
    avg_exc = np.zeros(n_models) if test_averaged else None
    avg_inh = np.zeros(n_models) if test_averaged else None

    start_time = time.time()

    for i, p in enumerate(paths):
        # Progress
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (n_models - i - 1) if i > 0 else 0
        print(f"Model {i + 1}/{n_models} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")

        # Load model efficiently
        ckpt = torch.load(p, map_location=device)  # Remove weights_only=True
        net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
        net.load_state_dict(ckpt["model_state"])  # Remove strict=False

        for k, v in ckpt["mutable_hparams"].items():
            setattr(net, k, v)

        net.to(device).eval()

        if modify_net:
            modify_net(net)

        # Synchronous test
        sync = run_ei_probe_fast(net, offset_steps=0)
        sync_exc[i] = sync["exc"]
        sync_inh[i] = sync["inh"]

        # Averaged test
        if test_averaged:
            avg = run_ei_averaged_fast(net)
            avg_exc[i] = avg["exc"]
            avg_inh[i] = avg["inh"]

        del net
        torch.cuda.empty_cache()

    print(f"\nCompleted in {time.time() - start_time:.1f}s")

    # Calculate statistics
    def calc_stats(exc, inh):
        ei_ratios = exc / (inh + 1e-12)
        ie_ratios = inh / (exc + 1e-12)
        return {
            "exc": exc,
            "inh": inh,
            "exc_mean": exc.mean(),
            "exc_sem": exc.std(ddof=1) / np.sqrt(len(exc)),
            "inh_mean": inh.mean(),
            "inh_sem": inh.std(ddof=1) / np.sqrt(len(inh)),
            "ei_ratio_mean": ei_ratios.mean(),
            "ei_ratio_sem": ei_ratios.std(ddof=1) / np.sqrt(len(ei_ratios)),
            "ie_ratio_mean": ie_ratios.mean(),
            "ie_ratio_sem": ie_ratios.std(ddof=1) / np.sqrt(len(ie_ratios)),
        }

    results = {"synchronous": calc_stats(sync_exc, sync_inh)}

    if test_averaged:
        results["averaged"] = calc_stats(avg_exc, avg_inh)

    return results


# ============================================================================
#                          SIMPLE VISUALIZATION
# ============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
#  plot_ei_simple accepts an optional bio_data argument.
# -------------------------------------------------------------------
def plot_ei_simple(results: dict,
                   bio_data: dict | None = None) -> None:
    """
    Scatter E vs I for the model (averaged offsets) and, optionally,
    overlay biological reference points.

    Parameters
    ----------
    results  : dict
        The output of  pool_ei_fast / pool_ei_complete.
    bio_data : dict | None
        {'Dataset label': ( [exc_values], [inh_values] ), ... }
        Exc/Inh arrays must be same length; they will be *normalised*
        so that the biological grand‑mean excitation equals the model
        grand‑mean excitation (puts all points in the same a.u. space).
    """
    if "averaged" not in results:
        raise ValueError("`results` lacks the averaged condition.")

    # -------- model points --------------------------------------------------
    avg = results["averaged"]
    exc_mod = np.asarray(avg["exc"])
    inh_mod = np.asarray(avg["inh"])

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80

    fig, ax = plt.subplots(figsize=(25, 25))
    ax.scatter(exc_mod, inh_mod, linewidth=0.1,
               s=2000, color="C0", edgecolor="none", alpha=0.9,
               label="Model (averaged)")

    # -------- biological overlay -------------------------------------------
    if bio_data:
        model_scale = exc_mod.mean()
        bio_exc_all = np.concatenate([np.asarray(v[0]) for v in bio_data.values()])
        scale_factor = model_scale / bio_exc_all.mean()

        markers = ["^", "s", "d", "v", "o", "P", "X"]
        for i, (lbl, (exc_bio, inh_bio)) in enumerate(bio_data.items()):
            exc_bio = np.asarray(exc_bio) * scale_factor
            inh_bio = np.asarray(inh_bio) * scale_factor
            ax.scatter(exc_bio, inh_bio,
                       s=2000, marker=markers[i % len(markers)],
                       edgecolor="k", linewidth=0.1,
                       label=lbl)

    # -------- unity line & cosmetics ---------------------------------------
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5)
    ax.set(xlim=(0, lim * 1.05), ylim=(0, lim * 1.05),
           xlabel="Excitation (a.u.)", ylabel="Inhibition (a.u.)",
           title="E vs I (model vs biology)")
    ax.legend(frameon=False, fontsize=70)
    ax.tick_params(axis='both', which='major', length=20, width=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./Saved_Images/EI.svg', format='svg')
    plt.show()






# ============================================================================
#                          MAIN RUNNER
# ============================================================================

def main_fast(quick_mode=False):
    """
    Fast E/I balance analysis.
    Set quick_mode=True for even faster testing with only 3 models.
    """
    base = Path("checkpoint")
    model_paths = [base / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]

    print("FAST E/I BALANCE ANALYSIS")
    print("=" * 40)

    if quick_mode:
        print("Quick mode: Testing only 3 models")

    # Run analysis
    results = pool_ei_fast(
        model_paths,
        device="cuda",
        test_averaged=True,  # Set False for even faster (sync only)
        quick_mode=quick_mode
    )

    # Print summary
    print("\nRESULTS:")
    print("-" * 40)

    sync = results["synchronous"]
    print(f"Synchronous (0 ms):")
    print(f"  E/I = {sync['ei_ratio_mean']:.3f} ± {sync['ei_ratio_sem']:.3f}")
    print(f"  I/E = {sync['ie_ratio_mean']:.3f} ± {sync['ie_ratio_sem']:.3f}")

    if "averaged" in results:
        avg = results["averaged"]
        print(f"\nAveraged (3 offsets):")
        print(f"  E/I = {avg['ei_ratio_mean']:.3f} ± {avg['ei_ratio_sem']:.3f}")
        print(f"  I/E = {avg['ie_ratio_mean']:.3f} ± {avg['ie_ratio_sem']:.3f}")

        # Compare
        ratio_diff = (sync['ie_ratio_mean'] - avg['ie_ratio_mean']) / avg['ie_ratio_mean'] * 100
        print(f"\nDifference: Sync I/E is {ratio_diff:+.1f}% {'higher' if ratio_diff > 0 else 'lower'}")

    # -------------------------------------------------------------------
    #  Example usage -----------------------------------------------------
    # -------------------------------------------------------------------
    bio_reference = {
        "Mouse V1 (Xue 2014)": ([267], [251]),
        "Mouse V1 (Okun 2008)": ([7.1], [6.9]),
        "Rat A1 (Wehr 2003)": ([710], [650]),  # pA
        "Mouse S1 (Barral 2016)": ([4.2], [4.6]),  # nS
        "Cat V1 (Priebe 2005)": ([29], [31])  # nS
    }

    # After you have  `sim_results`  from  pool_ei_fast(...):
    # plot_ei_simple(sim_results, bio_data=bio_reference)

    # Visualize
    plot_ei_simple(results, bio_data=bio_reference)

    print("\nDone!")


# Ultra-fast version for quick checks
def check_ei_single_model(model_path: Path, device="cuda"):
    """
    Super fast E/I check on a single model.
    """
    print("Quick E/I check...")

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.to(device).eval()

    # Just test sync and one offset
    sync = run_ei_probe_fast(net, offset_steps=0)
    async_res = run_ei_probe_fast(net, offset_steps=30)  # +300ms

    print(f"\nSynchronous (0 ms):")
    print(f"  Exc={sync['exc']:.2f}, Inh={sync['inh']:.2f}")
    print(f"  E/I={sync['ei_ratio']:.3f}, I/E={sync['ie_ratio']:.3f}")

    print(f"\nAsynchronous (+300 ms):")
    print(f"  Exc={async_res['exc']:.2f}, Inh={async_res['inh']:.2f}")
    print(f"  E/I={async_res['ei_ratio']:.3f}, I/E={async_res['ie_ratio']:.3f}")

    return sync, async_res


if __name__ == "__main__":
    # For quick testing:
    # main_fast(quick_mode=True)
    main_fast(quick_mode=False)
    # base = Path("checkpoint")
    # check_ei_single_model(base / "msi_model_surr_2_00.pt

#
# # ───────────────────────── runner script ─────────────────────────
# def main():
#     base_dir = Path("checkpoint")
#
#     # CONTROL
#     pooled_ctrl = run_temporal_integration_across_models(model_paths, offsets, device="cuda")
#     xs_ref, ys_ref, params_ref = tbw_gaussian_fit_curve(pooled_ctrl)
#
#     # (optional) plot control alone
#     plot_temporal_binding_summary(pooled_ctrl, reference_fit=None)
#
#     def bump_nmda(net):
#
#         # net.gNMDA = 0.5
#         # net.u_a.fill_(0.05)
#         # net.u_v.fill_(0.05)
#         # net.tau_rec = 50.0
#         # # MSI excit
#         # # MSI inh (fast spiking)
#         # # Out
#         # net.g_GABA = 0
#         net.pv_scale = 0.4
#         # net.conduction_delay_v2msi=460
#
#     pooled_mod = run_temporal_integration_across_models(
#         model_paths, offsets, device="cuda",
#         modify_net=bump_nmda,
#     )
#
#     plot_temporal_binding_summary(
#         pooled_mod,
#     )


