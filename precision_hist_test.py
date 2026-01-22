from Training import *
from matplotlib import font_manager


def _signed_errors_hybrid_fast(net, xA, xV, valid, loc_seqs,
                               integrate: bool = True,
                               use_msi_raw: bool = True,
                               batch_size: int = 1024):
    """Fast version with larger batches and less memory usage."""
    Ntotal, Tmax, N = xA.size(0), xA.size(1), net.n
    signed_errs = []

    for start in range(0, Ntotal, batch_size):
        end = min(start + batch_size, Ntotal)
        nB = end - start

        net.reset_state(nB)

        # Only store spikes for relevant frames
        event_frames = {}
        for i in range(nB):
            loc_seq = loc_seqs[start + i]
            pulse_ts = [t for t, v in enumerate(loc_seq) if v != 999]
            if pulse_ts:
                event_frames[i] = (max(pulse_ts[-1] - 9, 0), pulse_ts[-1])

        msi_accum = {}

        for t in range(Tmax):
            # Only store if needed
            store_frame = any(beg <= t <= end for beg, end in event_frames.values())

            *_, sSum = net.update_all_layers_batch(
                xA[start:end, t],
                xV[start:end, t],
                valid[start:end, t],
                return_spike_sum=integrate)

            if store_frame:
                if integrate:
                    for i, (beg, end_t) in event_frames.items():
                        if beg <= t <= end_t:
                            if i not in msi_accum:
                                msi_accum[i] = torch.zeros(N, device=net.device)
                            msi_accum[i] += sSum[i]
                else:
                    for i, (_, end_t) in event_frames.items():
                        if t == end_t:
                            msi_accum[i] = net._latest_sMSI[i].clone()

        # Decode
        for i, (beg_t, end_t) in event_frames.items():
            if i not in msi_accum:
                continue

            msi_vec = msi_accum[i]

            if use_msi_raw:
                # centre‑of‑mass of the MSI spike pattern
                pred_deg = decode_msi_location(
                    msi_vec.unsqueeze(0),  # shape (1 , n)
                    space_size=net.space_size,
                    method="com"  # ← main change
                )[0].item()
            else:
                logits = torch.clamp(torch.matmul(net.W_msi2out, msi_vec) + net.b_out, min=0)
                pred_deg = decode_msi_location(
                    logits.unsqueeze(0),
                    space_size=net.space_size,
                    method="com"
                )[0].item()
            # -----------------------------------------------------------

            true_deg = loc_seqs[start + i][end_t]

            diff = pred_deg - true_deg
            if diff > 90:  diff -= 180
            if diff < -90: diff += 180
            signed_errs.append(diff)

    return signed_errs


def generate_fixed_duration_event_sequences(space_size: int = 180,
                                            T: int = 30,
                                            D: int = 10,
                                            n_per_loc: int = 20):
    """
    One D-frame bimodal pulse centred at every integer azimuth (0…space_size-1),
    repeated *n_per_loc* times.  Returns the usual lists consumed by
    `generate_av_batch_tensor`.
    """
    loc_seqs, mod_seqs, offs, lens = [], [], [], []
    for loc in range(space_size):
        for _ in range(n_per_loc):
            start = np.random.randint(0, T - D + 1)
            loc_seq = [999] * T
            mod_seq = ['X'] * T
            for t in range(start, start + D):
                loc_seq[t] = loc
                mod_seq[t] = 'B'  # bimodal by default
            loc_seqs.append(loc_seq)
            mod_seqs.append(mod_seq)
            offs.append(False)
            lens.append(T)
    return loc_seqs, mod_seqs, offs, lens


def check_msi_enhancement_hybrid(net, *,
                                 space_size: int = 180,
                                 T: int = 30, D: int = 10, n_per_loc: int = 20,
                                 integrate: bool = True,
                                 use_msi_raw: bool = True,
                                 stimulus_intensity: float = 0.1,  # return delayed
                                 noise_std: float = 0.1):  # return delayed
    """
    Same as before but lets you dial stimulus strength & noise.
    Set intensity low (≈0.1–0.3) or raise noise to uncover inverse-effectiveness.
    """
    # 1. identical event set -------------------------------------------------
    loc_seqs, mod_seqs, offs, lens = generate_fixed_duration_event_sequences(
        space_size, T, D, n_per_loc)
    Tmax = max(lens)

    xA, xV, valid = generate_av_batch_tensor(
        loc_seqs, mod_seqs, offs,
        n=net.n, space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=noise_std,  # ← your knob
        loc_jitter_std=net.loc_jitter_std,
        stimulus_intensity=stimulus_intensity,  # ← your knob
        device=net.device, max_len=Tmax)

    err_A = _signed_errors_hybrid_fast(net, xA, torch.zeros_like(xV), valid,
                                       loc_seqs, integrate, use_msi_raw)
    err_V = _signed_errors_hybrid_fast(net, torch.zeros_like(xA), xV, valid,
                                       loc_seqs, integrate, use_msi_raw)
    err_AV = _signed_errors_hybrid_fast(net, xA, xV, valid,
                                        loc_seqs, integrate, use_msi_raw)

    x_min, x_max = -20, 20
    bins = np.linspace(x_min, x_max, 81)
    xs = np.linspace(x_min, x_max, 200)

    plt.figure(figsize=(24, 5))
    for k, (errs, ttl, col) in enumerate(
            [(err_A, "Audio-only", "blue"),
             (err_V, "Visual-only", "orange"),
             (err_AV, "Bimodal", "green")]):
        plt.subplot(1, 4, k + 1)
        sel = [e for e in errs if x_min <= e <= x_max]
        if len(sel) < 10:
            mu, sd = (0, 0)
        else:
            p10, p90 = np.percentile(sel, [10, 90])
            central = [x for x in sel if p10 <= x <= p90]
            mu, sd = stats.norm.fit(central) if len(central) > 5 else (0, 0)
        plt.hist(sel, bins=bins, density=True, edgecolor='k', color=col, alpha=.65)
        if sd > 0:
            plt.plot(xs, stats.norm.pdf(xs, mu, sd), 'r-', lw=2)
        plt.title(f"{ttl} (σ={sd:.2f}°)")
        plt.xlabel("Signed error (deg)")
        plt.xlim(x_min, x_max)

    # sensitivities
    plt.subplot(1, 4, 4)
    σA = np.std(err_A) if len(err_A) else 0
    σV = np.std(err_V) if len(err_V) else 0
    σB = np.std(err_AV) if len(err_AV) else 0
    sA = 0 if σA == 0 else 1 / σA
    sV = 0 if σV == 0 else 1 / σV
    sB = 0 if σB == 0 else 1 / σB
    sIdeal = np.sqrt(sA ** 2 + sV ** 2)
    bars = [sA, sV, sB, sIdeal]
    colors = ["blue", "orange", "green", "red"]
    labels = ["Audio", "Visual", "Bimodal", "Ideal"]
    plt.bar(range(4), bars, color=colors, alpha=.7)
    plt.xticks(range(4), labels)
    plt.ylabel("Sensitivity (1/σ)")
    plt.title(f"S_Bimodal={sB:.2f}  |  S_Ideal={sIdeal:.2f}")

    plt.tight_layout()
    plt.show()



# ─── utils_model.py ─────────────────────────────────────────────
def load_msi_model(path: Path, device="cuda"):
    """Return a ready‑to‑run `MultiBatchAudVisMSINetworkTime`."""
    ck = torch.load(path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ck["constructor_hparams"])
    net.load_state_dict(ck["model_state"])
    for k, v in ck["mutable_hparams"].items():
        setattr(net, k, v)
    net.eval()
    net.to(device)
    return net


# ─── hybrid_sensitivity.py ─────────────────────────────────────
def compute_hybrid_sensitivity_fast(
        net,
        *,
        stimulus_intensity=0.4,
        noise_std=0.1,
        batch_size=1024,  # larger batch
        **kw,
):
    """Fast version: runs A, V, AV in parallel as a 3x larger batch."""
    space_size = kw.get("space_size", 180)
    T = kw.get("T", 30)
    D = kw.get("D", 10)
    n_per_loc = kw.get("n_per_loc", 5)

    # Generate base stimuli
    loc_seqs, mod_seqs, offs, lens = generate_fixed_duration_event_sequences(
        space_size, T, D, n_per_loc)
    Tmax = max(lens)

    # Generate base A/V inputs
    xA_base, xV_base, valid = generate_av_batch_tensor(
        loc_seqs, mod_seqs, offs,
        n=net.n, space_size=net.space_size,
        sigma_in=net.sigma_in,
        noise_std=noise_std,
        stimulus_intensity=stimulus_intensity,
        device=net.device, max_len=Tmax)

    Ntotal = xA_base.size(0)

    # Stack 3 conditions: [A-only, V-only, AV]
    xA_3x = torch.cat([xA_base, torch.zeros_like(xA_base), xA_base], dim=0)
    xV_3x = torch.cat([torch.zeros_like(xV_base), xV_base, xV_base], dim=0)
    valid_3x = valid.repeat(3, 1)
    loc_seqs_3x = loc_seqs * 3

    # Run once for all conditions
    signed_errs_all = _signed_errors_hybrid_fast(
        net, xA_3x, xV_3x, valid_3x, loc_seqs_3x,
        batch_size=batch_size, **kw)

    # Split results
    err_A = signed_errs_all[:Ntotal]
    err_V = signed_errs_all[Ntotal:2 * Ntotal]
    err_AV = signed_errs_all[2 * Ntotal:]

    # Compute statistics
    σA = np.std(err_A) if err_A else 0
    σV = np.std(err_V) if err_V else 0
    σB = np.std(err_AV) if err_AV else 0
    sA = 0 if σA == 0 else 1 / σA
    sV = 0 if σV == 0 else 1 / σV
    sB = 0 if σB == 0 else 1 / σB
    sIdeal = math.sqrt(sA ** 2 + sV ** 2)

    return dict(
        err_A=err_A, err_V=err_V, err_AV=err_AV,
        sigma=(σA, σV, σB),
        sensitivity=np.array([sA, sV, sB, sIdeal], dtype=float),
    )


def pool_hybrid_sensitivity_fast(model_paths, *, modify_net=None, **sens_kw):
    """Fast pooling with better GPU memory management."""
    sens_vecs = []

    # Increase default batch size
    if 'batch_size' not in sens_kw:
        sens_kw['batch_size'] = 1024

    for p in model_paths:
        net = load_msi_model(Path(p))
        if callable(modify_net):
            modify_net(net)

        # Use fast version
        out = compute_hybrid_sensitivity_fast(net, **sens_kw)
        sens_vecs.append(out["sensitivity"])

        # Aggressive cleanup
        del net
        torch.cuda.empty_cache()

    mat = np.vstack(sens_vecs)
    mean = mat.mean(0)
    sem = mat.std(0, ddof=1) / math.sqrt(mat.shape[0])
    return mean, sem


def show_condition_plot(err_A, err_V, err_AV, title):
    x_min, x_max = -20, 20
    bins = np.arange(x_min - 0.5, x_max + 1.0, 1)  # 1° wide, centred on ints

    plt.figure(figsize=(24, 5))
    for k, (errs, ttl, col) in enumerate(
            [(err_A, "Audio", "blue"),
             (err_V, "Visual", "orange"),
             (err_AV, "Bimodal", "green")]
    ):
        plt.subplot(1, 4, k + 1)
        sel = [e for e in errs if x_min <= e <= x_max]

        sd = np.std(sel) if len(sel) > 0 else 0

        # Plot histogram without Gaussian fit
        plt.hist(sel, bins=bins, density=True,
                 edgecolor='k', color=col, alpha=.65)

        plt.title(f"{ttl} (σ={sd:.2f}°)")
        plt.xlabel("Signed error (deg)")
        plt.xlim(x_min, x_max)

    plt.subplot(1, 4, 4)  # keep layout: 3 hists + 1 bar plot
    σA, σV, σB = [np.std(e) if e else 0  # <‑‑ keep only the three
                  for e in (err_A, err_V, err_AV)]
    sens = [0 if σ == 0 else 1 / σ for σ in (σA, σV, σB)]

    plt.bar(range(3), sens,
            color=["blue", "orange", "green"], alpha=.7)
    plt.xticks(range(3), ["Audio", "Visual", "Bimodal"])
    plt.ylabel("Sensitivity (1/σ)")
    plt.title(title)
    plt.tight_layout()


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def plot_sensitivity_comparison(mean_ctrl, sem_ctrl,
                                mean_manip, sem_manip):
    mean_ctrl = mean_ctrl[:3]
    sem_ctrl = sem_ctrl[:3]
    mean_manip = mean_manip[:3]
    sem_manip = sem_manip[:3]

    # Calculate and print percentage differences
    labels = ["Audio", "Visual", "Bimodal"]
    print("\n=== Sensitivity Percentage Differences (Manip vs Control) ===")
    for i, label in enumerate(labels):
        pct_diff = ((mean_manip[i] - mean_ctrl[i]) / mean_ctrl[i]) * 100
        print(f"{label}: {pct_diff:+.1f}%")
    print("=" * 60 + "\n")

    print("=== Bimodal Enhancement (vs Best Unisensory) ===")
    # Control
    max_uni_ctrl = max(mean_ctrl[0], mean_ctrl[1])  # max(Audio, Visual)
    bimodal_enh_ctrl = ((mean_ctrl[2] - max_uni_ctrl) / max_uni_ctrl) * 100
    print(f"Control: {bimodal_enh_ctrl:+.1f}%")

    # Manipulation
    max_uni_manip = max(mean_manip[0], mean_manip[1])
    bimodal_enh_manip = ((mean_manip[2] - max_uni_manip) / max_uni_manip) * 100
    print(f"Manipulation: {bimodal_enh_manip:+.1f}%")
    print("=" * 60 + "\n")

    labels = ["Audio", "Visual", "Bimodal"]
    colours = ["blue", "orange", "green"]
    x_pos = np.arange(3)
    width = 0.8

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 70
    plt.rcParams['xtick.labelsize'] = 70
    plt.rcParams['ytick.labelsize'] = 70
    plt.rcParams['axes.titlesize'] = 70
    plt.rcParams['axes.labelsize'] = 70
    plt.rcParams['legend.fontsize'] = 70

    fig, ax = plt.subplots(figsize=(25, 16.5))
    for i, (x, col) in enumerate(zip(x_pos, colours)):
        # manipulation (filled)
        ax.bar(x, mean_manip[i], width,
               color=col, yerr=sem_manip[i], capsize=3,
               label="Manip" if i == 0 else "_nolegend_", zorder=2)
        # control (outline)
        ax.bar(x, mean_ctrl[i], width,
               facecolor="none", edgecolor="black", linestyle=":",
               linewidth=2, yerr=sem_ctrl[i], ecolor="black", capsize=3,
               label="Control" if i == 0 else "_nolegend_", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sensitivity (1/σ)")
    ax.set_title(None)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', length=20, width=1)
    ax.legend(frameon=False, loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./Saved_Images/sens_lowNMDA.svg', format='svg')
    plt.show()


# ===============================================================
# ===============================================================
import numpy as np
import scipy.stats as stats
from typing import Sequence, Tuple


def _freedman_diaconis_bin_width(data: np.ndarray) -> float:
    """Return optimal bin width via the Freedman–Diaconis rule."""
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    if iqr == 0 or n < 2:
        return 1.0  # fall back
    return 2 * iqr / (n ** (1 / 3))


def _trimmed_gaussian_fit(errors: np.ndarray,
                          trim_q: float = 0.05) -> Tuple[float, float]:
    """
    Fit N(μ,σ²) to the central *1‑2·trim_q* fraction of the data.
    Returns μ, σ.
    """
    lo, hi = np.quantile(errors, [trim_q, 1 - trim_q])
    core = errors[(errors >= lo) & (errors <= hi)]
    if len(core) < 10:
        return 0.0, 0.0
    return stats.norm.fit(core)  # MLE


def _separate_spike(errors: Sequence[float],
                    spike_thr: float) -> Tuple[np.ndarray, float]:
    """
    Split the sample into
        • spike errors  (|e| ≤ spike_thr)      – returned as mass p0
        • tail errors   (the rest)             – returned as ndarray
    """
    err = np.asarray(errors, dtype=float)
    if err.size == 0:
        return np.empty(0), 0.0
    spike_mask = np.abs(err) <= spike_thr
    p0 = spike_mask.mean()
    tails = err[~spike_mask]
    return tails, p0


# ===========================================================
#  Histogram‑only visualisation + σ‑based sensitivities
# ===========================================================
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Tuple


def show_error_histograms(err_A: Sequence[float],
                          err_V: Sequence[float],
                          err_AV: Sequence[float],
                          *,
                          x_lim: Tuple[int, int] = (-20, 20),
                          nbins: int = 80,
                          title: str = "CONTROL",
                          sem_vals: Sequence[float] = None) -> None:  # Added sem_vals parameter
    """
    Plot histograms of signed localisation errors for the three
    conditions and show sensitivities 1/σ in a fourth panel.

    Parameters
    ----------
    err_A , err_V , err_AV : sequence of float
        Signed error samples (in degrees) for Audio, Visual, Bimodal.
    x_lim : (lo, hi)
        Range of the x‑axis (degrees) for the histograms.
    nbins : int
        Number of bins for `plt.hist`.
    title : str
        Title for the sensitivity bar chart.
    sem_vals : sequence of float, optional
        Standard errors for the sensitivity values.
    """
    conditions = [("Audio", "blue", err_A),
                  ("Visual", "orange", err_V),
                  ("Bimodal", "green", err_AV)]

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 40
    plt.rcParams['xtick.labelsize'] = 40
    plt.rcParams['ytick.labelsize'] = 40
    plt.rcParams['axes.titlesize'] = 40
    plt.rcParams['axes.labelsize'] = 40
    plt.rcParams['legend.fontsize'] = 40

    x_min, x_max = x_lim
    fig, axes = plt.subplots(1, 4, figsize=(48, 10),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})

    sigma_vals = []

    # ---------------- histograms -----------------------------
    for ax, (label, colour, errs) in zip(axes[:3], conditions):
        ax.hist(errs, bins=nbins, range=x_lim,
                density=True, color=colour, alpha=.8,
                edgecolor='k', linewidth=0.5)  # Added linewidth=0.5
        ax.set_xlim(x_lim)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Signed error (deg)")

        sigma = float(np.std(errs, ddof=1)) if len(errs) >= 2 else 0.0
        ax.set_title(f"{label}  (σ={sigma:.2f}°)")
        ax.grid(False)
        ax.tick_params(axis='both', which='major', length=20, width=1)
        sigma_vals.append(sigma)

    # ---------------- sensitivities --------------------------
    sensitivities = [0 if s == 0 else 1 / s for s in sigma_vals]
    ax_s = axes[3]

    yerr = None
    if sem_vals is not None and len(sem_vals) >= 3:
        yerr = sem_vals[:3]  # Use only the first 3 values (A, V, B)

    ax_s.bar(["Audio", "Visual", "Bimodal"], sensitivities,
             color=["blue", "orange", "green"], alpha=.9,
             yerr=yerr, capsize=5)  # Added yerr and capsize
    ax_s.set_ylabel("Sensitivity (1/σ)")
    ax_s.set_title(None)
    ax_s.set_ylim(bottom=0)
    ax_s.grid(False)

    fig.tight_layout()
    ax_s.tick_params(axis='both', which='major', length=20, width=1)
    plt.rcParams['svg.fonttype'] = 'none'
    if title == "CONTROL":
        plt.savefig('./Saved_Images/Err_Hist.svg', format='svg')
    plt.show()


# Main execution

def main():
    # --- model list ----------------------------------------------------
    ckpt_dir = Path("checkpoint")
    paths = [ckpt_dir / f"msi_model_surr_10_{i:02d}.pt" for i in range(10)]

    # ---------- CONTROL -----------------------------------------------
    ctrl_errs = {"A": [], "V": [], "B": []}
    mean_ctrl, sem_ctrl = pool_hybrid_sensitivity_fast(paths)
    all_err_A = []
    all_err_V = []
    all_err_AV = []

    for path in paths:
        rep_net = load_msi_model(path)
        rep_out = compute_hybrid_sensitivity_fast(rep_net)
        all_err_A.extend(rep_out["err_A"])
        all_err_V.extend(rep_out["err_V"])
        all_err_AV.extend(rep_out["err_AV"])
        del rep_net
        torch.cuda.empty_cache()

    show_error_histograms(all_err_A, all_err_V, all_err_AV,
                          x_lim=(-20, 20),
                          nbins=40,
                          title="CONTROL",
                          sem_vals=sem_ctrl)  # Pass the error bars

    # del rep_net

    # ---------- MANIPULATION -------------------------
    def tweak_fn(net):
        # FF Inhibition manipulation
        # net.pv_nmda = 1
        # net.targ_ratio = 1.5

        # NMDA manipulation
        net.gNMDA = 0.02

        # Adaptation manipulation

    mean_mod, sem_mod = pool_hybrid_sensitivity_fast(paths, modify_net=tweak_fn)
    all_err_A_mod = []
    all_err_V_mod = []
    all_err_AV_mod = []

    for path in paths:
        rep_net2 = load_msi_model(path)
        tweak_fn(rep_net2)
        rep_out2 = compute_hybrid_sensitivity_fast(rep_net2)
        all_err_A_mod.extend(rep_out2["err_A"])
        all_err_V_mod.extend(rep_out2["err_V"])
        all_err_AV_mod.extend(rep_out2["err_AV"])
        del rep_net2
        torch.cuda.empty_cache()

    show_error_histograms(all_err_A_mod, all_err_V_mod, all_err_AV_mod,
                          x_lim=(-20, 20),
                          nbins=40,
                          title="MANIP",
                          sem_vals=sem_mod)  # Pass the error bars

    # ---------- combined bar‑plot --------------------------------------
    plot_sensitivity_comparison(mean_ctrl, sem_ctrl, mean_mod, sem_mod)


if __name__ == "__main__":
    main()
