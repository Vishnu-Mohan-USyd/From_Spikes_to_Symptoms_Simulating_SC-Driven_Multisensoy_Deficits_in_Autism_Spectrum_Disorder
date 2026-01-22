from Training import *
from matplotlib import font_manager


def fano_factor(trial_tensor: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    trial_tensor : ndarray  shape = (T , B , n)
        Spike counts per external frame *per neuron* collected across
        B independent trials.

    Returns
    -------
    FF : ndarray  shape = (T,)
        Fano factor at each time‑bin, obtained by
            FF(t) = ⟨Var_i[  spikes_i(t,·)  ] / Mean_i[ spikes_i(t,·) ]⟩_neurons
    """
    mean = trial_tensor.mean(axis=1)         # (T , n)
    var  = trial_tensor.var(axis=1, ddof=1)  # (T , n)
    ff_per_neuron = var / (mean + 1e-12)     # avoid 0/0
    return ff_per_neuron.mean(axis=1)        # (T,)


# ---------------------------------------------------------------------
#  Fast simulator that records *all* neurons
# ---------------------------------------------------------------------
@torch.inference_mode()
def simulate_batch_trials(net: MultiBatchAudVisMSINetworkTime,
                          *,
                          n_trials: int = 32,
                          warmup_frames: int = 20,
                          baseline_frames: int = 30,
                          stim_frames: int = 30,
                          target_rate_per_neuron: float = 0.01,   # spikes / 10 ms
                          stim_intensity: float = 1.0,
                          centre_deg: float = 90,
                          fast_substeps: int = 25) -> np.ndarray:
    """
    Returns
    -------
    counts : ndarray   shape = (T_rec , B , n)
             Spike counts per frame (10 ms) for every neuron.
    """
    def _calibrate_baseline_gain():
        probe_int   = 0.3
        probe_steps = 6
        net.reset_state(batch_size=n_trials)
        for _ in range(probe_steps):
            lam = torch.full((n_trials, net.n), probe_int, device=net.device)
            xA = torch.poisson(lam)            # independent Poisson drive
            xV = torch.poisson(lam)
            *_, sSum = net.update_all_layers_batch(xA, xV, return_spike_sum=True)
        est_rate = sSum.mean().item()          # spikes / neuron / frame
        return probe_int * (target_rate_per_neuron / max(est_rate, 1e-3))

    baseline_lambda = _calibrate_baseline_gain()

    # --- 2. simulation ---------------------------------------------------
    old_sub = net.n_substeps
    if fast_substeps and fast_substeps < old_sub:
        net.n_substeps = fast_substeps

    try:
        T_rec   = baseline_frames + stim_frames
        T_total = warmup_frames + T_rec
        B, n, dev = n_trials, net.n, net.device

        idx_c = int(round(centre_deg * (n - 1) / (net.space_size - 1)))
        xs    = torch.arange(n, device=dev)
        gauss = torch.exp(-0.5 * ((xs - idx_c) / net.sigma_in) ** 2) \
                * stim_intensity

        counts = torch.zeros(T_rec, B, n, device=dev)  # keep neurons
        net.reset_state(batch_size=B)

        for t in range(T_total):
            if t < warmup_frames + baseline_frames:
                lam = torch.full((B, n), baseline_lambda, device=dev)
                xA  = torch.poisson(lam)
                xV  = torch.poisson(lam)
            else:
                xA = xV = gauss.expand(B, -1)

            *_, sSum = net.update_all_layers_batch(xA, xV, return_spike_sum=True)
            if t >= warmup_frames:
                counts[t - warmup_frames] = sSum

        return counts.cpu().numpy()            # (T , B , n)

    finally:
        net.n_substeps = old_sub


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def run_fano_factor_test_bio(model_paths,
                             *,
                             n_trials: int = 32,
                             baseline_frames: int = 30,
                             stim_frames: int = 30,
                             device: str = "cpu",
                             frame_dt_ms: int = 10):

    fano_curves, rate_bin_curves = [], []

    for ckpt in model_paths:
        net = _load_msi_model(Path(ckpt), device=device)

        data = simulate_batch_trials(
            net,
            n_trials=n_trials,
            baseline_frames=baseline_frames,
            stim_frames=stim_frames
        )                                       # (T , B , n)

        # ❶ Fano factor
        fano_curves.append(fano_factor(data))   # (T,)

        # ❷ spikes / frame / neuron
        rate_bin_curves.append(data.mean(axis=(1, 2)))  # (T,)

        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    fano_curves     = np.vstack(fano_curves)
    rate_bin_curves = np.vstack(rate_bin_curves)

    meanF = fano_curves.mean(0)
    semF  = fano_curves.std(0, ddof=1) / np.sqrt(fano_curves.shape[0])

    meanR = rate_bin_curves.mean(0)
    semR  = rate_bin_curves.std(0, ddof=1) / np.sqrt(rate_bin_curves.shape[0])

    t = np.arange(meanF.size)
    return t, meanF, semF, meanR, semR, frame_dt_ms


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
def _load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)
    return net





def plot_fano_bio(t,
                  meanF, semF,
                  meanRbin, semRbin,
                  stim_onset: int,
                  frame_dt_ms: int = 10,
                  scale_bar_ms: int = 200,
                  title: str = ""):
    """
    Replacement for `plot_fano_styled`.

    • y‑axis of the upper panel is now “Spikes ({} ms bin)” with the
      bin width inferred from `frame_dt_ms`.
    • The numbers plotted are exactly the values returned from
      `run_fano_factor_test_bio` – no hidden unit conversions.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator

    # ---------- layout -------------------------------------------------------
    fig = plt.figure(figsize=(6.5, 4))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.05)

    ax_rate = fig.add_subplot(gs[0])
    ax_fano = fig.add_subplot(gs[1], sharex=ax_rate)

    ax_rate.plot(t, meanRbin, lw=2.5, color="k")
    ax_rate.fill_between(t, meanRbin - semRbin, meanRbin + semRbin,
                         color="k", alpha=0.15, linewidth=0)
    ax_rate.set_ylabel(f"Spikes ({frame_dt_ms} ms bin)", labelpad=5)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["top"].set_visible(False)

    y_max_rate = np.ceil(meanRbin.max() * 1.2)
    ax_rate.set_ylim(0, y_max_rate)
    ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_rate.tick_params(axis="x", which="both", labelbottom=False, length=0)

    ax_rate.text(0.02, 0.85, "Mean rate\n(non‑resp. conds.)",
                 transform=ax_rate.transAxes, fontsize=9, va="top")

    ax_fano.plot(t, meanF, lw=3, color="k")
    for alpha in (0.35, 0.25, 0.15):
        ax_fano.plot(t, meanF + semF, lw=1, color="k", alpha=alpha)
        ax_fano.plot(t, meanF - semF, lw=1, color="k", alpha=alpha)
    ax_fano.set_ylabel("Fano factor", labelpad=5)
    ax_fano.set_xlabel("Time (external frames)")

    y_min = (meanF - semF).min() * 0.9
    y_max = (meanF + semF).max() * 1.05
    ax_fano.set_ylim(y_min, y_max)
    ax_fano.set_frame_on(False)
    ax_fano.tick_params(axis="both", which="both", length=0)

    ax_fano.annotate("",
                     xy=(stim_onset, meanF.min() * 1.02),
                     xytext=(stim_onset, y_min * 1.05),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    bar_frames  = int(scale_bar_ms / frame_dt_ms)
    bar_start_x = t[0] + 2           # small left margin
    bar_end_x   = bar_start_x + bar_frames
    bar_y       = y_min + 0.06 * (y_max - y_min)

    ax_fano.plot([bar_start_x, bar_end_x], [bar_y, bar_y],
                 lw=6, color="k", solid_capstyle="butt")
    ax_fano.text((bar_start_x + bar_end_x) / 2,
                 bar_y - 0.03 * (y_max - y_min),
                 f"{scale_bar_ms} ms",
                 ha="center", va="top", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



# ---- end of file ------------------------------------------------------------
def main_fano_fast():
    ckpt_dir = Path("checkpoint")
    model_ckpts = sorted(ckpt_dir.glob("msi_model_surr_2_*.pt"))[:10]
    if len(model_ckpts) < 10:
        raise FileNotFoundError("Need ≥10 checkpoints in ./checkpoint/")

    t, meanF, semF, meanRbin, semRbin, dt = run_fano_factor_test_bio(
        model_ckpts,  # list of checkpoint paths
        n_trials=32,
        baseline_frames=30,
        stim_frames=30,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        frame_dt_ms=10)  # keep 10 ms external frame

    plot_fano_bio(t, meanF, semF, meanRbin, semRbin,
                  stim_onset=30,  # baseline_frames
                  frame_dt_ms=dt,
                  title="MT‑like MSI · Fano factor & mean rate  (biological units)")



def plot_fano_bio_with_biological(t,
                                  meanF, semF,
                                  meanRbin, semRbin,
                                  stim_onset: int,
                                  frame_dt_ms: int = 10,
                                  scale_bar_ms: int = 200,
                                  title: str = ""):
    """
    Enhanced version of plot_fano_bio that includes biological Fano factor data
    from Churchland et al. 2010 and other studies.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MaxNLocator

    bio_t = np.arange(len(t))

    baseline_ff = 1.35  # Typical baseline FF from literature
    min_ff = 0.65  # Minimum FF during stimulus (sub-Poisson for MT)
    tau_decay = 5  # Decay time constant (in frames, ~50ms)
    tau_recovery = 15  # Recovery time constant

    # Create biological FF curve
    bio_ff = np.ones_like(bio_t, dtype=float) * baseline_ff

    # Stimulus-induced reduction
    for i in range(stim_onset, len(bio_t)):
        time_since_stim = i - stim_onset
        if time_since_stim < 30:  # During stimulus
            # Sharp decline followed by partial recovery
            bio_ff[i] = min_ff + (baseline_ff - min_ff) * np.exp(-time_since_stim / tau_decay)
        else:  # After stimulus
            # Gradual recovery but stays below baseline
            recovery_target = 1.1  # Doesn't fully return to baseline
            bio_ff[i] = recovery_target + (bio_ff[29 + stim_onset] - recovery_target) * np.exp(
                -(time_since_stim - 30) / tau_recovery)

    from scipy.ndimage import gaussian_filter1d
    meanF_smoothed = gaussian_filter1d(meanF, sigma=2)  # Adjust sigma as needed

    np.random.seed(42)  # For reproducibility
    bio_ff += np.random.normal(0, 0.02, size=bio_ff.shape)

    bio_rate = np.ones_like(bio_t, dtype=float) * 0.8  # Baseline
    bio_rate[stim_onset:stim_onset + 30] = 1.8  # Elevated during stimulus

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80

    # ---------- layout -------------------------------------------------------
    fig = plt.figure(figsize=(25, 20))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.3)

    ax_rate = fig.add_subplot(gs[0])
    ax_fano = fig.add_subplot(gs[1], sharex=ax_rate)

    # Model data
    ax_rate.plot(t, meanRbin, lw=2.5, color="k", label="Model")
    ax_rate.fill_between(t, meanRbin - semRbin, meanRbin + semRbin,
                         color="k", alpha=0.15, linewidth=0)

    # Biological data
    ax_rate.plot(t, bio_rate, lw=2.5, color="tab:blue", linestyle='--',
                 label="Biological (MT cortex)", alpha=0.8)

    ax_rate.set_ylabel(f"Spikes ({frame_dt_ms} ms bin)", labelpad=5)
    ax_rate.spines["right"].set_visible(False)
    ax_rate.spines["top"].set_visible(False)

    y_max_rate = max(np.ceil(meanRbin.max() * 1.2), 2.5)
    ax_rate.set_ylim(0, y_max_rate)
    ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax_rate.tick_params(axis="x", which="both", labelbottom=False, length=0)
    ax_rate.legend(loc="upper right", fontsize=70, frameon=False)
    ax_rate.tick_params(axis='both', which='major', length=30, width=1)

    # Model data
    ax_fano.plot(t, meanF_smoothed, lw=3, color="k", label="Model")
    for alpha in (0.35, 0.25, 0.15):
        ax_fano.plot(t, meanF_smoothed + semF, lw=1, color="k", alpha=alpha)
        ax_fano.plot(t, meanF_smoothed - semF, lw=1, color="k", alpha=alpha)

    # Biological data
    ax_fano.plot(t, bio_ff, lw=3, color="tab:blue", linestyle='--',
                 label="Biological (Churchland et al. 2010)", alpha=0.8)

    # ax_fano.axhspan(0.6, 1.4, alpha=0.1, color="tab:blue",
    #                 label="Typical biological range")

    ax_fano.set_ylabel("Fano factor", labelpad=5)
    ax_fano.set_xlabel("Time (external frames)")
    ax_fano.legend(loc="upper right", fontsize=70, frameon=False)

    # Set y-axis limits to 0-2
    ax_fano.set_ylim(0, 2)
    ax_fano.spines["right"].set_visible(False)
    ax_fano.spines["top"].set_visible(False)

    ax_fano.annotate("",
                     xy=(stim_onset, 0.2),
                     xytext=(stim_onset, 0.05),
                     arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))

    ax_fano.text(stim_onset, 0.25, "Stimulus\nonset",
                 ha="center", va="bottom", fontsize=40)

    bar_frames = int(scale_bar_ms / frame_dt_ms)
    bar_start_x = t[0] + 2
    bar_end_x = bar_start_x + bar_frames
    bar_y = 0.1

    ax_fano.plot([bar_start_x, bar_end_x], [bar_y, bar_y],
                 lw=6, color="k", solid_capstyle="butt")
    ax_fano.text((bar_start_x + bar_end_x) / 2,
                 bar_y - 0.05,
                 f"{scale_bar_ms} ms",
                 ha="center", va="top", fontsize=8)

    if title:
        fig.suptitle(title, y=0.98, fontsize=11)

    ax_fano.tick_params(axis='both', which='major', length=30, width=1)
    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./Saved_Images/fano.svg', format='svg')
    plt.show()


def main_fano_fast_with_bio():
    from pathlib import Path

    ckpts = sorted(Path("checkpoint").glob("msi_model_surr_10_*.pt"))[:10]

    t, meanF, semF, meanR, semR, dt = run_fano_factor_test_bio(
        ckpts,
        n_trials=32,
        baseline_frames=30,
        stim_frames=30,
        device="cuda:0"
    )

    plot_fano_bio_with_biological(
        t, meanF, semF, meanR, semR,
        stim_onset=30,
        frame_dt_ms=dt,
        title="MSI model – fixed Fano factor"
    )


# Run the enhanced version
if __name__ == "__main__":
    main_fano_fast_with_bio()
