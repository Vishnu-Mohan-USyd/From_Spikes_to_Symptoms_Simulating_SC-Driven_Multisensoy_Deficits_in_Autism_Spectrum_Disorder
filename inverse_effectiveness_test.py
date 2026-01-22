from Training import *
from matplotlib import font_manager


def spatial_binding_diagnostics(
        net,
        *,
        separations_deg: Sequence[int] | None = None,
        example_seps: Sequence[int] | None = None,
        n_trials: int = 100,
        n_examples: int = 3,
        intensity: float = 0.5,
        duration: int = 20,
        method: str = "ratio"):
    """
    GPU-batched P(fusion) curve  +  illustrative rasters/profiles.
    Now draws a *linear fit* through the summary points, **and marks the 50 %-fusion
    threshold on the plot**.
    """
    # ───── helpers ────────────────────────────────────────────────────────
    N = net.n

    def idx(deg_arr):
        a = np.asarray(deg_arr, dtype=float)
        return np.round(a * (N - 1) / (net.space_size - 1)).astype(int)

    def make_gauss(idx_centres: torch.Tensor) -> torch.Tensor:
        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        return torch.exp(-0.5 * ((xs - idx_centres.unsqueeze(1)) /
                                 net.sigma_in) ** 2) * intensity

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode='wrap')
        if sm.max() < 1e-6:  # silent
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props['peak_heights'])[::-1][:2]]

        def valley(i, j):
            direct = abs(j - i)
            seg = sm[min(i, j): max(i, j) + 1] if direct <= N - direct else \
                np.r_[sm[max(i, j):], sm[:min(i, j) + 1]]
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.6

    # ───── default lists ─────────────────────────────────────────────────
    if separations_deg is None:
        separations_deg = list(range(0, 61, 5))  # 0 … 60°
    if example_seps is None:
        example_seps = [0, 20, 40, 60]

    # ───── 1) GPU-batched fusion-probability curve ───────────────────────
    n_sep = len(separations_deg)
    B = n_sep * n_trials
    rng = np.random.default_rng()

    base_deg = rng.integers(30, 150, size=B)
    sep_rep = np.repeat(separations_deg, n_trials)
    locA_deg = base_deg
    locV_deg = (base_deg + sep_rep) % net.space_size

    idxA = torch.as_tensor(idx(locA_deg), device=net.device)
    idxV = torch.as_tensor(idx(locV_deg), device=net.device)

    gA = make_gauss(idxA)
    gV = make_gauss(idxV)

    xA = torch.zeros(B, duration, N, device=net.device)
    xV = torch.zeros_like(xA)
    xA[:, :duration] = gA.unsqueeze(1)
    xV[:, :duration] = gV.unsqueeze(1)

    net.reset_state(batch_size=B)
    msi_sum = torch.zeros(B, N, device=net.device)
    for t in range(duration):
        net.update_all_layers_batch(xA[:, t], xV[:, t])
        msi_sum += net._latest_sMSI

    flags = np.zeros((n_sep, n_trials), dtype=bool)
    profs = msi_sum.cpu().numpy()
    for k in range(n_sep):
        s, e = k * n_trials, (k + 1) * n_trials
        for j in range(n_trials):
            flags[k, j] = is_fused(profs[s + j])

    fusion_prob = flags.mean(1)

    #  -- summary plot ----------------------------------------------------
    x = np.asarray(separations_deg, dtype=float)
    y = fusion_prob
    m, c = np.polyfit(x, y, 1)  # slope & intercept
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = m * x_fit + c

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=60, color='C0', zorder=3, label='data')
    plt.plot(x_fit, y_fit, color='C1', lw=2, label=f'slope = {m:.3f}')

    if m != 0:  # avoid divide-by-zero if slope is 0
        x50 = (0.5 - c) / m
        plt.axhline(0.5, ls='--', color='0.4', lw=1)
        plt.axvline(x50, ls='--', color='0.4', lw=1)
        plt.scatter([x50], [0.5], color='red', s=90, zorder=4)
        plt.text(x50, 0.53, f'50 % @ {x50:.1f}°', ha='center', va='bottom',
                 color='red', fontsize=8)

    plt.xlabel('Spatial disparity (°)')
    plt.ylabel('P(fusion)')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, x.max())
    plt.legend(frameon=False)
    plt.grid(alpha=.3)
    plt.title('Spatial binding window (linear fit)')
    plt.tight_layout()
    plt.show()

    for sep in example_seps:
        print(f"\n=== Raster & profile examples — separation {sep}° ===")
        for ex in range(n_examples):
            base = rng.integers(40, 140)
            loc1, loc2 = base, (base + sep) % net.space_size
            i1, i2 = idx(loc1), idx(loc2)

            g1 = make_gauss(torch.tensor([i1], device=net.device))[0]
            g2 = make_gauss(torch.tensor([i2], device=net.device))[0]

            net.reset_state(1)
            xA_ex, xV_ex = g1.unsqueeze(0), g2.unsqueeze(0)
            hist = []
            for _ in range(duration):
                net.update_all_layers_batch(xA_ex, xV_ex)
                hist.append(net._latest_sMSI[0].cpu().numpy())
            hist = np.stack(hist)
            prof = hist.sum(0)
            fused = is_fused(prof)

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(8, 3),
                gridspec_kw={'width_ratios': [2, 1]}
            )
            ax1.imshow(hist.T, aspect='auto', cmap='hot')
            ax1.set(
                title=f'A@{loc1}°  V@{loc2}°  →  {"FUSED" if fused else "SEPARATED"}',
                xlabel='time-step', ylabel='neuron'
            )
            ax2.plot(prof)
            ax2.axvline(i1, ls='--', c='r', label=f'A {loc1}°')
            ax2.axvline(i2, ls='--', c='g', label=f'V {loc2}°')
            ax2.set(xlabel='neuron index', ylabel='Σ spikes')
            ax2.legend()
            ax2.grid(alpha=.3)
            plt.tight_layout()
            plt.show()

    return {
        "separations_deg": separations_deg,
        "fusion_prob": fusion_prob.tolist(),
        "raw_flags": flags
    }



# Main execution

# ───────────────────────── imports ─────────────────────────
from pathlib import Path
from typing import Sequence
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ─────────────────── 0. checkpoint loader ───────────────────
def load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)  # make sure helpers pick this up
    return net


def spatial_binding_curve_fast(
        net,
        separations_deg=range(0, 61, 5),
        n_trials=100,
        duration=20,
        intensity=0.5):
    N, S = net.n, net.space_size
    rng = np.random.default_rng()

    # ----- build one giant batch ------------------------------------
    n_sep = len(separations_deg)
    B = n_sep * n_trials  # e.g. 13 × 100 = 1300

    base_deg = rng.integers(30, 150, size=B)
    sep_rep = np.repeat(separations_deg, n_trials)

    locA_deg = base_deg
    locV_deg = (base_deg + sep_rep) % S

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode='wrap')
        if sm.max() < 1e-6:  # silent
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props['peak_heights'])[::-1][:2]]

        def valley(i, j):
            direct = abs(j - i)
            seg = sm[min(i, j): max(i, j) + 1] if direct <= N - direct else \
                np.r_[sm[max(i, j):], sm[:min(i, j) + 1]]
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.6

    def to_idx(deg):  # degrees → neuron index
        return torch.round(
            torch.as_tensor(deg, device=net.device, dtype=torch.float32)
            * (N - 1) / (S - 1)).long()

    idxA, idxV = to_idx(locA_deg), to_idx(locV_deg)

    xs = torch.arange(N, device=net.device, dtype=torch.float32)
    g = lambda idx: torch.exp(-0.5 * ((xs - idx[:, None]) / net.sigma_in) ** 2) * intensity
    gA, gV = g(idxA), g(idxV)

    xA = torch.zeros(B, duration, N, device=net.device)
    xV = torch.zeros_like(xA)
    xA[:, :duration] = gA.unsqueeze(1)
    xV[:, :duration] = gV.unsqueeze(1)

    # ----- single forward pass --------------------------------------
    net.reset_state(batch_size=B)
    msi_sum = torch.zeros(B, N, device=net.device)

    for t in range(duration):  # only 20 calls now
        net.update_all_layers_batch(xA[:, t], xV[:, t])
        msi_sum += net._latest_sMSI

    # ----- decide “fused vs separated” ------------------------------
    profs = msi_sum.cpu().numpy()
    flags = np.zeros((n_sep, n_trials), dtype=bool)
    for k in range(n_sep):
        s, e = k * n_trials, (k + 1) * n_trials
        for j in range(n_trials):
            flags[k, j] = is_fused(profs[s + j])  # your existing helper

    return flags.mean(1)  #  P(fusion) curve


@torch.no_grad()
def compute_spatial_binding_curve(
        net,
        *,
        separations_deg: Sequence[int],
        n_trials: int = 100,
        intensity: float = 0.5,
        duration: int = 20,
        block_size: int = 32,  # max #trials simultaneously on GPU
):
    """
    Low‑memory computation of P(fusion) vs. spatial disparity.

    Each separation is handled in independent mini‑batches of size
    ≤ `block_size`, so GPU memory usage is essentially constant.
    """
    N = net.n
    space_deg = net.space_size
    rng = np.random.default_rng()

    # ---------- helper functions -------------------------------------
    def idx(deg_arr):
        a = np.asarray(deg_arr, dtype=float)
        return np.round(a * (N - 1) / (space_deg - 1)).astype(int)

    def make_gauss(idx_centres: torch.Tensor):
        xs = torch.arange(N, device=net.device, dtype=torch.float32)
        return torch.exp(
            -0.5 * ((xs - idx_centres.unsqueeze(1)) / net.sigma_in) ** 2
        ) * intensity

    def is_fused(profile: np.ndarray) -> bool:
        sm = gaussian_filter1d(profile, sigma=2, mode="wrap")
        if sm.max() < 1e-6:
            return True
        sm /= sm.max()
        peaks, props = find_peaks(sm, height=0.2, distance=10)
        if len(peaks) <= 1:
            return True
        p1, p2 = peaks[np.argsort(props["peak_heights"])[::-1][:2]]

        def valley(i, j):
            direct = abs(j - i)
            seg = (
                sm[min(i, j): max(i, j) + 1]
                if direct <= N - direct
                else np.r_[sm[max(i, j):], sm[: min(i, j) + 1]]
            )
            return seg.min()

        return valley(p1, p2) / (min(sm[p1], sm[p2]) + 1e-12) > 0.6

    # ---------- main loop over separations ---------------------------
    fusion_prob = []

    for sep in separations_deg:
        fused_trials = 0

        n_blocks = math.ceil(n_trials / block_size)
        for blk in range(n_blocks):
            bs = min(block_size, n_trials - blk * block_size)

            base_deg = rng.integers(30, 150, size=bs)
            locA_deg = base_deg
            locV_deg = (base_deg + sep) % space_deg

            idxA = torch.as_tensor(idx(locA_deg), device=net.device)
            idxV = torch.as_tensor(idx(locV_deg), device=net.device)

            gA = make_gauss(idxA)
            gV = make_gauss(idxV)

            xA = torch.zeros(bs, duration, N, device=net.device)
            xV = torch.zeros_like(xA)
            xA[:, :duration] = gA.unsqueeze(1)
            xV[:, :duration] = gV.unsqueeze(1)

            net.reset_state(batch_size=bs)
            msi_sum = torch.zeros(bs, N, device=net.device)

            for t in range(duration):
                net.update_all_layers_batch(xA[:, t], xV[:, t])
                msi_sum += net._latest_sMSI

            profs = msi_sum.cpu().numpy()
            for pr in profs:
                fused_trials += is_fused(pr)

            # Good GPU hygiene
            del xA, xV, msi_sum, gA, gV, idxA, idxV
            if net.device.type == "cuda":
                torch.cuda.empty_cache()

        fusion_prob.append(fused_trials / n_trials)

    return np.asarray(fusion_prob, dtype=float)


def fit_pedestal_curve(pooled, k_edge=4.0):
    x, y = pooled["separations_deg"], pooled["mean_prob"]
    p0 = [y.min(), y.max(), 25.0]  # base, top, half‑width
    popt, _ = curve_fit(
        lambda X, b, t, w: pedestal(X, b, t, w, k_edge), x, y, p0=p0
    )
    xs = np.linspace(x.min(), x.max(), 600)
    ys = pedestal(xs, *popt, k_edge)
    return xs, ys, popt  # you may want popt later


# ─────────────────── 2. run across checkpoints ───────────────────
def run_spatial_binding_across_models(
        model_paths,
        *,
        separations_deg,
        n_trials=20,
        intensity=1,
        duration=20,
        device="cpu",
        modify_net=None,  # optional modifier
):
    curves = []
    for p in model_paths:
        net = load_msi_model(Path(p), device=device)

        if callable(modify_net):  
            modify_net(net)  # tweak parameters *in‑place*

        curves.append(
            spatial_binding_curve_fast(
                net,
                separations_deg=separations_deg,
                n_trials=n_trials,
                intensity=intensity,
                duration=duration,
            )
        )
        del net
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    curves = np.vstack(curves)
    return {
        "separations_deg": np.asarray(separations_deg, float),
        "mean_prob": curves.mean(0),
        "sem_prob": curves.std(0, ddof=1) / np.sqrt(curves.shape[0]),
        "all_prob": curves,
    }


def gaussian(x, base, amp, mu, sigma):
    return base + amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ---------------- pedestal (3-parameter flattop) ----------------
def pedestal(x, base, top, half_width, k=4.0):
    """
    Smooth symmetric flattop:

        base                     (outside |x| > half_width)
        top                      (inside  |x| < half_width, up to logistic softness)
        half_width: positive; full width ≈ 2*half_width
        k : fixed edge steepness (deg). Smaller = sharper edges.
    """
    left = 1.0 / (1.0 + np.exp(-(x + half_width) / k))
    right = 1.0 / (1.0 + np.exp((x - half_width) / k))
    return base + (top - base) * left * right


def plot_spatial_binding_pedestal(
        pooled,
        *,
        k_edge=4.0,
        level=0.5,
        reference_fit=None,  # (xs_ref, ys_ref) or None
        ref_tint_color="lightcoral",
        ref_tint_alpha=0.18,
):
    import numpy as np, matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    x, y, err = (pooled[k] for k in ("separations_deg",
                                     "mean_prob",
                                     "sem_prob"))

    fit = lambda X, b, t, w: pedestal(X, b, t, w, k_edge)
    p0 = [y.min(), y.max(), 25.0]
    popt, _ = curve_fit(fit, x, y, p0=p0)
    xs_fit = np.linspace(x.min(), x.max(), 600)
    ys_fit = fit(xs_fit, *popt)

    signs = ys_fit - level
    inside = signs[:-1] * signs[1:] <= 0
    cross_m = [xs_fit[i] + (level - ys_fit[i]) *
               (xs_fit[i + 1] - xs_fit[i]) / (ys_fit[i + 1] - ys_fit[i])
               for i in np.where(inside)[0]]

    # ---------- canvas --------------------------------------------------
    y_bottom = -0.05
    y_top = max(1.05, ys_fit.max() * 1.05)
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(x, y, yerr=err, fmt="o", capsize=0,
                elinewidth=1.2, markeredgewidth=0.8,
                label="mean ± SEM (n=10)", zorder=3)
    ax.plot(xs_fit, ys_fit, color="C1", lw=2,
            label="Pedestal fit (manipulated)", zorder=2)

    # ---------- CONTROL pedestal (grey dashed) --------------------------
    if reference_fit is not None:
        xs_ref, ys_ref = reference_fit
        ax.plot(xs_ref, ys_ref, ls="--", color="0.5", lw=2,
                label="Control pedestal", zorder=1.5)

        ref_signs = ys_ref - level
        inside_r = ref_signs[:-1] * ref_signs[1:] <= 0
        cross_r = [xs_ref[i] + (level - ys_ref[i]) *
                   (xs_ref[i + 1] - xs_ref[i]) / (ys_ref[i + 1] - ys_ref[i])
                   for i in np.where(inside_r)[0]]

        for xc in cross_r:
            ax.vlines(xc, y_bottom, level, ls=":", color=ref_tint_color,
                      lw=1.3, zorder=1)
        if len(cross_r) == 2:
            mask_r = (xs_ref >= min(cross_r)) & (xs_ref <= max(cross_r))
            ax.fill_between(xs_ref, y_bottom, level, where=mask_r,
                            color=ref_tint_color, alpha=ref_tint_alpha,
                            zorder=0)

    mask_m = (xs_fit >= min(cross_m)) & (xs_fit <= max(cross_m))
    ax.fill_between(xs_fit, y_bottom, level, where=mask_m,
                    color="C1", alpha=0.15, zorder=1)

    # ---------- cosmetics ----------------------------------------------
    ax.axhline(level, ls=":", color="0.4")
    for xc in cross_m:
        ax.vlines(xc, y_bottom, level, ls=":", color="0.4")


    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])
    ax.set(xlabel="Spatial disparity (°)",
           ylabel="P(fusion)",
           title="Spatial binding window – Pedestal fits",
           ylim=(y_bottom, y_top),
           xlim=(xs_fit.min(), xs_fit.max()))
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_spatial_binding_gaussian(pooled):
    """
    Averaged SBW curve with Gaussian fit, central tint down to the x-axis,
    and vertical dotted lines that touch the axis.
    """
    # ---------- raw means & SEM ------------------------------------------
    x, y, err = pooled["separations_deg"], pooled["mean_prob"], pooled["sem_prob"]
    level = 0.5  # 50 % reference

    # ---------- Gaussian fit --------------------------------------------
    p0 = [y.min(), y.max() - y.min(), 0.0, 20.0]
    popt, _ = curve_fit(gaussian, x, y, p0=p0)
    base, amp, mu, sigma = popt
    xs_fit = np.linspace(x.min(), x.max(), 600)
    ys_fit = gaussian(xs_fit, *popt)

    # two x-values where Gaussian crosses 0.5
    inside = (ys_fit - level)[:-1] * (ys_fit - level)[1:] <= 0
    crossings = []
    for i in np.where(inside)[0]:
        x1, x2, y1, y2 = xs_fit[i], xs_fit[i + 1], ys_fit[i], ys_fit[i + 1]
        crossings.append(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
    crossings = np.asarray(crossings)  # left & right

    # ---------- figure ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))

    # mean ± SEM data
    ax.errorbar(
        x, y, yerr=err,
        fmt="o", capsize=0,
        elinewidth=1.2, markeredgewidth=0.8,
        label="mean ± SEM (n=10)", zorder=3,
    )

    # Gaussian fit
    ax.plot(xs_fit, ys_fit, color="C1", lw=2, label="Gaussian fit", zorder=2)

    y_bottom = -0.05
    y_top = max(1.05, ys_fit.max() * 1.05)
    ax.set_ylim(y_bottom, y_top)

    mask_cent = (xs_fit >= crossings.min()) & (xs_fit <= crossings.max())
    ax.fill_between(
        xs_fit, y_bottom, ys_fit,
        where=mask_cent,
        color="C1", alpha=0.15, zorder=1,
    )

    # 50 % reference line
    ax.axhline(level, ls=":", color="0.4")

    for xc in crossings:
        ax.vlines(xc, ymin=y_bottom, ymax=level, ls=":", color="0.4")
        ax.text(
            xc, level + 0.03,
            f"{abs(xc):.1f}°",
            ha="center", va="bottom", fontsize=8, color="0.25",
        )

    # x-ticks with absolute labels
    xticks = np.arange(-80, 85, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(t):d}" for t in xticks])

    # remaining cosmetics
    ax.set(
        xlabel="Spatial disparity (°)",
        ylabel="P(fusion)",
        title="Spatial binding window – 10-model mean (Gaussian fit)",
        xlim=(xs_fit.min(), xs_fit.max()),
    )
    ax.legend(frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # quick numeric log
    print(f"Gaussian fit: base={base:.3f}, amp={amp:.3f}, μ={mu:.2f}, σ={sigma:.2f}")
    if len(crossings) == 2:
        print(f"Spatial 50 % window: ±{abs(crossings[1]):.1f}°")


# ─────────────────── 4. driver script ───────────────────
# ───────────────────────── main driver ──────────────────────────
def main():
    # ---------- user‑editable parameters ----------
    MODEL_DIR = Path("checkpoint")  # 10 replicas live here
    INTENSITIES = np.array([0.05, .1, .2, .4, .8, 1.6], dtype=float)
    LOC_DEG = 90  # azimuth to stimulate
    SIGMA_IN = 5.0  # Gaussian width (neurons)
    PULSE_LEN = 10  # external frames (≈100 ms)
    N_FRAMES = 20  # simulate a quiet tail
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------

    def integrated_spikes(net, cond, intensity):
        """Return Σ MSI spikes for one trial."""
        # Build a single‑batch stimulus
        gauss = lambda: torch.exp(
            -0.5 * ((torch.arange(net.n, device=DEVICE) -
                     (LOC_DEG * (net.n - 1) / (net.space_size - 1))) / SIGMA_IN) ** 2
        ) * intensity
        xA = torch.zeros(N_FRAMES, net.n, device=DEVICE)
        xV = torch.zeros_like(xA)
        if cond in ("A", "B"):
            xA[:PULSE_LEN] = gauss()
        if cond in ("V", "B"):
            xV[:PULSE_LEN] = gauss()
        net.reset_state(batch_size=1)
        pop = 0.0
        for t in range(N_FRAMES):
            *_, sSum = net.update_all_layers_batch(xA[t:t + 1],
                                                   xV[t:t + 1],
                                                   return_spike_sum=True)
            pop += sSum.sum().item()
        return pop

    # ---------------- run all models -----------------
    models = sorted(MODEL_DIR.glob("msi_model_surr_10_*.pt"))
    resp_A = np.zeros((len(models), INTENSITIES.size))
    resp_V = np.zeros_like(resp_A)
    resp_AV = np.zeros_like(resp_A)

    for m_i, path in enumerate(models):
        net = load_msi_model(path, device=DEVICE)
        for j, I in enumerate(INTENSITIES):
            resp_A[m_i, j] = integrated_spikes(net, "A", I)
            resp_V[m_i, j] = integrated_spikes(net, "V", I)
            resp_AV[m_i, j] = integrated_spikes(net, "B", I)
        del net
        torch.cuda.empty_cache()

    # ---------- multisensory enhancement -------------
    max_uni = np.maximum(resp_A, resp_V)
    mei = (resp_AV - max_uni) / max_uni
    mei_mean = mei.mean(0)
    mei_sem = mei.std(0, ddof=1) / np.sqrt(mei.shape[0])

    # ---------------------------------------------------------------------------

    import matplotlib.pyplot as plt

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80


    fig = plt.figure(figsize=(25, 20))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])  # upper axis
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    for a in (ax, ax2):  # plot on both axes
        a.errorbar(INTENSITIES, mei_mean, yerr=mei_sem,
                   marker='o', capsize=3, ls='-', lw=1.5)

    y_lo, y_hi = 0.00, 0.50  # hide everything between 0 → 0.50
    ax.set_ylim(y_hi * 1.05, mei_mean.max() * 1.10)  # upper panel (> 0.50)
    ax2.set_ylim(y_lo, y_lo + 1e-6)  # lower panel, a sliver around zero

    ax.xaxis.set_visible(False)  # only ONE shared x‑axis (the bottom)
    ax2.set_yticks([])  # no y‑ticks / labels on the spacer

    # cosmetics & “zig‑zag” marks (unchanged)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    d = .015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs);
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs['transform'] = ax2.transAxes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs);
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax2.set_xscale('log')
    ax2.set_xlabel("Stimulus intensity (a.u., log scale)")
    ax.set_ylabel("MEI")
    ax.set_title("Inverse effectiveness in 10‑model ensemble")
    ax2.axhline(0, color='k', lw=.8)
    ax.tick_params(axis='both', which='major', length=30, width=1)
    ax2.tick_params(axis='both', which='major', length=30, width=1)
    plt.tight_layout()
    plt.savefig('./Saved_Images/inv_eff.svg', format='svg')
    plt.show()


# ───────────────────────── entry point ──────────────────────────
if __name__ == "__main__":
    main()



