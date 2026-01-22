from Training import *
from matplotlib import font_manager


def load_msi_model(ckpt_path: Path, *, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    net.to(device).eval()
    net.device = torch.device(device)  # make sure helpers pick this up
    return net

# ─────────────────── 4. driver script ───────────────────
# ───────────────────────── main driver ──────────────────────────
import argparse
import itertools, math, torch

def make_dataset(
        sigma_levels=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
        n_trials=20,
        aud_deg=80.0, vis_deg=100.0,
        stim_frames=10,
        n_neurons=180, space_size=180,
        base_intensity=30.0,        # ↑ bigger default
        gain_exp=2,               # ⬅️  set 1.0  or 2.0
        device="cuda"):

    combos = list(itertools.product(sigma_levels, sigma_levels))
    B = len(combos) * n_trials
    T = stim_frames + 1
    xA = torch.zeros(B, T, n_neurons, device=device)
    xV = torch.zeros_like(xA)
    meta = []

    sigma_ref = min(sigma_levels)
    xs = torch.arange(n_neurons, dtype=torch.float32, device=device)
    idx_a = int(round(aud_deg*(n_neurons-1)/(space_size-1)))
    idx_v = int(round(vis_deg*(n_neurons-1)/(space_size-1)))

    for c_idx, (sigA, sigV) in enumerate(combos):
        # constant‑area Gaussian kernels
        normA = 1/(sigA*math.sqrt(2*math.pi))
        normV = 1/(sigV*math.sqrt(2*math.pi))
        gainA = (sigma_ref/sigA)**gain_exp
        gainV = (sigma_ref/sigV)**gain_exp

        gA = base_intensity * gainA * normA * \
             torch.exp(-0.5*((xs-idx_a)/sigA)**2)
        gV = base_intensity * gainV * normV * \
             torch.exp(-0.5*((xs-idx_v)/sigV)**2)

        for tr in range(n_trials):
            b = c_idx*n_trials + tr
            xA[b,:stim_frames] = gA
            xV[b,:stim_frames] = gV
            meta.append((sigA, sigV))

    return xA, xV, meta


@torch.no_grad()
def reliability_sweep_batched(net, xA, xV, meta,
                              stim_frames=10, method="com"):
    """
    net : loaded MSI model already on correct device
    xA,xV : (B,T,n)  tensors
    meta : list[(σA,σV)] for B rows
    Returns dict keyed by (σA,σV) → list[w_V] (len = n_trials)
    """
    B, T, n = xA.shape
    device = net.device
    net.reset_state(batch_size=B)  # one big batch
    # -------- fast forward pass ----------
    spike_sum = torch.zeros(B, n, device=device)
    for t in range(T):
        net.update_all_layers_batch(xA[:,t], xV[:,t])
        spike_sum += net._latest_sMSI            # integrate over event

    est_deg = decode_msi_location(spike_sum, space_size=net.space_size,
                                  method=method)  # (B,)
    # visual weight
    SA, SV = 80.0, 100.0
    wV = (est_deg.cpu().numpy() - SA) / (SV - SA)

    # collate
    out = {}
    for i,(sigA,sigV) in enumerate(meta):
        out.setdefault((sigA,sigV), []).append(wV[i])
    return out

# ----------------------- aggregation helpers -----------------------------
def pool_to_mean_sem(single_model_dicts):
    pooled = {}
    for d in single_model_dicts:
        for k, vs in d.items():
            pooled.setdefault(k, []).extend(vs)
    out = []
    for (sigA,sigV), vals in pooled.items():
        arr = np.asarray(vals)
        out.append(dict(sigma_a=sigA,
                        sigma_v=sigV,
                        mean_w=arr.mean(),
                        sem_w=arr.std(ddof=1)/math.sqrt(arr.size)))
    return out


def plot_results(pooled, savefig=None):
    sigmas = sorted({d["sigma_a"] for d in pooled})
    Z = np.zeros((len(sigmas), len(sigmas)))
    E = np.zeros_like(Z)
    for d in pooled:
        i = sigmas.index(d["sigma_a"]);
        j = sigmas.index(d["sigma_v"])
        Z[i, j] = d["mean_w"];
        E[i, j] = d["sem_w"]

    # Apply Gaussian smoothing to Z
    from scipy.ndimage import gaussian_filter
    Z = gaussian_filter(Z, sigma=1.0)  # You can adjust sigma for more/less smoothing

    font_path = './fonts/Roboto-Regular.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Roboto'
    plt.rcParams['font.size'] = 80
    plt.rcParams['xtick.labelsize'] = 80
    plt.rcParams['ytick.labelsize'] = 80
    plt.rcParams['axes.titlesize'] = 50
    plt.rcParams['axes.labelsize'] = 80
    plt.rcParams['legend.fontsize'] = 80

    fig, ax = plt.subplots(figsize=(25, 23))
    im = ax.imshow(Z, origin="lower", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(sigmas)));
    ax.set_yticks(range(len(sigmas)))
    ax.set_xticklabels(sigmas);
    ax.set_yticklabels(sigmas)
    ax.set_xlabel("σV");
    ax.set_ylabel("σA")
    ax.set_title("Visual weight  $w_V$  (mean ± SEM)")
    # for i in range(len(sigmas)):
    #     for j in range(len(sigmas)):
    #         ax.text(j,i,f"{Z[i,j]:.2f}\n±{E[i,j]:.2f}",
    #                 ha="center",va="center",
    #                 color="white" if Z[i,j]<0.5 else "black",
    #                 fontsize=6)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("$w_V$")
    ax.tick_params(axis='both', which='major', length=30, width=1)
    plt.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig('./Saved_Images/cue_rel.svg', format='svg')
    plt.show()

# ---------------------------------------------------------------------
#  EXTRA PLOTS  –  scatter & error‑heat‑map
# ---------------------------------------------------------------------
def plot_weight_vs_prediction(pooled, savefig=None):
    """
    Scatter of empirical visual weight w_V versus the inverse‑variance
    prediction w_V*  with vertical SEM bars.
    """
    import numpy as np, matplotlib.pyplot as plt

    x_pred, y_emp, y_err = [], [], []
    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)  # σ⁻² / Σ σ⁻²
        x_pred.append(w_pred)
        y_emp.append(d["mean_w"])
        y_err.append(d["sem_w"])

    x_pred = np.asarray(x_pred)
    y_emp  = np.asarray(y_emp)
    y_err  = np.asarray(y_err)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.errorbar(x_pred, y_emp, yerr=y_err, fmt="o", capsize=2)
    ax.plot([0, 1], [0, 1], "--k", lw=1, label="$y=x$  (ideal)")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel="Predicted  $w_V^{*}$",
           ylabel="Model $w_V$",
           title="Cue‑reliability weighting")
    ax.legend(frameon=False)
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300)
    plt.show()


def plot_error_matrix(pooled, savefig=None):
    """
    Heat‑map of |w_emp - w_pred| for every (σA, σV).
    """
    import numpy as np, matplotlib.pyplot as plt
    sigmas = sorted({d["sigma_a"] for d in pooled})
    Z = np.zeros((len(sigmas), len(sigmas)))

    for d in pooled:
        i = sigmas.index(d["sigma_a"])
        j = sigmas.index(d["sigma_v"])
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        Z[i, j] = abs(d["mean_w"] - w_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Z, origin="lower", cmap="magma_r",
                   extent=[0, len(sigmas), 0, len(sigmas)],
                   vmin=0, vmax=Z.max())
    ax.set_xticks(np.arange(len(sigmas)) + 0.5)
    ax.set_yticks(np.arange(len(sigmas)) + 0.5)
    ax.set_xticklabels(sigmas)
    ax.set_yticklabels(sigmas)
    ax.set_xlabel("σV  (deg)")
    ax.set_ylabel("σA  (deg)")
    ax.set_title("|  $w_V$ − $w_V^{*}$  |")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04).set_label("abs. error")
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300)
    plt.show()

# ----------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="checkpoint")
    ap.add_argument("--pattern", default="msi_model_surr_10_*.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=20,
                    help="trials per (σA,σV)")
    ap.add_argument("--substeps", type=int, default=None,
                    help="override net.n_substeps during validation")
    args = ap.parse_args()

    device = args.device
    xA,xV,meta = make_dataset(n_trials=args.trials, device=device)

    model_paths = sorted(Path(args.model_dir).glob(args.pattern))[:10]
    if len(model_paths)<10: raise RuntimeError("Need ≥10 models")

    all_results = []
    t0=time.time()
    for idx,p in enumerate(model_paths,1):
        print(f"[{idx}/10]  {p.name}  …", end="", flush=True)
        net = load_msi_model(p, device=device)
        # speed lever: cut sub‑steps
        if args.substeps is not None:
            net.n_substeps = args.substeps
        # disable probes / plasticity bookkeeping
        net._probe = None
        net.allow_inhib_plasticity=False

        res = reliability_sweep_batched(net, xA, xV, meta)
        all_results.append(res)
        print(" done")
        del net; torch.cuda.empty_cache()
    print(f"✓ finished in {time.time()-t0:.1f}s")

    pooled = pool_to_mean_sem(all_results)
    deviations = []
    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        w_emp = d["mean_w"]
        deviations.append(abs(w_emp - w_pred) / w_pred * 100)

    print(f"\nAverage deviation from MLE predictions: {np.mean(deviations):.1f}%")

    mae_list = []
    squared_errors = []
    w_pred_all = []
    w_emp_all = []

    for d in pooled:
        var_a, var_v = d["sigma_a"] ** 2, d["sigma_v"] ** 2
        w_pred = 1.0 / var_v / (1.0 / var_a + 1.0 / var_v)
        w_emp = d["mean_w"]

        mae_list.append(abs(w_emp - w_pred))
        squared_errors.append((w_emp - w_pred) ** 2)
        w_pred_all.append(w_pred)
        w_emp_all.append(w_emp)

    print(f"Mean absolute error: {np.mean(mae_list):.3f}")
    print(f"RMSE: {np.sqrt(np.mean(squared_errors)):.3f}")
    print(f"R² correlation: {np.corrcoef(w_pred_all, w_emp_all)[0, 1] ** 2:.3f}")

    plot_results(pooled, savefig="cue_reliability_fast.png")


    plot_weight_vs_prediction(pooled, savefig="cue_reliability_scatter.png")


    plot_error_matrix(pooled, savefig="cue_reliability_abs_error.png")

if __name__ == "__main__":
    main()


