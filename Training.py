import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize


class Positive(nn.Module):
    def forward(self, θ):
        return F.softplus(θ) - math.log(2.0)

    # optional override
    def right_inverse(self, W):
        eps = 1e-6
        return torch.log(torch.exp(W + math.log(2.0)) - 1.0 + eps)


class NonNegative(nn.Module):
    """ Strictly ≥ 0 by using plain soft‑plus (no ln2 shift). """

    def forward(self, theta):
        return F.softplus(theta)  # ≥ 0

    def right_inverse(self, W):
        eps = 1e-6
        return torch.log(torch.exp(W) - 1.0 + eps)



class AMPANMDADebugger:
    """
    Light-weight accumulator that runs silently during training.
    Activate with  net._probe = AMPANMDADebugger()
    and call   net._probe.report(epoch)   after the epoch finishes.
    """

    def __init__(self):
        self.reset()

    # --------------------------------------------------
    def reset(self):
        self.t = 0
        self.sums = defaultdict(float)

    @torch.no_grad()
    def log_EI(self, Q_exc, Q_inh):
        self.sums["Q_exc"] += Q_exc.sum().item()
        self.sums["Q_inh"] += Q_inh.sum().item()

    # --------------------------------------------------
    @torch.no_grad()
    def log(self, Q_ampa, Q_nmda, I_M, sA, sV, sM,
            R_a, R_v, mg_gate,
            J_ampa_inst, J_nmda_inst, n_spikes):  # instantaneous current args
        """
        J_ampa / J_nmda     : Tensors (B,n) – effective contributions in this sub-step
        J_ampa_inst / J_nmda_inst : Tensors (B,n) - instantaneous currents from new spikes
        I_M                 : Tensor (B,n)  – net current after all terms
        sA,sV,sM            : Spikes (B,n)
        R_a, R_v            : Tsodyks resources (B,n)
        mg_gate             : Tensor (B,n)  – Mg block factor 0…1
        """
        # current totals
        self.sums["Q_ampa"] += Q_ampa.sum().item()
        self.sums["Q_nmda"] += Q_nmda.sum().item()
        self.sums["I_exc"] += torch.clamp(I_M, min=0).sum().item()
        self.sums["I_inh"] += -torch.clamp(I_M, max=0).sum().item()

        # instant current logging
        self.sums["J_ampa_inst"] += J_ampa_inst.sum().item()
        self.sums["J_nmda_inst"] += J_nmda_inst.sum().item()
        self.sums["n_spikes"] += n_spikes

        # stats
        self.sums["spk_A"] += sA.sum().item()
        self.sums["spk_V"] += sV.sum().item()
        self.sums["spk_M"] += sM.sum().item()
        self.sums["R_a"] += R_a.mean().item()
        self.sums["R_v"] += R_v.mean().item()
        self.sums["mg"] += mg_gate.mean().item()
        self.t += 1

    # --------------------------------------------------
    def report(self, net, tag=""):
        if self.t == 0:
            print("[probe] No samples collected.")
            return
        g = self.sums  # alias
        mean = lambda k: g[k] / self.t

        print("\n──────── AMPA vs NMDA probe", tag, "────────")
        print("  Charge delivered in one epoch (∫ I dt):")
        print(f"    Q_AMPA : {g['Q_ampa']:>12.3e}")
        print(f"    Q_NMDA : {g['Q_nmda']:>12.3e}")
        print(f"    Q_NMDA / Q_AMPA : {g['Q_nmda'] / max(g['Q_ampa'], 1e-9):9.5f}")

        print("\n  Instantaneous Current Comparison (per-spike impact):")
        print(f"    J_AMPA_inst total : {g['J_ampa_inst']:>10.1f}")
        print(f"    J_NMDA_inst total : {g['J_nmda_inst']:>10.1f}")
        ratio_inst = g['J_nmda_inst'] / max(g['J_ampa_inst'], 1e-9)
        print(f"    J_NMDA_inst / J_AMPA_inst : {ratio_inst:6.3f}")
        if self.sums["n_spikes"] > 0:
            print(f"  ⟨J_AMPA⟩/spk : {mean('J_ampa_inst') / mean('n_spikes'):.3e}")

        print("\n  General Network Stats:")
        print(f"    mean E/I ratio  : {mean('I_exc') / max(mean('I_inh'), 1e-9):6.3f}")
        print(f"    Exc charge : {g['Q_exc']:>12.3e}")
        print(f"    Inh charge : {g['Q_inh']:>12.3e}")
        print(f"    E/I charge ratio : {g['Q_exc'] / max(g['Q_inh'], 1e-9):9.5f}")
        print(f"    mean R_a, R_v   : {mean('R_a'):.3f}, {mean('R_v'):.3f}")
        print(f"    mean Mg-gate    : {mean('mg'):.3f}")
        print(f"    spikes A|V|M    : {int(g['spk_A'])} , "
              f"{int(g['spk_V'])} , {int(g['spk_M'])}")
        print("────────────────────────────────────────────\n")


########################################################
#             UTILITY / GENERATION FUNCTIONS
########################################################

def location_to_index(loc_deg, n, space_size=180):
    if n <= 1:
        return 0
    frac = loc_deg / float(space_size - 1)
    return int(round(frac * (n - 1)))


def index_to_location(idx, n, space_size=180):
    if n <= 1:
        return 0
    frac = idx / float(n - 1)
    return frac * (space_size - 1)


def make_gaussian_vector_batch_gpu(center_indices, size=180, sigma=5.0, device=None):
    xs = torch.arange(size, dtype=torch.float32, device=device)
    centers = center_indices.view(-1, 1)
    dist = torch.abs(xs - centers)
    return torch.exp(-0.5 * (dist / sigma) ** 2)


def generate_event_loc_seq_batch(batch_size=32,
                                 space_size=180,
                                 offset_probability=0.1,
                                 event_duration=5,
                                 p_start=0.2,
                                 p_shift_visual=None,
                                 p_shift_audio=None,
                                 offset_range_deg=3.0,
                                 temporal_jitter_max=0):
    """
    Create synthetic A/V sequences with spatial offsets and optional temporal jitter.

    Parameters
    ----------
    temporal_jitter_max : int
        Max A/V onset offset in frames (0 disables jitter).
    """
    if p_shift_visual is None:
        p_shift_visual = offset_probability
    if p_shift_audio is None:
        p_shift_audio = offset_probability

    T = 20
    D = event_duration

    loc_seqs, mod_seqs, offs, lens = [], [], [], []

    def _merge_mode(prev: str, new: str) -> str:
        """Merge per-frame modality tags into {'A','V','B'}."""
        if prev == 'X':
            return new
        if prev == new or prev == 'B':
            return prev
        return 'B'

    for _ in range(batch_size):
        loc = [999] * T
        mode = ['X'] * T
        offA = [0.0] * T
        offV = [0.0] * T

        t = 0
        while t <= T - D:
            if np.random.rand() < p_start:
                az = int(np.random.randint(0, space_size))
                r = np.random.rand()
                mode_tag = 'B' if r < 0.6 else ('A' if r < 0.8 else 'V')
                shiftA = (np.random.rand() < p_shift_audio)
                shiftV = (np.random.rand() < p_shift_visual)
                deltaA = np.random.uniform(-offset_range_deg, offset_range_deg) if shiftA else 0.0
                deltaV = np.random.uniform(-offset_range_deg, offset_range_deg) if shiftV else 0.0

                # -------------------------------
                # temporal jitter
                # -------------------------------
                dt_frames = 0
                if mode_tag == 'B' and temporal_jitter_max and temporal_jitter_max > 0:
                    max_shift = max(0, (T - D) - t)
                    dt_frames = int(np.random.randint(-temporal_jitter_max, temporal_jitter_max + 1))
                    if abs(dt_frames) > max_shift:
                        dt_frames = int(np.sign(dt_frames) * max_shift)

                a_start = t + (-dt_frames if dt_frames < 0 else 0)
                v_start = t + (dt_frames if dt_frames > 0 else 0)
                window_len = D + abs(dt_frames)

                # audio frames
                if mode_tag in ('A', 'B'):
                    for tau in range(D):
                        idx = a_start + tau
                        if idx >= T:
                            break
                        loc[idx] = az
                        mode[idx] = _merge_mode(mode[idx], 'A')
                        offA[idx] = deltaA

                # visual frames
                if mode_tag in ('V', 'B'):
                    for tau in range(D):
                        idx = v_start + tau
                        if idx >= T:
                            break
                        loc[idx] = az
                        mode[idx] = _merge_mode(mode[idx], 'V')
                        offV[idx] = deltaV

                t += window_len
            else:
                t += 1

        loc_seqs.append(loc)
        mod_seqs.append(mode)
        offs.append({"A": offA, "V": offV})
        lens.append(T)

    return loc_seqs, mod_seqs, offs, lens




def assign_unimodal_preferred_locations(net):
    """Assign preferred location markers to unimodal A/V neurons."""

    n = net.n
    sp_size = net.space_size

    # Evenly space across map

    net.unimodal_prefA = []
    net.unimodal_prefV = []
    for i in range(n):
        locA = (i / float(n - 1)) * (sp_size - 1)
        locV = (i / float(n - 1)) * (sp_size - 1)

        net.unimodal_prefA.append(locA)
        net.unimodal_prefV.append(locV)

    print("[INFO] Assigned genetic preferred loc for each unimodal neuron (A/V).")


def assign_msi_preferred_locations(net):
    """Assign preferred location markers to MSI excitatory neurons."""

    n = net.n
    sp_size = net.space_size

    net.msi_prefA = []
    net.msi_prefV = []
    for i in range(n):
        loc_val = (i / float(n - 1)) * (sp_size - 1)

        net.msi_prefA.append(loc_val)
        net.msi_prefV.append(loc_val)

    print("[INFO] Assigned 'genetic' preferred loc for each MSI excit neuron (A->MSI, V->MSI).")


# ----------------------------------------------------------------------
# Hebbian/Oja topographic anchor
# ----------------------------------------------------------------------
def apply_topographic_anchor_unimodal(net, layer="A", lr=1e-4, sigma=5.0):
    W = net.W_inA if layer == "A" else net.W_inV  # (n,n)
    spikes = net._latest_sA if layer == "A" else net._latest_sV  # (B,n)
    r = spikes.mean(0)  # (n,)

    n = net.n
    idx = torch.arange(n, device=W.device)
    G = torch.exp(-0.5 * ((idx[:, None] - idx[None, :]) / sigma) ** 2)  # (n,n)

    dW = lr * (r.unsqueeze(1) * G)  # Hebbian growth
    attr = "W_inA" if layer == "A" else "W_inV"
    net._p_add(attr, dW - lr * (r.unsqueeze(1) * W))  # Oja decay term


def apply_topographic_anchor_msi(net, layer="A", lr=1e-4, sigma=5.0):
    """Hebbian/Oja topographic anchoring for A/V->MSI feedforward weights."""
    # Select connection
    if layer == "A":
        W_AMPA = net.W_a2msi_AMPA
        W_NMDA = net.W_a2msi_NMDA
        spikes_in = net._latest_sA  # presyn
        # MSI spikes are post
    else:
        W_AMPA = net.W_v2msi_AMPA
        W_NMDA = net.W_v2msi_NMDA
        spikes_in = net._latest_sV

    s_post = net._latest_sMSI  # shape (B, n)
    r_post = s_post.mean(dim=0)  # average over batch => shape (n,)

    n_msi = net.n
    idx = torch.arange(n_msi, device=W_AMPA.device)
    G = torch.exp(-0.5 * ((idx[:, None] - idx[None, :]) / 2.0) ** 2)
    dW_ampa = lr * (r_post.unsqueeze(1) * G)  # shape (n,n)
    dW_ampa_decay = lr * (r_post.unsqueeze(1) * W_AMPA)
    net._p_add("W_a2msi_AMPA" if layer == "A" else "W_v2msi_AMPA", dW_ampa - dW_ampa_decay)
    dW_nmda = lr * (r_post.unsqueeze(1) * G)
    dW_nmda_decay = lr * (r_post.unsqueeze(1) * W_NMDA)
    net._p_add("W_a2msi_NMDA" if layer == "A" else "W_v2msi_NMDA", dW_nmda - dW_nmda_decay)


import torch


def decode_msi_location(
        spikes_t: torch.Tensor,
        space_size: int = 180,
        method: str = "com"
) -> torch.Tensor:
    """
    Parameters
    ----------
    spikes_t : (B, n) tensor
        Spike counts or rates of MSI excitatory neurons at one time-step
        *or* summed across the duration of an event.
    space_size : int
        Degrees represented by the map (same value you pass to
        `MultiBatchAudVisMSINetworkTime`, default 180).
    method : {"argmax", "com"}
        * "argmax": winner-take-all
        * "com"   : centre of mass
    Returns
    -------
    pred_deg : (B,) tensor
        Predicted azimuth in degrees for each item in the batch.
    """
    B, n = spikes_t.shape
    device = spikes_t.device
    idxs = torch.arange(n, device=device, dtype=torch.float32)  # 0 … n-1

    if method == "argmax":
        pred_idx = torch.argmax(spikes_t, dim=1).float()  # (B,)
    elif method == "com":
        num = torch.sum(spikes_t * idxs, dim=1)  # (B,)
        den = torch.sum(spikes_t, dim=1).clamp_min(1e-6)  # avoid /0
        pred_idx = num / den
    else:
        raise ValueError("method must be 'argmax' or 'com'")

    pred_deg = pred_idx * (space_size - 1) / (n - 1)
    return pred_deg


def decode_local_com(spikes, half_width=3):
    idx = torch.arange(spikes.size(-1), device=spikes.device)
    c = torch.argmax(spikes, -1, keepdim=True)  # peak index
    mask = (idx >= c - half_width) & (idx <= c + half_width)  # ±3 neighbours
    sp = spikes * mask
    return decode_msi_location(sp, method="com")  # same utility


def apply_local_competition_unimodal(
        net,
        layer="A",
        beta=1e-5,
        neighbor_dist=5
):
    """Local decorrelation for neighboring unimodal feedforward weights."""
    spikes = net._latest_sA if (layer == "A") else net._latest_sV
    spk_avg = spikes.mean(dim=0)  # shape (n,)

    # local range
    n = net.n
    W = net.W_inA if (layer == "A") else net.W_inV

    with torch.no_grad():
        for i in range(n):
            si = spk_avg[i].item()
            if si < 1e-9:
                continue
            # for j in [i-neighbor_dist..i+neighbor_dist], j!=i
            j_low = max(0, i - neighbor_dist)
            j_high = min(n, i + neighbor_dist + 1)
            for j in range(j_low, j_high):
                if j == i:
                    continue
                sj = spk_avg[j].item()
                if sj < 1e-9:
                    continue

                # minimal approach:
                W[i, :] -= beta * si * sj * (W[j, :] - W[i, :])
                W[j, :] -= beta * si * sj * (W[i, :] - W[j, :])


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def apply_local_competition_unimodal_fast(net,
                                          layer: str = "A",
                                          beta: float = 1e-4,
                                          neighbour_dist: int = 5,
                                          target_norm: float = None):
    """
    Lateral competition with heterosynaptic LTD **plus an L2 row clamp**.
    After subtractive LTD each row is renormalised to *target_norm*
    (default = median row-norm at call-time) so rows keep their total
    drive but are forced to concentrate on fewer presynaptic neurons.
    """
    W = net.W_inA if layer == "A" else net.W_inV
    spikes = net._latest_sA if layer == "A" else net._latest_sV
    r = spikes.mean(0)  # (n,)

    n = net.n
    idx = torch.arange(n, device=W.device)
    M = ((idx[:, None] - idx[None, :]).abs() <= neighbour_dist).float()
    s = torch.matmul(M, r)  # neighbour firing sum

    # heterosynaptic LTD (multiplicative)
    dW = -beta * (r * s).unsqueeze(1) * W
    net._p_add("W_inA" if layer == "A" else "W_inV", dW)

    # ---------------- L2 row-normalisation -----------------
    if target_norm is None:
        with torch.no_grad():
            target_norm = W.norm(p=2, dim=1, keepdim=True).median()

    with torch.no_grad():
        W_now = net.W_inA if layer == "A" else net.W_inV
        row_norm = W_now.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        W_now.mul_(target_norm / row_norm)


def apply_local_competition_msi_fast(net, beta=5e-4, neighbour_dist=5):
    """Heterosynaptic LTD for MSI feedforward weights (decorrelation)."""
    # MSI excit spikes
    s_msi = net._latest_sMSI  # shape (B, n)
    r = s_msi.mean(dim=0)  # shape (n,)

    n = net.n
    idx = torch.arange(n, device=s_msi.device)
    # Linear distance
    M = ((idx[:, None] - idx[None, :]).abs() <= neighbour_dist).float()

    # sum of neighbor firing
    s = torch.matmul(M, r)  # shape (n,)

    d_factor = -beta * (r * s).unsqueeze(1)

    # A->MSI
    net._p_add("W_a2msi_AMPA", d_factor * net.W_a2msi_AMPA)
    net._p_add("W_a2msi_NMDA", d_factor * net.W_a2msi_NMDA)

    # V->MSI
    net._p_add("W_v2msi_AMPA", d_factor * net.W_v2msi_AMPA)
    net._p_add("W_v2msi_NMDA", d_factor * net.W_v2msi_NMDA)


def soft_row_scaling(net, target_norm=1.0, eps=1e-3):
    for attr in ("W_inA", "W_inV"):
        W = getattr(net, attr)
        with torch.no_grad():
            row_norm = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            W.mul_(1.0 + eps * (target_norm / row_norm - 1.0))


# ----------------------------------------------------------------------
#  Long-timescale multiplicative scaling (biologically plausible)
# ----------------------------------------------------------------------
def slow_synaptic_scaling(W: torch.Tensor,
                          tau_hours: float = 2.0,
                          target_mean: float = 0.006,
                          dt_minutes: float = 1.0):
    """Slow homeostatic row-scaling toward target mean weight."""
    alpha = dt_minutes / (tau_hours * 60.0)
    with torch.no_grad():
        row_mean = W.mean(dim=1, keepdim=True).clamp_min(1e-9)
        scale = target_mean / row_mean
        W.mul_(1.0 + alpha * (scale - 1.0))


def generate_av_batch_tensor(
        loc_seqs,
        mod_seqs,
        offset_applied,
        n=180,
        space_size=180,
        sigma_in=5.0,
        noise_std=0.01,
        loc_jitter_std=0.0,
        stimulus_intensity=1.0,
        device=None,
        max_len=None
):
    """
    Build analog A/V inputs (batch_size, T, n) with optional spatial offsets.

    Backwards compatibility:
      • If offset_applied[b] is a bool:
          False → no offset (legacy).
          True  → legacy behavior: per-frame V-only jitter (±3 deg),
                  A remains unshifted (old code path).
      • If offset_applied[b] is a dict or (A_seq, V_seq):
          Use per-frame offsets for A and V respectively (new path).

    The rest (Gaussian profiles, additive noise, Poisson conversion downstream)
    is unchanged.
    """
    batch_size = len(loc_seqs)
    if max_len is None:
        max_len = max(len(seq) for seq in loc_seqs)

    xA_batch = torch.zeros((batch_size, max_len, n), dtype=torch.float32, device=device)
    xV_batch = torch.zeros((batch_size, max_len, n), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for b in range(batch_size):
        loc_seq = loc_seqs[b]
        mod_seq = mod_seqs[b]
        offs = offset_applied[b] if offset_applied is not None else False

        T = len(loc_seq)
        valid_mask[b, :T] = True

        a_off_seq = None
        v_off_seq = None
        if isinstance(offs, dict):
            a_off_seq = offs.get("A", None)
            v_off_seq = offs.get("V", None)
        elif isinstance(offs, (tuple, list)) and len(offs) == 2:
            a_off_seq, v_off_seq = offs[0], offs[1]

        for t in range(T):
            loc_t_deg = loc_seq[t]
            if loc_t_deg == 999 or mod_seq[t] == 'X':
                continue

            if loc_jitter_std > 0.0:
                loc_t_deg = float(np.clip(
                    loc_t_deg + np.random.normal(0, loc_jitter_std),
                    0, space_size - 1))

            # pick offsets for this frame
            if a_off_seq is not None:
                a_off = a_off_seq[t] if t < len(a_off_seq) else 0.0
            else:
                a_off = 0.0  # legacy path never shifted A

            if v_off_seq is not None:
                v_off = v_off_seq[t] if t < len(v_off_seq) else 0.0
            else:
                v_off = (np.random.uniform(-3, 3) if bool(offs) else 0.0)

            # convert to neuron indices
            center_idxA = location_to_index(loc_t_deg + a_off, n, space_size)
            center_idxV = location_to_index(loc_t_deg + v_off, n, space_size)

            # build analog Gaussians
            gaussA = make_gaussian_vector_batch_gpu(
                torch.tensor([center_idxA], device=device),
                n, sigma_in, device
            ) * stimulus_intensity

            gaussV = make_gaussian_vector_batch_gpu(
                torch.tensor([center_idxV], device=device),
                n, sigma_in, device
            ) * stimulus_intensity

            # respect modality for this frame
            if mod_seq[t] == 'A':
                gaussV.zero_()
            elif mod_seq[t] == 'V':
                gaussA.zero_()

            # additive noise (unchanged)
            if noise_std and noise_std > 0.0:
                gaussA += torch.randn(1, n, device=device) * noise_std
                gaussV += torch.randn(1, n, device=device) * noise_std

            xA_batch[b, t] = gaussA[0]
            xV_batch[b, t] = gaussV[0]

    return xA_batch, xV_batch, valid_mask


import torch
import numpy as np


def visualize_unimodal_gaussian_response_with_msi(
        net,
        center_deg=90,
        n_steps=15,
        sigma_in=5.0,
        layer="A",  # "A" or "V"
        pulse_duration=5,
        stimulus_intensity=1.0,
        device=None,
        figsize=(10, 12),
        style="seaborn-talk",
        show=True
):
    """Plot unimodal + MSI spike rasters for a Gaussian pulse."""

    if device is None:
        device = net.device

    if style is not None:
        plt.style.use(style)
    net.reset_state(batch_size=1)
    xA = torch.zeros((n_steps, net.n), dtype=torch.float32, device=device)
    xV = torch.zeros((n_steps, net.n), dtype=torch.float32, device=device)

    # Convert center_deg => index
    center_idx = int(round((center_deg / (net.space_size - 1)) * (net.n - 1)))
    center_idx = max(0, min(net.n - 1, center_idx))

    # 1D Gaussian vector
    xs = torch.arange(net.n, device=device, dtype=torch.float32)
    dist = xs - center_idx
    gauss_vec = torch.exp(-0.5 * (dist / sigma_in) ** 2) * stimulus_intensity

    if layer == "A":
        xA[:pulse_duration] = gauss_vec
    else:
        xV[:pulse_duration] = gauss_vec

    input_center_neuron = []
    for t in range(n_steps):
        if layer == "A":
            input_center_neuron.append(xA[t, center_idx].item())
        else:
            input_center_neuron.append(xV[t, center_idx].item())
    unimodal_spk_records = []  # [(t, [firing_neurons])]
    unimodal_spikes_per_t = []

    msi_spk_records = []  # same but for MSI
    msi_spikes_per_t = []

    for t in range(n_steps):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))

        # --- unimodal layer spikes ---
        if layer == "A":
            spikes_uni = net._latest_sA[0]
        else:
            spikes_uni = net._latest_sV[0]
        firing_uni = (spikes_uni > 0.5).nonzero(as_tuple=True)[0]
        unimodal_spk_records.append((t, firing_uni.detach().cpu().numpy()))
        unimodal_spikes_per_t.append(firing_uni.numel())

        # --- MSI spikes ---
        spikes_msi = net._latest_sMSI[0]
        firing_msi = (spikes_msi > 0.5).nonzero(as_tuple=True)[0]
        msi_spk_records.append((t, firing_msi.detach().cpu().numpy()))
        msi_spikes_per_t.append(firing_msi.numel())
    fig = plt.figure(figsize=figsize)

    fig.suptitle(
        f"Unimodal '{layer}' + MSI response to Gaussian\n"
        f"(center={center_deg}°, sigma={sigma_in}, pulse={pulse_duration} steps)",
        fontsize=16, fontweight='bold'
    )

    ax_input = fig.add_subplot(5, 1, 1)
    ax_input.plot(range(n_steps), input_center_neuron, marker='o', color='C0', label='Center Input')
    ax_input.set_ylabel("Input Amplitude", fontsize=12)
    ax_input.set_title("Stimulus at Center Neuron vs. Time", fontsize=12)
    ax_input.grid(True, alpha=0.3)
    ax_input.legend(loc="best")

    # -- (B) Unimodal Raster --
    ax_uni_raster = fig.add_subplot(5, 1, 2)
    all_t_uni = []
    all_idx_uni = []
    all_colors_uni = []
    for t, neuron_idxs in unimodal_spk_records:
        if len(neuron_idxs) > 0:
            all_t_uni.extend([t] * len(neuron_idxs))
            all_idx_uni.extend(neuron_idxs.tolist())
            all_colors_uni.extend(neuron_idxs.tolist())
    sc_uni = ax_uni_raster.scatter(all_t_uni, all_idx_uni, c=all_colors_uni, cmap='viridis', marker='|', s=80)
    ax_uni_raster.set_ylabel("Unimodal Neuron Index", fontsize=12)
    ax_uni_raster.set_title(f"{layer}-Layer Raster", fontsize=12)
    ax_uni_raster.set_ylim([-1, net.n])
    ax_uni_raster.grid(True, alpha=0.2)
    cb_uni = plt.colorbar(sc_uni, ax=ax_uni_raster, orientation='vertical', shrink=0.65)
    cb_uni.set_label('Neuron Index', fontsize=12)

    ax_uni_line = fig.add_subplot(5, 1, 3)
    ax_uni_line.plot(range(n_steps), unimodal_spikes_per_t, '-o', color='C1', label=f'Total Spikes ({layer})')
    ax_uni_line.set_ylabel("Spikes", fontsize=12)
    ax_uni_line.set_title(f"Unimodal '{layer}' Spikes per Time Step", fontsize=12)
    ax_uni_line.grid(True, alpha=0.3)
    ax_uni_line.legend(loc="best")

    # -- (D) MSI Raster --
    ax_msi_raster = fig.add_subplot(5, 1, 4)
    all_t_msi = []
    all_idx_msi = []
    all_colors_msi = []
    for t, neuron_idxs in msi_spk_records:
        if len(neuron_idxs) > 0:
            all_t_msi.extend([t] * len(neuron_idxs))
            all_idx_msi.extend(neuron_idxs.tolist())
            all_colors_msi.extend(neuron_idxs.tolist())
    sc_msi = ax_msi_raster.scatter(all_t_msi, all_idx_msi, c=all_colors_msi, cmap='plasma', marker='|', s=80)
    ax_msi_raster.set_ylabel("MSI Neuron Index", fontsize=12)
    ax_msi_raster.set_title("MSI Raster of Spikes", fontsize=12)
    ax_msi_raster.set_ylim([-1, net.n])
    ax_msi_raster.grid(True, alpha=0.2)
    cb_msi = plt.colorbar(sc_msi, ax=ax_msi_raster, orientation='vertical', shrink=0.65)
    cb_msi.set_label('Neuron Index', fontsize=12)

    ax_msi_line = fig.add_subplot(5, 1, 5)
    ax_msi_line.plot(range(n_steps), msi_spikes_per_t, '-o', color='C2', label='Total Spikes (MSI)')
    ax_msi_line.set_xlabel("Time step", fontsize=12)
    ax_msi_line.set_ylabel("Spikes", fontsize=12)
    ax_msi_line.set_title("MSI Spikes per Time Step", fontsize=12)
    ax_msi_line.grid(True, alpha=0.3)
    ax_msi_line.legend(loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()

    return fig


def visualize_unimodal_gaussian_response_with_msi_rate(
        net,
        center_deg=90,
        n_steps=15,
        sigma_in=5.0,
        layer="A",  # "A" or "V"
        pulse_duration=5,
        stimulus_intensity=1.0,
        device=None,
        figsize=(10, 12),
        style="seaborn-talk",
        show=True
):
    """Plot spike rasters with colour encoding mean firing rate."""

    if device is None:
        device = net.device
    if style:
        plt.style.use(style)
    net.reset_state(batch_size=1)
    xA = torch.zeros((n_steps, net.n), device=device)
    xV = torch.zeros_like(xA)

    idx_c = int(round((center_deg / (net.space_size - 1)) * (net.n - 1)))
    idx_c = max(0, min(net.n - 1, idx_c))

    xs = torch.arange(net.n, device=device, dtype=torch.float32)
    gauss_vec = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * stimulus_intensity
    if layer == "A":
        xA[:pulse_duration] = gauss_vec
    else:
        xV[:pulse_duration] = gauss_vec
    uni_spk = torch.zeros((n_steps, net.n), device=device)
    msi_spk = torch.zeros((n_steps, net.n), device=device)
    in_amp = (xA if layer == "A" else xV)[:, idx_c].cpu().tolist()

    for t in range(n_steps):
        net.update_all_layers_batch(xA[t].unsqueeze(0), xV[t].unsqueeze(0))
        if layer == "A":
            uni_spk[t] = net._latest_sA[0]
        else:
            uni_spk[t] = net._latest_sV[0]
        msi_spk[t] = net._latest_sMSI[0]
    uni_rate = uni_spk.mean(dim=0).cpu()  # (n,)
    msi_rate = msi_spk.mean(dim=0).cpu()  # (n,)

    # avoid division-by-zero colour scaling
    eps = 1e-9
    uni_rate_norm = (uni_rate - uni_rate.min()) / (uni_rate.max() - uni_rate.min() + eps)
    msi_rate_norm = (msi_rate - msi_rate.min()) / (msi_rate.max() - msi_rate.min() + eps)

    uni_colour_map = uni_rate_norm.numpy()
    msi_colour_map = msi_rate_norm.numpy()
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"'{layer}' + MSI response | colour = mean firing-rate",
        fontsize=16, fontweight="bold"
    )

    # (A) stimulus trace
    ax_in = fig.add_subplot(5, 1, 1)
    ax_in.plot(range(n_steps), in_amp, '-o')
    ax_in.set(ylabel="Input amp.", title="Stimulus at centre neuron")
    ax_in.grid(alpha=.3)

    def build_raster_lists(spk_tensor):
        times, ids, cols = [], [], []
        for t in range(n_steps):
            active = (spk_tensor[t] > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            if active:
                times.extend([t] * len(active))
                ids.extend(active)
        return times, ids

    # (B) unimodal raster
    t_uni, n_uni = build_raster_lists(uni_spk)
    c_uni = [uni_colour_map[i] for i in n_uni]
    ax_ru = fig.add_subplot(5, 1, 2)
    sc_u = ax_ru.scatter(t_uni, n_uni, c=c_uni, cmap="inferno", marker='|', s=80)
    ax_ru.set(ylabel=f"{layer} idx", title=f"{layer}-layer raster")
    ax_ru.set_ylim(-1, net.n);
    ax_ru.grid(alpha=.2)
    cb_u = plt.colorbar(sc_u, ax=ax_ru, shrink=.65)
    cb_u.set_label("Mean spikes/step")

    # (C) unimodal spike count over time
    ax_uc = fig.add_subplot(5, 1, 3)
    ax_uc.plot(range(n_steps), (uni_spk > 0.5).sum(1).cpu(), '-o', label="spikes / t")
    ax_uc.set(ylabel="count", title=f"Total {layer} spikes")
    ax_uc.grid(alpha=.3);
    ax_uc.legend()

    # (D) MSI raster
    t_msi, n_msi = build_raster_lists(msi_spk)
    c_msi = [msi_colour_map[i] for i in n_msi]
    ax_rm = fig.add_subplot(5, 1, 4)
    sc_m = ax_rm.scatter(t_msi, n_msi, c=c_msi, cmap="inferno", marker='|', s=80)
    ax_rm.set(ylabel="MSI idx", title="MSI raster")
    ax_rm.set_ylim(-1, net.n);
    ax_rm.grid(alpha=.2)
    cb_m = plt.colorbar(sc_m, ax=ax_rm, shrink=.65)
    cb_m.set_label("Mean spikes/step")

    # (E) MSI spike count over time
    ax_mc = fig.add_subplot(5, 1, 5)
    ax_mc.plot(range(n_steps), (msi_spk > 0.5).sum(1).cpu(), '-o', color='C2',
               label="spikes / t")
    ax_mc.set(xlabel="time-step", ylabel="count", title="Total MSI spikes")
    ax_mc.grid(alpha=.3);
    ax_mc.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
#  POPULATION-WIDTH METRICS & FEED-FORWARD DIAGNOSTICS
# ----------------------------------------------------------------------
def msi_pop_fwhm(spikes_1d: torch.Tensor, space_size: int = 180) -> float:
    """
    Return the full-width-at-half-maximum (degrees) of a 1-D MSI spike vector.
    Input may live on CPU or GPU; nothing is modified in-place.
    """
    spikes = spikes_1d.detach()
    if spikes.numel() <= 1:
        return 0.0
    c_idx = torch.argmax(spikes).item()
    half_peak = 0.5 * spikes[c_idx]

    l_idx = c_idx
    while l_idx > 0 and spikes[l_idx] >= half_peak:
        l_idx -= 1
    r_idx = c_idx
    n = spikes.numel()
    while r_idx < n - 1 and spikes[r_idx] >= half_peak:
        r_idx += 1

    width_neur = r_idx - l_idx
    return width_neur * (space_size - 1) / (n - 1)


def feedforward_row_stats(net, path: str = "A2MSI", sample_rows: int = 20):
    """
    Print the effective σ (neurons) of randomly sampled rows in any feed-forward
    weight matrix.
    """
    if path == "A2MSI":
        W = net.W_a2msi_AMPA + net.W_a2msi_NMDA
    elif path == "V2MSI":
        W = net.W_v2msi_AMPA + net.W_v2msi_NMDA
    elif path == "InA":
        W = net.W_inA
    elif path == "InV":
        W = net.W_inV
    else:
        raise ValueError("unknown path")

    W = W.detach().cpu()
    n = W.shape[0]
    rows = torch.linspace(0, n - 1, sample_rows).long()
    xs = torch.arange(n, dtype=torch.float32)

    print(f"[feedforward_row_stats]  path={path}")
    for i in rows:
        row = W[i]
        if row.sum() == 0:
            print(f"  row {i:3d}: EMPTY")
            continue
        mu = (row * xs).sum() / row.sum()
        var = (row * (xs - mu) ** 2).sum() / row.sum()
        print(f"  row {i:3d}   σ≈{var.sqrt():4.1f} neur.")


# -----------------------------------------------------------
# MSI activity visualization
# -----------------------------------------------------------
import matplotlib.pyplot as plt


def msi_activity_summary(
        net,
        centre_deg: float,
        sigma_in: float = 5.0,
        pulse_len: int = 6,
        n_steps: int = 25,
        modality: str = "A",  # "A" or "V"
        intensity: float = 1.0,
        style: str = "default",
        figsize=(10, 6)
):
    """Plot MSI raster, spike count per time, and spike count per neuron."""

    # ------------- switch to single batch -------------
    old_B = net.batch_size
    net.reset_state(batch_size=1)

    try:
        # ---------- build Gaussian pulse ----------
        n = net.n
        idx_c = int(round(centre_deg * (n - 1) / (net.space_size - 1)))
        xs = torch.arange(n, dtype=torch.float32, device=net.device)
        gauss = torch.exp(-0.5 * ((xs - idx_c) / sigma_in) ** 2) * intensity

        xA, xV = (torch.zeros(n_steps, n, device=net.device) for _ in range(2))
        (xA if modality == "A" else xV)[:pulse_len] = gauss

        # ---------- run & record ----------
        spikes = torch.zeros(n_steps, n, device=net.device)  # MSI only
        for t in range(n_steps):
            net.update_all_layers_batch(xA[t][None, :], xV[t][None, :])
            spikes[t] = net._latest_sMSI[0]

        # ---------- prepare plots ----------
        ts_count = spikes.sum(dim=1).cpu()  # spikes per time-step  (T,)
        nu_count = spikes.sum(dim=0).cpu()  # spikes per neuron    (n,)

        plt.style.use(style)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.35)

        # -- (1) raster -----------------------------------------------------
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(spikes.cpu(),
                   cmap="Greys", aspect='auto', origin='lower', interpolation="nearest")
        ax1.set_ylabel("MSI neuron")
        ax1.set_title(f"MSI activity  |  centre={centre_deg}°, σ={sigma_in}, mode={modality}")
        ax1.axvline(0, ls="--", lw=.8, color="tab:blue")
        ax1.axvline(pulse_len, ls="--", lw=.8, color="tab:blue")

        # -- (2) spikes / time-step ----------------------------------------
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.bar(range(n_steps), ts_count, width=0.8)
        ax2.set_ylabel("Σ spikes")
        ax2.set_ylim(0, ts_count.max() * 1.1)

        # -- (3) spikes / neuron ------------------------------------------
        ax3 = fig.add_subplot(gs[2])
        ax3.bar(range(n), nu_count, width=0.8)
        ax3.set_xlabel("neuron index")
        ax3.set_ylabel("Σ spikes")
        ax3.set_xlim(0, n - 1)
        ax3.set_ylim(0, nu_count.max() * 1.1)

        plt.tight_layout()
        plt.show()

    finally:
        # restore original batch size
        if old_B != 1:
            net.reset_state(batch_size=old_B)


########################################################
#       MULTI-BATCH GPU IZHIKEVICH NETWORK CLASS
########################################################

class MultiBatchAudVisMSINetworkTime(nn.Module):
    """
    Implements a multi-layer spiking network
    (Audio, Visual, MSI excitatory, MSI inhibitory, and Readout) with:
      - A->MSI & V->MSI split into AMPA/NMDA (both excitatory).
      - A->MSI_inh & V->MSI_inh also split into AMPA/NMDA (excitatory).
      - Dedicated MSI_inh -> MSI_exc GABA projection.
      - Dedicated inhibitory projection from unimodal layers (A_inh, V_inh) directly to MSI excit.
      - Tsodyks-Markram short-term depression on AMPA synapses.
      - Conduction delays, STDP for early layers, supervised readout training.
      - Lateral (surround) inhibition in MSI excit.
    """

    def __init__(
            self,
            n_neurons=30,
            batch_size=32,
            lr_unimodal=1e-4,
            lr_msi=1e-4,
            lr_readout=1e-4,
            sigma_in=5.0,
            sigma_teacher=3.0,
            noise_std=0.1,
            single_modality_prob=0.3,
            v_thresh=0.25,
            dt=0.1,
            tau_m=5.0,
            n_substeps=100,
            loc_jitter_std=0.0,
            space_size=180,
            conduction_delay_a2msi=5,
            conduction_delay_v2msi=5,
            conduction_delay_msi2out=5
    ):
        super().__init__()
        self.n = n_neurons  # number of excitatory neurons
        self.batch_size = batch_size
        self.space_size = space_size
        self.sigma_in = sigma_in
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # teacher scheduling
        self.sigma_teacher_init = 6.0
        self.sigma_teacher_final = 2.0
        self.curriculum_epochs = 10

        self.lr_uni = lr_unimodal
        self.lr_msi = lr_msi
        self.lr_out = lr_readout
        self.noise_std = noise_std
        self.v_thresh = v_thresh
        self.dt = dt
        self.tau_m = tau_m
        self.n_substeps = n_substeps
        self.loc_jitter_std = loc_jitter_std

        self.pv_nmda = 5.0
        self.targ_ratio = 5.0

        self.W_latA = torch.zeros((self.n, self.n), device=self.device)
        self.W_latV = torch.zeros((self.n, self.n), device=self.device)
        self.g_latA = 0.1  # Lateral inhibition gain for A
        self.g_latV = 0.1  # Lateral inhibition gain for V
        self._probe = AMPANMDADebugger()  # ← add near other debug fields

        self.tau_ampa_lp = 2.5  # ms  (same as self.tau_syn)
        self.ampa_alpha = 1.0  # scale factor per injection
        self.gAMPA_LP = 1.0  # gain when converting ampa_m → current
        self.Erev_ampa = 0.0  # mV, typical AMPA reversal potential

        self.ampa_m = torch.zeros((self.batch_size, self.n),
                                  dtype=torch.float32,
                                  device=self.device)

        def pos_init(shape, scale=3.0):
            """
            Return a Parameter θ such that
              W = softplus(θ) – log(2)
            has mean ≈ 0  (because softplus(0)=log 2).
            """
            theta = scale * torch.randn(*shape, device=self.device)  # θ ~ N(0,σ²)
            return nn.Parameter(theta, requires_grad=False)  # θ is stored

        # define an MSI inhibitory subpopulation
        self.n_inh = int(0.3 * n_neurons)
        if self.n_inh < 1:
            self.n_inh = 1

        self.input_scaling = 150.0
        self.gAMPA = 1.0  # add once in __init__
        self.Erev_ampa = 0.0  # mV, typical reversal

        # ------------- Weights: In -> Uni(A/V) --------------
        self.W_inA = pos_init((self.n, self.n), 0.1)
        self.W_inV = pos_init((self.n, self.n), 0.1)

        init_a2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)
        init_v2msi = torch.tensor(0.005 * np.random.randn(self.n, self.n),
                                  dtype=torch.float32, device=self.device)

        self.W_a2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_a2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_v2msi_AMPA = pos_init((self.n, self.n), 0.005 * 0.8)
        self.W_v2msi_NMDA = pos_init((self.n, self.n), 0.005 * 0.8)

        self.W_inA_inh = nn.Parameter(0.002 * torch.rand(self.n, self.n, device=self.device),
                                      requires_grad=False)  # U[0,0.002)

        self.W_inV_inh = nn.Parameter(0.002 * torch.rand(self.n, self.n, device=self.device),
                                      requires_grad=False)  # U[0,0.002)

        init_a2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)
        init_v2msi_inh = torch.tensor(0.005 * np.random.randn(self.n_inh, self.n),
                                      dtype=torch.float32, device=self.device)

        self.W_a2msiInh_AMPA = nn.Parameter(0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_a2msiInh_NMDA = nn.Parameter(0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_AMPA = nn.Parameter(0.005 * 5.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        self.W_v2msiInh_NMDA = nn.Parameter(0.005 * 15.0 * torch.rand(self.n_inh, self.n, device=self.device),
                                            requires_grad=False)
        # ------------- MSI_inh -> MSI_exc (GABA) --------------

        # ------------- MSI_inh -> MSI_exc (GABA) --------------
        self.W_msiInh2Exc_GABA = nn.Parameter(0.002 * torch.rand(self.n, self.n_inh, device=self.device),
                                              requires_grad=False)  # U[0,0.002)

        self.register_buffer(
            "W_msiInh2Exc_GABA_init", self.W_msiInh2Exc_GABA.clone()
        )

        # ------------- MSI -> Out --------------
        self.W_msi2out = torch.tensor(0.01 * np.random.randn(self.n, self.n),
                                      dtype=torch.float32, device=self.device)

        # ----- inhibitory iSTDP parameters -----
        self.rho0 = 5 / 1000  # target rate per sub-step (≈5 Hz)
        self.eta_i = 1e-2  # learning-rate
        self.tau_post_i = 150.0  # decay of postsyn trace (ms)

        self.allow_inhib_plasticity = True

        self.step_counter = 0
        self.inhib_scaling_T = 2000

        # One trace per *excitatory* postsynaptic neuron
        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)
        self.rate_avg_tau = 5000.0  # ms (50 s network time)
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         device=self.device)

        # ------------- Biases --------------
        self.b_uniA = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_uniV = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        self.b_msi_inh = torch.zeros(self.n_inh, dtype=torch.float32, device=self.device)
        self.b_out = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        self.EI_history = []  # will store ratios for plots/debug

        # ------------- STDP traces --------------
        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.pre_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Izhikevich params --------------
        # For unimodal excit
        self.aA, self.bA, self.cA, self.dA = 0.02, 0.2, -65.0, 8.0
        self.aV, self.bV, self.cV, self.dV = 0.02, 0.2, -65.0, 8.0
        # MSI excit
        self.aM, self.bM, self.cM, self.dM = 0.02, 0.2, -65.0, 8.0
        # MSI inh (fast spiking)
        self.aMi, self.bMi, self.cMi, self.dMi = 0.1, 0.2, -65.0, 2.0
        # Out
        self.aO, self.bO, self.cO, self.dO = 0.1, 0.2, -65.0, 2.0

        # ------------- Membrane potentials & recovery --------------
        # Unimodal excit
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # Out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, device=self.device)
        self.u_out = self.bO * self.v_out

        # ------------- Spikes from last substep --------------
        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        # ------------- Synaptic currents --------------
        self.tau_syn = 2.5
        self.I_A = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), device=self.device)

        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)

        # ------------- Conduction delay buffers --------------
        self.conduction_delay_a2msi = conduction_delay_a2msi
        self.conduction_delay_v2msi = conduction_delay_v2msi
        self.conduction_delay_msi2out = conduction_delay_msi2out

        # unimodal->MSI excit
        self.buffer_a2msi = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_a2msi)
        ], maxlen=self.conduction_delay_a2msi)
        self.buffer_v2msi = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_v2msi)
        ], maxlen=self.conduction_delay_v2msi)

        # unimodal->MSI inh (dedicated inhibitory path):
        self.conduction_delay_inA_inh = conduction_delay_a2msi + 20
        self.conduction_delay_inV_inh = conduction_delay_v2msi + 20
        self.buffer_inA_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_inA_inh)
        ], maxlen=self.conduction_delay_inA_inh)
        self.buffer_inV_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_inV_inh)
        ], maxlen=self.conduction_delay_inV_inh)

        # unimodal->MSI_inh excit:
        self.conduction_delay_a2msi_inh = conduction_delay_a2msi + 20
        self.conduction_delay_v2msi_inh = conduction_delay_v2msi + 20
        self.buffer_a2msi_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_a2msi_inh)
        ], maxlen=self.conduction_delay_a2msi_inh)
        self.buffer_v2msi_inh = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_v2msi_inh)
        ], maxlen=self.conduction_delay_v2msi_inh)

        # MSI_inh->MSI_exc
        self.conduction_delay_msi_inh2exc = max(self.conduction_delay_a2msi,
                                                self.conduction_delay_v2msi) + 50
        self.buffer_msi_inh2exc = deque([
            torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_msi_inh2exc)
        ], maxlen=self.conduction_delay_msi_inh2exc)

        # MSI->Out
        self.buffer_msi2out = deque([
            torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
            for _ in range(self.conduction_delay_msi2out)
        ], maxlen=self.conduction_delay_msi2out)

        ################################################################
        # NMDA parameters and state variables
        ################################################################
        self.gNMDA = 0.6
        self.tau_nmda = 40.0
        self.nmda_alpha = 0.1
        self.mg_k = 0.062
        self.Erev_nmda = 10.0
        self.tau_nmdaVolt = 200.0
        self.v_nmda_rest = -65.0
        self.nmda_vrest_offset = 7.0
        self.mg_vhalf = -35.0

        self.dend_coupling_alpha = 0.1
        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)

        # NMDA gating state (MSI excit)
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)

        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        ################################################################
        # Short-term depression (Tsodyks-Markram) for AMPA
        ################################################################
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.tau_rec = 400.0
        self.tau_fac = 20.0

        # --- debug counters -------------------------------------------------
        self._dbg_spk_A = 0.0  # accumulated spikes in layer A
        self._dbg_spk_V = 0.0  # accumulated spikes in layer V
        self._dbg_spk_MSI = 0.0  # accumulated spikes in MSI excit
        self._dbg_steps = 0  # how many external frames have been seen

        exc_names = [
            "W_inA", "W_inV",
            "W_a2msi_AMPA", "W_a2msi_NMDA",
            "W_v2msi_AMPA", "W_v2msi_NMDA",
        ]
        for name in exc_names:
            parametrize.register_parametrization(self, name, Positive())

        for attr in ["W_inA_inh", "W_inV_inh", "W_msiInh2Exc_GABA",
                     "W_a2msiInh_AMPA", "W_a2msiInh_NMDA",
                     "W_v2msiInh_AMPA", "W_v2msiInh_NMDA"
                     ]:
            parametrize.register_parametrization(self, attr, NonNegative())


        self.g_GABA = 0.7  # ① global scale  ↑  (was 0.4)


        with torch.no_grad():
            pos = torch.arange(self.n, device=self.device)
            dist = (pos[:, None] - pos[None, :]).abs().float()

        self.register_buffer("dist_mask", dist)  # (n,n) cyclic distance

        self.R_near = 4.0  # “near” radius (neurons)
        self.eta_H = 3e-4  # Hebbian rate  (centre potentiation)
        self.eta_AH = 1e-4  # anti-Hebbian  (surround strengthening)

        self.register_buffer("near_mask", torch.exp(-(dist / self.R_near) ** 2))
        self.register_buffer("far_mask", 1.0 - self.near_mask)

        # initialise learnable surround-inhibition matrix (non-negative)
        init_W = self.near_mask.clone()
        self.W_MSI_inh = nn.Parameter(init_W,
                                      requires_grad=False)
        parametrize.register_parametrization(self, "W_MSI_inh", Positive())

        self.register_buffer("W_MSI_inh_init", init_W.clone())

        self.g_GABA = 2  # was 3 – stronger Mexican-hat inhibition
        self.W_MSI_inh.mul_(1.1)  # start with higher surround weight
        self.g_FFinh = 5  # start neutral; can be auto‑calibrated

        # self.g_GABA *= 15
        #
        # self.W_msiInh2Exc_GABA.mul_(20)
        #
        # self.W_inA_inh.mul_(20)
        # self.W_inV_inh.mul_(20)

        # ------------------------------------------------------------------
        #                       calibration helpers
        # ------------------------------------------------------------------

    def set_inhib_plasticity(self, enable: bool):
        self.allow_inhib_plasticity = enable

    def disable_all_inhibition(self):
        """
        Sets all known inhibitory pathways to zero at the raw Parameter level
        *including* reparametrized 'original' for a2msiInh/v2msiInh AMPA/NMDA.

        After this, sums of W_a2msiInh_AMPA, W_a2msiInh_NMDA, etc.
        must all be zero in the final forward pass.
        """

        def forcibly_zero_reparam(attr: str):
            if attr in self.parametrizations:  # wrapped
                plist = self.parametrizations[attr]
                theta = plist.original
                if isinstance(plist[0], NonNegative):  # cannot hit zero exactly
                    theta.fill_(-20.0)  # ≈ 2 × 10⁻⁹ after soft‑plus
                else:  # Positive → exact 0 is OK
                    theta.zero_()
            else:  # not wrapped
                w = getattr(self, attr, None)
                if w is not None:
                    w.zero_()

        with torch.no_grad():
            forcibly_zero_reparam("W_inA_inh")
            forcibly_zero_reparam("W_inV_inh")
            forcibly_zero_reparam("W_a2msiInh_AMPA")
            forcibly_zero_reparam("W_a2msiInh_NMDA")
            forcibly_zero_reparam("W_v2msiInh_AMPA")
            forcibly_zero_reparam("W_v2msiInh_NMDA")
            forcibly_zero_reparam("W_msiInh2Exc_GABA")
            forcibly_zero_reparam("W_MSI_inh")
            self.g_GABA = 0.0
            self.allow_inhib_plasticity = False

        # Now we print
        print("[INFO] All known inhibition forcibly zeroed at raw param level. Summaries:")
        print(f"  W_inA_inh sum={self.W_inA_inh.sum().item()}")
        print(f"  W_inV_inh sum={self.W_inV_inh.sum().item()}")
        print(f"  W_a2msiInh_AMPA sum={self.W_a2msiInh_AMPA.sum().item()}")
        print(f"  W_a2msiInh_NMDA sum={self.W_a2msiInh_NMDA.sum().item()}")
        print(f"  W_v2msiInh_AMPA sum={self.W_v2msiInh_AMPA.sum().item()}")
        print(f"  W_v2msiInh_NMDA sum={self.W_v2msiInh_NMDA.sum().item()}")
        print(f"  W_msiInh2Exc_GABA sum={self.W_msiInh2Exc_GABA.sum().item()}")
        print(f"  W_MSI_inh sum={self.W_MSI_inh.sum().item()}")
        print(f"  g_GABA={self.g_GABA}, allow_inhib_plasticity={self.allow_inhib_plasticity}")

    # ------------------------------------------------------------------
    @staticmethod
    def _positive_update(W: torch.Tensor, dW: torch.Tensor):
        with torch.no_grad():
            W.copy_((W + dW).clamp_(min=0.0))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _p_add(self, attr: str, dW: torch.Tensor,
               eps: float = 1e-9,
               rel_clip: float = 0.25,
               abs_cap: float = 5.0):
        """Clipped additive update for parametrised weights."""
        with torch.no_grad():
            W = getattr(self, attr)  # current ≥0 view
            step = torch.clamp(dW, -rel_clip * W, rel_clip * W)
            W_new = (W + step).clamp(min=eps, max=abs_cap)  # safe candidate

            theta = self.parametrizations[attr].original
            proj = self.parametrizations[attr][0]  # Positive() / NonNegative()
            theta.copy_(proj.right_inverse(W_new))

    # ---------- MultiBatchAudVisMSINetworkTime.patch ----------

    def _probe_spike_sum(self, *, stim_peak: float = 1.0) -> float:
        """
        Deliver a 3‑frame audio pulse (no visual input) centred on the map and
        return the *integrated* MSI‑excit spike count produced with the *current*
        value of `self.input_scaling`.

        A healthy untrained network typically fires 200‑800 spikes here when
        `input_scaling` is in the right ball‑park.
        """
        self.reset_state(batch_size=1)

        pulse = torch.zeros((1, self.n), device=self.device)
        pulse[0, self.n // 2] = stim_peak  # centre neuron only

        tot = 0.0
        for _ in range(15):  # three external frames
            *_, sSum = self.update_all_layers_batch(
                pulse, torch.zeros_like(pulse),  # AUDIO‑only
                return_spike_sum=True  # <<< counts ALL sub‑steps
            )
            tot += sSum.sum().item()
        return tot

    def auto_calibrate_input_gain(self,
                                  target_MSI_spikes: int = 300,
                                  tol: int = 30,
                                  max_iter: int = 12,
                                  high_bound: float = 4000.0):
        """
        Binary‑search `self.input_scaling` so that a *single‑modality* pulse
        elicits `target_MSI_spikes` ± `tol`.  Then choose the smallest
        `g_FFinh` (in 0.05 increments) that halves that response.
        """
        print("[CAL] calibrating input_scaling (+ g_FFinh)")

        lo, hi = 0.0, high_bound
        best_gain, best_err = self.input_scaling, float("inf")

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        for _ in range(max_iter):
            self.input_scaling = 0.5 * (lo + hi)
            excit = self._probe_spike_sum()
            err = abs(excit - target_MSI_spikes)

            # print(f"    gain={self.input_scaling:6.1f}  spikes={excit:6.1f}")

            if err < best_err:
                best_gain, best_err = self.input_scaling, err
            if err <= tol:
                break
            if excit > target_MSI_spikes:
                hi = self.input_scaling
            else:
                lo = self.input_scaling

        self.input_scaling = best_gain

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        for g in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00, 1.50, 2.00):
            self.g_FFinh = g
            inhibited = self._probe_spike_sum()
            if inhibited < 0.6 * target_MSI_spikes:
                break  # first g that halves response

        print(f"    input_scaling={self.input_scaling:.1f}, "
              f"g_FFinh={self.g_FFinh:.2f}")

    def iSTDP_homeo(self, W_attr, pre_spk, post_spk, lr=1e-4, rho=0.05):
        """
        Vogels-Abbott rule.
        """
        dw = lr * torch.bmm((post_spk - rho).unsqueeze(2),
                            pre_spk.unsqueeze(1)).mean(0)

        # limit oversized RF jumps
        W_now = getattr(self, W_attr)
        dw.clamp_(-0.25 * W_now, 0.25 * W_now)  # ±25 % of current weight
        # ------------------------------------------------------------------

        self._p_add(W_attr, dw)


        with torch.no_grad():
            W = getattr(self, W_attr)
            row = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-9)
            init = getattr(self, f"{W_attr}_init").norm(p=2, dim=1, keepdim=True)
            mask = row > init
            W[mask.squeeze()] *= (init / row)[mask]

    def reset_state(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        # reset unimodal
        self.v_uniA = torch.full((self.batch_size, self.n), self.cA, dtype=torch.float32, device=self.device)
        self.u_uniA = self.bA * self.v_uniA
        self.v_uniV = torch.full((self.batch_size, self.n), self.cV, dtype=torch.float32, device=self.device)
        self.u_uniV = self.bV * self.v_uniV

        # reset MSI excit
        self.v_msi = torch.full((self.batch_size, self.n), self.cM, dtype=torch.float32, device=self.device)
        self.u_msi = self.bM * self.v_msi

        # reset MSI inh
        self.v_msi_inh = torch.full((self.batch_size, self.n_inh), self.cMi, dtype=torch.float32, device=self.device)
        self.u_msi_inh = self.bMi * self.v_msi_inh

        # reset out
        self.v_out = torch.full((self.batch_size, self.n), self.cO, dtype=torch.float32, device=self.device)
        self.u_out = self.bO * self.v_out

        self.I_A = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_V = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_M_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.I_O = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.I_ampa_filtered = torch.zeros((self.batch_size, self.n), device=self.device)

        self._latest_sA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self._latest_sMSI_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self._latest_sOut = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)

        self.pre_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inA = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_inV = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_a2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.post_trace_v2msi = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.pre_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_a2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.pre_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.post_trace_v2msi_nmda = torch.zeros((self.batch_size, self.n), device=self.device)
        self.ampa_m.zero_()  # clear low-pass AMPA state

        # clear conduction buffers
        self.buffer_a2msi.clear()
        self.buffer_v2msi.clear()
        self.buffer_inA_inh.clear()
        self.buffer_inV_inh.clear()
        self.buffer_a2msi_inh.clear()
        self.buffer_v2msi_inh.clear()
        self.buffer_msi_inh2exc.clear()
        self.buffer_msi2out.clear()
        if hasattr(self, "ampa_m"):
            self.ampa_m = torch.zeros((self.batch_size, self.n),
                                      dtype=torch.float32,
                                      device=self.device)
        # ------------------------------------------------

        for _ in range(self.conduction_delay_a2msi):
            self.buffer_a2msi.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_v2msi):
            self.buffer_v2msi.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_inA_inh):
            self.buffer_inA_inh.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_inV_inh):
            self.buffer_inV_inh.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_a2msi_inh):
            self.buffer_a2msi_inh.append(
                torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_v2msi_inh):
            self.buffer_v2msi_inh.append(
                torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_msi_inh2exc):
            self.buffer_msi_inh2exc.append(
                torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device))
        for _ in range(self.conduction_delay_msi2out):
            self.buffer_msi2out.append(torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device))

        # reset NMDA gating
        self.nmda_m = torch.zeros((self.batch_size, self.n), dtype=torch.float32, device=self.device)
        self.v_nmda = torch.full((self.batch_size, self.n), self.v_nmda_rest, dtype=torch.float32, device=self.device)
        self.nmda_m_inh = torch.zeros((self.batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        self.v_nmda_inh = torch.full((self.batch_size, self.n_inh), self.v_nmda_rest, dtype=torch.float32,
                                     device=self.device)

        self.v_dend_A = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_V = torch.full((self.batch_size, self.n), self.cM, device=self.device)
        self.v_dend_inhA = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)
        self.v_dend_inhV = torch.full((self.batch_size, self.n_inh), self.cMi, device=self.device)

        # reset STP
        self.R_a = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_a = torch.full((self.batch_size, self.n), 0.2, device=self.device)
        self.R_v = torch.ones((self.batch_size, self.n), device=self.device)
        self.u_v = torch.full((self.batch_size, self.n), 0.2, device=self.device)

        self.R_a_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_a_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)
        self.R_v_inh = torch.ones((self.batch_size, self.n_inh), device=self.device)
        self.u_v_inh = torch.full((self.batch_size, self.n_inh), 0.2, device=self.device)

        self.post_i_trace = torch.zeros((self.batch_size, self.n),
                                        dtype=torch.float32,
                                        device=self.device)

        # sync firing-rate tracker
        self.post_rate_avg = torch.zeros((self.batch_size, self.n),
                                         dtype=torch.float32,
                                         device=self.device)

        # --- debug counters -------------------------------------------------
        self._dbg_spk_A = 0.0  # accumulated spikes in layer A
        self._dbg_spk_V = 0.0  # accumulated spikes in layer V
        self._dbg_spk_MSI = 0.0  # accumulated spikes in MSI excit
        self._dbg_steps = 0  # how many external frames have been seen

        # reset RF tracking
        self.msi_rf_centers = torch.zeros((self.batch_size, self.n), device=self.device)
        self.msi_rf_certainty = torch.zeros((self.batch_size, self.n), device=self.device)

    def print_epoch_spike_summary(self, tag: str = "") -> None:
        """Print mean firing rates (Hz) for A, V, MSI layers."""
        if self._dbg_steps == 0:
            print(f"[rate] {tag} – no frames processed")
            return

        norm = self._dbg_steps * self.n  # total neuron-frames
        rA = self._dbg_spk_A / norm
        rV = self._dbg_spk_V / norm
        rM = self._dbg_spk_MSI / norm

        # Convert to Hz
        ms_per_frame = self.n_substeps * self.dt
        hz_fact = 1000.0 / ms_per_frame
        rA_hz, rV_hz, rM_hz = (x * hz_fact for x in (rA, rV, rM))

        print(f"[rate] {tag:10s}"
              f"  A={rA_hz:6.2f} Hz"
              f"  V={rV_hz:6.2f} Hz"
              f"  MSI={rM_hz:6.2f} Hz"
              f"   (target={self.rho0 * hz_fact:5.2f} Hz)")

        # ready for next epoch
        self._dbg_spk_A = self._dbg_spk_V = self._dbg_spk_MSI = 0.0
        self._dbg_steps = 0

    def update_all_layers_batch(self,
                                xA_batch,
                                xV_batch,
                                valid_mask=None,
                                record_voltages=False,
                                debug=False,
                                conduction_debug=False,
                                curr_debug=False,
                                epoch_idx=0,
                                return_delayed=False,
                                return_spike_sum=False):  # optional spike-sum
        """
        Forward-prop one external time-step (100 Izhikevich sub-steps).

        If `return_spike_sum` is True, an extra tensor
            sum_sM   (batch_size , n)
        containing the **total number of MSI spikes in this external frame**
        is appended to the return tuple.
        """

        batch_size = xA_batch.size(0)

        if return_spike_sum:
            sum_sM = torch.zeros(batch_size, self.n, device=self.device)

        if valid_mask is not None:
            mask = valid_mask.view(batch_size, 1)
            xA_batch = xA_batch * mask
            xV_batch = xV_batch * mask

        sA = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sV = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sM = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)
        sMi = torch.zeros((batch_size, self.n_inh), dtype=torch.float32, device=self.device)
        sO = torch.zeros((batch_size, self.n), dtype=torch.float32, device=self.device)

        decay_factor = 1.0 - self.dt / self.tau_syn
        spike_threshold = 30.0

        # Expand for unimodal feedforward
        xA_expanded = xA_batch.unsqueeze(1)
        xV_expanded = xV_batch.unsqueeze(1)

        W_inA_expanded = self.W_inA.unsqueeze(0).expand(batch_size, self.n, self.n)
        W_inV_expanded = self.W_inV.unsqueeze(0).expand(batch_size, self.n, self.n)

        # ---- feedforward input ----
        I_A_input = self.input_scaling * (xA_batch @ self.W_inA + self.b_uniA)
        I_V_input = self.input_scaling * (xV_batch @ self.W_inV + self.b_uniV)

        for sub_i in range(self.n_substeps):
            # --- Decay old currents ---
            self.I_A.mul_(decay_factor)
            self.I_V.mul_(decay_factor)
            self.I_M.mul_(decay_factor)
            self.I_M_inh.mul_(decay_factor)
            self.I_O.mul_(decay_factor)

            # Add external input (split across substeps)
            self.I_A.add_(I_A_input / float(self.n_substeps))
            self.I_V.add_(I_V_input / float(self.n_substeps))

            # --- Conduction-delay safe pops ---
            if self.conduction_delay_a2msi > 0:
                delayed_spikes_a2msi = self.buffer_a2msi.popleft()
            else:
                delayed_spikes_a2msi = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_v2msi > 0:
                delayed_spikes_v2msi = self.buffer_v2msi.popleft()
            else:
                delayed_spikes_v2msi = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_inA_inh > 0:
                delayed_spikes_inA_inh = self.buffer_inA_inh.popleft()
            else:
                delayed_spikes_inA_inh = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_inV_inh > 0:
                delayed_spikes_inV_inh = self.buffer_inV_inh.popleft()
            else:
                delayed_spikes_inV_inh = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_a2msi_inh > 0:
                delayed_spikes_a2msi_inh = self.buffer_a2msi_inh.popleft()
            else:
                delayed_spikes_a2msi_inh = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_v2msi_inh > 0:
                delayed_spikes_v2msi_inh = self.buffer_v2msi_inh.popleft()
            else:
                delayed_spikes_v2msi_inh = torch.zeros((batch_size, self.n), device=self.device)

            if self.conduction_delay_msi_inh2exc > 0:
                delayed_spikes_msi_inh2exc = self.buffer_msi_inh2exc.popleft()
            else:
                delayed_spikes_msi_inh2exc = torch.zeros((batch_size, self.n_inh), device=self.device)

            if self.conduction_delay_msi2out > 0:
                delayed_spikes_msi2out = self.buffer_msi2out.popleft()
            else:
                delayed_spikes_msi2out = torch.zeros((batch_size, self.n), device=self.device)

            # ============== A->MSI (AMPA+NMDA) ==============
            # (Tsodyks-Markram STP usage for A->MSI)
            W_a2msi_AMPA_expanded = self.W_a2msi_AMPA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_v2msi_AMPA_expanded = self.W_v2msi_AMPA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_a2msi_NMDA_expanded = self.W_a2msi_NMDA.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_v2msi_NMDA_expanded = self.W_v2msi_NMDA.unsqueeze(0).expand(batch_size, self.n, self.n)

            self.I_ampa_filtered.mul_(decay_factor)

            self.R_a += (1.0 - self.R_a) * (self.dt / self.tau_rec)
            use_A = self.u_a * self.R_a
            self.R_a -= use_A * delayed_spikes_a2msi
            I_M_a_AMPA = torch.bmm(
                W_a2msi_AMPA_expanded,
                (use_A * delayed_spikes_a2msi).unsqueeze(2)
            ).squeeze(2)

            self.R_v += (1.0 - self.R_v) * (self.dt / self.tau_rec)
            use_V = self.u_v * self.R_v
            self.R_v -= use_V * delayed_spikes_v2msi
            I_M_v_AMPA = torch.bmm(
                W_v2msi_AMPA_expanded,
                (use_V * delayed_spikes_v2msi).unsqueeze(2)
            ).squeeze(2)

            # -----------------------------------------------------------------
            # -----------------------------------------------------------------
            ampa_decay = 1.0 - self.dt / self.tau_ampa_lp
            self.ampa_m.mul_(ampa_decay)
            self.ampa_m.add_(self.ampa_alpha *
                             (I_M_a_AMPA + I_M_v_AMPA).clamp(min=0.0))

            # -------------------------------------------------------------
            I_AMPA_curr = (self.gAMPA
                           * (I_M_a_AMPA + I_M_v_AMPA)
                           * (self.Erev_ampa - self.v_msi))
            self.I_M.add_(I_AMPA_curr + self.b_msi)

            I_ampa_lp = self.gAMPA_LP * self.ampa_m * (self.Erev_ampa - self.v_msi)

            self.I_ampa_filtered.add_(I_M_a_AMPA + I_M_v_AMPA)
            ampa_release = (I_M_a_AMPA + I_M_v_AMPA).clamp(min=0)  # (B, n)
            I_ampa_step = self.gAMPA * ampa_release * (self.Erev_ampa - self.v_msi)

            # -------------------------------------------------------
            # -------------------------------------------------------
            scale_F = 0.8  # 0 = no STD, 1 = same as AMPA
            gate_A_nmda = 1.0 - scale_F * (1.0 - self.R_a)  # (B,n)
            gate_V_nmda = 1.0 - scale_F * (1.0 - self.R_v)

            pre_A_nmda = gate_A_nmda * delayed_spikes_a2msi  # (B,n)
            pre_V_nmda = gate_V_nmda * delayed_spikes_v2msi
            # ---------------------------------------------------------------

            nmda_a = torch.bmm(W_a2msi_NMDA_expanded,
                               pre_A_nmda.unsqueeze(2)).squeeze(2)
            nmda_v = torch.bmm(W_v2msi_NMDA_expanded,
                               pre_V_nmda.unsqueeze(2)).squeeze(2)
            inc_m_exc = self.nmda_alpha * (nmda_a + nmda_v)  # unchanged
            self.nmda_m.mul_(1.0 - self.dt / self.tau_nmda)
            self.nmda_m.add_(inc_m_exc)

            # Dend coupling
            d_va = self.dend_coupling_alpha * (self.v_msi - self.v_dend_A) / self.tau_m
            d_vv = self.dend_coupling_alpha * (self.v_msi - self.v_dend_V) / self.tau_m
            self.v_dend_A += self.dt * d_va
            self.v_dend_V += self.dt * d_vv

            dv_nmda = ((self.v_msi + self.nmda_vrest_offset) - self.v_nmda) / self.tau_nmdaVolt
            self.v_nmda += self.dt * dv_nmda
            mg_A = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_A - self.mg_vhalf)))
            mg_V = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_V - self.mg_vhalf)))
            I_nmda = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_nmda_step = self.gNMDA * inc_m_exc * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_nmda_lp = self.gNMDA * self.nmda_m * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)

            self.I_M.add_(I_nmda)

            dt_s = self.dt / 1000.0

            release = (I_M_a_AMPA + I_M_v_AMPA)  # what you already had
            I_AMPA_tp = self.gAMPA * release * (self.Erev_ampa - self.v_msi)  # current
            J_ampa_step = I_AMPA_tp.detach()
            I_nmda_step = self.gNMDA * inc_m_exc * (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)
            I_ampa_total = I_AMPA_curr + self.gAMPA_LP * self.ampa_m * (self.Erev_ampa - self.v_msi)
            I_nmda_total = I_nmda  # already includes nmda_m tail

            if self._probe is not None:
                # --- 1. true instantaneous currents -------------------------
                J_ampa_step = I_AMPA_curr.detach()
                J_nmda_step = I_nmda_step.detach()  # B

                dt_sec = self.dt / 1000.0  # 0.0001 s
                Q_ampa = I_ampa_total.detach() * dt_sec
                Q_nmda = I_nmda_total.detach() * dt_sec

                # ---- per‑spike injection (optional) ----
                J_ampa_inj = I_AMPA_curr.detach()
                J_nmda_inj = I_nmda_step.detach()

                # ---- spike counter for normalisation ----
                n_new_spk = (delayed_spikes_a2msi + delayed_spikes_v2msi).sum().item()

                # effective AMPA current
                # ------------------------------------------------------------------
                # ------------------------------------------------------------------
                release = (I_M_a_AMPA + I_M_v_AMPA)  # (B, n)

                # true AMPA current
                I_AMPA_step = self.gAMPA * release * (self.Erev_ampa - self.v_msi)
                J_ampa_inst = I_AMPA_curr.detach()  # <<<<<< line A
                J_nmda_inst = I_nmda_step.detach()  # unchanged
                # ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑#

                # (nothing else in this block changes)
                self._probe.log(
                    Q_ampa=Q_ampa,
                    Q_nmda=Q_nmda,
                    I_M=self.I_M.detach(),
                    sA=self._latest_sA, sV=self._latest_sV, sM=self._latest_sMSI,
                    R_a=self.R_a, R_v=self.R_v,
                    mg_gate=(mg_A + mg_V) / 2,
                    J_ampa_inst=J_ampa_inj,  # <<<<<< line B
                    J_nmda_inst=J_nmda_inj,
                    n_spikes=n_new_spk  # unchanged
                )

            W_inA_inh_expanded = self.W_inA_inh.unsqueeze(0).expand(batch_size, self.n, self.n)
            W_inV_inh_expanded = self.W_inV_inh.unsqueeze(0).expand(batch_size, self.n, self.n)
            I_inA_inh = torch.bmm(W_inA_inh_expanded, delayed_spikes_inA_inh.unsqueeze(2)).squeeze(2)
            I_inV_inh = torch.bmm(W_inV_inh_expanded, delayed_spikes_inV_inh.unsqueeze(2)).squeeze(2)
            self.I_M.sub_(self.g_FFinh
                          * (I_inA_inh + I_inV_inh))

            # ============== A->MSI_inh, V->MSI_inh ==============
            Wa2miA_expanded = self.W_a2msiInh_AMPA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            Wa2miN_expanded = self.W_a2msiInh_NMDA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            Wv2miA_expanded = self.W_v2msiInh_AMPA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)
            Wv2miN_expanded = self.W_v2msiInh_NMDA.unsqueeze(0).expand(batch_size, self.n_inh, self.n)

            self.R_a_inh += (1.0 - self.R_a_inh) * (self.dt / self.tau_rec)
            use_A_inh = self.u_a_inh * self.R_a_inh
            spike_sum_a = delayed_spikes_a2msi_inh.sum(dim=1, keepdim=True)
            self.R_a_inh -= use_A_inh * spike_sum_a

            raw_inp_a_AMPA = torch.bmm(Wa2miA_expanded, delayed_spikes_a2msi_inh.unsqueeze(2)).squeeze(2)
            I_Mi_a_AMPA = (use_A_inh * raw_inp_a_AMPA)

            raw_inp_a_NMDA = torch.bmm(
                Wa2miN_expanded,
                delayed_spikes_a2msi_inh.unsqueeze(2)
            ).squeeze(2)

            self.R_v_inh += (1.0 - self.R_v_inh) * (self.dt / self.tau_rec)
            use_V_inh = self.u_v_inh * self.R_v_inh
            spike_sum_v = delayed_spikes_v2msi_inh.sum(dim=1, keepdim=True)
            self.R_v_inh -= use_V_inh * spike_sum_v

            raw_inp_v_AMPA = torch.bmm(Wv2miA_expanded, delayed_spikes_v2msi_inh.unsqueeze(2)).squeeze(2)
            I_Mi_v_AMPA = (use_V_inh * raw_inp_v_AMPA)

            raw_inp_v_NMDA = torch.bmm(
                Wv2miN_expanded,
                delayed_spikes_v2msi_inh.unsqueeze(2)
            ).squeeze(2)

            self.I_M_inh.add_(I_Mi_a_AMPA + I_Mi_v_AMPA + self.b_msi_inh)
            self.nmda_m_inh.mul_(1.0 - self.dt / self.tau_nmda)
            self.nmda_m_inh.add_(self.nmda_alpha * (raw_inp_a_NMDA + raw_inp_v_NMDA))

            d_viA = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhA) / self.tau_m
            d_viV = self.dend_coupling_alpha * (self.v_msi_inh - self.v_dend_inhV) / self.tau_m
            self.v_dend_inhA += self.dt * d_viA
            self.v_dend_inhV += self.dt * d_viV

            dv_nmda_inh = ((self.v_msi_inh + self.nmda_vrest_offset) - self.v_nmda_inh) / self.tau_nmdaVolt
            self.v_nmda_inh += self.dt * dv_nmda_inh
            mg_iA = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhA - self.mg_vhalf)))
            mg_iV = 1.0 / (1.0 + torch.exp(-self.mg_k * (self.v_dend_inhV - self.mg_vhalf)))
            I_nmda_inh = self.gNMDA * self.nmda_m_inh * (mg_iA + mg_iV) * (self.Erev_nmda - self.v_msi_inh)
            self.I_M_inh.add_(I_nmda_inh)

            # MSI_inh->MSI_ex
            W_msiInh2Exc_expanded = self.W_msiInh2Exc_GABA.unsqueeze(0).expand(batch_size, self.n, self.n_inh)
            I_M_inh2exc = torch.bmm(W_msiInh2Exc_expanded, delayed_spikes_msi_inh2exc.unsqueeze(2)).squeeze(2)
            self.I_M.sub_(I_M_inh2exc)

            # MSI->Out
            W_msi2out_expanded = self.W_msi2out.unsqueeze(0).expand(batch_size, self.n, self.n)
            I_O_msi = torch.bmm(W_msi2out_expanded, delayed_spikes_msi2out.unsqueeze(2)).squeeze(2)
            self.I_O.add_((I_O_msi + self.b_out))

            I_latA = torch.mm(self._latest_sA, self.W_latA)  # shape (B, n)
            self.I_A.sub_(self.g_latA * I_latA)

            I_latV = torch.mm(self._latest_sV, self.W_latV)  # shape (B, n)
            self.I_V.sub_(self.g_latV * I_latV)

            # -------------- Izhikevich updates --------------
            # A
            dVA = (0.04 * self.v_uniA.pow(2) + 5.0 * self.v_uniA + 140.0
                   - self.u_uniA + self.I_A)
            self.v_uniA += self.dt * dVA
            self.u_uniA += self.dt * (self.aA * (self.bA * self.v_uniA - self.u_uniA))
            new_sA = (self.v_uniA >= spike_threshold).float()
            self.v_uniA[self.v_uniA >= spike_threshold] = self.cA
            self.u_uniA[self.v_uniA == self.cA] += self.dA

            # V
            dVV = (0.04 * self.v_uniV.pow(2) + 5.0 * self.v_uniV + 140.0
                   - self.u_uniV + self.I_V)
            self.v_uniV += self.dt * dVV
            self.u_uniV += self.dt * (self.aV * (self.bV * self.v_uniV - self.u_uniV))
            new_sV = (self.v_uniV >= spike_threshold).float()
            self.v_uniV[self.v_uniV >= spike_threshold] = self.cV
            self.u_uniV[self.v_uniV == self.cV] += self.dV

            # MSI excit
            dVM = (0.04 * self.v_msi.pow(2) + 5.0 * self.v_msi + 140.0 - self.u_msi + self.I_M)
            self.v_msi += self.dt * dVM
            self.u_msi += self.dt * (self.aM * (self.bM * self.v_msi - self.u_msi))
            new_sM = (self.v_msi >= spike_threshold).float()
            self.v_msi[self.v_msi >= spike_threshold] = self.cM
            self.u_msi[self.v_msi == self.cM] += self.dM

            # (A) compute surround inhibition current
            I_latM = torch.mm(new_sM, self.W_MSI_inh)  # shape (B, n)
            # (B) apply it
            self.I_M.sub_(self.g_GABA * I_latM)

            if self._probe is not None:
                I_total = self.I_M.detach()  # includes inhibition
                Q_exc = torch.clamp(I_total, min=0) * dt_sec
                Q_inh = -torch.clamp(I_total, max=0) * dt_sec
                self._probe.log_EI(Q_exc, Q_inh)  # add two extra slots

            if epoch_idx == 5:
                # instantaneous excitation
                exc_AMPA = ((I_M_a_AMPA + I_M_v_AMPA).clamp(min=0) *
                            self.gAMPA * (self.Erev_ampa - self.v_msi)).sum().item()
                exc_NMDA = (self.gNMDA * inc_m_exc *
                            (mg_A + mg_V) * (self.Erev_nmda - self.v_msi)).sum().item()

                # instantaneous inhibition
                inh_FF = (self.g_FFinh * (I_inA_inh + I_inV_inh)).sum().item()
                inh_lat = (self.g_GABA * I_latM).sum().item()
                inh_recur = I_M_inh2exc.sum().item()


            if return_spike_sum:  # ****
                sum_sM += new_sM  # ****

            # iSTDP trace
            decay_i = torch.tensor(-self.dt / self.tau_post_i, device=self.device, dtype=torch.float32)
            decay_i = torch.exp(decay_i)
            self.post_i_trace.mul_(decay_i)
            self.post_i_trace.add_(new_sM)

            alpha_rate = self.dt / self.rate_avg_tau
            self.post_rate_avg.mul_(1.0 - alpha_rate).add_(alpha_rate * new_sM)

            # MSI inh
            dVMi = (0.04 * self.v_msi_inh.pow(2) + 5.0 * self.v_msi_inh + 140.0 - self.u_msi_inh + self.I_M_inh)
            self.v_msi_inh += self.dt * dVMi
            self.u_msi_inh += self.dt * (self.aMi * (self.bMi * self.v_msi_inh - self.u_msi_inh))
            new_sMi = (self.v_msi_inh >= spike_threshold).float()
            self.v_msi_inh[self.v_msi_inh >= spike_threshold] = self.cMi
            self.u_msi_inh[self.v_msi_inh == self.cMi] += self.dMi

            # Out
            dVO = (0.04 * self.v_out.pow(2) + 5.0 * self.v_out + 140.0 - self.u_out + self.I_O)
            self.v_out += self.dt * dVO
            self.u_out += self.dt * (self.aO * (self.bO * self.v_out - self.u_out))
            new_sO = (self.v_out >= spike_threshold).float()
            self.v_out[self.v_out >= spike_threshold] = self.cO
            self.u_out[self.v_out == self.cO] += self.dO

            # iSTDP homeostasis on MSI_inh->MSI_ex if allowed
            if self.allow_inhib_plasticity:
                pre_ff_inh = torch.cat([delayed_spikes_inA_inh,
                                        delayed_spikes_inV_inh], dim=1)  # (B, 2n)

                W_ff_inh = torch.cat([self.W_inA_inh, self.W_inV_inh], dim=1)

                # Vogel‑Abbott update
                dw = self.eta_i * torch.bmm(
                    (self.post_i_trace - self.rho0).unsqueeze(2),
                    pre_ff_inh.unsqueeze(1)  # (B ,1 ,2n)
                ).mean(0)

                dw_A = dw[:, :self.n]
                dw_V = dw[:, self.n:]
                self._p_add("W_inA_inh", dw_A)
                self._p_add("W_inV_inh", dw_V)

            if valid_mask is not None:
                mask_sub = valid_mask.view(-1, 1)
                new_sA *= mask_sub
                new_sV *= mask_sub
                new_sM *= mask_sub
                new_sMi *= mask_sub
                new_sO *= mask_sub

            # Append next conduction if delay>0
            if self.conduction_delay_a2msi > 0:
                self.buffer_a2msi.append(new_sA.clone())
            if self.conduction_delay_v2msi > 0:
                self.buffer_v2msi.append(new_sV.clone())
            if self.conduction_delay_inA_inh > 0:
                self.buffer_inA_inh.append(new_sA.clone())
            if self.conduction_delay_inV_inh > 0:
                self.buffer_inV_inh.append(new_sV.clone())
            if self.conduction_delay_a2msi_inh > 0:
                self.buffer_a2msi_inh.append(new_sA.clone())
            if self.conduction_delay_v2msi_inh > 0:
                self.buffer_v2msi_inh.append(new_sV.clone())
            if self.conduction_delay_msi_inh2exc > 0:
                self.buffer_msi_inh2exc.append(new_sMi.clone())
            if self.conduction_delay_msi2out > 0:
                self.buffer_msi2out.append(new_sM.clone())

            sA, sV, sM, sMi, sO = new_sA, new_sV, new_sM, new_sMi, new_sO

            self._latest_sA = sA
            self._latest_sV = sV
            self._latest_sMSI = sM
            self._latest_sMSI_inh = sMi

            # ---------- epoch-level debug counters ------------------------------
            with torch.no_grad():  # tiny, avoid autograd tracking
                self._dbg_spk_A += sA.sum().item()
                self._dbg_spk_V += sV.sum().item()
                self._dbg_spk_MSI += sM.sum().item()
                self._dbg_steps += sA.size(0)  # B external frames just processed

            if curr_debug:
                self.debug_msi_current_and_stp(sub_i)

            # ------------------------------------------------------------------
            # ------------------------------------------------------------------
            if sub_i % 10 == 0:
                apply_topographic_anchor_unimodal(self, layer="A",
                                                  lr=0.1 * self.lr_uni,
                                                  sigma=3.0)
                apply_topographic_anchor_unimodal(self, layer="V",
                                                  lr=0.1 * self.lr_uni,
                                                  sigma=3.0)
                if epoch_idx > 25:
                    apply_topographic_anchor_msi(self, layer="A",
                                                 lr=0.8 * self.lr_msi,
                                                 sigma=3.0)
                    apply_topographic_anchor_msi(self, layer="V",
                                                 lr=0.8 * self.lr_msi,
                                                 sigma=3.0)

            self.step_counter += 1

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        apply_topographic_anchor_unimodal(self, layer="A",
                                          lr=1.0 * self.lr_uni,  # was 0.5
                                          sigma=2.5)  # was 3.0
        apply_topographic_anchor_unimodal(self, layer="V",
                                          lr=1.0 * self.lr_uni,
                                          sigma=2.5)

        apply_local_competition_unimodal_fast(self, "A",
                                              beta=2.0 * self.lr_uni,  # was 0.8
                                              neighbour_dist=4)
        apply_local_competition_unimodal_fast(self, "V",
                                              beta=2.0 * self.lr_uni,
                                              neighbour_dist=4)

        if epoch_idx > 25:
            apply_local_competition_msi_fast(self,
                                             beta=1.5 * self.lr_msi,
                                             neighbour_dist=6)

        soft_row_scaling(self)  # keeps norms near unity but *does not* freeze patterns

        # very slow scaling
        if (self.step_counter % 10) == 0:
            slow_synaptic_scaling(self.W_inA)
            slow_synaptic_scaling(self.W_inV)
            slow_synaptic_scaling(self.W_a2msi_AMPA)
            slow_synaptic_scaling(self.W_v2msi_AMPA)
            slow_synaptic_scaling(self.W_a2msi_NMDA)
            slow_synaptic_scaling(self.W_v2msi_NMDA)

        # --- Fast AGC (PV‑like) ---------------------------------------------
        exc_fast = self.I_ampa_filtered.clamp(min=0).mean().item()  # AMPA only
        inh_mean = (-self.I_M).clamp(min=0).mean().item()

        target_ratio = self.targ_ratio # retain original set‑point
        alpha_fast = 1e-3  # keep existing step size
        self.g_FFinh += alpha_fast * (exc_fast * target_ratio - inh_mean)
        self.g_FFinh = max(0.05, min(self.g_FFinh, 5.0))

        if (self.step_counter % 100) == 0:  # ≈ 1 s interval
            exc_mean_long = self.I_M.clamp(min=0).mean().item()
            inh_mean_long = (-self.I_M).clamp(min=0).mean().item()
            alpha_slow = 2e-4  # 5× slower
            self.g_FFinh += alpha_slow * (exc_mean_long * self.pv_nmda - inh_mean_long)
            self.g_FFinh = max(0.05, min(self.g_FFinh, 5.0))

        if return_delayed and return_spike_sum:
            return (sA, sV, sM, sO,
                    delayed_spikes_a2msi, delayed_spikes_v2msi,
                    sum_sM)  # 7 objs
        elif return_delayed:  # delayed spikes
            return (sA, sV, sM, sO,
                    delayed_spikes_a2msi, delayed_spikes_v2msi)  # 6 objs
        elif return_spike_sum:  # only spike accumulator
            return sA, sV, sM, sO, sum_sM  # 5 objs
        else:  # vanilla
            return sA, sV, sM, sO  # 4 objs

    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    def stdp_update_batch(self,
                          W_attr: str,
                          post_spk: torch.Tensor,
                          pre_spk: torch.Tensor,
                          post_trace: torch.Tensor,
                          pre_trace: torch.Tensor,
                          lr: float,
                          tau_pre: float = 0.9,
                          tau_post: float = 0.9,
                          A_plus: float = 1.0,
                          A_minus: float = 1.0,
                          debug=False):
        """
        Pair-based STDP update. Now with debug logs.
        """
        B = post_spk.size(0)

        # Update the eligibility traces
        pre_trace.mul_(tau_pre).add_(pre_spk)
        post_trace.mul_(tau_post).add_(post_spk)

        dW = torch.zeros_like(getattr(self, W_attr))
        for b in range(B):
            dW += A_plus * torch.ger(post_spk[b], pre_trace[b]) \
                  - A_minus * torch.ger(post_trace[b], pre_spk[b])

        dW.mul_(lr / B)

        # Apply via the positive reparam
        self._p_add(W_attr, dW)

        return pre_trace, post_trace, dW

    def normalize_rows_gpu(self, W):
        norms = torch.norm(W, dim=1, keepdim=True)
        norms[norms == 0] = 1.0
        W.div_(norms)

    def normalize_rows(self, attr: str, eps: float = 1e-8):
        with torch.no_grad():
            W = getattr(self, attr)  # current positive view
            theta = self.parametrizations[attr].original
            P = self.parametrizations[attr][0]

            Wnorm = W / (W.norm(2, 1, keepdim=True) + eps)
            theta.copy_(P.right_inverse(Wnorm))

    ########################################################
    #           UNSUPERVISED & SUPERVISED TRAINING
    ########################################################

    def sample_poisson_spikes_from_analog(self, analog_vec, max_rate=50.0, dt=0.001):
        """
        Convert an analog input (shape (B,n) or (n,)) into a 0/1 spike train
        by sampling from Poisson( lambda = analog_vec * max_rate ),
        each step scaled by dt.

        If analog_vec is (n,), we treat it as (1,n).
        """
        if analog_vec.dim() == 1:
            analog_vec = analog_vec.unsqueeze(0)  # (1, n)

        rate = analog_vec.clamp(min=0) * max_rate * dt  # shape (B, n)
        p = rate.clamp(max=1.0)
        spikes = torch.bernoulli(p)
        return spikes  # same shape (B, n)

    def train_unsupervised_batch(self,
                                 n_sequences,
                                 batch_size: int = 32,
                                 debug: bool = False,
                                 epoch_idx: int = 0):
        """
        Unsupervised STDP phase:
        • Generates random AV event sequences
        • Runs the network
        • Applies STDP
            – In  → Uni  : uses Poisson-sampled presyn spikes (same as before)
            – Uni → MSI
        """

        seq_counter = 0  # how many sequences processed so far
        while seq_counter < n_sequences:

            B = min(batch_size,  # current mini-batch
                    n_sequences - seq_counter)

            # --- 1. generate synthetic sequences ------------------------------
            loc_seqs, mod_seqs, offset_ok, seq_lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                offset_probability=0.6,
                temporal_jitter_max=2
            )

            T_max = max(seq_lens)

            batch_intensity = 10 ** torch.empty(1).uniform_(math.log10(0.4), 0).item()

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offset_ok,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=batch_intensity,
                device=self.device,
                max_len=T_max)

            self.reset_state(B)

            for t in range(T_max):
                # forward pass
                (sA, sV, sMSI, _,
                 dA2M, dV2M) = self.update_all_layers_batch(
                    xA[:, t],  # analog A input
                    xV[:, t],  # analog V input
                    valid[:, t],  # validity mask
                    epoch_idx=epoch_idx,
                    return_delayed=True)

                pre_inA = self.sample_poisson_spikes_from_analog(
                    xA[:, t], max_rate=300., dt=0.01)
                pre_inV = self.sample_poisson_spikes_from_analog(
                    xV[:, t], max_rate=300., dt=0.01)

                self.pre_trace_inA, self.post_trace_inA, _ = self.stdp_update_batch(
                    'W_inA',
                    post_spk=sA,
                    pre_spk=pre_inA,
                    post_trace=self.post_trace_inA,
                    pre_trace=self.pre_trace_inA,
                    lr=self.lr_uni,
                    debug=debug)

                self.pre_trace_inV, self.post_trace_inV, _ = self.stdp_update_batch(
                    'W_inV',
                    post_spk=sV,
                    pre_spk=pre_inV,
                    post_trace=self.post_trace_inV,
                    pre_trace=self.pre_trace_inV,
                    lr=self.lr_uni,
                    debug=debug)

                # Uni → MSI STDP
                self.pre_trace_a2msi, self.post_trace_a2msi, _ = self.stdp_update_batch(
                    'W_a2msi_AMPA',
                    post_spk=sMSI,
                    pre_spk=dA2M,  
                    post_trace=self.post_trace_a2msi,
                    pre_trace=self.pre_trace_a2msi,
                    lr=self.lr_msi,
                    debug=debug)

                self.pre_trace_v2msi, self.post_trace_v2msi, _ = self.stdp_update_batch(
                    'W_v2msi_AMPA',
                    post_spk=sMSI,
                    pre_spk=dV2M,  
                    post_trace=self.post_trace_v2msi,
                    pre_trace=self.pre_trace_v2msi,
                    lr=self.lr_msi,
                    debug=debug)

                # NMDA STDP (slower)

                nmda_lr_scale = 1  # NMDA learns slower
                self.pre_trace_a2msi_nmda, self.post_trace_a2msi_nmda, _ = self.stdp_update_batch(
                    'W_a2msi_NMDA',
                    post_spk=sMSI,
                    pre_spk=dA2M,
                    post_trace=self.post_trace_a2msi_nmda,
                    pre_trace=self.pre_trace_a2msi_nmda,
                    lr=self.lr_msi * nmda_lr_scale,
                    tau_pre=0.95,
                    tau_post=0.95,
                    A_plus=1,
                    A_minus=1,
                    debug=debug)

                self.pre_trace_v2msi_nmda, self.post_trace_v2msi_nmda, _ = self.stdp_update_batch(
                    'W_v2msi_NMDA',
                    post_spk=sMSI,
                    pre_spk=dV2M,
                    post_trace=self.post_trace_v2msi_nmda,
                    pre_trace=self.pre_trace_v2msi_nmda,
                    lr=self.lr_msi * nmda_lr_scale,
                    tau_pre=0.95,
                    tau_post=0.95,
                    A_plus=1,
                    A_minus=1,
                    debug=debug)

                # --- OPTIONAL: re-normalise AMPA/NMDA split -------------------
                with torch.no_grad():
                    def set_param_weight(attr, W_new):
                        """Helper to update a parametrized weight using right_inverse."""
                        parametrization = self.parametrizations[attr]
                        theta = parametrization.original
                        module = parametrization[0]
                        theta.copy_(module.right_inverse(W_new))

                    # A -> MSI connections
                    W_tot_a = self.W_a2msi_AMPA + self.W_a2msi_NMDA
                    # Redistribute the total synaptic strength
                    set_param_weight('W_a2msi_AMPA', 0.25 * W_tot_a)
                    set_param_weight('W_a2msi_NMDA', 0.75 * W_tot_a)

                    # V -> MSI connections
                    W_tot_v = self.W_v2msi_AMPA + self.W_v2msi_NMDA
                    # Redistribute the total synaptic strength
                    set_param_weight('W_v2msi_AMPA', 0.25 * W_tot_v)
                    set_param_weight('W_v2msi_NMDA', 0.75 * W_tot_v)

            # end-for t
            seq_counter += B

            # (Optional) print diagnostics once per mini-batch
            if debug:
                print(f"[unsup] processed {seq_counter}/{n_sequences} sequences")

        # ----------------------------------------------------------------------
        # apply slow updates
        # ----------------------------------------------------------------------
        apply_topographic_anchor_unimodal(self, layer="A", lr=self.lr_uni)
        apply_topographic_anchor_unimodal(self, layer="V", lr=self.lr_uni)
        soft_row_scaling(self)

    def evaluate_batch(self,
                       n_sequences: int,
                       condition: str = "both",  # "both", "audio_only", "visual_only"
                       batch_size: int = 32,
                       stimulus_intensity: float = 1.0,
                       decode: str = "argmax"  # "com" (centre-of-mass) or "argmax"
                       ) -> float:
        """
        Returns the mean absolute localisation error (degrees) for *n_sequences*
        synthetic AV trials under *condition*.  Each trial may contain several
        events, but **only the last event in the sequence is evaluated**.

        Parameters
        ----------
        decode : {"com", "argmax"}
            • "com"    – centre-of-mass decoder (robust, default)
            • "argmax" – winner-take-all decoder
        """
        errors: list[float] = []

        while n_sequences > 0:
            B = min(batch_size, n_sequences)

            loc_seqs, mod_seqs, offset, seq_lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                offset_probability=0.1
            )
            T_max = max(seq_lens)

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offset,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=stimulus_intensity,
                device=self.device,
                max_len=T_max
            )

            # optionally zero one modality
            if condition == "audio_only":
                xV.zero_()
            elif condition == "visual_only":
                xA.zero_()

            # --- 2. run the network -------------------------------------------
            self.reset_state(B)
            msi_hist = torch.zeros(T_max, B, self.n, device=self.device)

            for t in range(T_max):
                self.update_all_layers_batch(xA[:, t], xV[:, t], valid[:, t])
                msi_hist[t] = self._latest_sMSI

            for seq_i in range(B):
                loc_seq = loc_seqs[seq_i]

                # frames that actually carry a stimulus
                non_blank = [t for t, val in enumerate(loc_seq) if val != 999]
                if not non_blank:
                    continue  # this sequence had no events

                final_t = non_blank[-1] + 5  # last frame of last event

                first_t = final_t
                while first_t > 0 and loc_seq[first_t - 1] != 999:
                    first_t -= 1

                event_slice = slice(first_t, final_t + 1)
                spikes = msi_hist[event_slice, seq_i].sum(dim=0)

                # decode
                pred_deg = decode_msi_location(
                    spikes.unsqueeze(0),  # (1, n)
                    space_size=self.space_size,
                    method=decode
                )[0].item()

                true_deg = loc_seq[final_t]
                err = abs(pred_deg - true_deg)
                if err > 90:  # shortest angular distance
                    err = 180 - err
                errors.append(err)

            n_sequences -= B

        return 0.0 if not errors else float(np.mean(errors))

    # ---------------------------------------------------------------------------
    def evaluate_batch_argmax_10step(self,
                                     n_sequences: int,
                                     batch_size: int = 32,
                                     condition: str = "both") -> float:
        """
        Mean |error| (degrees) using
          • 10-frame stimulus pulse,
          • arg-max over total MSI spikes in those frames,
          • all spikes generated in each external frame.

        condition ∈ {"both", "audio_only", "visual_only"}
        """
        errors = []
        while n_sequences > 0:
            B = min(batch_size, n_sequences)

            # ---- build stimuli -------------------------------------------------
            loc_seqs, mod_seqs, offs, lens = generate_event_loc_seq_batch(
                batch_size=B,
                space_size=self.space_size,
                event_duration=10)  # <-- keep pulse length in sync
            T_max = max(lens)

            xA, xV, valid = generate_av_batch_tensor(
                loc_seqs, mod_seqs, offs,
                n=self.n,
                space_size=self.space_size,
                sigma_in=self.sigma_in,
                noise_std=self.noise_std,
                loc_jitter_std=self.loc_jitter_std,
                stimulus_intensity=1.0,
                device=self.device,
                max_len=T_max)

            if condition == "audio_only":
                xV.zero_()
            elif condition == "visual_only":
                xA.zero_()

            # ---- run network ---------------------------------------------------
            self.reset_state(B)
            msi_sum = torch.zeros(T_max, B, self.n, device=self.device)

            for t in range(T_max):
                *_, sSum = self.update_all_layers_batch(
                    xA[:, t], xV[:, t], valid[:, t],
                    return_spike_sum=True)
                msi_sum[t] = sSum

            # ---- decode each sequence -----------------------------------------
            for i in range(B):
                loc_seq = loc_seqs[i]
                pulse_frames = [t for t, v in enumerate(loc_seq) if v != 999]
                if not pulse_frames:
                    continue
                end = pulse_frames[-1]
                start = max(end - 9, 0)  # 10 frames

                spikes = msi_sum[start:end + 1, i].sum(0)
                pred_i = torch.argmax(spikes).item()
                pred_deg = index_to_location(pred_i, self.n, self.space_size)

                true_deg = loc_seq[end]
                err = abs(pred_deg - true_deg)
                if err > 90:  # shortest path on 0–180° circle
                    err = 180 - err
                errors.append(err)

            n_sequences -= B

        return 0.0 if not errors else float(np.mean(errors))

    # ─────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────
    def isolate_surround(self, enable: bool = True):
        """
        If *enable* is True, this zeroes out:
           • feed-forward A→MSI_ex & V→MSI_ex inhibition
           • MSI_inh → MSI_ex gate
        and freezes inhibitory plasticity, leaving the Mexican-hat
        surround (W_MSI_inh + g_GABA) untouched.
        Call again with False to restore learning.
        """
        z = 0.0 if enable else 1.0
        self.W_inA_inh.mul_(z)
        self.W_inV_inh.mul_(z)
        self.W_msiInh2Exc_GABA.mul_(z)
        self.allow_inhib_plasticity = not enable


def make_checkpoint(net,
                    epoch: int,
                    optim=None,  # pass your optimiser if you need it
                    comment="",
                    rng_tag=True):
    """
    Collect *all* state required to restore or resume the experiment.

    Parameters
    ----------
    net      : trained MultiBatchAudVisMSINetworkTime
    epoch    : int, last finished epoch  (for bookkeeping)
    optim    : torch.optim.Optimizer | None
               If provided, its state_dict is saved so you can resume training.
    comment  : str, optional text note
    rng_tag  : bool, also save torch RNG states (recommended)

    Returns
    -------
    checkpoint : dict  (ready to torch.save)
    """

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    constructor_hparams = dict(
        n_neurons=net.n,
        batch_size=net.batch_size,
        lr_unimodal=net.lr_uni,
        lr_msi=net.lr_msi,
        lr_readout=net.lr_out,
        sigma_in=net.sigma_in,
        noise_std=net.noise_std,
        v_thresh=net.v_thresh,
        dt=net.dt,
        tau_m=net.tau_m,
        n_substeps=net.n_substeps,
        loc_jitter_std=net.loc_jitter_std,
        space_size=net.space_size,
        conduction_delay_a2msi=net.conduction_delay_a2msi,
        conduction_delay_v2msi=net.conduction_delay_v2msi,
        conduction_delay_msi2out=net.conduction_delay_msi2out,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    mutable_hparams = dict(
        # learning hyper-parameters / curriculum
        sigma_teacher_init=net.sigma_teacher_init,
        sigma_teacher_final=net.sigma_teacher_final,
        curriculum_epochs=net.curriculum_epochs,

        # global gains & scaling
        input_scaling=net.input_scaling,
        g_latA=net.g_latA,
        g_latV=net.g_latV,
        g_GABA=net.g_GABA,
        g_FFinh=net.g_FFinh,

        # MSI surround-inhibition hyper-params
        R_near=net.R_near,
        eta_H=net.eta_H,
        eta_AH=net.eta_AH,

        # synaptic-current / receptor time-constants
        gNMDA=net.gNMDA,
        tau_syn=net.tau_syn,
        tau_nmda=net.tau_nmda,
        nmda_alpha=net.nmda_alpha,
        mg_k=net.mg_k,
        Erev_nmda=net.Erev_nmda,
        tau_nmdaVolt=net.tau_nmdaVolt,
        v_nmda_rest=net.v_nmda_rest,
        nmda_vrest_offset=net.nmda_vrest_offset,
        mg_vhalf=net.mg_vhalf,
        dend_coupling_alpha=net.dend_coupling_alpha,

        # Tsodyks–Markram STP
        tau_rec=net.tau_rec,
        tau_fac=net.tau_fac,

        # iSTDP homeostasis
        rho0=net.rho0,
        eta_i=net.eta_i,
        tau_post_i=net.tau_post_i,
        rate_avg_tau=net.rate_avg_tau,

        # conduction delays outside the constructor
        conduction_delay_inA_inh=net.conduction_delay_inA_inh,
        conduction_delay_inV_inh=net.conduction_delay_inV_inh,
        conduction_delay_a2msi_inh=net.conduction_delay_a2msi_inh,
        conduction_delay_v2msi_inh=net.conduction_delay_v2msi_inh,
        conduction_delay_msi_inh2exc=net.conduction_delay_msi_inh2exc,

        # Izhikevich intrinsic parameters
        aA=net.aA, bA=net.bA, cA=net.cA, dA=net.dA,
        aV=net.aV, bV=net.bV, cV=net.cV, dV=net.dV,
        aM=net.aM, bM=net.bM, cM=net.cM, dM=net.dM,
        aMi=net.aMi, bMi=net.bMi, cMi=net.cMi, dMi=net.dMi,
        aO=net.aO, bO=net.bO, cO=net.cO, dO=net.dO,

        # plasticity & toggles
        allow_inhib_plasticity=net.allow_inhib_plasticity,

        # diagnostic counters
        step_counter=net.step_counter,
    )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    rng_state = dict()
    if rng_tag:
        rng_state["torch_cpu"] = torch.get_rng_state()
        rng_state["torch_cuda"] = (torch.cuda.get_rng_state()
                                   if torch.cuda.is_available() else None)

    # ------------------------------------------------------------------
    # D. pack everything together
    # ------------------------------------------------------------------
    checkpoint = dict(
        model_state=net.state_dict(),  # *all* Parameters + buffers
        constructor_hparams=constructor_hparams,
        mutable_hparams=mutable_hparams,
        epoch=epoch,
        comment=comment,
        **({"optim_state": optim.state_dict()} if optim else {}),
        **rng_state,
        timestamp=datetime.utcnow().isoformat(timespec="seconds")
    )
    return checkpoint


# Diagnostics functions
import math
import torch
from collections import defaultdict
from typing import Literal


def run_sc_diagnostics(
        net,
        *,
        centre_deg: float = 90.0,
        modality: Literal["A", "V", "B"] = "B",  # A = audio, V = visual, B = bimodal
        sigma_in: float = 5.0,
        stimulus_intensity: float = 1.0,
        pulse_frames: int = 5,
        n_frames: int = 20,
        noise_std: float = 0.0,
        verbose: bool = True,
) -> dict:
    """
    Passes a brief Gaussian pulse through *net* and returns a dictionary
    of quantitative measures plus an optional human‑readable print‑out.

    Parameters
    ----------
    centre_deg          – azimuth of the pulse (0–179°)
    modality            – 'A', 'V', or 'B' (= both modalities active)
    sigma_in            – input Gaussian σ (neurons) used for the stimulus
    stimulus_intensity  – scale factor applied to the Gaussian input
    pulse_frames        – how many external frames the pulse lasts
    n_frames            – total number of frames simulated
    noise_std           – additive Gaussian noise on the stimulus
    verbose             – if True, prints a nicely formatted report

    Returns
    -------
    metrics : dict
        {
          "spike_rates"      : {layer: Hz, ...},
          "currents"         : {"exc": …, "inh": …, "ampa": …, "nmda": …},
          "I_E_ratio"        : inh / exc,
          "STP"              : {"R_a_mean": …, "R_v_mean": …},
          "raw_time_series"  : defaultdict(list)       # (optional) per‑frame traces
        }
    """
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    n_sub = net.n_substeps
    dt_ms = net.dt
    N = net.n
    device = net.device

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    net.reset_state(batch_size=1)
    net._dbg_spk_A = net._dbg_spk_V = net._dbg_spk_MSI = 0.0
    net._dbg_steps = 0

    xA = torch.zeros(n_frames, N, device=device)
    xV = torch.zeros_like(xA)

    def _gauss_vec(center_deg):
        idx = int(round(center_deg * (N - 1) / (net.space_size - 1)))
        idx = max(0, min(N - 1, idx))
        xs = torch.arange(N, dtype=torch.float32, device=device)
        g = torch.exp(-0.5 * ((xs - idx) / sigma_in) ** 2)
        g = g * stimulus_intensity
        if noise_std > 0:
            g += torch.randn_like(g) * noise_std
        return g

    g_vec = _gauss_vec(centre_deg)

    if modality in ("A", "B"):
        xA[:pulse_frames] = g_vec
    if modality in ("V", "B"):
        xV[:pulse_frames] = g_vec

    valid = torch.ones(n_frames, 1, device=device, dtype=torch.bool)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    ts_store = defaultdict(list)  # raw per‑frame traces (optional)

    for t in range(n_frames):
        _, _, _, _, sum_sM = net.update_all_layers_batch(
            xA[t].unsqueeze(0),
            xV[t].unsqueeze(0),
            valid_mask=None,
            return_spike_sum=True
        )

        # ---- mean currents (MSI excit) ----------------------------------
        I_M = net.I_M.detach()
        exc_curr = torch.clamp(I_M, min=0).mean().item()
        inh_curr = -torch.clamp(I_M, max=0).mean().item()

        mg_A = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_A - net.mg_vhalf)))
        mg_V = 1.0 / (1.0 + torch.exp(-net.mg_k * (net.v_dend_V - net.mg_vhalf)))
        I_nmda = (net.gNMDA
                  * net.nmda_m
                  * (mg_A + mg_V)
                  * (net.Erev_nmda - net.v_msi)).mean().item()
        ampa_curr = max(exc_curr - I_nmda, 0.0)  # safeguard floor

        ts_store["exc"].append(exc_curr)
        ts_store["inh"].append(inh_curr)
        ts_store["ampa"].append(ampa_curr)
        ts_store["nmda"].append(I_nmda)
        ts_store["MSI_spikes"].append(sum_sM.sum().item())

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    sim_time_s = n_frames * n_sub * dt_ms / 1_000.0  # seconds
    spike_rates = {
        "A": net._dbg_spk_A / (N * sim_time_s),
        "V": net._dbg_spk_V / (N * sim_time_s),
        "MSI": net._dbg_spk_MSI / (N * sim_time_s),
    }

    currents = {
        "exc": float(torch.tensor(ts_store["exc"]).mean()),
        "inh": float(torch.tensor(ts_store["inh"]).mean()),
        "ampa": float(torch.tensor(ts_store["ampa"]).mean()),
        "nmda": float(torch.tensor(ts_store["nmda"]).mean()),
    }
    currents["I_E_ratio"] = currents["inh"] / (currents["exc"] + 1e-12)

    stp_stats = {
        "R_a_mean": float(net.R_a.mean().item()),
        "R_v_mean": float(net.R_v.mean().item()),
    }

    metrics = dict(
        spike_rates=spike_rates,
        currents=currents,
        STP=stp_stats,
        raw_time_series=ts_store,
    )

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    if verbose:
        hdr = "=" * 72
        print(hdr)
        print("SC Multisensory‑Network Diagnostics")
        print(hdr)
        print(f"Stimulus  : {modality}  |  centre={centre_deg:.1f}°  |  σ={sigma_in} neur.")
        print(f"Intensity : {stimulus_intensity:.3f}  |  pulse={pulse_frames} frames")
        print(f"Sim time  : {sim_time_s * 1e3:.1f} ms  "
              f"({n_frames} frames × {n_sub} sub‑steps × {dt_ms:.1f} ms)")
        print("\n--- Mean firing rates (Hz) --------------------------------")
        for k, v in spike_rates.items():
            print(f"  {k:>4s}: {v:7.2f}")
        print("\n--- Membrane current @ MSI excit --------------------------")
        print(f"  Excitatory (all) : {currents['exc']:9.4f}")
        print(f"    – AMPA         : {currents['ampa']:9.4f}")
        print(f"    – NMDA         : {currents['nmda']:9.4f}")
        if currents['ampa'] > 0:
            print(f"      NMDA/AMPA    : {currents['nmda'] / currents['ampa']:.3f}")
        print(f"  Inhibitory (net) : {currents['inh']:9.4f}")
        print(f"  I/E ratio        : {currents['I_E_ratio']:.3f}")
        print("\n--- Short‑term plasticity resources -----------------------")
        print(f"  R_a (A→MSI) mean : {stp_stats['R_a_mean']:.3f}")
        print(f"  R_v (V→MSI) mean : {stp_stats['R_v_mean']:.3f}")
        print(hdr)

    return metrics


def analyze_late_nmda_vs_ampa(diagnostics, late_start=5):
    """
    diagnostics: the dict returned by run_sc_diagnostics(...),
                 which already contains raw_time_series in
                 diagnostics["raw_time_series"].

    late_start : int
        The time-step at which we start focusing on the NMDA fraction
        (e.g. skip the first 5 frames if you want).
    """
    ts = diagnostics["raw_time_series"]
    ampa_vals = ts["ampa"]  # or "exc" minus "nmda" if you prefer
    nmda_vals = ts["nmda"]
    steps = range(len(ampa_vals))
    plt.figure(figsize=(6, 4))
    plt.plot(steps, ampa_vals, label="AMPA current", color="C1")
    plt.plot(steps, nmda_vals, label="NMDA current", color="C0")

    # highlight or label the "late" region
    if late_start < len(ampa_vals):
        plt.axvspan(late_start, len(ampa_vals) - 1, color="gray", alpha=0.1,
                    label=f"Late window start={late_start}")

    plt.xlabel("External frame index")
    plt.ylabel("Mean Current (arbitrary units)")
    plt.title("AMPA vs. NMDA Over Time (SC Diagnostics)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ampa_late = ampa_vals[late_start:]
    nmda_late = nmda_vals[late_start:]

    if len(ampa_late) == 0:
        print(f"No data after late_start={late_start}. Nothing to average.")
        return

    ampa_mean = sum(ampa_late) / len(ampa_late)
    nmda_mean = sum(nmda_late) / len(nmda_late)

    ratio = nmda_mean / (ampa_mean + 1e-9)

    print(f"[Late-window analysis] (t >= {late_start})")
    print(f"  AMPA mean: {ampa_mean:.3f},  NMDA mean: {nmda_mean:.3f}")
    print(f"  NMDA/AMPA ratio: {ratio:.3f}")


def _init(mat, mean):  # positive soft‑plus wrappers already active
    with torch.no_grad():
        mat.copy_(torch.abs(torch.randn_like(mat)) * mean)


def generate_two_event_offset_seq(loc, T=60, D=5, offset=0, space_size=180):
    """
    A positive offset → visual lags audio by <offset> macro steps (10 ms each);
    a negative offset → visual leads; 0 → simultaneous.
    """
    loc_seq = [999] * T
    mod_seq = ['X'] * T
    aud_on = 0 if offset >= 0 else abs(offset)
    vis_on = 0 if offset <= 0 else offset
    for t in range(aud_on, aud_on + D):
        loc_seq[t] = loc
        mod_seq[t] = 'A'
    for t in range(vis_on, vis_on + D):
        loc_seq[t] = loc
        mod_seq[t] = 'V' if mod_seq[t] == 'X' else 'B'
    return loc_seq, mod_seq


def generate_flash_sound_batch(
        offsets,
        loc=90,
        T=50,
        D=5,
        space_size=180
):
    loc_seqs = []
    mod_seqs = []
    offset_applied = []
    seq_lengths = []

    for off in offsets:
        seq_loc, seq_mod = generate_two_event_offset_seq(
            loc=loc, T=T, D=D, offset=off, space_size=space_size
        )
        loc_seqs.append(seq_loc)
        mod_seqs.append(seq_mod)
        offset_applied.append(False)
        seq_lengths.append(T)

    return loc_seqs, mod_seqs, offset_applied, seq_lengths


def run_temporal_integration(net, offsets, *, loc=90,
                             T=60, D=5, extra=5, stim_in=1):
    """
    Evaluate MSI population response for a range of AV onset offsets.

    Parameters
    ----------
    net      : trained MultiBatchAudVisMSINetworkTime
    offsets  : list/1-D array of int
               AV onset asynchronies in *macro-steps* (10 ms each).
               Positive  ->  visual lags audio.
               Negative  ->  visual leads audio.
    loc      : spatial location in degrees (default 90).
    T, D     : see generate_two_event_offset_seq  (T time-bins, D duration).
    extra    : number of *extra* macro-steps added to the integration
               window after the burst finishes (default 5 -> 50 ms).

    Returns
    -------
    dict with keys
      'spike_raster' : ndarray (T, len(offsets))       pop. spikes / 10 ms
      'int_spikes'   : 1-D ndarray (len(offsets),)     integrated counts
      'offsets_ms'   : list of onset offsets in ms
    """
    loc_seqs, mod_seqs, off_flags, seq_lens = generate_flash_sound_batch(
        offsets, loc=loc, T=T, D=D, space_size=net.space_size
    )
    max_len = max(seq_lens)
    xA, xV, mask = generate_av_batch_tensor(
        loc_seqs, mod_seqs, off_flags,
        n=net.n, space_size=net.space_size, sigma_in=net.sigma_in,
        noise_std=0.0, device=net.device, max_len=max_len, stimulus_intensity=stim_in,
    )

    # 2 .  run the network
    net.reset_state(len(offsets))
    rast = torch.zeros((max_len, len(offsets)), device=net.device)

    for t in range(max_len):
        net.update_all_layers_batch(xA[:, t], xV[:, t], mask[:, t])
        # population spike count (MSI excit.)
        rast[t] = net._latest_sMSI.sum(dim=1)

    # 3 .  integrate *aligned* windows
    int_spikes = []
    for i_off, off in enumerate(offsets):
        later_onset = abs(off)  # macro-steps until later stimulus
        win_start = later_onset
        win_stop = min(win_start + D + extra, rast.size(0))
        int_spikes.append(rast[win_start:win_stop, i_off].sum().item())

    return {
        'spike_raster': rast.cpu().numpy(),
        'int_spikes': np.asarray(int_spikes),
        'offsets_ms': [o * 10 for o in offsets]  # 1 macro-step = 10 ms
    }


from scipy.optimize import curve_fit


def fit_tbw_curve(offs_ms, int_spikes, *, model="gaussian", p0=None):
    """
    Fit a bell-shaped curve to the temporal-binding-window (TBW) points.

    Parameters
    ----------
    offs_ms     : 1-D array-like
        Audio–visual onset asynchronies in milliseconds.
    int_spikes  : 1-D array-like
        Integrated spike counts (same ordering as offs_ms).
    model       : "gaussian" | "flattop"
        Which analytical shape to fit.
    p0          : list or tuple, optional
        Initial parameter guesses.  If None, sensible defaults are chosen.

    Returns
    -------
    fit_dict    : dict
        {
          "xs"       : densely sampled x-axis,
          "ys"       : fitted curve evaluated at xs,
          "params"   : best-fit parameters,
          "cov"      : covariance matrix from curve_fit,
          "fwhm"     : full-width at half maximum (for Gaussian),
        }
    """
    offs = np.asarray(offs_ms, dtype=float)
    ints = np.asarray(int_spikes, dtype=float)

    # ----------- choose the analytic form -----------------------------------
    if model == "gaussian":
        def _f(x, base, amp, mu, sigma):
            return base + amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        if p0 is None:
            p0 = [ints.min(), np.ptp(ints), 0.0, 60.0]

    elif model == "flattop":
        def _f(x, base, amp, lc, lk, rc, rk):
            left = 1.0 / (1.0 + np.exp(-(x - lc) / lk))
            right = 1.0 / (1.0 + np.exp((x - rc) / rk))
            return base + amp * left * right

        if p0 is None:
            p0 = [ints.min(), np.ptp(ints), -80.0, 10.0, 80.0, 10.0]

    else:
        raise ValueError("model must be 'gaussian' or 'flattop'")

    # ----------- non-linear least-squares fit --------------------------------
    popt, pcov = curve_fit(_f, offs, ints,
                           p0=p0)  # SciPy’s LM/Trust-Region optimiser :contentReference[oaicite:0]{index=0}

    xs = np.linspace(offs.min(), offs.max(), 600)
    ys = _f(xs, *popt)

    fwhm = None
    if model == "gaussian":
        sigma = popt[3]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma  # standard formula :contentReference[oaicite:1]{index=1}

    return {"xs": xs, "ys": ys, "params": popt, "cov": pcov, "fwhm": fwhm}


def plot_temporal_binding(results, *, fit_model="gaussian", **fit_kw):
    """
    Visualise the spike raster AND the TBW curve with an analytical fit,
    *and* print the key numerical values so they can be logged or pasted.

    Parameters
    ----------
    results : dict
        Output of run_temporal_integration.
    fit_model : {"gaussian", "flattop", None}
        Which model to super‑impose.  Pass None to disable fitting.
    fit_kw : dict
        Extra keywords forwarded to fit_tbw_curve.
    """
    rast = results["spike_raster"]
    ints = results["int_spikes"]
    offs = np.asarray(results["offsets_ms"])

    # —–––––––––––––––– heat‑map panel ––––––––––––––––––
    plt.figure(figsize=(8, 4))
    plt.imshow(rast,
               origin="lower", aspect="auto",
               extent=[offs[0], offs[-1], 0, 10 * rast.shape[0]])
    plt.colorbar(label="MSI pop‑spikes / 10 ms")
    plt.xlabel("Audio – Visual onset (ms)")
    plt.ylabel("Time (ms)")
    plt.title("MSI activity vs. AV asynchrony")

    # —–––––––––––––––– binding curve –––––––––––––––––––
    plt.figure(figsize=(4, 3))
    plt.plot(offs, ints, "o", label="data")

    if fit_model is not None:
        fit = fit_tbw_curve(offs, ints, model=fit_model, **fit_kw)
        plt.plot(fit["xs"], fit["ys"], "-", lw=2, label=f"{fit_model} fit")
        # annotate peak & width for Gaussian
        if fit_model == "gaussian":
            base, amp, mu, sigma = fit["params"]
            fwhm = fit["fwhm"]
            plt.annotate(
                f"μ = {mu:+.0f} ms\nFWHM = {fwhm:.0f} ms",
                xy=(mu, base + amp),
                xytext=(mu + 30, base + 0.6 * amp),
                arrowprops=dict(arrowstyle="->", lw=0.8),
                fontsize=8,
            )

    plt.axvline(0, ls="--", c="k", lw=0.7)
    plt.xlabel("Audio – Visual onset (ms)")
    plt.ylabel("Integrated spikes (0–100 ms)")
    plt.title("Temporal binding window")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

    print("Offsets (ms) :", offs.tolist())  # diagnostics
    print("Int. spikes  :", ints.tolist())  # diagnostics


def run_training(
        batch_size=256,
        n_unsup_epochs=80,
):
    """
    Main training run
    """
    print(f"Initializing network with batch size {batch_size}...")
    net = MultiBatchAudVisMSINetworkTime(
        n_neurons=180,
        batch_size=batch_size,
        lr_unimodal=2e-2,
        lr_msi=2e-2,
        lr_readout=8e-4,
        sigma_in=10.0,
        sigma_teacher=2.0,  # (not used directly in final, replaced by scheduling)
        noise_std=0.02,
        single_modality_prob=0.5,
        v_thresh=0.3,
        dt=0.1,
        tau_m=20.0,
        n_substeps=100,
        loc_jitter_std=0,
        space_size=180,
        conduction_delay_a2msi=250,
        conduction_delay_v2msi=400
    )

    init_W_inA = net.W_inA.clone().cpu().numpy()
    init_W_inV = net.W_inV.clone().cpu().numpy()

    net.set_inhib_plasticity(True)

    assign_unimodal_preferred_locations(net)

    print("  W_inA_inh sum =", net.W_inA_inh.sum().item())
    print("  W_inV_inh sum =", net.W_inV_inh.sum().item())
    print("  W_a2msiInh_AMPA sum =", net.W_a2msiInh_AMPA.sum().item())
    print("  W_a2msiInh_NMDA sum =", net.W_a2msiInh_NMDA.sum().item())
    print("  W_v2msiInh_AMPA sum =", net.W_v2msiInh_AMPA.sum().item())
    print("  W_v2msiInh_NMDA sum =", net.W_v2msiInh_NMDA.sum().item())
    print("  W_msiInh2Exc_GABA sum =", net.W_msiInh2Exc_GABA.sum().item())
    print("  W_MSI_inh sum =", net.W_MSI_inh.sum().item())
    print("  g_GABA =", net.g_GABA)

    net.b_uniA.data.fill_(0.0)
    net.b_uniV.data.fill_(0.0)

    _init(net.W_a2msi_AMPA, 0.004)
    _init(net.W_v2msi_AMPA, 0.004)
    _init(net.W_a2msi_NMDA, 0.004)
    _init(net.W_v2msi_NMDA, 0.004)

    with torch.no_grad():

        net.gNMDA = 0.05
        net.tau_nmda = 80.0
        net.nmda_alpha = 0.1
        net.Erev_nmda = 20.0
        net.tau_nmdaVolt = 100.0
        net.v_nmda_rest = -65.0
        net.nmda_vrest_offset = 7.0
        net.mg_vhalf = -35.0


    net.u_a.fill_(0.7)
    net.u_v.fill_(0.7)
    net.tau_rec = 400.0

    print("[tune_for_biology] coarse biological calibration applied")

    # net.auto_calibrate_input_gain(target_MSI_spikes=10)
    net.input_scaling = 2000
    net.g_FFinh = 0.6
    net.g_GABA = 10



    # Unsupervised STDP - Useless, no need
    print("\n--- STDP training (unsupervised) ---")
    unsup_start = time.time()
    last_ep = 0
    for epoch in range(n_unsup_epochs):
        last_ep = epoch
        epoch_start = time.time()
        if 2 <= epoch <= 79:  # choose any window you like
            if net._probe is None:
                net._probe = AMPANMDADebugger()
            else:
                net._probe.reset()
        with torch.no_grad():
            W_before = net.W_inA.clone()  # snapshot *before* training

        net.train_unsupervised_batch(1000, batch_size=256, debug=False, epoch_idx=epoch)  # run some sequences
        net.print_epoch_spike_summary(f"unsup {epoch + 1:02d}")

        if 2 <= epoch <= 79:
            net._probe.report(net, f"epoch {epoch}")

        with torch.no_grad():
            delta = (net.W_inA - W_before).abs().max().item()
            print("Δ‖W_inA‖ =", delta)
        epoch_time = time.time() - epoch_start
        print(f"  Unsup Epoch {epoch + 1}/{n_unsup_epochs} - Time: {epoch_time:.2f}s")
    unsup_time = time.time() - unsup_start
    print(f"Unsupervised training completed in {unsup_time:.2f}s")

    net.set_inhib_plasticity(False)

    ckpt = make_checkpoint(net,
                           epoch=last_ep,
                           optim=None,  # or None if you’re done training
                           comment="MSI model – paper Figure 3")

    save_path = Path("checkpoint") / "msi_redone_agc_fix_.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, save_path)
    print(f"✅  Full checkpoint written to  {save_path.resolve()}")

    net = None

    ckpt_path = Path("checkpoint/msi_redone_agc_fix_.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = MultiBatchAudVisMSINetworkTime(**ckpt["constructor_hparams"])
    net.load_state_dict(ckpt["model_state"])
    for k, v in ckpt["mutable_hparams"].items():
        setattr(net, k, v)
    if "torch_cpu" in ckpt:
        torch.set_rng_state(ckpt["torch_cpu"])
    if "torch_cuda" in ckpt and ckpt["torch_cuda"] is not None:
        torch.cuda.set_rng_state(ckpt["torch_cuda"])

    net.eval()  # or net.train() to keep learning
    print("✔️  Model ready for evaluation.")

    net.reset_state()

    net.set_inhib_plasticity(True)

    net.reset_state()


    offsets = list(range(-50, 51))  # −100 … +100 ms in 10 ms steps
    res_ti = run_temporal_integration(net, offsets, loc=90, T=60, D=5, stim_in=1)
    plot_temporal_binding(res_ti)
    net.reset_state()


    msi_activity_summary(net,
                         centre_deg=90,
                         sigma_in=5,
                         pulse_len=15,
                         n_steps=30,
                         modality="A",
                         style="ggplot")

    msi_activity_summary(net,
                         centre_deg=130,
                         sigma_in=5,
                         pulse_len=15,
                         n_steps=30,
                         modality="V",
                         style="ggplot")

    return {
        'network': net

    }


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def train_and_save(model_idx: int,
                   base_seed: int = 42,
                   out_dir: str = "checkpoint") -> Path:
    """
    Build ➜ train ➜ checkpoint one network replica.

    Parameters
    ----------
    model_idx : 0‑based integer label (0…9)
    base_seed : deterministic offset so each net sees a unique RNG stream
    out_dir   : folder where .pt files are written

    Returns
    -------
    Path to the file that was saved.
    """
    torch.manual_seed(base_seed + model_idx)
    np.random.seed(base_seed + model_idx)
    results = run_training()
    net = results["network"]
    ckpt = make_checkpoint(net,
                           epoch=0,
                           comment=f"replica {model_idx}",
                           rng_tag=True)

    # save to disk
    out_path = Path(out_dir) / f"msi_model_surr_16_{model_idx:02d}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    del net
    torch.cuda.empty_cache()

    return out_path


# ----------------------------------------------------------------------
if __name__ == "__main__":
    N_REPLICAS = 10
    saved = []
    for i in range(N_REPLICAS):
        print(f"\n=== TRAINING REPLICA {i + 1}/{N_REPLICAS} ===")
        path = train_and_save(i)
        saved.append(path)
        print(f"✔ Saved checkpoint ➜ {path}")
    print("\nAll replicas finished:")
    for p in saved:
        print("  •", p)

