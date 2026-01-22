# From Spikes to Symptoms: Simulating SC‑Driven Multisensory Deficits in Autism Spectrum Disorder

The Python code in this repository serves as a key resource for:

**Mohan & Rideaux — _From Spikes to Symptoms: Simulating SC‑Driven Multisensory Deficits in Autism Spectrum Disorder_** 

This project implements a biologically grounded **spiking neural network (SNN)** model of the **superior colliculus (SC)** and uses it to study how specific synaptic/cellular perturbations (e.g., NMDA conductance scaling, altered feedforward inhibition, reduced spike‑frequency adaptation) reshape **audiovisual multisensory integration**.

---

## Repository contents

### Training

- `Training.py`  
  Trains the SNN on **synthetic audiovisual event sequences** using unsupervised plasticity, then saves PyTorch checkpoints (`.pt`) to `checkpoint/`.

### Analyses / figure reproduction

Each analysis script loads pretrained checkpoints and saves figures to `Saved_Images/`.

- `SBW_test.py` — spatial binding window (SBW) / spatial fusion curves  
  **Outputs:** `Saved_Images/SBW_curve.svg`, `Saved_Images/sbw_inset.svg`

- `TBW_test.py` — temporal binding window (TBW) / temporal fusion curves  
  **Outputs:** `Saved_Images/TBW_curve.svg`, `Saved_Images/tbw_inset.svg`

- `cue_reliability_test.py` — cue reliability weighting  
  **Output:** `Saved_Images/cue_rel.svg`

- `inverse_effectiveness_test.py` — inverse effectiveness analysis  
  **Output:** `Saved_Images/inv_eff.svg`

- `response_latency_test.py` — response latency analysis  
  **Output:** `Saved_Images/Latency.svg`

- `precision_hist_test.py` — localisation error distributions / precision histograms  
  **Outputs:** `Saved_Images/Err_Hist.svg`, `Saved_Images/sens_lowNMDA.svg`

- `EI_balance_test.py` — excitation/inhibition balance  
  **Output:** `Saved_Images/EI.svg`

- `fano_factor_test.py` — spike‑count variability (Fano factor)  
  **Output:** `Saved_Images/fano.svg`

### Assets

- `checkpoint/` — pretrained model checkpoints (`*.pt`)
- `Saved_Images/` — generated figures (SVG/PNG)
- `fonts/` — fonts used for figure generation

---

## Model overview (high level)

- SC‑inspired architecture with **unisensory auditory and visual** populations projecting to a **multisensory integration (MSI)** population (excitatory + inhibitory).
- **Izhikevich‑type spiking neurons** (regular‑spiking excitatory; fast‑spiking inhibitory).
- Explicit synaptic currents for **AMPA**, **NMDA**, and **GABA**, plus conduction delays.
- Learning includes **STDP** (feedforward pathways) and homeostatic mechanisms to stabilize activity.

See the manuscript for the full methodological details.

---

## Requirements

- Python ≥ 3.9
- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas` (required for `response_latency_test.py`)

---

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install torch numpy scipy matplotlib pandas
```

If you plan to use an NVIDIA GPU, install a CUDA‑enabled build of PyTorch suitable for your system.

---

## Quickstart: reproduce figures with pretrained checkpoints

Most users should start here.

From the repository root:

```bash
python EI_balance_test.py
python fano_factor_test.py
python SBW_test.py
python TBW_test.py
python cue_reliability_test.py
python inverse_effectiveness_test.py
python precision_hist_test.py
python response_latency_test.py
```

Figures are written to `Saved_Images/`.

**Headless environments (SSH/HPC):** some scripts call `plt.show()`. If you are running without a display, comment out `plt.show()` or switch matplotlib to a non‑interactive backend.

---

## CPU vs GPU

Several scripts default to `device="cuda"`. To run on CPU, change the relevant `device` argument in the script to `"cpu"`.

`cue_reliability_test.py` also exposes a CLI flag:

```bash
python cue_reliability_test.py --device cpu
```

---

## Training (optional)

To train new networks from scratch:

```bash
python Training.py
```

`Training.py` trains an ensemble of networks (10 instantiations by default) and saves checkpoints into `checkpoint/`.

> **Note:** the analysis scripts are configured to load the provided
> `checkpoint/msi_model_surr_10_00.pt` … `checkpoint/msi_model_surr_10_09.pt`.
> If you train new checkpoints with different filenames, update the `model_paths`
> list inside each analysis script accordingly.

---

## Perturbation experiments

The manuscript explores different perturbations by modifying network parameters **after loading a trained checkpoint**.

Each analysis script contains a `tweak_fn` / `modify_net` hook near the bottom (inside the `main_*` function). Example:

```python
def tweak_fn(net):
    # 1) NMDA hypofunction / hyperfunction
    net.gNMDA = 0.02   # hypofunction
    # net.gNMDA = 0.15 # hyperfunction

    # 2) Reduced feedforward inhibition (scale to 30% of baseline)
    # net.g_FFinh *= 0.3

    # 3) Reduced spike‑frequency adaptation in MSI excitatory neurons
    # net.aM = 0.0001
    # net.bM = 0.2
    # net.cM = -60
    # net.dM = 0.01

    return net
```

A helper is also provided to isolate the surround (“Mexican hat”) component:

```python
net.isolate_surround(enable=True)
```

---

## Checkpoint format

All `.pt` files in `checkpoint/` are PyTorch checkpoints containing:

- `model_state` — model parameters
- `constructor_hparams` — constructor hyperparameters (used to recreate the network)
- `mutable_hparams` — additional parameters applied after construction

---

## Citing

If you use this code in academic work, please cite the accompanying manuscript.

---

## Contact & acknowledgements

Corresponding author: **reuben.rideaux@sydney.edu.au**

Funding: ARC DECRA (DE210100790) and NHMRC Investigator Grant (2026318). See the manuscript for full acknowledgements and references.

---

## License

No project‑wide license file is included in this archive. Font files in `fonts/` are provided under their own license (see `fonts/LICENSE.txt`).
