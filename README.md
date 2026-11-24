# afpool: Open-Set Radio-Frequency Identification of Individual Drones by Acquisition-Factor Pooling

This repository holds the core implementation of the paper *Open-Set Radio-Frequency Identification of Individual Drones by Acquisition-Factor Pooling* by Xinbin Zhang, Xixi Zhang, Zhengran He, Tiantian Tang, Guangjie Han, and Guan Gui (Nanjing University of Posts and Telecommunications; Hohai University), under review.

The system works on one pool built from several public drone and WiFi RF datasets, and reads no band, dataset, or receiver label at inference. It:

- names enrolled drones;
- rejects unregistered ones;
- groups the rejected recordings by individual, without knowing how many individuals there are.

## What is included

| Module | Paper | Content |
|---|---|---|
| `afpool/signal.py` | Sec. III-A, eq. (12) | burst windows, the transient view of f_T, the frequency-centered view of f_A, the pre-onset noise spectrum phi(u) |
| `afpool/encoder.py` | Sec. IV-B, eq. (5) | identity encoder (shared by f_T and f_A), unit embeddings, checkpoint loading |
| `afpool/training.py` | Sec. IV-B, Table II | training objective: channel augmentation, AWGN, bag cosine cross-entropy, VICReg, cross-bag and cross-receiver contrastive terms, optimizer settings |
| `afpool/recr.py` | Sec. IV-C, eqs. (6)–(10) | **relative-evidence consensus rejection (RECR)**: three relative scores, quantile consensus over two encoders, calibrated threshold |
| `afpool/rgd.py` | Sec. IV-D, eqs. (11)–(14) | **receiver-grouped discovery (RGD)**: WCCN, receiver groups from pre-onset spectra, within-group clustering, cross-group association, and the enrolled-only calibration of every threshold |
| `afpool/metrics.py` | Sec. III-E | AUROC, OSCR, FPR95, FAR, ARI, NMI, ACC, B-cubed F, cross-receiver split rate, false-merge rate |

## Installation

```bash
git clone https://github.com/Kilimetarlop/open-set-drone-rffi.git
cd open-set-drone-rffi
pip install -r requirements.txt
pip install -e .
```

Python 3.10 or later. The paper's labels need `hdbscan==0.8.44` for exact reproduction; other versions may break ties differently.

## Quick start

```bash
python examples/demo_synthetic.py   # RECR and RGD end to end on a synthetic pool (no data needed)
python -m pytest tests              # fast API checks
```

The demo builds a toy pool with the three classes of acquisition factors that the paper analyzes:

- **Identity-constant factors:** enrolled identities.
- **Identity-varying factors:** a receiver effect that splits each drone heard by two receivers.
- **Shared factors:** look-alike unregistered identities.

On this toy pool, RECR rejects most unregistered units, and RGD lowers the cross-receiver split rate. These numbers describe the synthetic pool only.

Typical use on your own data:

```python
from afpool import encoder, signal, recr, rgd

f_T = encoder.load_checkpoint('f_T.pt')   # or train with afpool.training
z_T, ok = encoder.unit_embedding(f_T, signal.transient_view, bursts)      # bursts: [n, 2, 16384] float32
z_A, ok = encoder.unit_embedding(f_A, signal.channel_view, bursts)
phi = signal.preonset_spectrum(list(bursts))

out = recr.recr(gallery, gallery_labels, calibration, queries, valid)    # dicts keyed by encoder: {'T': ..., 'A': ...}
labels, groups = rgd.rgd(rgd.Discovery(), x, ids, spectra, prototypes, within_cal, d_r, theta_x, delta_x)
```

The calibration functions are `rgd.calibrate_within`, `rgd.receiver_cut`, `rgd.receiver_episodes`, and `rgd.calibrate_cross`. They use enrolled data only: units of unregistered individuals enter no training, calibration, threshold, or model-selection step.

## Faithfulness to the paper

This package was checked against the sealed system that produced the paper's results. All 126 checks are exact equality, not tolerance:

- **Encoders:** the checkpoints load into `IdentityEncoder` and give identical views and embeddings.
- **Signal and training terms:** the spectrum, the augmentations, and the losses equal the original implementations.
- **RECR:** on all ten folds, it reproduces every rejection score, accept decision, decided identity, and threshold.
- **RGD:** on all ten folds, it reproduces the WCCN transform, every calibrated threshold, and every discovery label.

The verification script runs on the paper's per-fold embedding caches. It is part of the complete code release.

## Data

The pooled datasets are public, and their providers distribute them under their own licenses. This repository does not redistribute them; see the paper's references for each source:

- Hovering-UAV;
- WiSig;
- DroneRFb-DIR;
- DRFF-R2;
- DroneMMset.

## Complete code

The complete code will be released here after the paper's review. It includes:

- the data readers, training schedules, and experiment contracts;
- the published-method baselines;
- the figure and table scripts;
- the trained checkpoints;
- the verification script.

## Citation

If you use this code, please cite the paper:

```bibtex
@unpublished{zhang2026afpool,
  author = {Xinbin Zhang and Xixi Zhang and Zhengran He and Tiantian Tang and Guangjie Han and Guan Gui},
  title  = {Open-Set Radio-Frequency Identification of Individual Drones by Acquisition-Factor Pooling},
  note   = {Manuscript under review},
  year   = {2026}
}
```

The entry will be updated with the publication details after acceptance.

## License

MIT; see [LICENSE](LICENSE).
