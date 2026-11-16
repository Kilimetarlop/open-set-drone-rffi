"""End-to-end demonstration of RECR and RGD on synthetic embeddings (no recorded data needed).

The synthetic pool imitates the three acquisition-factor classes of the paper:
  - enrolled identities with gallery, calibration, and later query units;
  - a receiver effect on every unit recorded by the second receiver, with a common part and a part that differs
    per identity (an identity-varying factor that splits each drone heard by both receivers and that whitening
    cannot remove), and a receiver-specific pre-onset noise spectrum;
  - unregistered look-alike identities placed near enrolled ones (a shared factor).
The numbers it prints describe this toy pool only; they are not results of the paper.

Run:  python examples/demo_synthetic.py
"""
import numpy as np

from afpool import metrics, recr, rgd

rng = np.random.default_rng(0)
D = 64


def unitv(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_identities(n, spread=1.0):
    return unitv(rng.normal(size=(n, 2, D)) * spread)  # one mean per identity and encoder


def units(mean, n, receiver, offset, noise=0.12):
    """n unit embeddings per encoder for one identity; receiver 1 adds the identity's receiver effect."""
    x = mean[None] + noise * rng.normal(size=(n, 2, D)) + (offset[None] if receiver == 1 else 0)
    return unitv(x)


def receiver_effect():
    return 0.3 * common + 2.0 * unitv(rng.normal(size=(2, D)))


def spectrum(receiver, n):
    """Pre-onset spectra: flat for receiver 0, peaked at the center for receiver 1, plus noise."""
    f = np.linspace(-1, 1, 64)
    shape = np.zeros(64) if receiver == 0 else 1.5 * np.exp(-(f / .35) ** 2)
    s = shape[None] + .15 * rng.normal(size=(n, 64))
    return s - s.mean(1, keepdims=True)


common = unitv(rng.normal(size=(2, D)))
enrolled = make_identities(12)
lookalike = unitv(enrolled[:3] + 0.35 * unitv(rng.normal(size=(3, 2, D))))   # unregistered, close to identities 0-2
strangers = make_identities(3)                                                # unregistered, far from all
unregistered = np.concatenate([lookalike, strangers])
effect_enrolled = [receiver_effect() for _ in enrolled]
effect_unreg = [receiver_effect() for _ in unregistered]

# ---------------------------------------------------------------- enrolled data (both receivers)
# The first four enrolled identities were recorded by both receivers (label prefix 'M|', as the receiver-diverse
# calibration expects); the others by receiver 0 only, like a single-receiver dataset.
gal, gal_lab, cal, cal_lab, qk, qk_lab, rx_enrolled = [], [], [], [], [], [], []
for i, m in enumerate(enrolled):
    name, receivers = (f'M|id{i}', (0, 1)) if i < 4 else (f'S|id{i}', (0,))
    for r in receivers:
        gal.append(units(m, 6, r, effect_enrolled[i])); gal_lab += [name] * 6; rx_enrolled += [r] * 6
        cal.append(units(m, 4, r, effect_enrolled[i])); cal_lab += [name] * 4
        qk.append(units(m, 5, r, effect_enrolled[i])); qk_lab += [name] * 5
gal, cal, qk = np.concatenate(gal), np.concatenate(cal), np.concatenate(qk)

# ---------------------------------------------------------------- unregistered units (both receivers)
qu, qu_lab, qu_rx = [], [], []
for j, m in enumerate(unregistered):
    for r in (0, 1):
        qu.append(units(m, 6, r, effect_unreg[j])); qu_lab += [f'new{j}'] * 6; qu_rx += [r] * 6
qu = np.concatenate(qu)

# ---------------------------------------------------------------- RECR: identification and rejection
queries = np.concatenate([qk, qu])
by_encoder = lambda a: {'T': a[:, 0], 'A': a[:, 1]}
valid = {m: np.ones(len(queries), bool) for m in ('T', 'A')}
out = recr.recr(by_encoder(gal), gal_lab, by_encoder(cal), by_encoder(queries), valid)
kn, un = list(range(len(qk))), list(range(len(qk), len(queries)))
truth = qk_lab + qu_lab
print('RECR', {k: round(v, 3) for k, v in metrics.rejection_measures(out['score'], out['tau'], out['identity'], truth, kn, un).items()})

# ---------------------------------------------------------------- RGD: discovery of the unregistered units
disc = rgd.Discovery()
joint = lambda a: rgd.joint_embedding(a[:, 0], a[:, 1])
# As in the paper, WCCN and the within-group thresholds use the single-receiver enrolled identities; the enrolled
# drones heard by both receivers set d_r, the receiver-diverse episodes, and prototypes.
single_g = [i for i, l in enumerate(gal_lab) if not l.startswith('M|')]
single_c = [i for i, l in enumerate(cal_lab) if not l.startswith('M|')]
t = rgd.wccn(np.concatenate([joint(gal[single_g]), joint(cal[single_c])]),
             [gal_lab[i] for i in single_g] + [cal_lab[i] for i in single_c])
fit_z = t(joint(gal))
cal_w = rgd.calibrate_within(disc, fit_z[single_g], [gal_lab[i] for i in single_g], t(joint(cal[single_c])),
                             [cal_lab[i] for i in single_c])
protos, proto_ids = disc.prototypes(fit_z, gal_lab)
spec_enrolled = np.concatenate([spectrum(r, 1) for r in rx_enrolled])
dual = [i for i, l in enumerate(gal_lab) if l.startswith('M|')]
d_r = rgd.receiver_cut(spec_enrolled[dual], [rx_enrolled[i] for i in dual])
episodes, by = rgd.receiver_episodes(gal_lab, rx_enrolled, sizes=(2, 3, 4), per_size=6)
cx = rgd.calibrate_cross(disc, fit_z, spec_enrolled, gal_lab, episodes, by, protos, proto_ids, cal_w, d_r)

x = t(joint(qu))
ids = [f'u{i:04d}' for i in range(len(x))]
spec_q = np.concatenate([spectrum(r, 1) for r in qu_rx])
single = disc.single_group(x, ids, protos, cal_w)
grouped, groups = rgd.rgd(disc, x, ids, spec_q, protos, cal_w, d_r, cx['theta_x'], cx['delta_x'])
for name, lab in (('without receiver groups', single), ('RGD', grouped)):
    m = metrics.discovery_measures(qu_lab, lab.tolist())
    m.update(metrics.pair_rates(lab.tolist(), qu_lab, qu_rx))
    print(f'{name:24s}', {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
print('receiver groups found:', len(set(groups.tolist())), '| d_r = %.3f, theta_x = %.2f, delta_x = %.2f' % (d_r, cx['theta_x'], cx['delta_x']))
