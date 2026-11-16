"""Training objective of the identity encoders (Section IV-B and Table II), as a reference implementation.

Per update, one source dataset contributes one batch:
  - burst datasets (64 windows): two random-phase views of every window, each channel-augmented for f_A; a cosine
    classification head per dataset on the mean embedding of each recording ("bag"), scale 16; VICReg (weight 0.04)
    on the 64-256-256 projector; for f_A, a supervised contrastive term (temperature 0.1, weight 0.1) whose
    positives are recordings of the same identity in other bags, which for the drones heard by both receivers are
    the recordings by the other receiver;
  - WiFi preambles (32 windows, single view): cosine head plus a supervised contrastive term whose positives share
    the transmitter across another receiver and another day.
Optimizer AdamW (lr 3e-4, weight decay 1e-4), gradient clip 5. f_T trains on clean transient views of the four
development datasets for 8000 updates; f_A adds channel augmentation and, in one of nine updates, a batch of the
enrolled drones heard by both receivers, for 9000 updates.

The full pipeline with its data readers, schedules, and guards is in the complete code release; this module keeps
the arithmetic of every term, so new datasets can be trained with the same objective.
"""
import torch
from torch.nn import functional as F

from .signal import normalize_iq, rotate_phase, transient_view, center_frequency, WINDOW

ADAMW = dict(lr=3e-4, weight_decay=1e-4, betas=(.9, .999), eps=1e-8)
GRAD_CLIP = 5.0
HEAD_SCALE = 16.0
VICREG_WEIGHT = 0.04
SUPCON_TEMPERATURE, SUPCON_WEIGHT = 0.1, 0.1
AUG_PROB, MAX_TAPS, MAX_DRIFT = 1.0, 8, 2e-5
SNR_DB = (5.0, 30.0)


def channel_augment(x, generator, prob=AUG_PROB):
    """Random multipath (1-8 taps, exponential power profile) and slow frequency drift, then RMS renormalization."""
    b, _, n = x.shape
    u = lambda *shape: torch.rand(shape, generator=generator, dtype=torch.float64)
    apply = u(b) < prob
    taps = torch.floor(u(b) * MAX_TAPS).long() + 1
    tau = 0.5 + 2.5 * u(b)
    lag = torch.arange(MAX_TAPS, dtype=torch.float64)
    power = torch.exp(-lag[None, :] / tau[:, None]) * (lag[None, :] < taps[:, None])
    gains = torch.complex(torch.randn(b, MAX_TAPS, generator=generator, dtype=torch.float64),
                          torch.randn(b, MAX_TAPS, generator=generator, dtype=torch.float64)) * power.sqrt()
    gains = gains / gains.abs().square().sum(1, keepdim=True).sqrt()
    delta = (u(b) * 2 - 1) * MAX_DRIFT
    phi = u(b) * 2 * torch.pi
    identity = torch.zeros(b, MAX_TAPS, dtype=torch.complex128)
    identity[:, 0] = 1
    gains = torch.where(apply[:, None], gains, identity)
    delta = torch.where(apply, delta, torch.zeros_like(delta))
    phi = torch.where(apply, phi, torch.zeros_like(phi))
    z = torch.complex(x[:, 0].double(), x[:, 1].double())
    g = gains.to(x.device)
    y = torch.zeros_like(z)
    for lag_ in range(MAX_TAPS):
        y = y + g[:, lag_:lag_ + 1] * (F.pad(z, (lag_, 0))[:, :n] if lag_ else z)
    t = torch.arange(n, dtype=torch.float64, device=x.device)
    y = y * torch.exp(1j * (2 * torch.pi * delta.to(x.device)[:, None] * t[None, :] + phi.to(x.device)[:, None]))
    y = y / y.abs().square().mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    return torch.stack((y.real, y.imag), dim=1).float()


def awgn(x, generator, snr_db=SNR_DB):
    """Additive white Gaussian noise at a random SNR per window (unit-power signal)."""
    snr = snr_db[0] + (snr_db[1] - snr_db[0]) * torch.rand(len(x), generator=generator, dtype=torch.float64)
    sigma = (10 ** (-snr / 10) / 2).sqrt().float().to(x.device)
    return x + sigma[:, None, None] * torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x.device)


def channel_training_view(x, generator):
    """Training view of f_A: channel augmentation, AWGN, frequency centering, transient view."""
    v = awgn(channel_augment(x, generator), generator)
    if v.shape[-1] == WINDOW:
        v = normalize_iq(center_frequency(v).contiguous())
    return transient_view(v)


def vicreg_loss(p):
    """VICReg on aligned projections p [B, 2, 256]: 25 invariance + 25 variance + covariance."""
    batch, dim = p.shape[0], p.shape[-1]
    a, b = p[:, 0], p[:, 1]
    invariance = (a - b).square().mean()
    std_a, std_b = torch.sqrt(a.var(0, correction=1) + 1e-4), torch.sqrt(b.var(0, correction=1) + 1e-4)
    variance = (F.relu(1 - std_a).mean() + F.relu(1 - std_b).mean()) / 2
    ca, cb = a - a.mean(0, keepdim=True), b - b.mean(0, keepdim=True)
    off = ~torch.eye(dim, dtype=torch.bool, device=p.device)
    cov = (((ca.T @ ca) / (batch - 1))[off].square().sum() + ((cb.T @ cb) / (batch - 1))[off].square().sum()) / dim
    return 25 * invariance + 25 * variance + cov


def bag_cosine_ce(e, target, bag_keys, head):
    """Cosine cross-entropy on the mean embedding of each recording bag; e [B, 2 views, 64].

    Returns the loss, the bag embeddings [bags, 2, 64], and the identity of each bag."""
    groups = {}
    for i, key in enumerate(bag_keys):
        groups.setdefault(key, []).append(i)
    bags, labels = [], []
    for rows in groups.values():
        m = e[rows].mean(0)
        bags.append(m / torch.linalg.vector_norm(m, dim=1, keepdim=True))
        labels += [int(target[rows[0]])] * 2
    bag = torch.stack(bags)
    loss = F.cross_entropy(HEAD_SCALE * head(bag.reshape(-1, e.shape[-1])), torch.tensor(labels, device=e.device))
    return loss, bag, [int(target[rows[0]]) for rows in groups.values()]


def cross_bag_supcon(emb, identity, group, temperature=SUPCON_TEMPERATURE):
    """Supervised contrastive term whose positives share the identity in another bag (another recording)."""
    sim = (emb @ emb.T) / temperature
    eye = torch.eye(len(emb), dtype=torch.bool, device=emb.device)
    same_group = group[:, None] == group[None, :]
    positive = (identity[:, None] == identity[None, :]) & ~same_group
    candidates = ~eye & ~(same_group & ~eye)
    has = positive.any(1)
    if not bool(has.any()):
        return (emb * 0).sum()
    logprob = sim - torch.logsumexp(sim.masked_fill(~candidates, -torch.inf), dim=1, keepdim=True)
    loss = -(logprob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))
    return loss[has].mean()


def cross_receiver_day_supcon(e, tx, rx, day, temperature=SUPCON_TEMPERATURE):
    """WiFi term: positives share the transmitter across another receiver and another day."""
    u = e / torch.linalg.vector_norm(e, dim=1, keepdim=True)
    logits = (u @ u.T) / temperature
    eye = torch.eye(len(e), device=e.device, dtype=torch.bool)
    same = tx[:, None] == tx[None, :]
    positive = same & (rx[:, None] != rx[None, :]) & (day[:, None] != day[None, :]) & ~eye
    candidates = ~eye & (~same | positive)
    logprob = logits - torch.logsumexp(logits.masked_fill(~candidates, -torch.inf), dim=1)[:, None]
    return -(logprob.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1)).mean()


def burst_batch_loss(model, head, iq, target, bag_keys, phi, generator=None, contrastive=True):
    """Loss of one burst batch: two random-phase views (channel-augmented when generator is given, i.e. for f_A),
    bag cosine CE + 0.04 VICReg (+ 0.1 cross-bag contrastive)."""
    views = [rotate_phase(iq, phi[:, k]) for k in (0, 1)]
    views = [channel_training_view(v, generator) if generator is not None else transient_view(v) for v in views]
    out = model(torch.stack(views, 1).reshape(-1, 2, iq.shape[-1]))
    e = out['e'].reshape(len(iq), 2, -1)
    p = model.projector(out['z']).reshape(len(iq), 2, -1)
    ce, bag, bag_ids = bag_cosine_ce(e, target, bag_keys, head)
    loss = ce + VICREG_WEIGHT * vicreg_loss(p)
    if contrastive:
        ident = torch.tensor([i for i in bag_ids for _ in (0, 1)], device=e.device)
        grp = torch.tensor([g for g in range(len(bag_ids)) for _ in (0, 1)], device=e.device)
        loss = loss + SUPCON_WEIGHT * cross_bag_supcon(bag.reshape(-1, e.shape[-1]), ident, grp)
    return loss


def optimizer_for(model, heads):
    params = list(model.parameters()) + [p for h in heads for p in h.parameters()]
    return torch.optim.AdamW(params, **ADAMW), params
