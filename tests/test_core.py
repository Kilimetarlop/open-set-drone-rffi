"""Fast checks of the public API on synthetic data (no recorded data, no checkpoints). Run: python -m pytest tests"""
import numpy as np
import torch

from afpool import encoder, metrics, recr, rgd, signal, training


def _bursts(n=3, seed=0):
    return signal.normalize_iq(torch.randn(n, 2, signal.WINDOW, generator=torch.Generator().manual_seed(seed)))


def test_views_and_spectrum():
    x = _bursts()
    for view in (signal.transient_view, signal.channel_view):
        v = view(x)
        assert v.shape == x.shape and torch.isfinite(v).all()
        assert torch.allclose(v.double().square().sum(1).mean(1), torch.ones(len(x), dtype=torch.float64), atol=1e-5)
    wifi = torch.randn(2, 2, 256)
    assert torch.equal(signal.transient_view(wifi), wifi)
    phi = signal.preonset_spectrum([w for w in x.numpy()])
    assert phi.shape == (64,) and abs(phi.mean()) < 1e-9
    assert np.isnan(signal.preonset_spectrum([np.zeros((2, 256))])).all()


def test_encoder_and_unit_embedding():
    model = encoder.IdentityEncoder().eval()
    out = model(signal.transient_view(_bursts()))
    assert out['e'].shape == (3, 64) and bool(out['valid'].all())
    z, ok = encoder.unit_embedding(model, signal.channel_view, _bursts().numpy())
    assert ok and abs(np.linalg.norm(z) - 1) < 1e-9


def test_training_terms_are_finite():
    model, head = encoder.IdentityEncoder(), encoder.CosineHead(2)
    x = _bursts(4)
    target = torch.tensor([0, 0, 1, 1])
    phi = torch.rand(4, 2) * 6.28
    loss = training.burst_batch_loss(model, head, x, target, ['a', 'a', 'b', 'b'], phi, torch.Generator().manual_seed(1))
    loss.backward()
    assert torch.isfinite(loss)


def test_recr_rejects_far_queries():
    rng = np.random.default_rng(0)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    means = unit(rng.normal(size=(6, 64)))
    def draw(ids, n):
        return unit(np.concatenate([means[i] + .1 * rng.normal(size=(n, 64)) for i in ids])), [f'id{i}' for i in ids for _ in range(n)]
    g, gl = draw(range(4), 8)
    c, _ = draw(range(4), 5)
    q_known, _ = draw(range(4), 5)
    q_new, _ = draw([4, 5], 5)
    q = np.concatenate([q_known, q_new])
    out = recr.recr({'T': g}, gl, {'T': c}, {'T': q}, {'T': np.ones(len(q), bool)})
    kn, un = list(range(len(q_known))), list(range(len(q_known), len(q)))
    assert metrics.auroc(out['score'][kn], out['score'][un]) > .9


def test_rgd_receiver_groups_and_readout():
    rng = np.random.default_rng(1)
    spectra = np.vstack([rng.normal(0, .1, size=(10, 64)), rng.normal(0, .1, size=(10, 64)) + 2])
    groups = rgd.receiver_groups(spectra, d_r=5.0)
    assert len(set(groups[:10])) == 1 and len(set(groups[10:])) == 1 and groups[0] != groups[10]
    assert rgd.receiver_cut(spectra, [0] * 10 + [1] * 10) > 0
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    x = unit(np.vstack([rng.normal(0, .05, size=(10, 8)) + np.eye(8)[0], rng.normal(0, .05, size=(10, 8)) + np.eye(8)[1]]))
    protos = unit(np.eye(8)[2:4])
    cal = {'theta': .9, 'delta': 0.0, 'tau_a': .5, 'delta_u': 0.0}
    lab, _ = rgd.rgd(rgd.Discovery(), x, [f'u{i:02d}' for i in range(20)], spectra, protos, cal, 5.0, .5, 0.0)
    assert metrics.discovery_measures(['a'] * 10 + ['b'] * 10, lab.tolist())['ARI'] > .9
