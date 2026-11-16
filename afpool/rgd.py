"""Receiver-grouped discovery, RGD (Section IV-D).

RGD takes the embeddings and pre-onset spectra of the unregistered units and returns one cluster per individual:
  1. WCCN (eq. (11)) normalizes the joint embedding so that within-identity variation is equal in all directions;
  2. receiver groups (eq. (13)): average-linkage clustering of the pre-onset spectra, cut at d_r, which lies midway
     between same-receiver and cross-receiver spectrum distances of enrolled units; units without spectra form one
     group;
  3. within each group, HDBSCAN (minimum cluster size 3, minimum samples 2) forms candidate clusters; two candidates
     merge when the cosine of their centroids reaches theta_w and exceeds the larger cosine to an enrolled
     prototype by delta_w;
  4. across groups, two clusters merge when each is the other's nearest cluster in the other group and their cosine
     passes (theta_x, delta_x) (eq. (14));
  5. an unclustered unit joins its nearest cluster when the cosine reaches theta_o and exceeds its largest prototype
     cosine by delta_o; otherwise it stays unclustered (-1).
Every threshold is calibrated on enrolled data only (calibrate_within, receiver_cut, calibrate_cross).

HDBSCAN runs on rows sorted by unit identifier, so the labels do not depend on the input order. Exact reproduction of
the paper's labels needs hdbscan 0.8.44.
"""
import hashlib
import itertools
import math

import numpy as np

GRID_THETA = [.2, .3, .4, .5, .6, .7, .8, .9]
GRID_DELTA = [-.3, -.2, -.1, -.05, 0.0, .05, .1]
MIN_CLUSTER_SIZE, MIN_SAMPLES = 3, 2


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def unit(m):
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def _near(a, protos):
    """Largest cosine of each row of a to an enrolled prototype; -inf when no prototype remains (then the merge and
    attachment tests reduce to their cosine thresholds)."""
    return (a @ protos.T).max(1) if len(protos) else np.full(len(a), -np.inf)


# ------------------------------------------------------------------ WCCN
def wccn(pool_z, pool_labels, shrink=0.1):
    """Return the WCCN transform h(u) = unit(W z(u)) fitted on labeled enrolled units (eq. (11)).

    W = S^(-1/2) with S = (1 - shrink) S_w / (trace(S_w) / d) + shrink I, where S_w is the within-identity covariance
    of the units, scaled to unit average variance."""
    pool_z = np.asarray(pool_z)
    groups = {}
    for i, l in enumerate(pool_labels):
        groups.setdefault(l, []).append(i)
    d = pool_z.shape[1]
    sw, cnt = np.zeros((d, d)), 0
    for rows in groups.values():
        if len(rows) >= 2:
            x = pool_z[rows] - pool_z[rows].mean(0)
            sw += x.T @ x
            cnt += len(rows)
    sw /= cnt
    s = (1 - shrink) * sw / (np.trace(sw) / d) + shrink * np.eye(d)
    vals, vecs = np.linalg.eigh(s)
    w = vecs @ np.diag(vals ** -.5) @ vecs.T
    return lambda z: unit(np.asarray(z) @ w)


def joint_embedding(z_t, z_a):
    """Joint embedding [z_T; z_A] / sqrt(2) of the two encoders."""
    return np.hstack([z_t, z_a]) / np.sqrt(2)


# ------------------------------------------------------------------ clustering primitives
class Discovery:
    """Within-group clustering, merging, and attachment with fixed HDBSCAN settings."""

    def __init__(self, hdbscan_module=None, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES):
        if hdbscan_module is None:
            import hdbscan as hdbscan_module
        self.pkg, self.mcs, self.ms = hdbscan_module, min_cluster_size, min_samples

    def hdb(self, x, ids):
        import joblib
        order = sorted(range(len(ids)), key=lambda i: str(ids[i]))
        m = self.pkg.HDBSCAN(min_cluster_size=self.mcs, min_samples=self.ms, metric='euclidean', alpha=1.0, p=None,
                             algorithm='best', leaf_size=40, memory=joblib.Memory(location=None, verbose=0),
                             approx_min_span_tree=False, gen_min_span_tree=False, core_dist_n_jobs=1,
                             cluster_selection_method='eom', allow_single_cluster=False, prediction_data=False,
                             branch_detection_data=False, match_reference_implementation=False,
                             cluster_selection_epsilon=0.0, cluster_selection_epsilon_max=math.inf,
                             cluster_selection_persistence=0.0, max_cluster_size=0)
        sel = np.asarray(m.fit_predict(np.ascontiguousarray(np.asarray(x, dtype=np.float64)[order])), dtype=np.int64)
        lab = np.empty(len(ids), dtype=np.int64)
        lab[order] = sel
        return lab

    @staticmethod
    def centroids(x, cand):
        ks = sorted(set(cand[cand >= 0].tolist()))
        return ks, (unit(np.stack([x[cand == k].mean(0) for k in ks])) if ks else np.zeros((0, x.shape[1])))

    @staticmethod
    def prototypes(ref, labels):
        ids = sorted(set(labels))
        return unit(np.stack([ref[[k for k, l in enumerate(labels) if l == c]].mean(0) for c in ids])), ids

    @staticmethod
    def merge(cand, ks, cs, protos, theta, delta):
        """Union of candidate clusters whose centroid cosine reaches theta and exceeds the larger prototype cosine
        by delta; returns relabeled clusters (0..K-1) or a single cluster when everything merges."""
        n = len(cand)
        if len(ks) <= 1:
            return np.zeros(n, dtype=np.int64)
        near = _near(cs, protos)
        parent = list(range(len(ks)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        sims = cs @ cs.T
        for a, b in itertools.combinations(range(len(ks)), 2):
            if sims[a, b] >= theta and sims[a, b] - max(near[a], near[b]) >= delta:
                parent[find(a)] = find(b)
        roots = sorted({find(i) for i in range(len(ks))})
        if len(roots) < 2:
            return np.zeros(n, dtype=np.int64)
        remap = {k: roots.index(find(i)) for i, k in enumerate(ks)}
        return np.asarray([remap[c] if c >= 0 else -1 for c in cand], dtype=np.int64)

    @staticmethod
    def attach(x, labels, protos, theta_o, delta_o):
        labels = labels.copy()
        ks = sorted(set(labels[labels >= 0].tolist()))
        noise = np.flatnonzero(labels < 0)
        if not ks or not len(noise):
            return labels
        cents = unit(np.stack([x[labels == k].mean(0) for k in ks]))
        s = x[noise] @ cents.T
        best = s.argmax(1)
        top = s[np.arange(len(noise)), best]
        ok = (top >= theta_o) & (top - _near(x[noise], protos) >= delta_o)
        labels[noise[ok]] = np.asarray(ks)[best[ok]]
        return labels

    def single_group(self, x, ids, protos, cal):
        """Clustering of one group: HDBSCAN, merge (theta_w, delta_w), attach (theta_o, delta_o)."""
        n = len(ids)
        if n < max(self.mcs, self.ms + 1):
            return np.zeros(n, dtype=np.int64)
        cand = self.hdb(x, ids)
        ks, cs = self.centroids(x, cand)
        return self.attach(x, self.merge(cand, ks, cs, protos, cal['theta'], cal['delta']), protos,
                           cal['tau_a'], cal['delta_u'])


def calibration_ari(t, p):
    """ARI with every unclustered unit as its own cluster (used inside the calibration searches)."""
    from collections import Counter
    p = [x if x >= 0 else -(i + 2) for i, x in enumerate(p)]
    c2 = lambda x: x * (x - 1) / 2
    nij, a_, b_ = Counter(zip(t, p)), Counter(t), Counter(p)
    s = sum(c2(v) for v in nij.values())
    sa, sb = sum(c2(v) for v in a_.values()), sum(c2(v) for v in b_.values())
    e = sa * sb / c2(len(t))
    return (s - e) / ((sa + sb) / 2 - e) if (sa + sb) / 2 != e else 0.0


# ------------------------------------------------------------------ within-group calibration (theta_w, delta_w, theta_o, delta_o)
def percentile_calibration(disc, fit_z, fit_labels, held_z, held_labels, seed=20261029):
    """Percentile calibration on enrolled units: merge thresholds from cluster pairs within one enrolled identity,
    attachment thresholds as 5th percentiles of an enrolled unit's cosine to its own identity centroid and of its
    margin over the other prototypes."""
    fit_ids = sorted(set(fit_labels))
    fp, _ = disc.prototypes(fit_z, fit_labels)
    he, hl = held_z, list(held_labels)
    rng, th, st = np.random.default_rng(seed), [], []
    for ident in sorted(set(hl)):
        rows = [i for i, l in enumerate(hl) if l == ident]
        if len(rows) < 3:
            continue
        other = [k for k, c in enumerate(fit_ids) if c != ident]
        for _ in range(10):
            keep = min(len(rows), max(3, math.ceil(float(rng.uniform(.5, 1.)) * len(rows))))
            pick = [rows[k] for k in sorted(rng.choice(len(rows), keep, replace=False))]
            if keep < max(disc.mcs, disc.ms + 1):
                continue
            ks, cents = disc.centroids(he[pick], disc.hdb(he[pick], [f'h{i:05d}' for i in pick]))
            if len(ks) >= 2:
                sims = cents @ cents.T
                near = (cents @ fp[other].T).max(1)
                for a, b in itertools.combinations(range(len(ks)), 2):
                    th.append(float(sims[a, b]))
                    st.append(float(sims[a, b] - max(near[a], near[b])))
    s1s, s2s = [], []
    for ident in sorted(set(hl)):
        rows = [i for i, l in enumerate(hl) if l == ident]
        if len(rows) < 3:
            continue
        other = [k for k, c in enumerate(fit_ids) if c != ident]
        for h in rows:
            c = he[[r for r in rows if r != h]].mean(0)
            s1 = float(he[h] @ (c / np.linalg.norm(c)))
            s1s.append(s1)
            s2s.append(s1 - float((fp[other] @ he[h]).max()))
    return {'theta': float(np.percentile(th, 5)) if len(th) >= 30 else 1.0,
            'delta': float(np.percentile(st, 5)) if len(st) >= 30 else 0.0,
            'tau_a': float(np.percentile(s1s, 5)), 'delta_u': float(np.percentile(s2s, 5))}


def simulated_episodes(pool_labels, seed=20261107, sizes=(2, 3, 5, 8), per_size=20, min_units=6, cap=30):
    """Simulated discovery episodes from enrolled identities: units of each identity in a seeded hash order (at most
    `cap`), identities drawn without replacement, `per_size` episodes per number of identities."""
    by_id = {}
    for i in sorted(range(len(pool_labels)), key=lambda i: _sha(f'{pool_labels[i]}|{i}')):
        by_id.setdefault(pool_labels[i], []).append(i)
    ids = sorted(c for c, r in by_id.items() if len(r) >= min_units)
    by_id = {c: by_id[c][:cap] for c in ids}
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(ids, r, replace=False).tolist()) for r in sizes for _ in range(per_size)], by_id


def episode_calibration(disc, fit_z, fit_labels, pool_z, pool_labels, pc, episodes, by_id):
    """Pick (theta_w, delta_w) maximizing the mean ARI over the episodes, never below the percentile values."""
    fit_ids = sorted(set(fit_labels))
    fp, _ = disc.prototypes(fit_z, fit_labels)
    acc = np.zeros((len(GRID_THETA), len(GRID_DELTA)))
    for ep in episodes:
        rows = [i for c in ep for i in by_id[c]]
        x, truth = pool_z[rows], [pool_labels[i] for i in rows]
        protos = fp[[k for k, c in enumerate(fit_ids) if c not in ep]]
        cand = disc.hdb(x, [f'e{i:06d}' for i in rows])
        ks, cs = disc.centroids(x, cand)
        for ti, t in enumerate(GRID_THETA):
            for di, d in enumerate(GRID_DELTA):
                lab = disc.attach(x, disc.merge(cand, ks, cs, protos, t, d), protos, pc['tau_a'], pc['delta_u'])
                acc[ti, di] += calibration_ari(truth, lab.tolist())
    acc /= len(episodes)
    best = max(((acc[ti, di], GRID_THETA[ti], GRID_DELTA[di]) for ti in range(len(GRID_THETA))
                for di in range(len(GRID_DELTA))))
    return {**pc, 'theta': max(pc['theta'], best[1]), 'delta': max(pc['delta'], best[2]),
            'theta_e': best[1], 'delta_e': best[2], 'episode_ari': float(best[0])}


def calibrate_within(disc, fit_z, fit_labels, held_z, held_labels):
    """Within-group thresholds from the enrolled fit units and the held-out enrolled units (calibration and held
    units of enrolled identities). Returns theta (theta_w), delta (delta_w), tau_a (theta_o), delta_u (delta_o)."""
    pc = percentile_calibration(disc, fit_z, fit_labels, held_z, held_labels)
    pool_z = np.vstack([fit_z, held_z])
    pool_labels = list(fit_labels) + list(held_labels)
    episodes, by_id = simulated_episodes(pool_labels)
    return episode_calibration(disc, fit_z, fit_labels, pool_z, pool_labels, pc, episodes, by_id)


# ------------------------------------------------------------------ receiver groups and cross-group association
def receiver_cut(spectra, receivers):
    """d_r (eq. (13)): midpoint of the 95th percentile of same-receiver and the 5th percentile of cross-receiver
    Euclidean spectrum distances between enrolled units heard by both receivers."""
    x = np.asarray(spectra)
    dist = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(x), 1)
    same = np.asarray([receivers[i] == receivers[j] for i, j in zip(*iu)])
    if not (same.any() and (~same).any()):
        raise ValueError('the enrolled units must cover two receivers')
    return float((np.percentile(dist[iu][same], 95) + np.percentile(dist[iu][~same], 5)) / 2)


def receiver_groups(spectra, d_r):
    """Average-linkage groups of the pre-onset spectra cut at d_r; units without spectra (NaN) form one group."""
    from scipy.cluster.hierarchy import fcluster, linkage
    spectra = np.asarray(spectra)
    ok = ~np.isnan(spectra).any(1)
    g = np.full(len(spectra), -1, dtype=np.int64)
    idx = np.flatnonzero(ok)
    if len(idx) == 1:
        g[idx] = 0
    elif len(idx) > 1:
        g[idx] = fcluster(linkage(spectra[idx], method='average', metric='euclidean'), t=d_r, criterion='distance') - 1
    g[~ok] = (g.max() + 1) if ok.any() else 0
    return g


def within_groups(disc, x, ids, groups, cal, protos):
    """Group-local HDBSCAN and merge; unique labels across groups (-1 unclustered) and the group of each cluster."""
    lab = np.full(len(ids), -1, dtype=np.int64)
    cgroup, nxt = {}, 0
    for g in sorted(set(groups.tolist())):
        rows = np.flatnonzero(groups == g)
        if len(rows) < max(disc.mcs, disc.ms + 1):
            continue
        xs = x[rows]
        cand = disc.hdb(xs, [ids[i] for i in rows])
        ks, cs = disc.centroids(xs, cand)
        sub = disc.merge(cand, ks, cs, protos, cal['theta'], cal['delta'])
        for c in sorted(set(sub[sub >= 0].tolist())):
            lab[rows[sub == c]] = nxt
            cgroup[nxt] = g
            nxt += 1
    return lab, cgroup


def cross_groups(x, lab, cgroup, protos, theta_x, delta_x):
    """Join clusters that are mutual nearest clusters across two groups and pass (theta_x, delta_x) (eq. (14))."""
    ks = sorted(cgroup)
    if len(set(cgroup.values())) < 2:
        return lab.copy()
    cents = {k: unit(x[lab == k].mean(0)) for k in ks}
    near = {k: float(_near(cents[k][None], protos)[0]) for k in ks}
    parent = {k: k for k in ks}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for g, h in itertools.combinations(sorted(set(cgroup.values())), 2):
        a_ = [k for k in ks if cgroup[k] == g]
        b_ = [k for k in ks if cgroup[k] == h]
        if not a_ or not b_:
            continue
        s = np.asarray([[float(cents[i] @ cents[j]) for j in b_] for i in a_])
        for ia, ka in enumerate(a_):
            jb = int(s[ia].argmax())
            if int(s[:, jb].argmax()) != ia:
                continue
            c = s[ia, jb]
            if c >= theta_x and c - max(near[ka], near[b_[jb]]) >= delta_x:
                parent[find(ka)] = find(b_[jb])
    roots = sorted({find(k) for k in ks})
    remap = {k: roots.index(find(k)) for k in ks}
    return np.asarray([remap[l] if l >= 0 else -1 for l in lab], dtype=np.int64)


def rgd(disc, x, ids, spectra, protos, cal, d_r, theta_x, delta_x, pre=None):
    """RGD readout (Algorithm 1, steps 4-5). x: WCCN-normalized embeddings of the unregistered units; ids: unit
    identifiers; spectra: pre-onset spectra (NaN rows allowed); protos: enrolled prototypes; cal: within-group
    thresholds from calibrate_within. Returns (cluster labels with -1 unclustered, receiver groups)."""
    groups = receiver_groups(spectra, d_r)
    if len(set(groups.tolist())) == 1:
        return disc.single_group(x, ids, protos, cal), groups
    lab, cgroup = pre if pre is not None else within_groups(disc, x, ids, groups, cal, protos)
    merged = cross_groups(x, lab, cgroup, protos, theta_x, delta_x)
    return disc.attach(x, merged, protos, cal['tau_a'], cal['delta_u']), groups


def receiver_episodes(labels, receivers, dual_prefix='M|', seed=20261131, sizes=(2, 3, 5, 8), per_size=20,
                      min_units=6, cap=30):
    """Receiver-diverse episodes for calibrating (theta_x, delta_x): identities whose label starts with dual_prefix
    were recorded by both receivers; their units are taken alternately from each receiver, and every episode holds
    one or two of them (alternating) plus single-receiver identities."""
    by = {}
    for i in sorted(range(len(labels)), key=lambda i: _sha(f'{labels[i]}|{i}')):
        by.setdefault(labels[i], []).append(i)
    dual = sorted(c for c in by if c.startswith(dual_prefix))
    for c in dual:
        dv = [receivers[i] for i in by[c]]
        lists = [[i for i, d in zip(by[c], dv) if d == dd] for dd in sorted(set(dv))]
        take = []
        while len(take) < cap and any(lists):
            for lst in lists:
                if lst and len(take) < cap:
                    take.append(lst.pop(0))
        by[c] = take
    single = sorted(c for c in by if not c.startswith(dual_prefix) and len(by[c]) >= min_units)
    for c in single:
        by[c] = by[c][:cap]
    rng = np.random.default_rng(seed)
    episodes = []
    for r in sizes:
        for j in range(per_size):
            nm = 2 if j % 2 == 0 else 1
            ep = list(rng.choice(dual, nm, replace=False)) + list(rng.choice(single, r - nm, replace=False))
            episodes.append(sorted(str(c) for c in ep))
    return episodes, by


def calibrate_cross(disc, x, spectra, labels, episodes, by, protos_all, proto_ids, cal, d_r):
    """Pick (theta_x, delta_x) maximizing the mean ARI of the RGD readout over receiver-diverse enrolled episodes."""
    acc = np.zeros((len(GRID_THETA), len(GRID_DELTA)))
    for ep in episodes:
        rows = [i for c in ep for i in by[c]]
        xe, se, truth = x[rows], spectra[rows], [labels[i] for i in rows]
        protos = protos_all[[k for k, c in enumerate(proto_ids) if c not in ep]]
        ids = [f'r{i:06d}' for i in rows]
        groups = receiver_groups(se, d_r)
        pre = None if len(set(groups.tolist())) == 1 else within_groups(disc, xe, ids, groups, cal, protos)
        for ti, t in enumerate(GRID_THETA):
            for di, d in enumerate(GRID_DELTA):
                lab, _ = rgd(disc, xe, ids, se, protos, cal, d_r, t, d, pre=pre)
                acc[ti, di] += calibration_ari(truth, lab.tolist())
    acc /= len(episodes)
    best = max(((acc[ti, di], GRID_THETA[ti], GRID_DELTA[di]) for ti in range(len(GRID_THETA))
                for di in range(len(GRID_DELTA))))
    return {'theta_x': best[1], 'delta_x': best[2], 'episode_ari': float(best[0]), 'd_r': d_r}
