"""Relative-evidence consensus rejection, RECR (Section IV-C).

A query is accepted only when three relative scores, each a similarity minus a reference similarity, all rank high
among the enrolled calibration units, under both encoders:
  r_m(q)      relative nearest-neighbor score: similarity to the nearest gallery unit minus that unit's mean
              similarity to its five nearest gallery units (eq. (6));
  gamma_m(q)  identity margin: lead of the most similar identity over the second (eq. (7));
  c_m(q)      cohort-relative score: top-identity similarity minus the mean similarity to the five most similar
              other queries whose top identity differs (eq. (8)); computed from the unlabeled query pool.
Each score is mapped to its empirical CDF over the calibration set C, the minimum of the three is taken per
encoder, and the two encoders are averaged (eq. (9)). The threshold tau is the 5th percentile of the fused
calibration scores, so 95% of calibration units are accepted (eq. (10)).

All embeddings are unit-norm rows. Invalid queries (NaN embeddings) score SENTINEL and are rejected.
"""
import numpy as np

SENTINEL = -9.0
K_REF = K_COHORT = 5


def _cdf(ref, x):
    return np.searchsorted(np.sort(ref), x, side='right') / len(ref)


def encoder_parts(gallery, gallery_labels, calibration, queries, query_valid):
    """Per-encoder scores. gallery [G, d] with labels; calibration [C, d] (enrolled identities, disjoint units);
    queries [Q, d] with a validity mask. Returns (query parts, calibration parts, per-identity query similarity
    [Q, I], identity list); parts are dicts with keys 'rel', 'margin', 'cohort'."""
    g = gallery @ gallery.T
    np.fill_diagonal(g, -np.inf)
    local = np.sort(g, axis=1)[:, -K_REF:].mean(1)
    ids = sorted(set(gallery_labels))
    members = [np.array([k for k, l in enumerate(gallery_labels) if l == c]) for c in ids]
    q_all = np.nan_to_num(queries)
    qv = np.asarray(query_valid, bool)

    def components(e, v):
        sim = np.nan_to_num(e @ gallery.T, nan=SENTINEL)
        j = sim.argmax(1)
        rel = np.where(v, sim[np.arange(len(e)), j] - local[j], SENTINEL)
        per_id = np.stack([sim[:, m].max(1) for m in members], 1)
        top = np.sort(per_id, 1)
        return rel, np.where(v, top[:, -1] - top[:, -2], SENTINEL), per_id.argmax(1), top[:, -1], per_id

    q_rel, q_margin, q_top, q_s1, q_pid = components(queries, qv)
    c_valid = np.ones(len(calibration), bool)
    c_rel, c_margin, c_top, c_s1, _ = components(calibration, c_valid)

    def cohort(e, top, s1, v, self_rows=None):
        sim = np.nan_to_num(e @ q_all.T, nan=SENTINEL)
        sim[:, ~qv] = SENTINEL
        sim[top[:, None] == q_top[None, :]] = SENTINEL
        if self_rows is not None:
            sim[np.arange(len(e)), self_rows] = SENTINEL
        return np.where(v, s1 - np.sort(sim, 1)[:, -K_COHORT:].mean(1), SENTINEL)

    q_coh = cohort(queries, q_top, q_s1, qv, np.arange(len(queries)))
    c_coh = cohort(calibration, c_top, c_s1, c_valid)
    return ({'rel': q_rel, 'margin': q_margin, 'cohort': q_coh}, {'rel': c_rel, 'margin': c_margin, 'cohort': c_coh},
            q_pid, ids)


def consensus(q_parts, c_parts, keys=('rel', 'margin', 'cohort')):
    """Quantile map of each score over calibration, minimum over the scores (one encoder)."""
    q = np.minimum.reduce([_cdf(c_parts[k], q_parts[k]) for k in keys])
    c = np.minimum.reduce([_cdf(c_parts[k], c_parts[k]) for k in keys])
    return q, c


def recr(gallery, gallery_labels, calibration, queries, query_valid, percentile=5.0, keys=('rel', 'margin', 'cohort')):
    """RECR over several encoders.

    gallery, calibration, queries, query_valid: dicts keyed by encoder name (for the paper, 'T' and 'A') holding the
    arrays of encoder_parts; gallery_labels: one label list shared by the encoders. keys selects the scores
    (the ablation without the cohort-relative score uses ('rel', 'margin')).
    Returns a dict with the consensus score s(q) (SENTINEL for invalid queries), the accept mask, the decided
    identity (y-hat, eq. (10)), and the threshold tau.
    """
    names = sorted(gallery)
    q_fused, c_fused, pid_sum, valid, ids = 0.0, 0.0, 0.0, None, None
    for m in names:
        qp, cp, pid, ids = encoder_parts(gallery[m], gallery_labels, calibration[m], queries[m], query_valid[m])
        q_s, c_s = consensus(qp, cp, keys)
        q_fused = q_fused + np.where(query_valid[m], q_s, SENTINEL)
        c_fused = c_fused + c_s
        pid_sum = pid_sum + pid
        v = np.asarray(query_valid[m], bool)
        valid = v if valid is None else valid & v
    q_fused, c_fused = q_fused / len(names), c_fused / len(names)
    tau = float(np.percentile(c_fused, percentile))
    score = np.where(valid, q_fused, SENTINEL)
    accepted = valid & (q_fused >= tau)
    identity = [ids[k] for k in (pid_sum / len(names)).argmax(1)]
    return {'score': score, 'accepted': accepted, 'identity': identity, 'tau': tau}
