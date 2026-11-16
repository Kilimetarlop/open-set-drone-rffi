"""Evaluation measures (Section III-E).

Rejection (higher score = more likely enrolled; kn / un = index lists of known and unregistered queries):
  auroc       P(score_known > score_unregistered), ties count 1/2
  oscr        area under the correct-classification rate (known, accepted, identity correct) against the
              false-positive rate (unregistered accepted) over all thresholds (eq. (3))
  fpr95       unregistered accept rate at the threshold that accepts 95% of known queries
  far         false-accept rate: unregistered accept rate at the calibrated threshold
Discovery (unclustered = -1, each becomes its own cluster): ARI, NMI, ACC (Hungarian), B-cubed F, and the
cross-receiver split rate (eq. (4)), same-receiver split rate, and false-merge rate.
"""
from collections import Counter

import numpy as np


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None]) + .5 * (pos[:, None] == neg[None])).mean())


def oscr(s, correct, kn, un):
    """Exact area under the CCR-FPR step curve; tied scores enter at one threshold."""
    s = np.asarray(s, float)
    ck = np.asarray([bool(correct[i]) for i in kn], float)
    sk, su = s[kn], s[un]
    thr = np.unique(np.r_[sk, su])[::-1]
    ccr = np.r_[0.0, [(ck * (sk >= t)).sum() / len(kn) for t in thr]]
    fpr = np.r_[0.0, [(su >= t).mean() for t in thr]]
    return float(np.sum((fpr[1:] - fpr[:-1]) * (ccr[1:] + ccr[:-1]) / 2))


def fpr95(s, kn, un):
    s = np.asarray(s, float)
    sk = np.sort(s[kn])
    return float((s[un] >= sk[int(np.floor(.05 * len(sk)))]).mean())


def rejection_measures(s, tau, identity, truth, kn, un):
    s = np.asarray(s, float)
    correct = {i: identity[i] == truth[i] for i in kn}
    acc = s >= tau
    return {'AUROC': auroc(s[kn], s[un]), 'OSCR': oscr(s, correct, kn, un), 'FPR95': fpr95(s, kn, un),
            'FAR': float(acc[un].mean()), 'known_correct': float(np.mean([bool(acc[i]) and correct[i] for i in kn]))}


def singletons(lab):
    lab = [int(v) for v in lab]
    nxt = max([v for v in lab if v >= 0], default=-1) + 1
    out = []
    for v in lab:
        if v < 0:
            v, nxt = nxt, nxt + 1
        out.append(v)
    return out


def bcubed_f(truth, lab):
    lab = singletons(lab)
    n = len(truth)
    pc, tc, both = Counter(lab), Counter(truth), Counter(zip(truth, lab))
    p = sum(both[(truth[i], lab[i])] / pc[lab[i]] for i in range(n)) / n
    r = sum(both[(truth[i], lab[i])] / tc[truth[i]] for i in range(n)) / n
    return 2 * p * r / (p + r)


def cluster_acc(truth, lab):
    from scipy.optimize import linear_sum_assignment
    lab = singletons(lab)
    ts, ls = sorted(set(truth)), sorted(set(lab))
    m = np.zeros((len(ts), len(ls)))
    for t, l in zip(truth, lab):
        m[ts.index(t), ls.index(l)] += 1
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / len(truth))


def discovery_measures(truth, lab):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    s = singletons(lab)
    return {'ARI': float(adjusted_rand_score(truth, s)), 'NMI': float(normalized_mutual_info_score(truth, s)),
            'ACC': cluster_acc(truth, lab), 'BCubed_F': bcubed_f(truth, lab),
            'K': len({v for v in lab if v >= 0}), 'unclustered': int(sum(v < 0 for v in lab))}


def pair_rates(lab, truth, receiver):
    """Cross-receiver split rate (eq. (4)), same-receiver split rate, and false-merge rate over unit pairs."""
    cs = ct = ss = st = fm = ft = 0
    n = len(lab)
    for i in range(n):
        for j in range(i + 1, n):
            together = lab[i] >= 0 and lab[i] == lab[j]
            if truth[i] == truth[j]:
                if receiver[i] != receiver[j]:
                    ct += 1
                    cs += not together
                else:
                    st += 1
                    ss += not together
            else:
                ft += 1
                fm += together
    return {'cross_split': cs / max(ct, 1), 'same_split': ss / max(st, 1), 'false_merge': fm / max(ft, 1)}
