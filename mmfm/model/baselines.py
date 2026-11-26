"""Count baselines with the same factorised likelihood (bits/event), fit on train only.

B0 zero-intelligence : independent per-field marginals, no conditioning on past or other fields.
B1 Markov            : chain p(act|c) p(side|act,c) p(px|act,side,c) p(size|act,side,px,c) p(dt|act,c),
                       context c = (prev act, prev side, spread bucket). Add-alpha smoothing.
"""
from __future__ import annotations

import numpy as np

V = dict(act=3, side=2)


class B0:
    def fit(self, chunks, vocab, alpha=0.5):
        self.vocab = vocab
        self.p = {}
        for k, f in enumerate(("act", "side", "px", "size", "dt")):
            c = np.full(vocab[f], alpha)
            for ch in chunks:
                c += np.bincount(ch["tokens"][1:, k], minlength=vocab[f])
            self.p[f] = np.log(c / c.sum())
        return self

    def logp(self, ch):
        t = ch["tokens"][1:]
        return np.stack([self.p[f][t[:, k]] for k, f in enumerate(("act", "side", "px", "size", "dt"))], 1)


def _spread_bucket(feats_raw_spread):
    return np.digitize(feats_raw_spread, [-0.5, 0.0, 0.5, 1.0])  # on standardized log-spread


class B1:
    def fit(self, chunks, vocab, alpha=0.5):
        self.vocab, self.alpha = vocab, alpha
        self.c = {}
        for ch in chunks:
            for key, tgt, nv in self._iter(ch, vocab):
                d = self.c.setdefault(key, np.zeros(nv))
                np.add.at(d, tgt, 1.0)
        return self

    @staticmethod
    def _ctx(ch):
        prev = ch["tokens"][:-1]
        sb = _spread_bucket(ch["feats"][:-1, 0])
        return prev[:, 0] * 2 * 5 + prev[:, 1] * 5 + sb  # 30 contexts

    def _iter(self, ch, vocab):
        c = self._ctx(ch)
        t = ch["tokens"][1:]
        a, s, p, z = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
        for cc in np.unique(c):
            m = c == cc
            yield ("act", cc), t[m, 0], vocab["act"]
            for aa in range(3):
                mm = m & (a == aa)
                if mm.any():
                    yield ("side", cc, aa), s[mm], vocab["side"]
                    yield ("dt", cc, aa), t[mm, 4], vocab["dt"]
                    for ss in range(2):
                        m3 = mm & (s == ss)
                        if m3.any():
                            yield ("px", cc, aa, ss), p[m3], vocab["px"]
                            for pb in np.unique(p[m3] // 4):
                                m4 = m3 & (p // 4 == pb)
                                yield ("size", aa, ss, pb), z[m4], vocab["size"]

    def _lp(self, key, tgt, nv):
        d = self.c.get(key)
        d = np.zeros(nv) if d is None else d
        return np.log((d[tgt] + self.alpha) / (d.sum() + self.alpha * nv))

    def logp(self, ch):
        v = self.vocab
        c = self._ctx(ch)
        t = ch["tokens"][1:]
        out = np.zeros((len(t), 5))
        for i in range(len(t)):
            a, s, p, z, dt = t[i]
            cc = c[i]
            out[i] = (self._lp(("act", cc), a, v["act"]), self._lp(("side", cc, a), s, v["side"]),
                      self._lp(("px", cc, a, s), p, v["px"]), self._lp(("size", a, s, p // 4), z, v["size"]),
                      self._lp(("dt", cc, a), dt, v["dt"]))
        return out
