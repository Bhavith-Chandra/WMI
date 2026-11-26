"""LOB-Impact-style estimators (2610.27100 protocol, re-implemented) + additions (TOST, placebo).

Conventions (documented, fixed before any model is run):
  * impact I in units of the day's Parkinson sigma (sigma = range / kappa_P, kappa_P = sqrt(4 ln 2))
  * q = cumulative child volume / day market volume V
  * law I = a q^delta ; binned estimator: 18 quantile bins, signed conditional means, OLS (log-log) on positive means
  * Y = a * q_edge^(delta - 1/2)   (amplitude at the data edge, sqrt-law convention)
  * antisymmetrised I = (I_buy - I_sell)/2 on identical seeds
  * day = independent unit for bootstrap
"""
import numpy as np

KAPPA_P = np.sqrt(4 * np.log(2))


def binned_law(q, I, nbins=18):
    q = np.asarray(q).ravel()
    I = np.asarray(I).ravel()
    edges = np.quantile(q, np.linspace(0, 1, nbins + 1))
    idx = np.clip(np.searchsorted(edges, q, side='right') - 1, 0, nbins - 1)
    mq, mi = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        mq.append(q[m].mean())
        mi.append(I[m].mean())
    mq, mi = np.array(mq), np.array(mi)
    pos = (mi > 0) & (mq > 0)
    if pos.sum() < 3:
        return np.nan, np.nan
    A = np.vstack([np.log(mq[pos]), np.ones(pos.sum())]).T
    sl, ic = np.linalg.lstsq(A, np.log(mi[pos]), rcond=None)[0]
    return sl, np.exp(ic)


def identifiable(q, I):
    """Direct L2 profile over delta grid; law identified iff min not on a bound and |Y|>=0.05."""
    q = np.asarray(q).ravel()
    I = np.asarray(I).ravel()
    grid = np.arange(0.05, 1.5001, 0.01)
    best, bd, ba = np.inf, None, None
    for d in grid:
        qd = q ** d
        a = (I * qd).sum() / (qd * qd).sum()
        r = ((I - a * qd) ** 2).sum()
        if r < best:
            best, bd, ba = r, d, a
    Y = ba * q.max() ** (bd - 0.5)
    ok = (abs(Y) >= 0.05) and (bd > grid[0] + 1e-9) and (bd < grid[-1] - 1e-9)
    return ok, bd, Y


def law_from_pairs(Ib, qb):
    """Ib: (n_roll, n_child) antisymmetrised impact in sigma units; qb same shape."""
    d, a = binned_law(qb, Ib)
    if not np.isfinite(d):
        return np.nan, np.nan
    Y = a * qb.max() ** (d - 0.5)
    return d, Y


def day_bootstrap(Ib, qb, day_id, B=400, seed=0):
    rng = np.random.default_rng(seed)
    days = np.unique(day_id)
    groups = [np.where(day_id == d)[0] for d in days]
    ds, Ys = [], []
    for _ in range(B):
        pick = rng.integers(0, len(days), len(days))
        ii = np.concatenate([groups[p] for p in pick])
        d, Y = law_from_pairs(Ib[ii], qb[ii])
        ds.append(d)
        Ys.append(Y)
    return np.array(ds), np.array(Ys)


def sign_acf_gamma(signs, lmin=10, lmax=500):
    s = signs[signs != 0].astype(float)
    if len(s) < 5 * lmax:
        return np.nan
    s = s - s.mean()
    n = len(s)
    f = np.fft.rfft(s, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:lmax + 1] / np.arange(n, n - lmax - 1, -1)
    ac = ac / ac[0]
    lags = np.unique(np.logspace(np.log10(lmin), np.log10(lmax), 20).astype(int))
    c = ac[lags]
    m = c > 0
    if m.sum() < 5:
        return np.nan
    return -np.polyfit(np.log(lags[m]), np.log(c[m]), 1)[0]


def hurst(mid, kmin=4, kmax=13):
    lags = 2 ** np.arange(kmin, kmax + 1)
    v = [np.var(mid[l:] - mid[:-l]) for l in lags]
    return 0.5 * np.polyfit(np.log(lags), np.log(v), 1)[0]


def relaxation(Irel, n_exec):
    """Irel: (n_blocks,) mean antisymmetrised impact; returns end/peak and slope (% peak per exec duration)."""
    peak = Irel[n_exec - 1]
    end = Irel[-10:].mean()
    sil = Irel[n_exec:]
    x = np.arange(len(sil)) / n_exec
    half = len(sil) // 2
    slope = np.polyfit(x[half:], sil[half:] / peak, 1)[0] * 100
    return end / peak, slope


def tost_pass(boot_d, lo=0.4, hi=0.6, level=0.90):
    a = (1 - level) / 2
    ql, qh = np.nanquantile(boot_d, [a, 1 - a])
    return bool(ql >= lo and qh <= hi), (ql, qh)


def ladder(ref, res, ow_ep=None):
    """C1..C5 verdicts (LOB-Impact criteria) + TOST addition.
    ref: dict(gamma, H) of target market; res: measured dict."""
    C1 = (abs(res['gamma'] - ref['gamma']) <= 0.10 and abs(res['H'] - ref['H']) <= 0.02
          and abs(res['drift']) <= 5)
    C2 = bool(res['identified'])
    C3 = res['vb'] >= 3 if res.get('vb') is not None else None
    C4 = (abs(res['delta'] - 0.5) <= 0.10 and 0.5 <= res['Y'] <= 1.0) if C2 else None
    ep_bar = ow_ep if ow_ep is not None else res.get('ep_target', 2 / 3)
    C5 = (res['ep'] <= ep_bar and res['slope'] <= 0 and abs(res['drift']) < 20) if C2 else None
    return dict(C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, TOST=res.get('tost'))


def relax_bootstrap(Irows, day_id, n_exec, B=300, seed=0):
    """Day-clustered bootstrap CI of end/peak from per-rollout relaxation curves Irows (n_roll, n_blocks)."""
    rng = np.random.default_rng(seed)
    days = np.unique(day_id)
    groups = [np.where(day_id == d)[0] for d in days]
    eps = []
    for _ in range(B):
        pick = rng.integers(0, len(days), len(days))
        ii = np.concatenate([groups[p] for p in pick])
        eps.append(relaxation(Irows[ii].mean(0), n_exec)[0])
    return np.nanquantile(eps, [0.025, 0.975])
