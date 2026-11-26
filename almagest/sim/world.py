"""Ground-truth order-level market ("World") with known causal structure.

Shared action space (used by World and by neural generators):
  type 0 = limit add, 1 = cancel, 2 = market
  side 0 = buy (bid side for limit/cancel), 1 = sell
  lvl  : limit -> u in [-1, D-1] stored as u+1 in [0, D]; offset from own best (u=-1 improves inside spread)
         cancel -> level index from own best in [0, D-1]
         market -> 0
  size : market size in [1, SMAX]; limit/cancel size 1

World dynamics (causal mechanism, all known):
  * background market-order signs from n_split concurrent hidden splitters with Pareto(alpha) lengths
    (Lillo-Mike-Farmer -> long-memory signs, gamma ~ alpha-1)
  * contrarian/liquidity response to z = (mid - r)/z_scale where r is a multi-timescale (log-spaced EMA)
    memory of mid. This is the resilience/replenishment channel that makes impact concave and transient.
Blind mode: agents act on a VIEW book that never contains injected orders; a REAL book receives
agent events + injected children. Agent event stream in blind mode is bit-identical to the no-injection run.
"""
import numpy as np
from numba import njit, prange

P = 4096
D = 20
SMAX = 4
NEMA = 6
BURN = 400_000  # pre-day burn-in (>= 5x slowest EMA tau 70k): every day starts from a stationary state


@njit(cache=True)
def _best_bid(bid, start):
    i = start
    while i >= 0 and bid[i] == 0:
        i -= 1
    return i


@njit(cache=True)
def _best_ask(ask, start):
    i = start
    n = ask.shape[0]
    while i < n and ask[i] == 0:
        i += 1
    return i


@njit(cache=True)
def init_book(bid, ask, p0, depth, vol):
    bid[:] = 0
    ask[:] = 0
    for k in range(depth):
        bid[p0 - 1 - k] = vol
        ask[p0 + 1 + k] = vol


@njit(cache=True)
def apply_action(bid, ask, bb, ba, typ, side, lvl, size):
    """Apply action expressed relative to `this` book. Returns (bb, ba, signed traded volume)."""
    traded = 0
    if typ == 0:
        u = lvl - 1
        if side == 0:
            p = bb - u
            if u == -1 and bb + 1 >= ba:
                p = bb
            if p >= ba:
                p = ba - 1
            if p < 1:
                p = 1
            bid[p] += size
            if p > bb:
                bb = p
        else:
            p = ba + u
            if u == -1 and ba - 1 <= bb:
                p = ba
            if p <= bb:
                p = bb + 1
            if p > P - 2:
                p = P - 2
            ask[p] += size
            if p < ba:
                ba = p
    elif typ == 1:
        if side == 0:
            p = bb - lvl
            if p >= 0 and bid[p] > 0:
                bid[p] -= min(size, bid[p])
                if p == bb and bid[p] == 0:
                    bb = _best_bid(bid, bb)
        else:
            p = ba + lvl
            if p < P and ask[p] > 0:
                ask[p] -= min(size, ask[p])
                if p == ba and ask[p] == 0:
                    ba = _best_ask(ask, ba)
    else:
        rem = size
        if side == 0:
            while rem > 0 and ba < P:
                take = min(rem, ask[ba])
                ask[ba] -= take
                rem -= take
                traded += take
                if ask[ba] == 0:
                    ba = _best_ask(ask, ba)
        else:
            while rem > 0 and bb >= 0:
                take = min(rem, bid[bb])
                bid[bb] -= take
                rem -= take
                traded -= take
                if bid[bb] == 0:
                    bb = _best_bid(bid, bb)
    return bb, ba, traded


@njit(cache=True)
def apply_abs(bid, ask, bb, ba, typ, side, price, size):
    """Apply an agent event at an ABSOLUTE price (used to mirror view events onto the real book)."""
    traded = 0
    if typ == 0:
        if side == 0:
            p = price
            if p >= ba:
                p = ba - 1
            bid[p] += size
            if p > bb:
                bb = p
        else:
            p = price
            if p <= bb:
                p = bb + 1
            ask[p] += size
            if p < ba:
                ba = p
    elif typ == 1:
        if side == 0:
            if bid[price] > 0:
                bid[price] -= min(size, bid[price])
                if price == bb and bid[price] == 0:
                    bb = _best_bid(bid, bb)
        else:
            if ask[price] > 0:
                ask[price] -= min(size, ask[price])
                if price == ba and ask[price] == 0:
                    ba = _best_ask(ask, ba)
    else:
        bb, ba, traded = apply_action(bid, ask, bb, ba, 2, side, 0, size)
    return bb, ba, traded


@njit(cache=True)
def abs_price(bb, ba, typ, side, lvl):
    if typ == 0:
        u = lvl - 1
        if side == 0:
            p = bb - u
            if u == -1 and bb + 1 >= ba:
                p = bb
            if p >= ba:
                p = ba - 1
            return p
        p = ba + u
        if u == -1 and ba - 1 <= bb:
            p = ba
        if p <= bb:
            p = bb + 1
        return p
    if typ == 1:
        return bb - lvl if side == 0 else ba + lvl
    return -1


# params vector layout
# 0 w_mkt, 1 w_lim, 2 nu_cancel(per lot), 3 rho_limit_geom, 4 kappa_lim, 5 kappa_mkt, 6 rho_split(prob splitter sign),
# 7 alpha_pareto, 8 n_split, 9 z_scale, 10 kappa_cancel, 11..16 EMA weights (NEMA), 17..22 EMA taus,
# 23 sigma_F (fundamental RW step, ticks/event), 24 kappa_F (informed sign strength), 25 p_inf (informed share of mkt)
# 26 lambda_K (ticks per lot of remembered signed flow: market learns from flow), 27 tau_K (flow memory, events)
# 28 p_anchor (share of limit orders placed uniformly within K_lat ticks of the slow reference r: latent liquidity,
#    LLOB-style, density grows with distance from current price), 29 K_lat (ticks)
NPAR = 30
IF = NEMA       # ema[IF]  = fundamental F
IM = NEMA + 1   # ema[IM]  = signed-flow memory M (what the market has inferred from trades)


def default_params():
    p = np.zeros(NPAR)
    p[0] = 0.10   # market weight
    p[1] = 0.50   # limit weight
    p[2] = 0.004  # cancel per resting lot within D
    p[3] = 0.70   # geometric decay of limit placement depth
    p[4] = 0.20   # limit side skew strength
    p[5] = 1.00   # contrarian market strength
    p[6] = 0.70   # prob market sign comes from splitters
    p[7] = 1.5    # Pareto alpha of splitter lengths
    p[8] = 6      # concurrent splitters
    p[9] = 4.0    # z scale (ticks)
    p[10] = 0.0   # cancel skew
    taus = np.array([30., 150., 700., 3000., 15000., 70000.])
    w = taus ** -0.0
    p[11:17] = w / w.sum()
    p[17:23] = taus
    p[23] = 0.0
    p[24] = 0.5
    p[25] = 0.0
    p[26] = 0.0
    p[27] = 50000.0
    p[28] = 0.0
    p[29] = 10.0
    return p


@njit(cache=True)
def _sample_level_cancel(vol, best, sgn, total_w):
    # pick resting lot uniformly within D levels of own best
    r = np.random.random() * total_w
    acc = 0.0
    for k in range(D):
        p = best - k if sgn == 0 else best + k
        if p < 0 or p >= P:
            break
        acc += vol[p]
        if acc >= r:
            return k
    return 0


@njit(cache=True)
def world_policy(bid, ask, bb, ba, par, ema, split_sign, split_rem):
    """Sample one agent action from World given book + memory. Mutates splitter state."""
    ema[IF] += par[23] * np.random.standard_normal()
    ema[IM] *= 1.0 - 1.0 / par[27]
    mid = 0.5 * (bb + ba)
    r = 0.0
    for i in range(NEMA):
        r += par[11 + i] * ema[i]
    z = (mid - r) / par[9]
    nb = 0
    na = 0
    for k in range(D):
        if bb - k >= 0:
            nb += bid[bb - k]
        if ba + k < P:
            na += ask[ba + k]
    wm = par[0]
    wl = par[1]
    wc = par[2] * (nb + na)
    u = np.random.random() * (wm + wl + wc)
    if u < wm:
        typ = 2
        ui = np.random.random()
        if ui < par[25]:
            pb = 1.0 / (1.0 + np.exp(-par[24] * (ema[IF] + par[26] * ema[IM] - mid)))
            side = 0 if np.random.random() < pb else 1
        elif np.random.random() < par[6]:
            j = np.random.randint(int(par[8]))
            if split_rem[j] <= 0:
                split_sign[j] = np.random.randint(2)
                # Pareto(alpha) length, min 1
                split_rem[j] = int(np.random.random() ** (-1.0 / par[7]))
            side = split_sign[j]
            split_rem[j] -= 1
        else:
            pbuy = 1.0 / (1.0 + np.exp(par[5] * z))
            side = 0 if np.random.random() < pbuy else 1
        s = np.random.random()
        size = 1 if s < 0.4 else (2 if s < 0.7 else (3 if s < 0.88 else 4))
        ema[IM] += size if side == 0 else -size
        return typ, side, 0, size
    elif u < wm + wl:
        typ = 0
        pbuy = 0.5 - 0.5 * np.tanh(par[4] * z)
        side = 0 if np.random.random() < pbuy else 1
        spread = ba - bb
        if np.random.random() < par[28]:
            k = np.random.randint(int(par[29]) + 1)
            if side == 0:
                pr = int(np.floor(r)) - k
                if pr >= ba:
                    pr = ba - 1
                u = bb - pr
            else:
                pr = int(np.ceil(r)) + k
                if pr <= bb:
                    pr = bb + 1
                u = pr - ba
            if u < -1:
                u = -1
            if u > D - 1:
                u = D - 1
            lvl = u + 1
        elif spread > 1 and np.random.random() < 0.3:
            lvl = 0
        else:
            k = 0
            while k < D - 1 and np.random.random() < par[3]:
                k += 1
            lvl = k + 1
        return typ, side, lvl, 1
    else:
        typ = 1
        wb = nb * (1.0 + par[10] * np.tanh(z))
        wa = na * (1.0 - par[10] * np.tanh(z))
        if wb + wa <= 0:
            return 0, 0, 1, 1
        side = 0 if np.random.random() * (wb + wa) < wb else 1
        if side == 0:
            lvl = _sample_level_cancel(bid, bb, 0, nb)
        else:
            lvl = _sample_level_cancel(ask, ba, 1, na)
        return typ, side, lvl, 1


@njit(cache=True)
def _update_ema(ema, par, mid):
    for i in range(NEMA):
        a = 1.0 / par[17 + i]
        ema[i] += a * (mid - ema[i])


@njit(cache=True)
def _guard(bid, ask, bb, ba):
    # keep both sides alive (hard floor liquidity, part of World's physics)
    if bb < 0 or bb < ba - D * 3:
        bb = ba - 1 if ba < P else P // 2
        bid[bb] += 1
    if ba >= P or ba > bb + D * 3:
        ba = bb + 1
        ask[ba] += 1
    return bb, ba


@njit(cache=True)
def warm_state(par, seed, n_warm, bid, ask, ema, split_sign, split_rem):
    np.random.seed(seed)
    p0 = P // 2
    init_book(bid, ask, p0, D, 3)
    bb = p0 - 1
    ba = p0 + 1
    for i in range(NEMA + 1):
        ema[i] = p0
    ema[IM] = 0.0
    split_sign[:] = 0
    split_rem[:] = 0
    for t in range(n_warm + BURN):
        typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, split_sign, split_rem)
        bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
        bb, ba = _guard(bid, ask, bb, ba)
        _update_ema(ema, par, 0.5 * (bb + ba))
    return bb, ba


@njit(cache=True)
def run_branch(par, day_seed, n_warm, roll_seed, n_child, gap, n_silent_blocks, child_side, child_size,
               mode, out_mid):
    """One rollout. mode 0 = no injection, 1 = visible, 2 = blind.
    out_mid[k] = REAL-book mid at end of block k (block = child + gap agent events),
    blocks 0..n_child-1 execution, then n_silent_blocks silent blocks of `gap` events.
    Returns (start_mid, hash of agent event stream, agent market volume)."""
    bid = np.zeros(P, np.int64)
    ask = np.zeros(P, np.int64)
    ema = np.zeros(NEMA + 2)
    ss = np.zeros(int(par[8]) + 1, np.int64)
    sr = np.zeros(int(par[8]) + 1, np.int64)
    bb, ba = warm_state(par, day_seed, n_warm, bid, ask, ema, ss, sr)
    np.random.seed(roll_seed)
    vb = bid.copy()
    va = ask.copy()
    vbb = bb
    vba = ba
    mid0 = 0.5 * (bb + ba)
    h = np.uint64(1469598103934665603)
    mvol = 0
    nb = n_child + n_silent_blocks
    for blk in range(nb):
        if blk < n_child and mode != 0:
            bb, ba, _ = apply_action(bid, ask, bb, ba, 2, child_side, 0, child_size)
            bb, ba = _guard(bid, ask, bb, ba)
            if mode == 1:
                # visible: the market observes the child trade (learns from flow); blind: it never does
                ema[IM] += child_size if child_side == 0 else -child_size
        for g in range(gap):
            if mode == 2:
                typ, side, lvl, size = world_policy(vb, va, vbb, vba, par, ema, ss, sr)
                pr = abs_price(vbb, vba, typ, side, lvl)
                vbb, vba, _ = apply_action(vb, va, vbb, vba, typ, side, lvl, size)
                vbb, vba = _guard(vb, va, vbb, vba)
                if typ == 2:
                    bb, ba, _ = apply_action(bid, ask, bb, ba, 2, side, 0, size)
                else:
                    if pr >= 0 and pr < P:
                        bb, ba, _ = apply_abs(bid, ask, bb, ba, typ, side, pr, size)
                bb, ba = _guard(bid, ask, bb, ba)
                _update_ema(ema, par, 0.5 * (vbb + vba))
            else:
                typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
                pr = abs_price(bb, ba, typ, side, lvl)
                bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
                bb, ba = _guard(bid, ask, bb, ba)
                _update_ema(ema, par, 0.5 * (bb + ba))
            if typ == 2:
                mvol += size
            code = np.uint64(typ * 1000003 + side * 10007 + (pr + 7) * 31 + size)
            h = (h ^ code) * np.uint64(1099511628211)
        out_mid[blk] = 0.5 * (bb + ba)
    return mid0, h, mvol


@njit(cache=True, parallel=True)
def run_batch(par, day_seeds, n_warms, roll_seeds, n_child, gap, n_silent, side, child_size, mode):
    n = day_seeds.shape[0]
    out = np.zeros((n, n_child + n_silent))
    mid0 = np.zeros(n)
    hs = np.zeros(n, np.uint64)
    mv = np.zeros(n, np.int64)
    for i in prange(n):
        m0, hh, v = run_branch(par, day_seeds[i], n_warms[i], roll_seeds[i], n_child, gap, n_silent,
                               side, child_size, mode, out[i])
        mid0[i] = m0
        hs[i] = hh
        mv[i] = v
    return out, mid0, hs, mv


@njit(cache=True)
def free_run(par, seed, n, out_mid, out_sign, out_vol):
    """Free generation from seed: per-event mid and trade sign (0 if no trade)."""
    bid = np.zeros(P, np.int64)
    ask = np.zeros(P, np.int64)
    ema = np.zeros(NEMA + 2)
    ss = np.zeros(int(par[8]) + 1, np.int64)
    sr = np.zeros(int(par[8]) + 1, np.int64)
    bb, ba = warm_state(par, seed, 20000, bid, ask, ema, ss, sr)
    for t in range(n):
        typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
        bb, ba, tr = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
        bb, ba = _guard(bid, ask, bb, ba)
        _update_ema(ema, par, 0.5 * (bb + ba))
        out_mid[t] = 0.5 * (bb + ba)
        out_sign[t] = 0 if typ != 2 else (1 if side == 0 else -1)
        out_vol[t] = size if typ == 2 else 0


NFEAT = 2 * 10 + 1


@njit(cache=True)
def book_feat(bid, ask, bb, ba, out):
    for k in range(10):
        out[k] = bid[bb - k] if bb - k >= 0 else 0
        out[10 + k] = ask[ba + k] if ba + k < P else 0
    out[20] = ba - bb


@njit(cache=True)
def gen_stream(par, seed, n, ev, feat, mids, cm):
    """Observational training stream: ev[t] = (typ, side, lvl, size) sampled given book BEFORE event t,
    feat[t] = book features before event t, mids[t] = mid before event t."""
    bid = np.zeros(P, np.int64)
    ask = np.zeros(P, np.int64)
    ema = np.zeros(NEMA + 2)
    ss = np.zeros(int(par[8]) + 1, np.int64)
    sr = np.zeros(int(par[8]) + 1, np.int64)
    bb, ba = warm_state(par, seed, 20000, bid, ask, ema, ss, sr)
    for t in range(n):
        book_feat(bid, ask, bb, ba, feat[t])
        mids[t] = 0.5 * (bb + ba)
        for k in range(D):
            cm[t, 0, k] = 1 if (bb - k >= 0 and bid[bb - k] > 0) else 0
            cm[t, 1, k] = 1 if (ba + k < P and ask[ba + k] > 0) else 0
        typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
        ev[t, 0] = typ
        ev[t, 1] = side
        ev[t, 2] = lvl
        ev[t, 3] = size
        bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
        bb, ba = _guard(bid, ask, bb, ba)
        _update_ema(ema, par, 0.5 * (bb + ba))


# ---------------------------------------------------------------------------------------------------------------
# Window snapshots: run each day ONCE, store full state at each window start. Rollouts start from the snapshot and
# reseed with their rollout seed -> bit-identical to run_branch (which warms from the day seed then reseeds), ~50x faster.
NSS = 16  # max splitters supported in snapshots


@njit(cache=True, parallel=True)
def day_snapshots(par, day_seeds, warms, sb, sa, sbb, sba, sema, sss, ssr):
    """warms: sorted 1-D array of warm counts (same for every day). Outputs indexed [day, window, ...]."""
    nd = day_seeds.shape[0]
    nwin = warms.shape[0]
    ns = int(par[8]) + 1
    for d in prange(nd):
        bid = np.zeros(P, np.int64)
        ask = np.zeros(P, np.int64)
        ema = np.zeros(NEMA + 2)
        ss = np.zeros(ns, np.int64)
        sr = np.zeros(ns, np.int64)
        bb, ba = warm_state(par, day_seeds[d], warms[0], bid, ask, ema, ss, sr)
        done = warms[0]
        for w in range(nwin):
            for t in range(warms[w] - done):
                typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
                bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
                bb, ba = _guard(bid, ask, bb, ba)
                _update_ema(ema, par, 0.5 * (bb + ba))
            done = warms[w]
            sb[d, w, :] = bid
            sa[d, w, :] = ask
            sbb[d, w] = bb
            sba[d, w] = ba
            sema[d, w, :] = ema
            sss[d, w, :ns] = ss
            ssr[d, w, :ns] = sr


@njit(cache=True)
def run_from(par, bid, ask, bb, ba, ema, ss, sr, roll_seed, n_child, gap, n_silent_blocks, child_side, child_size,
             mode, out_mid):
    """Same dynamics as run_branch after its warm-up; state arrays are consumed (pass copies)."""
    np.random.seed(roll_seed)
    vb = bid.copy()
    va = ask.copy()
    vbb = bb
    vba = ba
    mid0 = 0.5 * (bb + ba)
    h = np.uint64(1469598103934665603)
    mvol = 0
    nb = n_child + n_silent_blocks
    for blk in range(nb):
        if blk < n_child and mode != 0:
            bb, ba, _ = apply_action(bid, ask, bb, ba, 2, child_side, 0, child_size)
            bb, ba = _guard(bid, ask, bb, ba)
            if mode == 1:
                ema[IM] += child_size if child_side == 0 else -child_size
        for g in range(gap):
            if mode == 2:
                typ, side, lvl, size = world_policy(vb, va, vbb, vba, par, ema, ss, sr)
                pr = abs_price(vbb, vba, typ, side, lvl)
                vbb, vba, _ = apply_action(vb, va, vbb, vba, typ, side, lvl, size)
                vbb, vba = _guard(vb, va, vbb, vba)
                if typ == 2:
                    bb, ba, _ = apply_action(bid, ask, bb, ba, 2, side, 0, size)
                else:
                    if pr >= 0 and pr < P:
                        bb, ba, _ = apply_abs(bid, ask, bb, ba, typ, side, pr, size)
                bb, ba = _guard(bid, ask, bb, ba)
                _update_ema(ema, par, 0.5 * (vbb + vba))
            else:
                typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
                pr = abs_price(bb, ba, typ, side, lvl)
                bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
                bb, ba = _guard(bid, ask, bb, ba)
                _update_ema(ema, par, 0.5 * (bb + ba))
            if typ == 2:
                mvol += size
            code = np.uint64(typ * 1000003 + side * 10007 + (pr + 7) * 31 + size)
            h = (h ^ code) * np.uint64(1099511628211)
        out_mid[blk] = 0.5 * (bb + ba)
    return mid0, h, mvol


@njit(cache=True, parallel=True)
def run_batch_snap(par, sb, sa, sbb, sba, sema, sss, ssr, row_d, row_w, roll_seeds, n_child, gap, n_silent, side,
                   child_size, mode):
    n = row_d.shape[0]
    out = np.zeros((n, n_child + n_silent))
    mid0 = np.zeros(n)
    hs = np.zeros(n, np.uint64)
    mv = np.zeros(n, np.int64)
    ns = int(par[8]) + 1
    for i in prange(n):
        d = row_d[i]
        w = row_w[i]
        m0, hh, v = run_from(par, sb[d, w].copy(), sa[d, w].copy(), sbb[d, w], sba[d, w], sema[d, w].copy(),
                             sss[d, w, :ns].copy(), ssr[d, w, :ns].copy(), roll_seeds[i], n_child, gap, n_silent,
                             side, child_size, mode, out[i])
        mid0[i] = m0
        hs[i] = hh
        mv[i] = v
    return out, mid0, hs, mv


class Snapshots:
    """Holds per-(day, window) World states for a set of days and sorted warm counts."""
    def __init__(self, par, day_seeds, warms):
        import numpy as _np
        self.day_seeds = _np.asarray(day_seeds, _np.int64)
        self.warms = _np.asarray(sorted(set(int(x) for x in warms)), _np.int64)
        nd, nw = len(self.day_seeds), len(self.warms)
        self.sb = _np.zeros((nd, nw, P), _np.int64)
        self.sa = _np.zeros((nd, nw, P), _np.int64)
        self.sbb = _np.zeros((nd, nw), _np.int64)
        self.sba = _np.zeros((nd, nw), _np.int64)
        self.sema = _np.zeros((nd, nw, NEMA + 2))
        self.sss = _np.zeros((nd, nw, NSS), _np.int64)
        self.ssr = _np.zeros((nd, nw, NSS), _np.int64)
        assert int(par[8]) + 1 <= NSS
        day_snapshots(par, self.day_seeds, self.warms, self.sb, self.sa, self.sbb, self.sba, self.sema, self.sss,
                      self.ssr)
        self.par = par
        self._di = {int(s): i for i, s in enumerate(self.day_seeds)}
        self._wi = {int(w): i for i, w in enumerate(self.warms)}

    def index(self, day_seeds, warms):
        import numpy as _np
        return (_np.array([self._di[int(s)] for s in day_seeds], _np.int64),
                _np.array([self._wi[int(w)] for w in warms], _np.int64))

    def run(self, day_seeds, warms, roll_seeds, n_child, gap, n_silent, side, child, mode):
        rd, rw = self.index(day_seeds, warms)
        return run_batch_snap(self.par, self.sb, self.sa, self.sbb, self.sba, self.sema, self.sss, self.ssr, rd, rw,
                              roll_seeds, n_child, gap, n_silent, side, child, mode)


@njit(cache=True)
def gen_stream_inj(par, seed, n, inj, inj_side, inj_size, ev, feat, mids, cm, src):
    """Interventional training stream: at steps with inj[t]=1 an EXOGENOUS child market order (no information,
    seen by the market exactly as in the visible evaluation arm) is executed instead of an agent event and recorded
    with src[t]=1. Otherwise identical to gen_stream (src=0)."""
    bid = np.zeros(P, np.int64)
    ask = np.zeros(P, np.int64)
    ema = np.zeros(NEMA + 2)
    ss = np.zeros(int(par[8]) + 1, np.int64)
    sr = np.zeros(int(par[8]) + 1, np.int64)
    bb, ba = warm_state(par, seed, 20000, bid, ask, ema, ss, sr)
    for t in range(n):
        book_feat(bid, ask, bb, ba, feat[t])
        mids[t] = 0.5 * (bb + ba)
        for k in range(D):
            cm[t, 0, k] = 1 if (bb - k >= 0 and bid[bb - k] > 0) else 0
            cm[t, 1, k] = 1 if (ba + k < P and ask[ba + k] > 0) else 0
        if inj[t] == 1:
            typ, side, lvl, size = 2, inj_side[t], 0, inj_size[t]
            ema[IM] += size if side == 0 else -size
            src[t] = 1
        else:
            typ, side, lvl, size = world_policy(bid, ask, bb, ba, par, ema, ss, sr)
            src[t] = 0
        ev[t, 0] = typ
        ev[t, 1] = side
        ev[t, 2] = lvl
        ev[t, 3] = size
        bb, ba, _ = apply_action(bid, ask, bb, ba, typ, side, lvl, size)
        bb, ba = _guard(bid, ask, bb, ba)
        _update_ema(ema, par, 0.5 * (bb + ba))


@njit(cache=True, parallel=True)
def oracle_labels(par, bid, ask, bb, ba, ema, ss, sr, seeds, out):
    """DAgger oracle: for each row, sample the World's action at the CURRENT (possibly model-visited) state.
    The World's hidden state (ema incl. F, flow memory IM, splitters) is advanced as the World would, except that the
    flow memory is NOT credited with the oracle's own hypothetical trade (the executed trade is credited separately by
    `credit_trade`). out[i] = (typ, side, lvl, size)."""
    n = bid.shape[0]
    for i in prange(n):
        np.random.seed(seeds[i])
        im0 = ema[i, IM] * (1.0 - 1.0 / par[27])
        typ, side, lvl, size = world_policy(bid[i], ask[i], bb[i], ba[i], par, ema[i], ss[i], sr[i])
        ema[i, IM] = im0
        out[i, 0] = typ
        out[i, 1] = side
        out[i, 2] = lvl
        out[i, 3] = size


@njit(cache=True, parallel=True)
def credit_trade(par, ema, typ, side, size, bb, ba):
    """After the executed (model's) event: credit its trade to the flow memory and update the mid EMAs."""
    for i in prange(ema.shape[0]):
        if typ[i] == 2:
            ema[i, IM] += size[i] if side[i] == 0 else -size[i]
        _update_ema(ema[i], par, 0.5 * (bb[i] + ba[i]))
