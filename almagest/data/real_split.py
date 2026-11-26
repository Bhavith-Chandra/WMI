"""Real-data (Coinbase) tokenised cache + chronological calendar split (EVAL_PROTOCOL Track B).

* Only COMPLETED UTC hours are used (the live hour is never read into any split).
* One calendar for all products (cross-asset leakage): hours sorted; fit = first 70%, select = next 10%, report = last
  20%; one embargo hour dropped at each boundary. The split is a pure function of the list of complete hours, recorded
  with every output. Report hours are touched once per frozen model.
"""
import os, glob, json, time
import numpy as np
from almagest.data import coinbase_tokenize as T

PRODUCTS = ('BTC-USD', 'ETH-USD', 'SOL-USD')


def complete_hours(root, now=None):
    now = time.time() if now is None else now
    cur = time.strftime('%Y%m%dT%H', time.gmtime(now))
    hrs = set()
    for p in PRODUCTS:
        for f in glob.glob(os.path.join(root, p, '*.jsonl.gz')):
            h = os.path.basename(f).split('_')[0]
            if h < cur:
                hrs.add(h)
    return sorted(hrs)


def calendar_split(hours, fit=0.7, sel=0.1):
    n = len(hours)
    a = int(round(n * fit))
    b = int(round(n * (fit + sel)))
    fit_h = hours[:max(0, a - 1)]          # last fit hour = embargo
    sel_h = hours[a:max(a, b - 1)]         # last select hour = embargo
    rep_h = hours[b:]
    return dict(fit=fit_h, select=sel_h, report=rep_h, embargo=[h for h in hours if h not in fit_h + sel_h + rep_h])


def hour_of(t):
    return time.strftime('%Y%m%dT%H', time.gmtime(t))


def build_cache(root, out_dir, hours_allowed):
    """Tokenise each product/segment (book state needs the whole segment from its snapshot), keep only events whose
    exchange time falls in a COMPLETE hour, save per product-segment npz with per-event hour labels."""
    os.makedirs(out_dir, exist_ok=True)
    allowed = set(hours_allowed)
    made = []
    for p in PRODUCTS:
        for seg_files in T.segments(root, p):
            seg = seg_files[0].rsplit('_seg', 1)[1].split('.')[0]
            out = os.path.join(out_dir, f'{p}_seg{seg}.npz')
            last_h = os.path.basename(seg_files[-1]).split('_')[0]
            meta_p = out + '.json'
            if os.path.exists(meta_p) and json.load(open(meta_p)).get('last_complete') == max(allowed):
                made.append(out)
                continue
            files = [f for f in seg_files if os.path.basename(f).split('_')[0] in allowed or
                     os.path.basename(f).split('_')[0] == last_h]
            o = T.tokenize_segment(files, p)
            hrs = np.array([hour_of(t) for t in o['times']])
            keep = np.isin(hrs, list(allowed))
            np.savez_compressed(out, ev=o['ev'][keep], feat=o['feat'][keep], mids=o['mids'][keep], cm=o['cm'][keep],
                                times=o['times'][keep], hours=hrs[keep])
            json.dump(dict(product=p, seg=seg, n=int(keep.sum()), stats=o['stats'], last_complete=max(allowed)),
                      open(meta_p, 'w'))
            made.append(out)
    return made
