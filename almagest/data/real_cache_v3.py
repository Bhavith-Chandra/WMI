"""v3 (replay-exact) real-data cache. One npz per product-segment holding the WHOLE segment's tokens (book state needs
the segment from its snapshot) with per-token hour labels; loaders select hours (split) — tokens of other hours are
context only and are never targets/metrics of the selected split. Checkpoints + hidden updates make the full engine
book available at any token (exact deep seeding). Only COMPLETE hours are ever read.
usage: real_cache_v3.py <raw_root> <out_dir>
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, __file__.rsplit('/almagest/', 1)[0])
from almagest.data import coinbase_exact as X, coinbase_tokenize as T
from almagest.data.real_split import complete_hours, hour_of, PRODUCTS


def build(root, out_dir, hours_allowed):
    os.makedirs(out_dir, exist_ok=True)
    allowed = set(hours_allowed)
    made = []
    for p in PRODUCTS:
        for seg_files in T.segments(root, p):
            files = [f for f in seg_files if os.path.basename(f).split('_')[0] in allowed]
            if not files:
                continue
            seg = seg_files[0].rsplit('_seg', 1)[1].split('.')[0]
            if os.path.basename(seg_files[0]).split('_')[0] not in allowed:
                continue                       # segment's snapshot file is not complete/allowed -> skip whole segment
            out = os.path.join(out_dir, f'{p}_seg{seg}.npz')
            meta_p = out + '.json'
            last = max(os.path.basename(f).split('_')[0] for f in files)
            if os.path.exists(meta_p) and json.load(open(meta_p)).get('last_hour') == last:
                made.append(out); continue
            t0 = time.time()
            st = X.ExactTokenizer(p)
            for f in files:
                for m in T.read_lines(f):
                    st.feed(m)
            st.flush()
            o = st.pack()
            hrs = np.array([hour_of(t) for t in o['times']])
            keep_ok = np.isin(hrs, list(allowed))
            np.savez_compressed(out, ev=o['ev'], lots=o['lots'], qv=o['qv'], feat=o['feat'], mids=o['mids'], cm=o['cm'],
                                times=o['times'], hours=hrs, ok=keep_ok, hid=o['hid'], **X.ck_arrays(o['ck']))
            json.dump(dict(product=p, seg=seg, n=len(o['ev']), stats=o['stats'], last_hour=last,
                           kappa=float(st.real.k) if st.real else None, q0=X.Q0[p], DR=X.DR, KI=X.KI, NF=X.NF,
                           sec=time.time() - t0), open(meta_p, 'w'))
            print(p, seg, len(o['ev']), round(time.time() - t0, 1), 's', flush=True)
            made.append(out)
    return made


if __name__ == '__main__':
    root, out = sys.argv[1], sys.argv[2]
    hrs = complete_hours(root)
    build(root, out, hrs)
    json.dump(dict(hours=hrs, built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
              open(os.path.join(out, 'BUILD.json'), 'w'))
