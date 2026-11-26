"""Score a v3 B2 result json against the pre-registered rule (EVAL_PROTOCOL 4.1). usage: r3_gate.py <b2.json>"""
import sys, json, math

DIST = ('spread_L1', 'depth_W1', 'band_W1', 'imb_W1', 'lvl_L1', 'dt_L1')
RAT = ('mshare_ratio', 'vol10_ratio', 'vol100_ratio', 'vol1000_ratio')
ACF = ('acf1', 'acf10')


def gate(res, key='model_vs_real'):
    g, f = res[key], res['floor_real_vs_real']
    rows = []
    for k in DIST:
        thr = 1.5 * f[k] + 0.02
        rows.append((k, g[k], f[k], thr, g[k] <= thr))
    for k in RAT:
        lg, lf = abs(math.log(max(g[k], 1e-9))), abs(math.log(max(f[k], 1e-9)))
        thr = 1.5 * lf + math.log(1.25)
        rows.append((k, g[k], f[k], thr, lg <= thr))
    for k in ACF:
        dg, df = abs(g[k][0] - g[k][1]), abs(f[k][0] - f[k][1])
        thr = 1.5 * df + 0.05
        rows.append((k, g[k], f[k], thr, dg <= thr))
    return rows


if __name__ == '__main__':
    res = json.load(open(sys.argv[1]))
    for key in ('model_vs_real', 'oracle_vs_real', 'oracle_sampled_sizes_vs_real'):
        if key not in res:
            continue
        rows = gate(res, key)
        print(f'== {key}: {sum(r[-1] for r in rows)}/{len(rows)} pass')
        for k, gv, fv, thr, ok in rows:
            print(f'  {k:14s} gen {gv}  floor {fv}  thr {thr:.3f}  {"PASS" if ok else "FAIL"}')
    print('oracle log-depth err @10/100/1000/gen', res['oracle_logdepth_err_at_10_100_1000_gen'])
