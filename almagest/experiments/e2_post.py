"""Stage-3 post-training: tune ONE scalar (tag_gain: sensitivity to the own-order channel) so the model's
counterfactual impact law matches the ground-truth World on TRAINING-range days (split 'tune'), then freeze it.
Dev/test are never used here. usage: e2_post.py <base.pt> <out.pt> [gains...]
Objective = |log(Y_m/Y_t)| + |delta_m - delta_t| + |R100_m - R100_t| + 5 * collapse_rate(F arm).
"""
import sys, os, json, subprocess, tempfile
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def objective(r):
    """All protocol targets at once (so no single fix can trade one failure for another unnoticed)."""
    t, m = r['truth'], r['model']
    if not (np.isfinite(m['Y']) and m['Y'] > 0 and np.isfinite(m['delta']) and np.isfinite(m['ep100'])):
        return 1e9
    return (abs(np.log(m['Y'] / t['Y'])) + abs(m['delta'] - t['delta']) + abs(m['ep100'] - t['ep100'])
            + abs(np.log(max(m['free_sd'], 1e-6) / t['free_sd'])) + 0.01 * abs(m['slope100'] - t['slope100'])
            + 5 * m['collapse_rate']['F'])


def main():
    base, out = sys.argv[1], sys.argv[2]
    other = os.environ.get('INTERP_WITH')          # model soup: weights (1-a)*base + a*other, grid over a
    grid = [float(x) for x in sys.argv[3:]] or ([0.25, 0.5, 0.75] if other else [0.5, 1.0, 1.5, 2.0, 3.0])
    ck = torch.load(base, weights_only=False)
    if other:
        ck2 = torch.load(other, weights_only=False)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)))
    log = []
    for g in grid:
        c = dict(ck); c['cfg'] = dict(ck['cfg'])
        if other:
            c['state'] = {k: (1 - g) * v + g * ck2['state'][k] if v.is_floating_point() else v for k, v in ck['state'].items()}
        else:
            c['cfg']['tag_gain'] = g
        p = os.path.join(tmp, f'g{g}.pt'); torch.save(c, p)
        js = os.path.join(tmp, f'g{g}.json')
        env = dict(os.environ, OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, os.path.join(HERE, 'e2_eval.py'), p, js, 'tune', '16', '16'], env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        r = json.load(open(js))
        L = objective(r)
        log.append(dict(param=g, objective=L, Y=r['model']['Y'], Y_truth=r['truth']['Y'], delta=r['model']['delta'],
                        delta_truth=r['truth']['delta'], ep100=r['model']['ep100'], ep100_truth=r['truth']['ep100'],
                        free_sd=r['model']['free_sd'], free_sd_truth=r['truth']['free_sd'],
                        slope100=r['model']['slope100'], slope100_truth=r['truth']['slope100'],
                        collapse=r['model']['collapse_rate']['F']))
        print(json.dumps(log[-1]), flush=True)
    best = min(log, key=lambda x: x['objective'])
    c = dict(ck); c['cfg'] = dict(ck['cfg'])
    if other:
        a = best['param']
        c['state'] = {k: (1 - a) * v + a * ck2['state'][k] if v.is_floating_point() else v for k, v in ck['state'].items()}
        c['post'] = dict(method='weight interpolation (model soup) on TUNE split', with_=other, log=log, chosen=a)
    else:
        c['cfg']['tag_gain'] = best['param']
        c['post'] = dict(method='grid over tag_gain on TUNE split (training range)', log=log, chosen=best['param'])
    torch.save(c, out)
    print('chosen', best['param'], flush=True)


if __name__ == '__main__':
    main()
