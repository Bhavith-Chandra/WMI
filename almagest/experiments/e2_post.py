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
    t, m = r['truth'], r['model']
    if not (np.isfinite(m['Y']) and m['Y'] > 0 and np.isfinite(m['delta']) and np.isfinite(m['ep100'])):
        return 1e9
    return (abs(np.log(m['Y'] / t['Y'])) + abs(m['delta'] - t['delta']) + abs(m['ep100'] - t['ep100'])
            + 5 * m['collapse_rate']['F'])


def main():
    base, out = sys.argv[1], sys.argv[2]
    gains = [float(x) for x in sys.argv[3:]] or [0.5, 1.0, 1.5, 2.0, 3.0]
    ck = torch.load(base, weights_only=False)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)))
    log = []
    for g in gains:
        c = dict(ck); c['cfg'] = dict(ck['cfg']); c['cfg']['tag_gain'] = g
        p = os.path.join(tmp, f'g{g}.pt'); torch.save(c, p)
        js = os.path.join(tmp, f'g{g}.json')
        env = dict(os.environ, OMP_NUM_THREADS='1')
        subprocess.run([sys.executable, os.path.join(HERE, 'e2_eval.py'), p, js, 'tune', '16', '16'], env=env,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        r = json.load(open(js))
        L = objective(r)
        log.append(dict(gain=g, objective=L, Y=r['model']['Y'], Y_truth=r['truth']['Y'], delta=r['model']['delta'],
                        delta_truth=r['truth']['delta'], ep100=r['model']['ep100'], ep100_truth=r['truth']['ep100'],
                        collapse=r['model']['collapse_rate']['F']))
        print(json.dumps(log[-1]), flush=True)
    best = min(log, key=lambda x: x['objective'])
    c = dict(ck); c['cfg'] = dict(ck['cfg']); c['cfg']['tag_gain'] = best['gain']
    c['post'] = dict(method='grid over tag_gain on TUNE split (training range)', log=log, chosen=best['gain'])
    torch.save(c, out)
    print('chosen tag_gain', best['gain'], flush=True)


if __name__ == '__main__':
    main()
