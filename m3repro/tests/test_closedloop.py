"""Closed-loop simulator guarantees (regression tests for the adversarial-review findings)."""
import numpy as np
import pytest
import torch

from m3.closedloop import ClosedLoop
from m3.engine import Book, decode_to_orders, replay


@pytest.fixture(scope="module")
def start(tok, example):
    codes, lob, st = example
    anchor = float(st[0, 4])
    bp, sp = replay(Book.from_snapshot(lob[0].numpy()), *decode_to_orders(tok.decode(codes)[0].numpy(), anchor))
    return codes, lob, st, bp, float(sp.t[-1])


def _sim(tiny, tok, start, B, seed=3, **kw):
    codes, lob, st, bp, t0 = start
    return ClosedLoop(tiny, tok, codes, lob, st, bp, t0, B, block=16, seed=seed, log_every=0, **kw)


def test_baseline_row_independent_of_other_rows_schedule(tiny, tok, start):
    """Critical: a no-trade row must evolve identically whether or not another row trades (review #1)."""
    H = 6.0
    a = _sim(tiny, tok, start, 2)
    a.run(H, [[], []], max_steps=1500)
    b = _sim(tiny, tok, start, 2)
    b.run(H, [[], [(0.5, 1, 3000), (2.0, 1, 3000), (4.0, 0, 3000)]], max_steps=1500)
    assert sum(not x["skipped"] for x in b.injected[1]) >= 2, "treatment row did not trade (vacuous)"
    n = min(len(a.codes[0]), len(b.codes[0]))
    assert n > 700
    assert a.codes[0][:n] == b.codes[0][:n], "baseline row was perturbed by another row's injections"
    assert a.codes[1][:n] != b.codes[1][:n], "treatment had no effect on its own row (vacuous)"
    assert a.mid_path[0][: len(b.mid_path[0])] == b.mid_path[0][: len(a.mid_path[0])]


def test_cache_with_masked_injections_and_roll_matches_full_recompute(tiny, tok, start):
    """Logits from the incremental cache (masked pads, explicit positions, rolls) == fresh forward on the row's
    real context [prefix, BOS, last-roll window + everything since]."""
    s = _sim(tiny, tok, start, 2, reserve=440)  # small roll_at -> several rolls within the test
    assert s.roll_at == 72
    s.run(5.0, [[], [(0.3, 1, 2000), (1.0, 0, 2000), (2.0, 1, 2000)]], max_steps=400)
    assert s.n_rolls >= 2
    for i in range(2):
        real = s.pos[i].item() - (s.m.P + 1)  # number of real codes in the current window
        ctx = torch.tensor([s.codes[i][-real:]])
        ids = torch.cat([torch.full((1, 1), s.m.bos, dtype=torch.long), ctx], 1)
        ref = s.m.llama(inputs_embeds=s.m.embeds(ids, s.lob[:1], s.st[:1])).logits[0, -1, : s.m.K].float()
        assert torch.allclose(ref, s.logits[i], atol=2e-3), f"row {i}: cache state != recompute"


def test_children_execute_strictly_before_horizon_and_after_schedule(tiny, tok, start):
    H = 6.0
    sch = [[(k - 0.5) * H / 4, 1, 1000] for k in range(1, 5)]
    s = _sim(tiny, tok, start, 1)
    s.run(H, [sch], max_steps=3000)
    ex = [x for x in s.injected[0] if not x["skipped"]]
    assert len(ex) == 4, "not all children executed (vacuous / horizon bug)"
    for x in ex:
        assert x["t_sched"] <= x["t_exec"] < H
    # every child's events are in the codes the model conditioned on
    assert all(x["n_events"] >= 1 for x in ex)


def test_ioc_leaves_no_resting_residual(tiny, tok, start):
    s = _sim(tiny, tok, start, 1)
    s.run(3.0, [[(0.5, 1, 400000)]], max_steps=2000)  # far larger than visible depth -> partial fill
    x = [x for x in s.injected[0] if not x["skipped"]][0]
    assert 0 < x["filled"] < 400000 and x["n_events"] == 2  # add + cancel of residual


def test_sampler_row_stream_independent_of_other_rows(tiny, tok, start):
    s1, s2 = _sim(tiny, tok, start, 2, seed=11), _sim(tiny, tok, start, 2, seed=11)
    s2.logits[1] = torch.randn_like(s2.logits[1]) * 5  # perturb row 1's distribution only
    assert torch.equal(s1._sample()[0], s2._sample()[0])
