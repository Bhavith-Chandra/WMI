"""M3 reproduction / audit experiments on the released checkpoints + released held-out example.

Usage: python -I scripts/run_audit.py <exp> [--rollouts S] [--new N]
  exp in {scaling, nll, tokenizer, rollout}
All outputs -> results/<exp>*.json (+ .npz for raw rollouts). Every number written is computed here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from m3.model import M3, VQTokenizer, n_params  # noqa: E402
from m3.engine import Book, decode_to_orders, replay  # noqa: E402
from m3.provenance import stamp  # noqa: E402

REL = Path("/home/user/ext/m3-market-microstructure")
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SIZES = ["tiny", "small", "base"]
torch.set_num_threads(4)


def load_example():
    codes = torch.from_numpy(np.load(REL / "examples/prompt_ids.npy").astype(np.int64))
    c = np.load(REL / "examples/conditioning.npz")
    return codes, torch.tensor(c["lob"]).float(), torch.tensor(c["initial_state_float"]).float()


USED = {"scaling": ["tiny", "small", "base"], "nll": ["tiny", "small", "base", "example_prompt_ids", "example_conditioning"],
        "tokenizer": ["tokenizer", "example_prompt_ids", "example_conditioning"],
        "rollout": ["tokenizer", "tiny", "small", "base", "example_prompt_ids", "example_conditioning"],
        "decode_ablation": ["tokenizer", "example_prompt_ids", "example_conditioning"]}


def dump(name, obj):
    if not isinstance(obj, dict):
        obj = {"rows": obj}
    obj["_provenance"] = stamp(USED[name])
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, default=float))
    print(f"-> {p}")


# ------------------------------------------------------------------ scaling law
def exp_scaling(_):
    E, A, a, B, b = 5.3697, 1.5987, 0.3795, 1.0027, 0.3117  # paper Eq. (2); N in M params, D in B tokens
    rows = []
    for s in SIZES:
        m = M3.from_checkpoint(REL / s / "best.pt")
        ta, met = m.meta["train_args"], m.meta["metrics"]
        N = n_params(m) / 1e6
        tok_per_step = ta["batch_size"] * ta["grad_accum_steps"] * ta["sample_seq_len"]
        D = m.meta["step"] * tok_per_step / 1e9
        pred = E + A / N**a + B / D**b
        rows.append(dict(size=s, N_M=N, step=m.meta["step"], max_steps=ta["max_steps"], tokens_per_step=tok_per_step,
                         D_B=D, ckpt_val_loss=met["val_loss"], eq2_pred=pred,
                         rel_err_pct=100 * (pred - met["val_loss"]) / met["val_loss"],
                         val_split=ta["val_split"], train_split=ta["train_split"], val_tokens=met["val_n_tokens"],
                         val_batches=met["val_batches"]))
        print(rows[-1])
    dump("scaling", rows)


# ------------------------------------------------------------------ NLL + LOB counterfactuals
def _perturb(lob, st, kind):
    lob, st = lob.clone(), st.clone()
    if kind == "true":
        pass
    elif kind.startswith("shift"):  # shift whole book & state prices by x%  (keeps anchor-relative geometry)
        f = 1 + float(kind[5:]) / 100
        lob[..., 0] *= f
        st[:, :3] *= f
    elif kind == "swap_depth":  # swap bid/ask volumes level-wise (imbalance flipped)
        v = lob[..., 1].clone()
        lob[:, 0, :, 1], lob[:, 1, :, 1] = v[:, 1], v[:, 0]
    elif kind == "vol_x10":
        lob[..., 1] *= 10
    elif kind == "vol_div10":
        lob[..., 1] /= 10
    elif kind == "late_session":  # same book, pretend 3h after open
        st[:, 3] = 3 * 3600.0
    return lob, st


def exp_nll(_):
    codes, lob, st = load_example()
    kinds = ["true", "shift+1", "shift-1", "swap_depth", "vol_x10", "vol_div10", "late_session"]
    res = {}
    for s in SIZES:
        m = M3.from_checkpoint(REL / s / "best.pt")
        r = {}
        base = m.token_nll(codes, lob, st)[0]
        for k in kinds:
            l2, s2 = _perturb(lob, st, k)
            nll = m.token_nll(codes, l2, s2)[0]
            r[k] = dict(mean=float(nll.mean()), first64=float(nll[:64].mean()), last256=float(nll[-256:].mean()),
                        delta_vs_true=float((nll - base).mean()),
                        delta_first64=float((nll - base)[:64].mean()))
        nl0 = m.token_nll(codes, lob, st, use_lob=False)[0]
        r["no_prefix_OOD"] = dict(mean=float(nl0.mean()))
        r["ckpt_val_loss"] = m.meta["metrics"]["val_loss"]
        r["per_position_nll_true"] = base.tolist()
        res[s] = r
        print(s, {k: round(v["mean"], 4) for k, v in r.items() if isinstance(v, dict)})
    dump("nll", res)


# ------------------------------------------------------------------ tokenizer audits
def exp_tokenizer(_):
    codes, lob, st = load_example()
    tok = VQTokenizer.from_checkpoint(REL / "tokenizer/base/best.pt")
    anchor = float(st[0, 4])
    f = tok.decode(codes)  # [1,512,5]
    re = tok.encode(f)
    agree = float((re == codes).float().mean())
    f2 = tok.decode(re)
    # decode-context sensitivity: tokens [256:512] decoded alone vs inside the full 512 context
    alone = tok.decode(codes[:, 256:])[0]
    inctx = f[0, 256:]
    px_ctx = np.rint(anchor * (1 + inctx[:, 0].double().numpy()) / 0.01)
    px_alone = np.rint(anchor * (1 + alone[:, 0].double().numpy()) / 0.01)
    tick_diff = np.abs(px_ctx - px_alone)
    raw_px = anchor * (1 + f[0, :, 0].double().numpy()) / 0.01
    frac = np.abs(raw_px - np.rint(raw_px))
    out = dict(
        anchor_open_mid=anchor,
        reencode_code_agreement=agree,
        reencode_price_tick_mae=float(np.mean(np.abs(np.rint(anchor * (1 + f2[0, :, 0].double().numpy()) / .01)
                                                     - np.rint(raw_px)))),
        context_sensitivity=dict(
            tick_exact_same=float(np.mean(tick_diff == 0)), tick_mae=float(tick_diff.mean()),
            tick_p90=float(np.percentile(tick_diff, 90)), tick_max=float(tick_diff.max()),
            action_flip=float((alone[:, 3] != inctx[:, 3]).float().mean()),
            side_flip=float((alone[:, 4] != inctx[:, 4]).float().mean()),
            logvol_mae=float((alone[:, 1] - inctx[:, 1]).abs().mean()),
            dt_mae=float((alone[:, 2] - inctx[:, 2]).abs().mean())),
        decoded_price_off_grid_mean_ticks=float(frac.mean()),
        negative_dt_frac=float((f[0, :, 2] < 0).float().mean()),
        add_frac=float(f[0, :, 3].mean()), bid_frac=float(f[0, :, 4].mean()),
        released_tokenizer_metrics={k: v for k, v in tok.meta["metrics"].items() if "tick" in k or "mae" in k},
        feature_clip=tok.meta["feature_clip"],
        val_split=tok.meta["train_args"]["val_split"], clip_stats_file=tok.meta["train_args"]["feature_clip_stats"],
    )
    print(json.dumps(out, indent=1, default=float)[:2500])
    dump("tokenizer", out)


# ------------------------------------------------------------------ rollouts + engine replay
def _summ(stats, mid0):
    m = stats.mid
    ok = np.isfinite(m)
    r = np.log(m[ok][-1] / mid0) * 1e4 if ok.any() else np.nan
    return dict(events=stats.n_events, elapsed_s=float(stats.t[-1]), add_frac=stats.n_add / stats.n_events,
                cancel_unmatched_qty_frac=stats.cancel_unmatched_qty / max(stats.cancel_qty, 1),
                cancel_fully_unmatched_frac=stats.cancel_events_fully_unmatched / max(stats.n_cancel, 1),
                cancel_unmatched_outside_known_frac=stats.cancel_unmatched_outside_known / max(stats.n_cancel, 1),
                marketable_add_frac=stats.marketable_adds / max(stats.n_add, 1), traded_qty=stats.traded_qty,
                one_side_empty_frac=stats.one_side_empty_events / stats.n_events,
                negative_dt_frac=stats.negative_dt_frac, zero_qty_frac=stats.zero_qty_events / stats.n_events,
                end_ret_bp=float(r), mid_min=float(np.nanmin(m)) if ok.any() else np.nan,
                mid_max=float(np.nanmax(m)) if ok.any() else np.nan,
                spread_ticks_med=float(np.nanmedian((stats.ask - stats.bid) / 0.01)))


def exp_rollout(args):
    codes, lob, st = load_example()
    tok = VQTokenizer.from_checkpoint(REL / "tokenizer/base/best.pt")
    anchor = float(st[0, 4])
    book0 = Book.from_snapshot(lob[0].numpy())
    mid0 = book0.mid()
    # real prompt: decode tokens (lossy) and replay from the snapshot -> reference path
    fp = tok.decode(codes)[0].numpy()
    o = decode_to_orders(fp, anchor)
    book_p, sp = replay(book0, *o)
    res = dict(meta=dict(anchor=anchor, mid0=mid0, S=args.rollouts, new=args.new, temperature=1.0,
                         prompt_len=int(codes.shape[1])), prompt_replay=_summ(sp, mid0))
    print("prompt", res["prompt_replay"])
    raw = {}
    for s in SIZES:
        m = M3.from_checkpoint(REL / s / "best.pt")
        g = torch.Generator().manual_seed(1234)
        S = args.rollouts
        t0 = time.time()
        gen, ent = m.generate(codes.expand(S, -1), lob.expand(S, -1, -1, -1), st.expand(S, -1), args.new,
                              generator=g)
        el = time.time() - t0
        # decode A (release-style): generated tokens alone, positions restart at 0
        fa = tok.decode(gen).numpy()
        # decode B (context-aware for first 512 generated): prompt + gen[:512] in one 1024 window
        fb = tok.decode(torch.cat([codes.expand(S, -1), gen[:, :512]], 1))[:, 512:].numpy()
        per = []
        ctx_tick = []
        for i in range(S):
            oa = decode_to_orders(fa[i], anchor)
            _, sa = replay(book_p, *oa)  # continue from the book reached by the real prompt
            per.append(_summ(sa, book_p.mid()))  # return measured from the post-prompt mid
            ob = decode_to_orders(fb[i], anchor)
            ctx_tick.append(np.abs(oa[0][:512] - ob[0]))
        ctx_tick = np.concatenate(ctx_tick)
        agg = {k: dict(mean=float(np.nanmean([p[k] for p in per])), median=float(np.nanmedian([p[k] for p in per])),
                       p10=float(np.nanpercentile([p[k] for p in per], 10)),
                       p90=float(np.nanpercentile([p[k] for p in per], 90))) for k in per[0]}
        pos = np.arange(codes.shape[1] + 1, codes.shape[1] + 1 + args.new)  # absolute position after BOS
        e = ent.mean(0).numpy()
        trained_len = m.meta["train_args"]["sample_seq_len"] + 1  # BOS + 1024 codes (+8 prefix)
        res[s] = dict(gen_seconds=el, agg=agg,
                      decode_context_tick_diff=dict(exact_same=float(np.mean(ctx_tick == 0)),
                                                    mae=float(ctx_tick.mean()),
                                                    p90=float(np.percentile(ctx_tick, 90))),
                      entropy_in_trained_range=float(e[pos <= trained_len].mean()),
                      entropy_beyond_trained=float(e[pos > trained_len].mean()) if (pos > trained_len).any() else None,
                      entropy_by_256=[float(e[j:j + 256].mean()) for j in range(0, len(e), 256)],
                      entropy_by_64=[float(e[j:j + 64].mean()) for j in range(0, len(e), 64)],
                      entropy_bins_abs_start_pos=[int(pos[j]) for j in range(0, len(e), 64)])
        raw[f"{s}_gen"] = gen.numpy()
        raw[f"{s}_entropy"] = ent.numpy()
        print(s, f"{el:.0f}s", {k: round(v['median'], 4) for k, v in agg.items()}, res[s]["decode_context_tick_diff"],
              "H in/out", res[s]["entropy_in_trained_range"], res[s]["entropy_beyond_trained"])
    np.savez_compressed(OUT / "rollout_raw.npz", **raw)
    dump("rollout", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["scaling", "nll", "tokenizer", "rollout", "decode_ablation"])
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--new", type=int, default=1024)
    a = ap.parse_args()
    if a.exp != "decode_ablation":
        dict(scaling=exp_scaling, nll=exp_nll, tokenizer=exp_tokenizer, rollout=exp_rollout)[a.exp](a)


# ------------------------------------------------------------------ decode-protocol ablation on saved rollouts
def exp_decode_ablation(_):
    """Same generated tokens, first 512 generated events, replayed under 2 decode protocols:
    A = release default (include_prompt=False: generated ids decoded alone, positions restart at 0)
    B = context-aware (prompt + gen[:512] decoded jointly in one 1024 window, matching training layout)."""
    codes, lob, st = load_example()
    tok = VQTokenizer.from_checkpoint(REL / "tokenizer/base/best.pt")
    anchor = float(st[0, 4])
    book0 = Book.from_snapshot(lob[0].numpy())
    mid0 = book0.mid()
    _, sp = replay(book0, *decode_to_orders(tok.decode(codes)[0].numpy(), anchor))
    book_p = _
    raw = np.load(OUT / "rollout_raw.npz")
    res = {}
    for s in SIZES:
        gen = torch.from_numpy(raw[f"{s}_gen"])[:, :512]
        S = gen.shape[0]
        fa = tok.decode(gen).numpy()
        fb = tok.decode(torch.cat([codes.expand(S, -1), gen], 1))[:, 512:].numpy()
        out = {}
        for name, F in (("A_release_gen_only", fa), ("B_context_aware", fb)):
            per = [_summ(replay(book_p, *decode_to_orders(F[i], anchor))[1], book_p.mid()) for i in range(S)]  # post-prompt mid
            out[name] = {k: dict(mean=float(np.nanmean([p[k] for p in per])),
                                 median=float(np.nanmedian([p[k] for p in per])),
                                 frac_negative=float(np.mean([p[k] < 0 for p in per])) if k == "end_ret_bp" else None)
                         for k in ("end_ret_bp", "elapsed_s", "cancel_unmatched_qty_frac", "cancel_fully_unmatched_frac",
                                   "marketable_add_frac", "negative_dt_frac", "add_frac")}
            out[name]["bid_frac_mean"] = float(F[..., 4].mean())
            out[name]["rel_open_mean"] = float(F[..., 0].mean())
        res[s] = out
        print(s, json.dumps({n: {k: (v["median"] if isinstance(v, dict) else v) for k, v in o.items()} for n, o in out.items()}))
    res["prompt_real_replay_end_ret_bp"] = float(np.log(sp.mid[-1] / mid0) * 1e4)
    res["prompt_decoded_bid_frac"] = float(tok.decode(codes)[0, :, 4].float().mean())
    dump("decode_ablation", res)


if __name__ == "__main__" and sys.argv[1:2] == ["decode_ablation"]:
    exp_decode_ablation(None)
