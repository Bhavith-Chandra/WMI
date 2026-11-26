"""No-lookahead guarantees: every component may only see the past.

A future perturbation must leave every earlier output bit-for-bit (to fp tolerance) unchanged.
"""
import torch

J = 200  # perturbation index


def test_ar_model_is_causal(tiny, example):
    codes, lob, st = example
    c = codes[:, :300].clone()
    c2 = c.clone()
    c2[0, J:] = (c2[0, J:] + 777) % tiny.K
    a = tiny.token_nll(c, lob, st)
    b = tiny.token_nll(c2, lob, st)
    # token_nll[t] = NLL of code t given codes<t: targets t<J see only unperturbed inputs
    assert torch.allclose(a[0, :J], b[0, :J], atol=1e-4), "AR model leaks future codes"
    assert not torch.allclose(a[0, J:], b[0, J:]), "perturbation had no effect (test is vacuous)"


def test_tokenizer_encoder_is_causal(tok, example):
    codes, _, _ = example
    f = tok.decode(codes[:, :300])
    f2 = f.clone()
    f2[0, J:, 0] += 0.01
    f2[0, J:, 1] += 0.5
    a, b = tok.encode(f), tok.encode(f2)
    assert torch.equal(a[0, :J], b[0, :J]), "tokenizer encoder leaks future features"
    assert not torch.equal(a[0, J:], b[0, J:]), "vacuous"


def test_tokenizer_decoder_is_causal(tok, example):
    codes, _, _ = example
    c = codes[:, :300].clone()
    c2 = c.clone()
    c2[0, J:] = (c2[0, J:] + 1234) % 32768
    a, b = tok.decode(c), tok.decode(c2)
    assert torch.allclose(a[0, :J], b[0, :J], atol=1e-5), "tokenizer decoder leaks future codes"
    assert not torch.allclose(a[0, J:], b[0, J:]), "vacuous"


def test_kv_cache_generation_matches_full_recompute(tiny, example):
    """The cached sampler must produce exactly the distribution a full forward pass gives (no state bugs)."""
    codes, lob, st = example
    p = codes[:, :64]
    g = torch.Generator().manual_seed(0)
    gen, ent = tiny.generate(p, lob, st, 12, generator=g)
    full = torch.cat([p, gen], 1)
    ids = torch.cat([torch.full((1, 1), tiny.bos, dtype=torch.long), full], 1)
    logits = tiny.llama(inputs_embeds=tiny.embeds(ids, lob, st)).logits[:, :, : tiny.K].float()
    P = tiny.P
    for k in range(12):
        lp = torch.log_softmax(logits[0, P + 64 + k], -1)
        h = -(lp.exp() * lp).sum()
        assert abs(float(h) - float(ent[0, k])) < 1e-3, f"cache mismatch at step {k}"


def test_seeded_generation_is_deterministic(tiny, example):
    codes, lob, st = example
    a, _ = tiny.generate(codes[:, :64], lob, st, 16, generator=torch.Generator().manual_seed(5))
    b, _ = tiny.generate(codes[:, :64], lob, st, 16, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
