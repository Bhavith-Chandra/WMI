"""Independent re-implementation of the M3 inference stack.

Written from the paper (arXiv 2608.19227) and the released checkpoint
state-dict layout. Parameter names match the release so `load_state_dict(strict=True)`
acts as an architecture-equivalence check. Checkpoints are always loaded with
`torch.load(..., weights_only=True)` (no pickle code execution).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

FEATURES = ("relative_open_price", "log_volume", "delta_time_seconds", "action", "side")


def load_ckpt(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def _causal(n: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def _enc_layer(d, h, ff, act="gelu"):
    return nn.TransformerEncoderLayer(d, h, ff, dropout=0.0, activation=act, batch_first=True, norm_first=True)


# --------------------------------------------------------------------------- tokenizer
class FeatureEmbedding(nn.Module):
    def __init__(self, dim: int, d_model: int):
        super().__init__()
        self.continuous_projs = nn.ModuleList(nn.Linear(1, dim) for _ in range(3))
        self.action_emb = nn.Embedding(2, dim)
        self.side_emb = nn.Embedding(2, dim)
        self.out_proj = nn.Linear(dim * 5, d_model)

    def forward(self, x):
        cont = x[..., :3].float()
        a = torch.round(x[..., 3]).long().clamp(0, 1)
        s = torch.round(x[..., 4]).long().clamp(0, 1)
        parts = [p(cont[..., i : i + 1]) for i, p in enumerate(self.continuous_projs)]
        return self.out_proj(torch.cat(parts + [self.action_emb(a), self.side_emb(s)], -1))


class _Codebook(nn.Module):
    def __init__(self, k: int, d: int):
        super().__init__()
        self.register_buffer("initted", torch.zeros(1))
        self.register_buffer("cluster_size", torch.zeros(1, k))
        self.register_buffer("embed_avg", torch.zeros(1, k, d))
        self.register_buffer("embed", torch.zeros(1, k, d))


class _VQ(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self._codebook = _Codebook(k, d)


class _Quantizer(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.vq = _VQ(k, d)

    @property
    def embed(self):
        return self.vq._codebook.embed[0]

    def nearest(self, z):  # euclidean nearest code (paper: argmin ||z-e||^2)
        e = self.embed
        d = (z * z).sum(-1, keepdim=True) - 2 * z @ e.T + (e * e).sum(-1)
        return d.argmin(-1)


class _Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.continuous = nn.Linear(d, 3)
        self.action = nn.Linear(d, 2)
        self.side = nn.Linear(d, 2)


class _TimeHead(nn.Module):  # legacy, unused at decode (decode_time_mode="reconstruction")
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))


class VQTokenizer(nn.Module):
    def __init__(self, cfg: dict, has_time_head: bool):
        super().__init__()
        d, h, ff, L = cfg["d_model"], cfg["n_heads"], cfg["ff_dim"], cfg["n_layers"]
        self.cfg = cfg
        self.embedding = FeatureEmbedding(cfg["feature_embed_dim"], d)
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], d)
        self.encoder = nn.TransformerEncoder(_enc_layer(d, h, ff), L, enable_nested_tensor=False)
        self.encoder_ln = nn.LayerNorm(d)
        self.to_latent = nn.Linear(d, cfg["latent_dim"])
        self.quantizer = _Quantizer(cfg["codebook_size"], cfg["latent_dim"])
        self.from_latent = nn.Linear(cfg["latent_dim"], d)
        self.decoder = nn.TransformerEncoder(_enc_layer(d, h, ff), L, enable_nested_tensor=False)
        self.decoder_ln = nn.LayerNorm(d)
        self.head = _Head(d)
        if has_time_head:
            self.time_head = _TimeHead(d)
        self.clip_lo = None
        self.clip_hi = None

    @classmethod
    def from_checkpoint(cls, path):
        p = load_ckpt(path)
        sd = p["model_state_dict"]
        m = cls(p["model_config"], any(k.startswith("time_head.") for k in sd))
        m.load_state_dict(sd, strict=True)
        fc = p["feature_clip"]  # order: rel_mid, rel_open, log_vol, dt  -> tokenizer uses [1,2,3]
        m.clip_lo = torch.tensor(fc["lower"][1:], dtype=torch.float32)
        m.clip_hi = torch.tensor(fc["upper"][1:], dtype=torch.float32)
        m.meta = {k: p[k] for k in ("metrics", "train_args", "feature_clip", "decode_policy", "step")}
        return m.eval()

    def _pos(self, n, dev):
        if n > self.pos_emb.num_embeddings:
            raise ValueError(f"tokenizer window {n} > {self.pos_emb.num_embeddings}")
        return self.pos_emb(torch.arange(n, device=dev))[None]

    @torch.no_grad()
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        x = feats.float().clone()
        x[..., :3] = torch.maximum(torch.minimum(x[..., :3], self.clip_hi), self.clip_lo)
        n = x.shape[1]
        h = self.embedding(x) + self._pos(n, x.device)
        h = self.encoder_ln(self.encoder(h, mask=_causal(n, x.device)))
        return self.quantizer.nearest(self.to_latent(h))

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim == 1:
            codes = codes[None]
        q = self.quantizer.embed[codes]
        n = q.shape[1]
        h = self.from_latent(q) + self._pos(n, q.device)
        h = self.decoder_ln(self.decoder(h, mask=_causal(n, q.device)))
        cont = self.head.continuous(h)
        a = self.head.action(h).argmax(-1).float()
        s = self.head.side(h).argmax(-1).float()
        return torch.cat([cont, a[..., None], s[..., None]], -1)


# --------------------------------------------------------------------------- AR model
class LobPrefixEncoder(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        H = c["hidden_size"]
        self.c = c
        self.level_input = nn.Linear(3, H)
        self.side_embedding = nn.Embedding(2, H)
        self.level_embedding = nn.Embedding(10, H)
        self.level_encoder = nn.TransformerEncoder(
            _enc_layer(H, c["lob_encoder_heads"], H * c["lob_encoder_ff_multiplier"]), c["lob_encoder_layers"],
            enable_nested_tensor=False,
        )
        self.query_tokens = nn.Parameter(torch.zeros(c["lob_prefix_tokens"], H))
        self.prefix_pool = nn.MultiheadAttention(H, c["lob_encoder_heads"], batch_first=True)
        self.state_projection = nn.Sequential(nn.Linear(4, H), nn.GELU(), nn.Linear(H, H))
        self.output_norm = nn.LayerNorm(H)

    @staticmethod
    def ref_mid(st):
        mid, bid, ask, opn = st[:, 0], st[:, 1], st[:, 2], st[:, 4]
        sp = 0.5 * (bid + ask)
        vm, vo, vs = (torch.isfinite(t) & (t > 0) for t in (mid, opn, sp))
        mid = torch.where(vm, mid, torch.where(vo, opn, sp))
        mid = torch.where(vs | vm | vo, mid, torch.ones_like(mid))
        return mid.clamp_min(1e-6)

    def forward(self, lob, st):
        B = lob.shape[0]
        st = st.float()
        mid = self.ref_mid(st)
        p, v = lob[..., 0].float(), lob[..., 1].float()
        valid = torch.isfinite(p) & (p > 0) & torch.isfinite(v) & (v >= 0)
        clip = float(self.c["lob_price_clip"])
        rel = ((p - mid[:, None, None]) / mid[:, None, None]).clamp(-clip, clip)
        f = torch.stack([rel, torch.log1p(v.clamp_min(0)), valid.float()], -1)
        f = torch.nan_to_num(f, 0.0, 0.0, 0.0).reshape(B, 20, 3)
        side = torch.arange(2, device=lob.device).repeat_interleave(10)
        lev = torch.arange(10, device=lob.device).repeat(2)
        t = self.level_input(f) + self.side_embedding(side)[None] + self.level_embedding(lev)[None]
        if self.c.get("lob_include_state", True):
            spread = ((st[:, 2] - st[:, 1]) / mid).clamp(-0.2, 0.2)
            tfrac = (st[:, 3] / 14400.0).clamp(0, 1)
            orel = ((st[:, 4] - mid) / mid).clamp(-0.2, 0.2)
            r = st[:, 5].clamp(-0.2, 0.2)
            sf = torch.nan_to_num(torch.stack([spread, tfrac, orel, r], -1), 0.0, 0.0, 0.0)
            t = t + self.state_projection(sf)[:, None]
        t = self.level_encoder(t)
        q = self.query_tokens[None].expand(B, -1, -1)
        out, _ = self.prefix_pool(q, t, t, need_weights=False)
        return self.output_norm(out)


class M3(nn.Module):
    def __init__(self, c: dict):
        super().__init__()
        self.c = c
        lc = LlamaConfig(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"], max_position_embeddings=c["max_position_embeddings"],
            rms_norm_eps=c["rms_norm_eps"], rope_theta=c["rope_theta"], tie_word_embeddings=c["tie_word_embeddings"],
            pad_token_id=c["pad_token_id"], bos_token_id=c["bos_token_id"], eos_token_id=c["eos_token_id"],
            hidden_act="silu", attention_bias=False, mlp_bias=False, use_cache=True,
        )
        self.llama = LlamaForCausalLM(lc)
        self.lob_encoder = LobPrefixEncoder(c)
        self.K = c["codebook_size"]
        self.bos = c["bos_token_id"]
        self.P = c["lob_prefix_tokens"]

    @classmethod
    def from_checkpoint(cls, path):
        p = load_ckpt(path)
        m = cls(p["ar_config"])
        m.load_state_dict(p["model_state_dict"], strict=True)
        m.meta = {k: p[k] for k in ("metrics", "train_args", "step", "ar_config")}
        return m.eval()

    def embeds(self, ids, lob, st):
        e = self.llama.get_input_embeddings()(ids)
        return torch.cat([self.lob_encoder(lob, st).to(e.dtype), e], 1)

    @torch.no_grad()
    def token_nll(self, codes, lob, st, use_lob=True):
        """Per-token NLL of order codes (BOS prepended, prefix/BOS labels masked, as in training)."""
        B = codes.shape[0]
        ids = torch.cat([torch.full((B, 1), self.bos, dtype=torch.long), codes], 1)
        if use_lob:
            x = self.embeds(ids, lob, st)
            off = self.P
        else:
            x = self.llama.get_input_embeddings()(ids)
            off = 0
        logits = self.llama(inputs_embeds=x).logits.float()
        lg = logits[:, off : off + ids.shape[1] - 1]  # positions predicting codes[0..T-1]
        lp = torch.log_softmax(lg, -1)
        return -lp.gather(-1, codes[..., None])[..., 0]

    @torch.no_grad()
    def generate(self, prompt, lob, st, n_new, temperature=1.0, generator=None, record_entropy=True):
        """Batched KV-cache sampling restricted to codebook ids (mirrors released sampler: no rolling window)."""
        B = prompt.shape[0]
        ids = torch.cat([torch.full((B, 1), self.bos, dtype=torch.long), prompt], 1)
        out = self.llama(inputs_embeds=self.embeds(ids, lob, st), use_cache=True, logits_to_keep=1)
        pkv = out.past_key_values
        logits = out.logits[:, -1, : self.K].float()
        gen, ent = [], []
        for i in range(n_new):
            lp = torch.log_softmax(logits / max(temperature, 1e-6), -1)
            if record_entropy:
                ent.append(-(lp.exp() * lp).sum(-1))
            nxt = torch.multinomial(lp.exp(), 1, generator=generator)
            gen.append(nxt)
            if i == n_new - 1:
                break
            out = self.llama(input_ids=nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, : self.K].float()
        g = torch.cat(gen, 1)
        e = torch.stack(ent, 1) if ent else None
        return g, e


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
