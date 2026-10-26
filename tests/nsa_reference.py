#!/usr/bin/env python3
"""Reference forward pass for Idiap/gated-deltanet-nsa-0.4B-10B.

This is a slow PyTorch port of the fla reference math. It is only used to
validate the C++ implementation; the shipped inference path does not use
Python. Run with the project environment:

  uv run --project . python tests/nsa_reference.py \
      --model gated-deltanet-nsa-0.4B-10B \
      --prompt "The capital of France is" \
      --dump /tmp/ref.bin
"""

from __future__ import annotations

import argparse
import json
import math
import struct

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def l2_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x / torch.clamp(x.pow(2).sum(-1, keepdim=True).sqrt(), min=eps)


def rope(x: torch.Tensor, base: float) -> torch.Tensor:
    # x: [T, H, D], split-half (GPT-NeoX) rotation, same as fla
    t, _, d = x.shape
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    pos = torch.arange(t, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    cos = torch.cos(freqs)[:, None, :]
    sin = torch.sin(freqs)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Model:
    def __init__(self, path: str):
        self.cfg = json.load(open(f"{path}/config.json"))
        self.w = load_file(f"{path}/model.safetensors")
        self.n_layer = self.cfg["num_hidden_layers"]
        self.eps = self.cfg["norm_eps"]
        self.attn = self.cfg["attn"]
        self.nsa_layers = set(self.attn["layers"])
        self.hidden = self.cfg["hidden_size"]

    def w_(self, name: str) -> torch.Tensor:
        return self.w[name].float()

    def gdn(self, x: torch.Tensor, il: int) -> torch.Tensor:
        p = f"model.layers.{il}.attn."
        T = x.shape[0]
        H, K = self.cfg["num_heads"], self.cfg["head_dim"]  # 8 x 128
        V = K

        q = x @ self.w_(p + "q_proj.weight").T
        k = x @ self.w_(p + "k_proj.weight").T
        v = x @ self.w_(p + "v_proj.weight").T

        def conv(z: torch.Tensor, name: str) -> torch.Tensor:
            w = self.w_(p + name)  # [C, 1, 4]
            z = z.T.unsqueeze(0)   # [1, C, T]
            z = F.pad(z, (3, 0))
            z = F.conv1d(z, w, groups=z.shape[1])
            return F.silu(z).squeeze(0).T

        q, k, v = conv(q, "q_conv1d.weight"), conv(k, "k_conv1d.weight"), conv(v, "v_conv1d.weight")
        q = l2_norm(q.reshape(T, H, K), self.eps)
        k = l2_norm(k.reshape(T, H, K), self.eps)
        v = v.reshape(T, H, V)

        beta = torch.sigmoid(x @ self.w_(p + "b_proj.weight").T)      # [T, H]
        alpha = x @ self.w_(p + "a_proj.weight").T                    # [T, H]
        dt = self.w_(p + "dt_bias")
        g = -torch.exp(self.w_(p + "A_log")) * F.softplus(alpha + dt)  # [T, H]

        scale = 1.0 / math.sqrt(K)
        q = q * scale

        state = torch.zeros(H, K, V)
        out = torch.zeros(T, H, V)
        for t in range(T):
            state = state * g[t].exp()[:, None, None]
            pred = (state * k[t][:, :, None]).sum(1)
            dv = (v[t] - pred) * beta[t][:, None]
            state = state + k[t][:, :, None] * dv[:, None, :]
            out[t] = (q[t][:, :, None] * state).sum(1)

        z = (x @ self.w_(p + "g_proj.weight").T).reshape(T, H, V)
        out = rms_norm(out, self.w_(p + "o_norm.weight"), self.eps) * F.silu(z)
        return (out.reshape(T, H * V) @ self.w_(p + "o_proj.weight").T)

    def nsa(self, x: torch.Tensor, il: int) -> torch.Tensor:
        p = f"model.layers.{il}.attn."
        T = x.shape[0]
        Hq, Hkv, D = self.attn["num_heads"], self.attn["num_kv_heads"], self.attn["head_dim"]
        BS, TOPK, WIN = self.attn["block_size"], self.attn["block_counts"], self.attn["window_size"]
        scale = 1.0 / math.sqrt(D)

        q = (x @ self.w_(p + "q_proj.weight").T).reshape(T, Hq, D)
        k = (x @ self.w_(p + "k_proj.weight").T).reshape(T, Hkv, D)
        v = (x @ self.w_(p + "v_proj.weight").T).reshape(T, Hkv, D)
        q, k = rope(q, self.attn["rope_theta"]), rope(k, self.attn["rope_theta"])
        g = torch.sigmoid((x @ self.w_(p + "g_proj.weight").T).reshape(T, Hq, 3))

        nb = T // BS
        if nb:
            kc = k[: nb * BS].reshape(nb, BS, Hkv, D).mean(1)
            vc = v[: nb * BS].reshape(nb, BS, Hkv, D).mean(1)
        else:
            kc = torch.zeros(0, Hkv, D)
            vc = torch.zeros(0, Hkv, D)

        group = Hq // Hkv
        kc_e = kc.repeat_interleave(group, dim=1)
        vc_e = vc.repeat_interleave(group, dim=1)
        k_e = k.repeat_interleave(group, dim=1)
        v_e = v.repeat_interleave(group, dim=1)
        out_cmp = torch.zeros(T, Hq, D)
        out_slc = torch.zeros(T, Hq, D)
        out_swa = torch.zeros(T, Hq, D)

        for t in range(T):
            n_complete = (t + 1) // BS
            lse = torch.zeros(Hq)
            if n_complete:
                s = torch.einsum("hd,bhd->hb", q[t], kc_e[:n_complete]) * scale
                lse = torch.logsumexp(s, dim=1)
                prob = torch.softmax(s, dim=1)
                out_cmp[t] = torch.einsum("hb,bhd->hd", prob, vc_e[:n_complete])

            ic = t // BS
            best: list[tuple[float, int]] = []
            for ib in range(0, ic + 1):
                if ib < nb:
                    score = 0.0
                    for h in range(Hq):
                        score += float(torch.exp(torch.dot(q[t, h], kc[ib, h // group]) * scale - lse[h]))
                else:
                    score = float(Hq)
                if ib == 0 or ib == ic or ib == ic - 1:
                    score += 1e9
                best.append((score, ib))
            best.sort(reverse=True)
            sel = [ib for _, ib in best[:TOPK] if ib >= 0]

            idx = []
            for ib in sel:
                for j in range(ib * BS, min((ib + 1) * BS, T)):
                    if j <= t:
                        idx.append(j)
            if idx:
                s = torch.einsum("hd,khd->hk", q[t], k_e[idx]) * scale
                prob = torch.softmax(s, dim=1)
                out_slc[t] = torch.einsum("hk,khd->hd", prob, v_e[idx])

            lo = max(0, t - WIN + 1)
            idx = list(range(lo, t + 1))
            s = torch.einsum("hd,khd->hk", q[t], k_e[idx]) * scale
            prob = torch.softmax(s, dim=1)
            out_swa[t] = torch.einsum("hk,khd->hd", prob, v_e[idx])

        out = g[..., 0:1] * out_cmp + g[..., 1:2] * out_slc + g[..., 2:3] * out_swa
        out = out.reshape(T, Hq * D) @ self.w_(p + "o_proj.weight").T
        return out

    def forward(self, ids: list[int], layers: list | None = None) -> torch.Tensor:
        x = self.w_("model.embeddings.weight")[ids]
        for il in range(self.n_layer):
            if layers is not None:
                layers.append(x.clone())
            h = rms_norm(x, self.w_(f"model.layers.{il}.attn_norm.weight"), self.eps)
            x = x + (self.nsa(h, il) if il in self.nsa_layers else self.gdn(h, il))
            h = rms_norm(x, self.w_(f"model.layers.{il}.mlp_norm.weight"), self.eps)
            x = x + (F.silu(h @ self.w_(f"model.layers.{il}.mlp.gate_proj.weight").T) *
                     (h @ self.w_(f"model.layers.{il}.mlp.up_proj.weight").T)) @ \
                self.w_(f"model.layers.{il}.mlp.down_proj.weight").T
        x = rms_norm(x, self.w_("model.norm.weight"), self.eps)
        return x @ self.w_("model.embeddings.weight").T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--compare")
    ap.add_argument("--dump-layers")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tok(args.prompt)["input_ids"]
    print("tokens:", ids)

    torch.set_grad_enabled(False)
    model = Model(args.model)
    layers = [] if args.dump_layers else None
    logits = model.forward(ids, layers)
    if args.dump_layers:
        for il, x in enumerate(layers):
            x.numpy().astype("<f4").tofile("%s/ref_layer_%02d.f32" % (args.dump_layers, il))
    print("logits:", tuple(logits.shape))

    if args.dump:
        with open(args.dump, "wb") as f:
            f.write(struct.pack("<ii", logits.shape[0], logits.shape[1]))
            f.write(logits.numpy().astype("<f4").tobytes())
        print("wrote", args.dump)

    if args.compare:
        with open(args.compare, "rb") as f:
            n, v = struct.unpack("<ii", f.read(8))
            other = torch.frombuffer(bytearray(f.read()), dtype=torch.float32).reshape(n, v)
        d = (logits - other).abs()
        print("compare: max %.6g mean %.6g" % (d.max().item(), d.mean().item()))
        for i in range(d.shape[0]):
            print("  tok %d max %.4f mean %+.4f" % (i, d[i].max().item(), (logits[i]-other[i]).mean().item()))
        print("top1 ref ", logits.argmax(-1).tolist())
        print("top1 cpp ", other.argmax(-1).tolist())


if __name__ == "__main__":
    main()
