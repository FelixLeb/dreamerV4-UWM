"""Two-stream tokenizer ported from the GR00T-WholeBodyControl reference
(`/home/mim-server/robot/GR00T-WholeBodyControl/models.py`). The wrapper at
the bottom mirrors `TokenizerWrapper`'s interface so callers don't have to
know which architecture they got.

Routing differs from the single-stream tokenizer in `tokenizer.py`:
  - encoder: latents attend to {latents ⊕ patches (⊕ proprio)}; patches
    attend to {patches} only; via separate attention modules per stream.
  - decoder: latents attend to {latents}; image readouts attend to
    {latents ⊕ patches}; proprio readouts are isolated from image readouts
    via a static spatial mask.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel


# ---------------------------------------------------------------------------
# KV cache (rolling buffer)
# ---------------------------------------------------------------------------

class KVCache:
    def __init__(self, max_seq_len: int, batch_size: int, num_heads: int,
                 head_dim: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len = max_seq_len
        self.curr_len = 0
        self.k_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                                   device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim,
                                   device=device, dtype=dtype)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        B, H, S_new, D = k_new.shape
        if self.curr_len + S_new > self.max_seq_len:
            shift = (self.curr_len + S_new) - self.max_seq_len
            self.k_cache = torch.roll(self.k_cache, shifts=-shift, dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=-shift, dims=2)
            self.curr_len -= shift
        self.k_cache[:, :, self.curr_len:self.curr_len + S_new, :] = k_new
        self.v_cache[:, :, self.curr_len:self.curr_len + S_new, :] = v_new
        self.curr_len += S_new
        return (self.k_cache[:, :, :self.curr_len, :],
                self.v_cache[:, :, :self.curr_len, :])

    def reset(self):
        self.curr_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class FeedForwardSwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: Optional[int] = None,
                 dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_hidden is None:
            d_hidden = (8 * d_model) // 3
        self.up = nn.Linear(d_model, d_hidden, bias=bias)
        self.gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.down = nn.Linear(d_hidden, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.silu(self.up(x))
        g = self.gate(x)
        h = self.drop(u * g)
        return self.down(h)


# ---------------------------------------------------------------------------
# MHA + GQA + RoPE + KV cache
# ---------------------------------------------------------------------------

class MHA_GQA(nn.Module):
    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int,
                 head_dim: int, dropout: float = 0.0,
                 max_seq_len: int = 8192, rope_base: float = 10000.0):
        super().__init__()
        assert d_model == num_heads_q * head_dim
        assert num_heads_q % num_heads_kv == 0
        assert head_dim % 2 == 0
        self.d_model = d_model
        self.Hq = num_heads_q
        self.G = num_heads_kv
        self.Dh = head_dim
        self.Wq = nn.Linear(d_model, self.Hq * self.Dh, bias=False)
        self.Wk = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.Wv = nn.Linear(d_model, self.G * self.Dh, bias=False)
        self.out = nn.Linear(self.Hq * self.Dh, d_model, bias=False)
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_rope_cache(max_seq_len)

    def _build_rope_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(positions, self.inv_freq)
        cos_cache = freqs.cos().repeat_interleave(2, dim=-1)
        sin_cache = freqs.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _apply_rope(self, x: torch.Tensor,
                    position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        *prefix, num_heads, seq_len, head_dim = x.shape
        if position_ids is None:
            cos = self.cos_cache[:seq_len]
            sin = self.sin_cache[:seq_len]
        else:
            cos = self.cos_cache[position_ids]
            sin = self.sin_cache[position_ids]
        cos = cos.to(x.dtype).unsqueeze(-3)
        sin = sin.to(x.dtype).unsqueeze(-3)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.unflatten(-1, (-1, 2))
        x_rot = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
        return x_rot.flatten(-2)

    def _proj_q(self, x):
        q = self.Wq(x)
        q = q.view(*q.shape[:-1], self.Hq, self.Dh)
        return q.transpose(-3, -2)

    def _proj_kv(self, x):
        k = self.Wk(x).view(*x.shape[:-1], self.G, self.Dh).transpose(-3, -2)
        v = self.Wv(x).view(*x.shape[:-1], self.G, self.Dh).transpose(-3, -2)
        return k, v

    def forward(self, q_src, kv_src, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                q_position_ids: Optional[torch.Tensor] = None,
                kv_position_ids: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                update_cache: bool = True):
        q = self._proj_q(q_src)
        k, v = self._proj_kv(kv_src)

        if kv_cache is not None:
            k = self._apply_rope(k, position_ids=kv_position_ids)
            if update_cache:
                k, v = kv_cache.update(k, v)
            else:
                cache_k = kv_cache.k_cache[:, :, :kv_cache.curr_len, :]
                cache_v = kv_cache.v_cache[:, :, :kv_cache.curr_len, :]
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
            q = self._apply_rope(q, position_ids=q_position_ids)
        else:
            q = self._apply_rope(q, position_ids=q_position_ids)
            k = self._apply_rope(k, position_ids=kv_position_ids)

        # Causal flag is meaningless for a single-token query (only one position).
        effective_causal = is_causal if q.shape[-2] != 1 else False

        prefix_shape = q.shape[:-3]
        if len(prefix_shape) > 1:
            prefix_prod = math.prod(prefix_shape)
            q = q.reshape(prefix_prod, *q.shape[-3:])
            k = k.reshape(prefix_prod, *k.shape[-3:])
            v = v.reshape(prefix_prod, *v.shape[-3:])
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    attn_mask=attn_mask, is_causal=effective_causal,
                    dropout_p=(self.dropout if self.training else 0.0),
                    enable_gqa=(self.G != self.Hq),
                )
            out = out.view(*prefix_shape, *out.shape[-3:])
        else:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    attn_mask=attn_mask, is_causal=effective_causal,
                    dropout_p=(self.dropout if self.training else 0.0),
                    enable_gqa=(self.G != self.Hq),
                )

        out = out.transpose(-3, -2).contiguous()
        out = out.view(*out.shape[:-2], self.Hq * self.Dh)
        return self.out(out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class BlockCausalEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads_q: int, num_heads_kv: int,
                 seq_len: int, num_latents: int, num_patches: int,
                 dropout: float = 0.0, mlp_ratio: float = 4.0,
                 temporal: bool = False, is_last: bool = False,
                 num_proprio: int = 0):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last

        self.attn_latent = MHA_GQA(
            d_model, num_heads_q, num_heads_kv, Dh, dropout=dropout,
            max_seq_len=seq_len if temporal else num_patches + num_latents + num_proprio,
        )
        if not is_last:
            self.attn_patch = MHA_GQA(
                d_model, num_heads_q, num_heads_q, Dh, dropout=dropout,
                max_seq_len=seq_len if temporal else num_patches,
            )

        self.ln_L_q = nn.RMSNorm(d_model)
        self.ln_L_kv = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_q = nn.RMSNorm(d_model)
            self.ln_P_kv = nn.RMSNorm(d_model)

        self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        if not is_last:
            self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)
        self.ln_L_ff = nn.RMSNorm(d_model)
        if not is_last:
            self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, P: torch.Tensor, L: torch.Tensor,
                Prop: Optional[torch.Tensor] = None
                ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if not self.temporal:
            kv_parts = [L, P] if Prop is None else [L, P, Prop]
            union_sp = torch.cat(kv_parts, dim=2)
            L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(union_sp), None, False)
            L_out = L + self.dropout(L_att)
            L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
            if not self.is_last:
                P_att = self.attn_patch(self.ln_P_q(P), self.ln_P_kv(P),
                                        attn_mask=None, is_causal=False)
                P_out = P + self.dropout(P_att)
                P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
            else:
                P_out = None
            return P_out, L_out

        # temporal
        L_t = L.permute(0, 2, 1, 3)  # [B, Nl, T, d]
        if not self.is_last:
            P_t = P.permute(0, 2, 1, 3)
            P_att = self.attn_patch(self.ln_P_q(P_t), self.ln_P_kv(P_t),
                                    attn_mask=None, is_causal=True)
            P_out = P_t + self.dropout(P_att)
            P_out = P_out + self.dropout(self.ffn_P(self.ln_P_ff(P_out)))
            P_out = P_out.permute(0, 2, 1, 3)
        else:
            P_out = None

        L_att = self.attn_latent(self.ln_L_q(L_t), self.ln_L_kv(L_t),
                                 attn_mask=None, is_causal=True)
        L_out = L_t + self.dropout(L_att)
        L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
        L_out = L_out.permute(0, 2, 1, 3)
        return P_out, L_out


class DreamerV4Encoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int], patch_size: int,
                 d_model: int, n_layers: int, num_heads_q: int,
                 num_heads_kv_latent: int, seq_len: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 n_latents: int = 256, bottleneck_dim: int = 16,
                 temporal_every: int = 1, in_channels: int = 3,
                 mae_max_mask_prob: float = 0.9, activate_masking: bool = False,
                 use_proprio: bool = False, proprio_dim: int = 0):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.mae_max_mask_prob = mae_max_mask_prob
        self.activate_masking = activate_masking

        self.patch_embed = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.latent_tokens = nn.Parameter(torch.randn(n_latents, d_model) / math.sqrt(d_model))

        layers = []
        for i in range(n_layers):
            temporal = (i % 2 == 1) if temporal_every == 1 else (((i + 1) % temporal_every) == 0)
            is_last = (i == n_layers - 1) or (
                i == n_layers - 2 and (n_layers % temporal_every) == 0
            )
            layers.append(BlockCausalEncoderLayer(
                d_model=d_model, num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv_latent, seq_len=seq_len,
                num_latents=self.Nl, num_patches=self.Np,
                dropout=dropout, mlp_ratio=mlp_ratio,
                temporal=temporal, is_last=is_last,
                num_proprio=1 if use_proprio else 0,
            ))
        self.layers = nn.ModuleList(layers)

        self.down_proj = nn.Linear(d_model, bottleneck_dim)

        self.use_proprio = use_proprio
        if use_proprio:
            assert proprio_dim > 0
            self.proprio_proj = nn.Linear(proprio_dim, d_model)
            self.mask_proprio_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
            nn.init.trunc_normal_(self.mask_proprio_token, std=0.02)
            self.proprio_mask_prob = 0.5

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        tok = self.patch_embed(x)
        tok = tok.flatten(2).transpose(1, 2)
        return tok.view(B, T, self.Np, self.d)

    def forward(self, video: torch.Tensor,
                proprio: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None):
        B, T, C, H, W = video.shape
        assert (H, W) == (self.H, self.W)

        P = self.patchify(video)

        if (self.training or self.activate_masking) and self.mae_max_mask_prob > 0.0:
            B, T, Np, D = P.shape
            drop_p = torch.rand(B, T, device=P.device, dtype=P.dtype) * self.mae_max_mask_prob
            rand = torch.rand(B, T, Np, device=P.device, dtype=P.dtype)
            mask = rand < drop_p.unsqueeze(-1) if mask is None else mask
            P = torch.where(mask.unsqueeze(-1), self.mask_token.to(P.dtype), P)

        Prop = None
        if self.use_proprio and proprio is not None:
            Prop = self.proprio_proj(proprio).unsqueeze(2)
            if self.training:
                prop_mask = torch.rand(B, T, device=Prop.device, dtype=Prop.dtype) < self.proprio_mask_prob
                Prop = torch.where(prop_mask[:, :, None, None],
                                   self.mask_proprio_token.to(Prop.dtype), Prop)

        L0 = self.latent_tokens[None, None, :, :].expand(B, T, self.Nl, self.d).contiguous()
        P_enc, L_enc = P, L0

        for layer in self.layers:
            prop_input = Prop if (not layer.temporal) else None
            P_enc, L_enc = layer(P_enc, L_enc, Prop=prop_input)

        # fp32 tanh: under bf16, the saturated tail's gradient becomes bit-zero,
        # which under sustained reconstruction pressure produces a one-way
        # collapse to a dead, constant code.
        pre_tanh = self.down_proj(L_enc).float()
        Z_fp32 = torch.tanh(pre_tanh)
        Z = Z_fp32.to(L_enc.dtype)

        if self.training:
            with torch.no_grad():
                self._last_pretanh_abs_mean = pre_tanh.detach().abs().mean()
                self._last_z_sat_frac = (Z_fp32.detach().abs() > 0.99).float().mean()

        if self.activate_masking:
            return P_enc, L_enc, Z, mask
        return P_enc, L_enc, Z


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class BlockCausalDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads_q: int,
                 num_heads_kv_latent: int, seq_len: int,
                 num_latents: int, num_patches: int,
                 dropout: float = 0.0, mlp_ratio: float = 4.0,
                 temporal: bool = False, is_last: bool = False,
                 use_proprio: bool = False, num_proprio: int = 1):
        super().__init__()
        Dh = d_model // num_heads_q
        self.temporal = temporal
        self.is_last = is_last
        self.use_proprio = use_proprio
        self.num_proprio = num_proprio if use_proprio else 0

        spatial_kv_len = num_latents + num_patches + self.num_proprio

        if not is_last:
            self.attn_latent = MHA_GQA(d_model, num_heads_q, num_heads_kv_latent, Dh,
                                       dropout=dropout,
                                       max_seq_len=seq_len if temporal else num_latents)
        self.attn_patch = MHA_GQA(d_model, num_heads_q, num_heads_q, Dh,
                                  dropout=dropout,
                                  max_seq_len=seq_len if temporal else spatial_kv_len)

        if not is_last:
            self.ln_L_q = nn.RMSNorm(d_model)
            self.ln_L_kv = nn.RMSNorm(d_model)
        self.ln_P_q = nn.RMSNorm(d_model)
        self.ln_P_kv = nn.RMSNorm(d_model)
        if not is_last:
            self.ffn_L = FeedForwardSwiGLU(d_model, None, dropout)
        self.ffn_P = FeedForwardSwiGLU(d_model, None, dropout)
        if not is_last:
            self.ln_L_ff = nn.RMSNorm(d_model)
        self.ln_P_ff = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_proprio:
            Nr = num_patches + num_proprio
            Nkv = num_latents + num_patches + num_proprio
            mask = torch.zeros(Nr, Nkv)
            mask[:num_patches, num_latents + num_patches:] = -1e9   # img Q ↛ proprio K
            mask[num_patches:, num_latents:num_latents + num_patches] = -1e9  # proprio Q ↛ img K
            self.register_buffer("proprio_spatial_mask", mask, persistent=True)

    def forward(self, R: torch.Tensor, L: torch.Tensor,
                kv_cache_l: Optional[KVCache] = None,
                kv_cache_r: Optional[KVCache] = None,
                position_ids: Optional[torch.Tensor] = None,
                update_cache: bool = True):
        if not self.temporal:
            union_sp = torch.cat([L, R], dim=2)
            if not self.is_last:
                L_att = self.attn_latent(self.ln_L_q(L), self.ln_L_kv(L),
                                         attn_mask=None, is_causal=False)
                L_out = L + self.dropout(L_att)
                L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
            else:
                L_out = None

            spatial_mask = (self.proprio_spatial_mask.to(dtype=R.dtype)
                            if self.use_proprio else None)
            R_att = self.attn_patch(self.ln_P_q(R), self.ln_P_kv(union_sp),
                                    attn_mask=spatial_mask, is_causal=False)
            R_out = R + self.dropout(R_att)
            R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))
            return R_out, L_out

        # temporal
        B, T, Np, d = R.shape
        R_flat = R.transpose(1, 2).reshape(B * Np, T, d)
        R_att = self.attn_patch(
            self.ln_P_q(R_flat), self.ln_P_kv(R_flat), is_causal=True,
            kv_cache=kv_cache_r, q_position_ids=position_ids,
            kv_position_ids=position_ids, update_cache=update_cache,
        )
        R_out = R_flat + self.dropout(R_att)
        R_out = R_out + self.dropout(self.ffn_P(self.ln_P_ff(R_out)))
        R_out = R_out.view(B, Np, T, d).transpose(1, 2)

        if not self.is_last:
            _, _, Nl, _ = L.shape
            L_flat = L.transpose(1, 2).reshape(B * Nl, T, d)
            L_att = self.attn_latent(
                self.ln_L_q(L_flat), self.ln_L_kv(L_flat), is_causal=True,
                kv_cache=kv_cache_l, q_position_ids=position_ids,
                kv_position_ids=position_ids, update_cache=update_cache,
            )
            L_out = L_flat + self.dropout(L_att)
            L_out = L_out + self.dropout(self.ffn_L(self.ln_L_ff(L_out)))
            L_out = L_out.view(B, Nl, T, d).transpose(1, 2)
        else:
            L_out = None

        return R_out, L_out


class DreamerV4Decoder(nn.Module):
    def __init__(self, image_size: Tuple[int, int], patch_size: int,
                 d_model: int, n_layers: int, num_heads_q: int,
                 num_heads_kv_latent: int, bottleneck_dim: int, seq_len: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 n_latents: int = 256, in_channels: int = 3,
                 temporal_every: int = 1, use_proprio: bool = False,
                 proprio_dim: int = 0, num_proprio: int = 1):
        super().__init__()
        H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.H, self.W = H, W
        self.P = patch_size
        self.Np = (H // patch_size) * (W // patch_size)
        self.Nl = n_latents
        self.d = d_model
        self.d_b = bottleneck_dim
        self.C = in_channels
        self.use_proprio = use_proprio
        self.num_proprio = num_proprio if use_proprio else 0

        if use_proprio:
            assert proprio_dim > 0
            self.proprio_dim = proprio_dim
            self.readout_proprio_token = nn.Parameter(
                torch.randn(num_proprio, d_model) / math.sqrt(d_model))
            self.proprio_head = nn.Linear(d_model, proprio_dim)

        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.readout_tokens = nn.Parameter(torch.randn(self.Np, d_model) / math.sqrt(d_model))

        layers = []
        for i in range(n_layers):
            temporal = (i % 2 == 1) if temporal_every == 1 else (((i + 1) % temporal_every) == 0)
            is_last = (i == n_layers - 1) or (
                i == n_layers - 2 and (n_layers % temporal_every) == 0
            )
            layers.append(BlockCausalDecoderLayer(
                d_model=d_model, num_heads_q=num_heads_q,
                num_heads_kv_latent=num_heads_kv_latent, seq_len=seq_len,
                num_latents=self.Nl, num_patches=self.Np,
                dropout=dropout, mlp_ratio=mlp_ratio,
                temporal=temporal, is_last=is_last,
                use_proprio=use_proprio, num_proprio=num_proprio,
            ))
        self.layers = nn.ModuleList(layers)

        self.patch_head = nn.Linear(d_model, (patch_size ** 2) * in_channels)

        self.caches_l: list = []
        self.caches_r: list = []

    def _build_readout(self, B: int, T: int) -> torch.Tensor:
        R0 = self.readout_tokens[None, None].expand(B, T, self.Np, self.d).contiguous()
        if self.use_proprio:
            R0_prop = self.readout_proprio_token[None, None].expand(
                B, T, self.num_proprio, self.d).contiguous()
            R0 = torch.cat([R0, R0_prop], dim=2)
        return R0

    def _decode_output(self, R_dec: torch.Tensor, B: int, T: int):
        R_img = R_dec[:, :, :self.Np, :]
        patches = self.patch_head(R_img).view(B, T, self.Np, self.P, self.P, self.C)
        Hp, Wp = self.H // self.P, self.W // self.P
        patches = patches.view(B, T, Hp, Wp, self.P, self.P, self.C).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x_hat = patches.view(B, T, self.H, self.W, self.C).permute(0, 1, 4, 2, 3).contiguous()
        if self.use_proprio:
            R_prop = R_dec[:, :, self.Np:, :]
            prop_hat = self.proprio_head(R_prop.squeeze(2))
            return R_dec, x_hat, prop_hat
        return R_dec, x_hat

    def forward(self, Z: torch.Tensor):
        B, T, Nl, d_b = Z.shape
        assert Nl == self.Nl and d_b == self.d_b
        L = self.up_proj(Z)
        R_dec, L_dec = self._build_readout(B, T), L
        # Compile boundary: prevents _build_readout fusion with layer-0 Triton
        # kernel, which under bf16 produces accumulation NaN. Zero overhead in
        # eager mode.
        torch._dynamo.graph_break()
        for layer in self.layers:
            R_dec, L_dec = layer(R_dec, L_dec)
        return self._decode_output(R_dec, B, T)

    def init_cache(self, batch_size: int, device: torch.device, max_seq_len: int):
        self.caches_l = []
        self.caches_r = []
        total_readout = self.Np + self.num_proprio
        for layer in self.layers:
            if layer.temporal:
                cache_l = (KVCache(
                    max_seq_len=max_seq_len,
                    batch_size=batch_size * self.Nl,
                    num_heads=layer.attn_latent.Hq,
                    head_dim=layer.attn_latent.Dh,
                    device=device, dtype=self.up_proj.weight.dtype)
                    if not layer.is_last else None)
                cache_r = KVCache(
                    max_seq_len=max_seq_len,
                    batch_size=batch_size * total_readout,
                    num_heads=layer.attn_patch.Hq,
                    head_dim=layer.attn_patch.Dh,
                    device=device, dtype=self.up_proj.weight.dtype)
                self.caches_l.append(cache_l)
                self.caches_r.append(cache_r)
            else:
                self.caches_l.append(None)
                self.caches_r.append(None)

    def forward_step(self, Z: torch.Tensor, start_step_idx: int,
                     update_cache: bool = True):
        B, T, _, _ = Z.shape
        L = self.up_proj(Z)
        R_dec, L_dec = self._build_readout(B, T), L
        pos_ids = torch.arange(start_step_idx, start_step_idx + T,
                               device=Z.device, dtype=torch.long).unsqueeze(0)
        for i, layer in enumerate(self.layers):
            if layer.temporal:
                R_dec, L_dec = layer(R_dec, L_dec,
                                     kv_cache_l=self.caches_l[i],
                                     kv_cache_r=self.caches_r[i],
                                     position_ids=pos_ids,
                                     update_cache=update_cache)
            else:
                R_dec, L_dec = layer(R_dec, L_dec)
        return self._decode_output(R_dec, B, T)


# ---------------------------------------------------------------------------
# Wrapper — same interface as TokenizerWrapper from `tokenizer.py`
# ---------------------------------------------------------------------------

class MultiStreamTokenizerWrapper(nn.Module):
    """Two-stream tokenizer wrapper; drop-in for `TokenizerWrapper`.

    Image convention is [0, 1] in / [0, 1] out (clamped on output) — matches
    the reference checkpoint and the single-stream wrapper's external contract.
    Internally the reference operates on raw [0, 1] (no [-1, 1] remap), so
    multi-stream wrappers do **not** apply the `*2 - 1` shift that the
    single-stream wrapper does.
    """

    def __init__(self, cfg: DictConfig, max_num_forward_steps: Optional[int] = None):
        super().__init__()
        self.cfg = cfg

        tcfg = cfg.tokenizer
        ms = tcfg.get("multi_stream", {}) or {}

        seq_len = max_num_forward_steps if max_num_forward_steps is not None else tcfg.max_sequence_length
        in_channels = int(ms.get("in_channels", 3))
        use_proprio = bool(ms.get("use_proprio", False))
        proprio_dim = int(ms.get("proprio_dim", 0))
        num_proprio = int(ms.get("num_proprio", 1))
        mlp_ratio = float(ms.get("mlp_ratio", 4.0))
        temporal_every = int(ms.get("temporal_every", 1))
        mae_max_mask_prob = float(ms.get("mae_max_mask_prob", 0.0))
        activate_masking = bool(ms.get("activate_masking", False))

        # Map shared field names.
        d_model = int(tcfg.model_dim)
        bottleneck_dim = int(tcfg.latent_dim)
        n_latents = int(tcfg.num_latent_tokens)
        n_heads_q = int(tcfg.n_heads)
        n_heads_kv = int(tcfg.get("n_kv_heads", tcfg.n_heads))
        dropout = float(tcfg.get("dropout_prob", 0.0))
        patch_size = int(tcfg.patch_size)

        # Image size from dataset config (matches single-stream wrapper).
        image_size = tuple(int(x) for x in cfg.dataset.resolution)

        self.use_proprio = use_proprio

        self.encoder = DreamerV4Encoder(
            image_size=image_size, patch_size=patch_size, d_model=d_model,
            n_layers=int(tcfg.enc_num_layers), num_heads_q=n_heads_q,
            num_heads_kv_latent=n_heads_kv, seq_len=seq_len,
            mlp_ratio=mlp_ratio, dropout=dropout, n_latents=n_latents,
            bottleneck_dim=bottleneck_dim, temporal_every=temporal_every,
            in_channels=in_channels, mae_max_mask_prob=mae_max_mask_prob,
            activate_masking=activate_masking, use_proprio=use_proprio,
            proprio_dim=proprio_dim,
        )
        self.decoder = DreamerV4Decoder(
            image_size=image_size, patch_size=patch_size, d_model=d_model,
            n_layers=int(tcfg.dec_num_layers), num_heads_q=n_heads_q,
            num_heads_kv_latent=n_heads_kv, bottleneck_dim=bottleneck_dim,
            seq_len=seq_len, mlp_ratio=mlp_ratio, dropout=dropout,
            n_latents=n_latents, in_channels=in_channels,
            temporal_every=temporal_every, use_proprio=use_proprio,
            proprio_dim=proprio_dim, num_proprio=num_proprio,
        )

    # --- Public interface (mirrors TokenizerWrapper) ----------------------

    def forward(self, images: torch.Tensor,
                proprio: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reconstruct images and (when use_proprio) proprio.

        Returns ``(recon_images, recon_proprio_or_None)``. The single-stream
        wrapper has the same signature; both arches can be called identically
        from training code.
        """
        return self._reconstruct(images, proprio)

    def _reconstruct(self, images: torch.Tensor,
                     proprio: Optional[torch.Tensor]
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        enc_out = self.encoder(images, proprio=proprio)
        Z = enc_out[2]
        dec_out = self.decoder(Z)
        if self.use_proprio:
            _, x_hat, prop_hat = dec_out
        else:
            _, x_hat = dec_out
            prop_hat = None
        recon = torch.clamp(x_hat, 0.0, 1.0)
        return recon, prop_hat

    def encode(self, images: torch.Tensor,
               proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(images, proprio=proprio)[2]

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        dec_out = self.decoder(latents)
        x_hat = dec_out[1]
        return torch.clamp(x_hat, 0.0, 1.0)

    def decode_step(self, x: torch.Tensor, start_step_idx: int,
                    update_cache: bool = True) -> torch.Tensor:
        dec_out = self.decoder.forward_step(x, start_step_idx, update_cache)
        x_hat = dec_out[1]
        return torch.clamp(x_hat, 0.0, 1.0)

    def init_cache(self, batch_size: int, context_length: int,
                   device: torch.device, dtype: torch.dtype):
        # `dtype` is consumed by the underlying decoder via its up_proj weight
        # dtype; not separately wired here, matching the reference's behavior.
        del dtype
        self.decoder.init_cache(batch_size, device, context_length)
