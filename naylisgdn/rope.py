import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE avec support du partial rotary factor (Qwen3.5 style).

    Si `rope_dim` < `dim`, la rotation n'est appliquée qu'aux premières
    `rope_dim` dimensions de q/k ; les dimensions restantes passent sans
    modification. Cela correspond au `partial_rotary_factor` de Qwen3.5
    (ex. rope_dim=64 sur head_dim=256 → factor=0.25).

    Si `rope_dim` est None ou égal à `dim`, comportement standard : toutes
    les dimensions sont rotées (compatibilité ascendante).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
        rope_base: int = 0,
        rope_dim: Optional[int] = None,
        device=None,
        use_yarn: bool = False,
        yarn_scale: float = 1.0,
        yarn_original_max_len: int = 1024,
    ):
        super().__init__()
        self.dim                   = dim
        self.max_seq_len           = max_seq_len
        self.base                  = rope_base if rope_base > 0 else base
        self.rope_dim              = rope_dim if rope_dim is not None else dim
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        assert self.rope_dim <= dim and self.rope_dim % 2 == 0, \
            f"rope_dim ({self.rope_dim}) doit être ≤ dim ({dim}) et pair"

        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim)
            )

        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self) -> torch.Tensor:
        freqs         = torch.arange(0, self.rope_dim, 2).float() / self.rope_dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        if self.yarn_scale == 1.0:
            return inv_freq_base
        alpha = self.yarn_scale
        beta  = max(self.rope_dim // 2, int(self.rope_dim * 0.25))
        dims  = torch.arange(0, self.rope_dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.rope_dim - beta),
        )
        return inv_freq_base / scale

    def _update_cos_sin_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len   = q.shape[2]
        total_len = seq_len + position_offset
        cos, sin  = self._update_cos_sin_cache(total_len, q.device, q.dtype)
        cos = cos[position_offset : position_offset + seq_len][None, None, :, :]
        sin = sin[position_offset : position_offset + seq_len][None, None, :, :]

        if self.rope_dim == self.dim:
            return (
                (q * cos) + (self._rotate_half(q) * sin),
                (k * cos) + (self._rotate_half(k) * sin),
            )

        q_rot  = q[..., : self.rope_dim]
        q_pass = q[..., self.rope_dim :]
        k_rot  = k[..., : self.rope_dim]
        k_pass = k[..., self.rope_dim :]

        q_out = torch.cat(
            [(q_rot * cos) + (self._rotate_half(q_rot) * sin), q_pass], dim=-1
        )
        k_out = torch.cat(
            [(k_rot * cos) + (self._rotate_half(k_rot) * sin), k_pass], dim=-1
        )
        return q_out, k_out

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_pos_emb(q, k, position_offset)
