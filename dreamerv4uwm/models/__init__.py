"""Model package public surface.

`make_tokenizer` is the only place that decides between the single-stream
tokenizer (`tokenizer.py`) and the two-stream port from the reference impl
(`tokenizer_multi_stream.py`). All scripts and notebooks should call this
factory instead of constructing a wrapper class directly, so the choice is
purely config-driven.
"""

from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig


def make_tokenizer(cfg: DictConfig,
                   max_num_forward_steps: Optional[int] = None) -> nn.Module:
    """Construct the tokenizer wrapper selected by `cfg.tokenizer.architecture`.

    Defaults to ``"single_stream"`` when the field is missing, so older configs
    keep working unchanged. Both wrappers expose the same interface:
    ``forward``, ``encode``, ``decode``, ``decode_step``, ``init_cache``.
    """
    arch = cfg.tokenizer.get("architecture", "single_stream")
    if arch == "single_stream":
        from .tokenizer import TokenizerWrapper
        return TokenizerWrapper(cfg, max_num_forward_steps=max_num_forward_steps)
    if arch == "multi_stream":
        from .tokenizer_multi_stream import MultiStreamTokenizerWrapper
        return MultiStreamTokenizerWrapper(cfg, max_num_forward_steps=max_num_forward_steps)
    raise ValueError(
        f"Unknown tokenizer.architecture={arch!r}; expected 'single_stream' or 'multi_stream'."
    )


__all__ = ["make_tokenizer"]
