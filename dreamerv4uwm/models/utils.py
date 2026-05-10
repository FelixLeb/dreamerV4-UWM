import torch
import torch.nn as nn
from omegaconf import DictConfig

from . import make_tokenizer
from .dynamics import DenoiserWrapper


@torch.no_grad()
def load_tokenizer(cfg: DictConfig, device: torch.device,
                   max_num_forward_steps=None, model_key: str = "model",
                   ckpt_format: str = None) -> nn.Module:
    """Load tokenizer (encoder+decoder) and heads from checkpoint.

    The architecture is selected by ``cfg.tokenizer.architecture`` via
    ``make_tokenizer``. ``ckpt_format`` is auto-detected from the checkpoint's
    top-level keys when not passed explicitly:

      - ``"native"``: ``state[model_key]`` is a flat wrapper-level state_dict
        (this project's training scripts save in this layout).
      - ``"reference"``: ``state["enc"]`` / ``state["dec"]`` (GR00T training
        layout). Only valid with ``cfg.tokenizer.architecture == "multi_stream"``.
    """
    tokenizer_wrapper = make_tokenizer(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.tokenizer_ckpt, map_location=device)

    if ckpt_format is None:
        if model_key in state:
            ckpt_format = "native"
        elif "enc" in state:
            ckpt_format = "reference"
        else:
            raise ValueError(
                f"cannot infer ckpt_format from {cfg.tokenizer_ckpt}: "
                f"top-level keys = {list(state.keys())}; expected "
                f"{model_key!r} (native) or 'enc'/'dec' (reference)."
            )

    if ckpt_format == "native":
        sd = state[model_key]
        tokenizer_wrapper.load_state_dict(sd, strict=True)
    elif ckpt_format == "reference":
        if cfg.tokenizer.get("architecture", "single_stream") != "multi_stream":
            raise ValueError(
                "ckpt_format='reference' requires cfg.tokenizer.architecture='multi_stream'."
            )
        # Reference checkpoints are saved from a `torch.compile`-wrapped module,
        # which prefixes every key with `_orig_mod.`. We load into the bare
        # encoder/decoder, so strip the prefix.
        def _strip(sd):
            return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

        tokenizer_wrapper.encoder.load_state_dict(_strip(state["enc"]), strict=True)
        if "dec" in state:
            tokenizer_wrapper.decoder.load_state_dict(_strip(state["dec"]), strict=True)
        else:
            print("[load_tokenizer] WARNING: 'dec' missing from reference checkpoint")
    else:
        raise ValueError(f"Unknown ckpt_format={ckpt_format!r}")

    return tokenizer_wrapper


@torch.no_grad()
def load_denoiser(cfg: DictConfig, device: torch.device,
                  max_num_forward_steps=None, model_key: str = "model",
                  strict: bool = True) -> nn.Module:
    """Load DreamerV4 denoiser from checkpoint."""
    denoiser = DenoiserWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.dynamics_ckpt, map_location=device)
    sd = state[model_key]
    denoiser.load_state_dict(sd, strict=strict)
    return denoiser
