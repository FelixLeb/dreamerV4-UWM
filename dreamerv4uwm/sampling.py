import math 
import numpy as np
import torch

def get_noise_index(noise_level, num_noise_levels):
    return int(np.clip(noise_level, 0., (num_noise_levels-1)/num_noise_levels)*num_noise_levels)

def get_step_index(step_length, num_noise_levels):
    num_steps = int(1./step_length)
    max_pow2 = int(math.log2(num_noise_levels))
    step_index = max_pow2 - int(math.log2(num_steps)) # Convention adopted in my loss
    return step_index

@torch.no_grad
def forward_dynamics_no_cache(
    denoiser,
    ctx_latents,
    actions=None,
    num_pred_steps=1,
    num_diffusion_steps=4,
    context_cond_tau=0.9,
    ):
    
    if actions is not None:
        assert actions.shape[1] == num_pred_steps + ctx_latents.shape[1], 'You should have one action per each context and prediction frames'
        actions = actions.to(device=ctx_latents.device, dtype=ctx_latents.dtype)
    
    B, _, N_lat, D_lat = ctx_latents.shape
    num_context_frames = ctx_latents.shape[1]

    # 1. Initialize pure noise at τ=0
    z = torch.randn(
        B,
        num_pred_steps+num_context_frames,
        N_lat,
        D_lat,
        device=ctx_latents.device,
        dtype=ctx_latents.dtype,
    )
    # Add a slight noise to the context tokens for robustness reasons according to the paper
    latents_cond = ctx_latents.clone()
    latents_cond = (1.0 - context_cond_tau) * torch.randn_like(latents_cond).to(latents_cond.device) + context_cond_tau * latents_cond
    z[:, :num_context_frames, ...] = latents_cond
    
    # Compute the discrete step index  
    step_size = 1.0 / num_diffusion_steps
    denoising_step_index = get_step_index(step_size, denoiser.cfg.denoiser.num_noise_levels)
    step_index_tensor = torch.full(
        (B, num_context_frames+num_pred_steps),
        denoising_step_index,
        dtype=torch.long,
        device=ctx_latents.device,
    )
    
    # Compute the discrete noise level for the context frames with slight noise added on them
    tau_cond_idx = get_noise_index(context_cond_tau, denoiser.cfg.denoiser.num_noise_levels) 

    # Start the shortcut denoising process
    for k in range(num_diffusion_steps):
        
        tau_current = k/num_diffusion_steps # Compute the noise level of the current step (parameter tau in the paper)
        tau_current_idx = get_noise_index(tau_current, denoiser.cfg.denoiser.num_noise_levels)
        tau_index_tensor = torch.full(
            (B, num_context_frames+num_pred_steps),
            tau_current_idx,
            dtype=torch.long,
            device=ctx_latents.device,
        )
        # ste the proper noise level for the context frames 
        tau_index_tensor[:,:num_context_frames] = tau_cond_idx 

        # Denoising
        z_hat = denoiser(
            noisy_z=z,
            action=actions,
            sigma_idx=tau_index_tensor,
            step_idx=step_index_tensor,
        )
        # v = (z_1 - z_τ) / (1 - τ)
        velocity = (z_hat - z) / (1.0 - tau_current)
        # Note: We only apply the denoising process on the future frames
        z[:,num_context_frames:] = z[:,num_context_frames:] + (velocity * step_size)[:,num_context_frames:]
    
    # return torch.cat([latents[:, :num_context_frames, ...], z_hat[:, num_context_frames:]], dim=-3) # z_hat is the output of the last denoiser step which is the predicted clean latents
    # z_hat is the output of the last denoiser step which is the predicted clean latents
    return z_hat[:, num_context_frames:] 

@torch.no_grad()
def forward_dynamics_flowmatching_no_cache(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    ctx_actions,              # (B, T_ctx, n_act)
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
):
    """
    Euler sampler for the plain flowmatching denoiser (no shortcut).

    step_idx is always 0 (finest step = d_min), matching compute_flowmatching_loss.
    Both obs latents and actions are jointly denoised.

    Returns
    -------
    pred_obs  : (B, num_pred_steps, N_lat, D_lat)
    pred_act  : (B, num_pred_steps, n_act)
    """
    device = ctx_latents.device
    dtype  = ctx_latents.dtype

    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    n_act   = ctx_actions.shape[-1]
    T_total = T_ctx + num_pred_steps

    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    # flowmatching uses only the finest step (step_idx == 0 == d_min)
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # 1) Initialize pure noise for the full sequence
    z     = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    z_act = torch.randn(B, T_total, n_act,         device=device, dtype=dtype)

    # 2) Replace context frames with slightly noised clean tokens
    z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
    a_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions
    z    [:, :T_ctx] = z_ctx
    z_act[:, :T_ctx] = a_ctx

    # discrete tau index for context frames
    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)

    # stride is exact because both num_noise_levels and num_diffusion_steps are powers of 2
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        tau_index_tensor = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        tau_index_tensor[:, :T_ctx] = tau_cond_idx   # context frames stay at tau_c

        z_hat, act_hat, _ = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= tau_index_tensor,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= tau_index_tensor,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)   # (B, T, n_act)

        denom = max(1.0 - tau_current, 1e-5)
        v_obs = (z_hat  - z    ) / denom
        v_act = (act_hat - z_act) / denom

        # integrate only future frames
        z    [:, T_ctx:] = z    [:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]
        z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    return z[:, T_ctx:], z_act[:, T_ctx:]


@torch.no_grad()
def unified_flowmatching_sampler(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    mode='wm',                # 'policy', 'video', 'wm', 'id', 'forcing'
    ctx_actions=None,         # (B, T_ctx, n_act) — required for policy, wm, id, forcing
    all_actions=None,         # (B, T_ctx + num_pred_steps, n_act) — alternative: pass all actions at once
    all_latents=None,         # (B, T_ctx + num_pred_steps, N_lat, D_lat) — required for 'id' mode
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
    action_noise_std=1.0,
):
    """
    Unified Euler sampler for all UWM training modes (no shortcut, no cache).

    Modes match the training loss in UWMForwardProcess / compute_uwm_loss:

      policy  — context: clean obs + clean act; denoise future obs + future act
      video   — context: clean obs, no actions; denoise future obs only
      wm      — context: clean obs + clean act; denoise future obs with all actions clean
      id      — all obs clean; denoise actions across all frames
      forcing — context obs/act conditioned; denoise future obs + future act

    Returns
    -------
    pred_obs : (B, num_pred_steps, N_lat, D_lat) or None
    pred_act : (B, num_pred_steps, n_act)         or None
        None is returned for the modality that is not denoised in a given mode.
    """
    assert mode in ('policy', 'video', 'wm', 'id', 'forcing'), \
        f"Unknown mode '{mode}'"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    device = ctx_latents.device
    dtype  = ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    # --- resolve actions tensor ---
    if mode == 'video':
        # video mode: no action signal at all
        n_act = ctx_actions.shape[-1] if ctx_actions is not None else (
            all_actions.shape[-1] if all_actions is not None else 1
        )
    elif mode == 'wm':
        # wm needs actions for ALL frames (context + future)
        assert all_actions is not None, "wm mode requires all_actions (B, T_total, n_act)"
        n_act = all_actions.shape[-1]
    elif mode == 'id':
        # id needs all obs clean
        assert all_latents is not None, "id mode requires all_latents (B, T_total, N_lat, D_lat)"
        if all_actions is not None:
            n_act = all_actions.shape[-1]
        elif ctx_actions is not None:
            n_act = ctx_actions.shape[-1]
        else:
            raise ValueError("id mode requires ctx_actions or all_actions to determine n_act")
    else:
        # policy, forcing
        assert ctx_actions is not None, f"{mode} mode requires ctx_actions"
        n_act = ctx_actions.shape[-1]

    # ====================================================================
    # Initialize obs latents
    # ====================================================================
    denoise_obs = mode != 'id'

    if mode == 'id':
        # obs are fully clean everywhere
        assert all_latents.shape == (B, T_total, N_lat, D_lat)
        z = all_latents.clone().to(device=device, dtype=dtype)
        # set obs sigma to "clean" for all frames
        obs_clean_idx = num_noise_levels - 1
    else:
        z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
        z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
        z[:, :T_ctx] = z_ctx

    # ====================================================================
    # Initialize actions
    # ====================================================================
    denoise_act = mode in ('policy', 'id', 'forcing')

    if mode == 'wm':
        # actions are fully clean for all frames
        z_act = all_actions.to(device=device, dtype=dtype)
        act_clean_idx = num_noise_levels - 1
    elif mode == 'video':
        # actions are pure noise everywhere (never denoised, never conditioned)
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    elif mode in ('policy', 'forcing'):
        # context actions are slightly noised clean; future actions start as noise
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
        a_ctx = (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions.to(device=device, dtype=dtype)
        z_act[:, :T_ctx] = a_ctx
    elif mode == 'id':
        # actions start as noise everywhere
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)

    # ====================================================================
    # Euler integration loop
    # ====================================================================
    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        # --- obs sigma indices ---
        if mode == 'id':
            obs_sigma_idx = torch.full((B, T_total), obs_clean_idx, dtype=torch.long, device=device)
        else:
            obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            obs_sigma_idx[:, :T_ctx] = tau_cond_idx

        # --- act sigma indices ---
        if mode == 'wm':
            act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)
        elif mode == 'video':
            # actions are pure noise (tau=0)
            act_sigma_idx = torch.zeros((B, T_total), dtype=torch.long, device=device)
        elif mode == 'id':
            # denoising actions across all frames at same noise level
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        else:
            # policy, forcing: context actions conditioned, future denoised
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            act_sigma_idx[:, :T_ctx] = tau_cond_idx

        z_hat, act_hat, _ = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)  # (B, T, n_act)

        denom = max(1.0 - tau_current, 1e-5)

        # --- integrate obs ---
        if denoise_obs:
            v_obs = (z_hat - z) / denom
            z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

        # --- integrate act ---
        if denoise_act:
            v_act = (act_hat - z_act) / denom
            if mode == 'id':
                # denoise actions for ALL frames
                z_act = z_act + v_act * step_size
            else:
                # denoise only future actions
                z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    # ====================================================================
    # Return results
    # ====================================================================
    out_obs = z[:, T_ctx:] if denoise_obs else None
    out_act = z_act[:, T_ctx:] if denoise_act else None

    # For modes that don't denoise a modality but still have meaningful output
    if mode == 'id':
        out_obs = None   # obs were given as input, not predicted
    if mode in ('video', 'wm'):
        out_act = None   # actions were not predicted

    return out_obs, out_act


@torch.no_grad()
def unified_action_sampler(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat) — context obs latents
    ctx_actions,              # (B, T_ctx, n_act) — only used to infer n_act and dtype/device
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    ctx_obs_init_tau=0.5,     # tau for context obs at start (0=pure noise, 1=clean)
    action_noise_std=1.0,
):
    """
    Action-prediction Euler sampler.

    Setup:
      - context: obs AND actions both start at noise level `ctx_obs_init_tau`
        (context actions are partially noised versions of `ctx_actions`) and
        are denoised together on the context schedule.
      - horizon: obs and actions both start as pure noise (tau = 0) and are
        denoised together on the horizon schedule.
      - The two regions have independent per-step tau and Euler step sizes,
        so each reaches the clean signal in `num_diffusion_steps` steps from
        its own starting noise level.

    Returns
    -------
    pred_obs : (B, T_total, N_lat, D_lat) — full denoised obs (ctx + horizon)
    pred_act : (B, T_total, n_act)         — full denoised actions
    """
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"
    assert 0.0 <= ctx_obs_init_tau < 1.0, \
        "ctx_obs_init_tau must be in [0, 1)"

    device = ctx_latents.device
    dtype  = ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps
    n_act = ctx_actions.shape[-1]
    ctx_actions = ctx_actions.to(device=device, dtype=dtype)

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # ---- obs init ----
    # horizon: pure noise (tau = 0)
    z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    # context: partially noised clean latents at tau = ctx_obs_init_tau
    # z_ctx = (1.0 - ctx_obs_init_tau) * torch.randn_like(ctx_latents) + ctx_obs_init_tau * ctx_latents
    # z[:, :T_ctx] = z_ctx
    z[:, :T_ctx] = ctx_latents

    # ---- act init ----
    # horizon: pure noise (tau = 0)
    z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    # context: partially noised clean actions at tau = ctx_obs_init_tau (same as ctx obs)
    a_ctx = (1.0 - ctx_obs_init_tau) * action_noise_std * torch.randn_like(ctx_actions) + ctx_obs_init_tau * ctx_actions
    z_act[:, :T_ctx] = a_ctx

    # Per-region step sizes (each region traverses its own tau range in N steps)
    horizon_step_size = 1.0 / num_diffusion_steps
    ctx_step_size     = (1.0 - ctx_obs_init_tau) / num_diffusion_steps

    for k in range(num_diffusion_steps):
        # Per-region current tau (obs and actions share the same per-region schedule)
        tau_horizon = k / num_diffusion_steps
        tau_ctx     = ctx_obs_init_tau + (1.0 - ctx_obs_init_tau) * (k / num_diffusion_steps)

        tau_horizon_idx = get_noise_index(tau_horizon, num_noise_levels)
        tau_ctx_idx     = get_noise_index(tau_ctx,     num_noise_levels)

        obs_sigma_idx = torch.full((B, T_total), tau_horizon_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, :T_ctx] = tau_ctx_idx

        act_sigma_idx = torch.full((B, T_total), tau_horizon_idx, dtype=torch.long, device=device)
        act_sigma_idx[:, :T_ctx] = tau_ctx_idx

        z_hat, act_hat, _ = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)

        # Independent Euler integration: context region (obs+act) and horizon region (obs+act)
        denom_horizon = max(1.0 - tau_horizon, 1e-5)
        denom_ctx     = max(1.0 - tau_ctx,     1e-5)

        v_obs_horizon = (z_hat[:, T_ctx:] - z[:, T_ctx:]) / denom_horizon
        v_obs_ctx     = (z_hat[:, :T_ctx] - z[:, :T_ctx]) / denom_ctx
        v_act_horizon = (act_hat[:, T_ctx:] - z_act[:, T_ctx:]) / denom_horizon
        v_act_ctx     = (act_hat[:, :T_ctx] - z_act[:, :T_ctx]) / denom_ctx

        z    [:, T_ctx:] = z    [:, T_ctx:] + v_obs_horizon * horizon_step_size
        z    [:, :T_ctx] = z    [:, :T_ctx] + v_obs_ctx     * ctx_step_size
        z_act[:, T_ctx:] = z_act[:, T_ctx:] + v_act_horizon * horizon_step_size
        z_act[:, :T_ctx] = z_act[:, :T_ctx] + v_act_ctx     * ctx_step_size

    return z, z_act


@torch.no_grad()
def unified_video_sampler(
    denoiser,
    noisy_latents,             # (B, T_total, N_lat, D_lat) — unified noisy sequence (ctx + horizon)
    current_tau,               # scalar in [0, 1): current noise level of noisy_latents
    n_act,                     # action dimension
    num_diffusion_steps=4,     # power of two; traverses (1 - current_tau) in this many Euler steps
    action_noise_std=1.0,
    use_shortcut=False,        # if True, use shortcut step_idx; else flowmatching (step_idx=0)
):
    """
    Video-mode Euler sampler.

    Treats the entire sequence (context + horizon) as one unified latent at a
    single current noise level `current_tau`. Denoising is applied uniformly
    to every frame — no context/horizon split, no partial conditioning.
    Actions are pure noise for all frames throughout the trajectory, and the
    action noise-level conditioning signal is set to "pure noise" (tau = 0).

    Returns
    -------
    pred_latents : (B, T_total, N_lat, D_lat) — denoised video latents
    """
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"
    assert 0.0 <= current_tau < 1.0, "current_tau must be in [0, 1)"

    device = noisy_latents.device
    dtype  = noisy_latents.dtype
    B, T_total = noisy_latents.shape[:2]

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    step_size = (1.0 - current_tau) / num_diffusion_steps
    if use_shortcut:
        denoising_step_index = get_step_index(1.0 / num_diffusion_steps, num_noise_levels)
    else:
        denoising_step_index = 0
    step_index_tensor = torch.full(
        (B, T_total), denoising_step_index, dtype=torch.long, device=device
    )

    z = noisy_latents.clone().to(device=device, dtype=dtype)

    # Actions: pure noise everywhere, with tau=0 conditioning signal
    z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    act_sigma_idx = torch.zeros((B, T_total), dtype=torch.long, device=device)

    for k in range(num_diffusion_steps):
        tau_k = current_tau + (1.0 - current_tau) * (k / num_diffusion_steps)
        tau_k_idx = get_noise_index(tau_k, num_noise_levels)

        obs_sigma_idx = torch.full((B, T_total), tau_k_idx, dtype=torch.long, device=device)

        z_hat, _, _ = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )

        denom = max(1.0 - tau_k, 1e-5)
        v_obs = (z_hat - z) / denom
        z = z + v_obs * step_size

    return z


@torch.no_grad()
def worldmodel_dynamics_flowmatching_no_cache(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)  — context obs frames
    all_actions,              # (B, T_ctx + num_pred_steps, n_act) — ALL actions, clean
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=0.9,
):
    """
    World-model Euler sampler for the plain flowmatching denoiser (no shortcut).

    Actions are treated as clean conditions for all frames (context + future).
    Only obs latents are denoised: context frames are conditioned with slight noise,
    future frames start as pure noise and are integrated toward the clean signal.

    Returns
    -------
    pred_obs : (B, num_pred_steps, N_lat, D_lat)
    """
    device = ctx_latents.device
    dtype  = ctx_latents.dtype

    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    assert all_actions.shape == (B, T_total, all_actions.shape[-1]), \
        "all_actions must be (B, T_ctx + num_pred_steps, n_act)"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    # flowmatching uses only the finest step for both modalities
    step_index_tensor = torch.zeros((B, T_total), dtype=torch.long, device=device)

    # actions are fully clean — highest tau index (training convention: tau = (N-1)/N)
    act_clean_idx = num_noise_levels - 1
    act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)

    # 1) Initialize obs: context = slightly noised clean, future = pure noise
    z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
    z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
    z[:, :T_ctx] = z_ctx

    # discrete tau index for context obs frames
    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)

    # stride is exact because both num_noise_levels and num_diffusion_steps are powers of 2
    stride    = num_noise_levels // num_diffusion_steps
    step_size = 1.0 / num_diffusion_steps

    # clean actions passed as-is for all frames (no integration needed)
    clean_act = all_actions.to(device=device, dtype=dtype)

    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        obs_sigma_idx[:, :T_ctx] = tau_cond_idx   # context obs stays at tau_c

        z_hat, _, _ = denoiser(
            noisy_act    = clean_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )

        denom = max(1.0 - tau_current, 1e-5)
        v_obs = (z_hat - z) / denom

        # integrate only future obs frames; context and actions are fixed
        z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

    return z[:, T_ctx:]


@torch.no_grad()
def unified_shortcut_sampler(
    denoiser,
    ctx_latents,              # (B, T_ctx, N_lat, D_lat)
    mode='wm',                # 'policy', 'video', 'wm', 'id', 'forcing'
    ctx_actions=None,         # (B, T_ctx, n_act) — required for policy, wm, id, forcing
    all_actions=None,         # (B, T_ctx + num_pred_steps, n_act)
    all_latents=None,         # (B, T_ctx + num_pred_steps, N_lat, D_lat) — required for 'id' mode
    num_pred_steps=1,
    num_diffusion_steps=4,    # power of two
    context_cond_tau=127/128,
    action_noise_std=1.0,
):
    """
    Unified Euler sampler for all UWM training modes with the shortcut
    denoiser. Behaviourally identical to `unified_flowmatching_sampler`,
    except that `step_idx` is set to
    `get_step_index(1/num_diffusion_steps, num_noise_levels)` instead of 0,
    matching compute_bootstrap_uwm_loss.
    """
    assert mode in ('policy', 'video', 'wm', 'id', 'forcing'), \
        f"Unknown mode '{mode}'"
    assert (num_diffusion_steps & (num_diffusion_steps - 1)) == 0, \
        "num_diffusion_steps must be a power of two"

    device = ctx_latents.device
    dtype  = ctx_latents.dtype
    B, T_ctx, N_lat, D_lat = ctx_latents.shape
    T_total = T_ctx + num_pred_steps

    num_noise_levels = denoiser.cfg.denoiser.num_noise_levels

    step_size = 1.0 / num_diffusion_steps
    denoising_step_index = get_step_index(step_size, num_noise_levels)
    step_index_tensor = torch.full(
        (B, T_total), denoising_step_index, dtype=torch.long, device=device
    )

    tau_cond_idx = get_noise_index(context_cond_tau, num_noise_levels)
    stride = num_noise_levels // num_diffusion_steps

    # --- resolve actions tensor ---
    if mode == 'video':
        n_act = ctx_actions.shape[-1] if ctx_actions is not None else (
            all_actions.shape[-1] if all_actions is not None else 1
        )
    elif mode == 'wm':
        assert all_actions is not None, "wm mode requires all_actions (B, T_total, n_act)"
        n_act = all_actions.shape[-1]
    elif mode == 'id':
        assert all_latents is not None, "id mode requires all_latents (B, T_total, N_lat, D_lat)"
        if all_actions is not None:
            n_act = all_actions.shape[-1]
        elif ctx_actions is not None:
            n_act = ctx_actions.shape[-1]
        else:
            raise ValueError("id mode requires ctx_actions or all_actions to determine n_act")
    else:
        assert ctx_actions is not None, f"{mode} mode requires ctx_actions"
        n_act = ctx_actions.shape[-1]

    # ---- obs init ----
    denoise_obs = mode != 'id'

    if mode == 'id':
        assert all_latents.shape == (B, T_total, N_lat, D_lat)
        z = all_latents.clone().to(device=device, dtype=dtype)
        obs_clean_idx = num_noise_levels - 1
    else:
        z = torch.randn(B, T_total, N_lat, D_lat, device=device, dtype=dtype)
        z_ctx = (1.0 - context_cond_tau) * torch.randn_like(ctx_latents) + context_cond_tau * ctx_latents
        z[:, :T_ctx] = z_ctx

    # ---- act init ----
    denoise_act = mode in ('policy', 'id', 'forcing')

    if mode == 'wm':
        z_act = all_actions.to(device=device, dtype=dtype)
        act_clean_idx = num_noise_levels - 1
    elif mode == 'video':
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
    elif mode in ('policy', 'forcing'):
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)
        a_ctx = (1.0 - context_cond_tau) * action_noise_std * torch.randn_like(ctx_actions) + context_cond_tau * ctx_actions.to(device=device, dtype=dtype)
        z_act[:, :T_ctx] = a_ctx
    elif mode == 'id':
        z_act = action_noise_std * torch.randn(B, T_total, n_act, device=device, dtype=dtype)

    # ---- Euler loop ----
    for k in range(num_diffusion_steps):
        tau_current_idx = k * stride
        tau_current     = tau_current_idx / float(num_noise_levels)

        if mode == 'id':
            obs_sigma_idx = torch.full((B, T_total), obs_clean_idx, dtype=torch.long, device=device)
        else:
            obs_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            obs_sigma_idx[:, :T_ctx] = tau_cond_idx

        if mode == 'wm':
            act_sigma_idx = torch.full((B, T_total), act_clean_idx, dtype=torch.long, device=device)
        elif mode == 'video':
            act_sigma_idx = torch.zeros((B, T_total), dtype=torch.long, device=device)
        elif mode == 'id':
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
        else:
            act_sigma_idx = torch.full((B, T_total), tau_current_idx, dtype=torch.long, device=device)
            act_sigma_idx[:, :T_ctx] = tau_cond_idx

        z_hat, act_hat, _ = denoiser(
            noisy_act    = z_act,
            noisy_obs    = z,
            obs_sigma_idx= obs_sigma_idx,
            obs_step_idx = step_index_tensor,
            act_sigma_idx= act_sigma_idx,
            act_step_idx = step_index_tensor,
        )
        act_hat = act_hat.squeeze(-2)

        denom = max(1.0 - tau_current, 1e-5)

        if denoise_obs:
            v_obs = (z_hat - z) / denom
            z[:, T_ctx:] = z[:, T_ctx:] + (v_obs * step_size)[:, T_ctx:]

        if denoise_act:
            v_act = (act_hat - z_act) / denom
            if mode == 'id':
                z_act = z_act + v_act * step_size
            else:
                z_act[:, T_ctx:] = z_act[:, T_ctx:] + (v_act * step_size)[:, T_ctx:]

    out_obs = z[:, T_ctx:] if denoise_obs else None
    out_act = z_act[:, T_ctx:] if denoise_act else None

    if mode == 'id':
        out_obs = None
    if mode in ('video', 'wm'):
        out_act = None

    return out_obs, out_act



class AutoRegressiveForwardDynamics:
    """KV-cached autoregressive sampler for the UWM two-stream flowmatching denoiser.

    Modes
    -----
    'wm'     : world-model. The caller supplies the next action at each `step()`;
               the action is treated as clean (sigma = clean_idx) and only the
               next observation is denoised. Context actions are also clean.
    'policy' : policy. `step()` takes no input; the next observation and the
               next action are denoised jointly. Context actions are slightly
               noised at the same `tau_cond` as context observations to match
               policy-mode training.

    Only the flow-matching denoiser path is supported here: `step_idx` is held at
    0 (finest step, ↔ d_min) throughout.
    """

    def __init__(self,
                 denoiser,
                 tokenizer,
                 mode='wm',
                 context_length=32,
                 max_forward_steps=5000,
                 context_cond_tau=0.9,
                 denoising_step_count=4,
                 device="cuda",
                 dtype=torch.float32,
                 cond_class=None):
        assert mode in ('wm', 'policy'), f"mode must be 'wm' or 'policy', got '{mode}'"

        self.denoiser = denoiser
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.context_length = context_length
        self.context_cond_tau = context_cond_tau
        self.denoising_step_count = denoising_step_count
        self.max_forward_steps = max_forward_steps
        self.current_frame_index = 0
        # AdaLN class conditioning: an int class index (or None = unconditioned,
        # bit-equivalent to a non-conditioned model). Materialized into a (B,)
        # tensor in reset(); passed to every denoiser.forward_step call.
        self.cond_class = cond_class
        self.cond_class_t = None

        N = denoiser.cfg.denoiser.num_noise_levels
        assert (denoising_step_count & (denoising_step_count - 1)) == 0, \
            "denoising_step_count must be a power of two"
        assert N % denoising_step_count == 0, \
            "num_noise_levels must be a multiple of denoising_step_count"
        self.num_noise_levels = N
        self.cond_tau_idx = get_noise_index(context_cond_tau, N)
        self.clean_idx = N - 1
        self.flow_step_idx = 0  # flowmatching: always finest step (d_min)
        self.n_act = denoiser.cfg.denoiser.n_actions

    # ------------------------------------------------------------------
    # context priming
    # ------------------------------------------------------------------
    @torch.no_grad
    def reset(self, imgs_init, actions_init):
        """Prime the denoiser + tokenizer KV caches with the context window.

        Args
        ----
        imgs_init    : (B, T_ctx, C, H, W)
        actions_init : (B, T_ctx, n_act) — required for both modes.
        """
        assert actions_init is not None, "actions_init is required for both wm and policy modes"

        self.current_frame_index = 0
        actions_ctx = actions_init.to(device=self.device, dtype=self.dtype)

        latents = self.tokenizer.encode(imgs_init)  # (B, T_ctx, N_lat, D_lat)
        B, T_ctx = latents.shape[:2]
        assert actions_ctx.shape[:2] == (B, T_ctx), \
            f"actions_init shape {tuple(actions_ctx.shape)} inconsistent with imgs_init T_ctx={T_ctx}"

        self.current_z = latents[:, -1:].clone()
        self.current_act = actions_ctx[:, -1:].clone()

        # Build the (B,) class-conditioning tensor once per rollout.
        self.cond_class_t = (
            None if self.cond_class is None
            else torch.full((B,), int(self.cond_class), dtype=torch.long, device=self.device)
        )

        # ---- tokenizer decoder cache: prime with clean context latents ----
        self.tokenizer.init_cache(batch_size=B,
                                  context_length=self.context_length,
                                  device=self.device,
                                  dtype=self.dtype)
        self.tokenizer.decode_step(latents.clone(),
                                   start_step_idx=0,
                                   update_cache=True)

        # ---- denoiser cache: prime with slightly-noised obs (and noised actions in policy mode) ----
        self.denoiser.init_cache(batch_size=B,
                                 context_length=self.context_length,
                                 device=self.device,
                                 dtype=self.dtype)

        latents_cond = (1.0 - self.context_cond_tau) * torch.randn_like(latents) \
                       + self.context_cond_tau * latents

        if self.mode == 'policy':
            actions_cond = (1.0 - self.context_cond_tau) * torch.randn_like(actions_ctx) \
                           + self.context_cond_tau * actions_ctx
            act_sigma_val = self.cond_tau_idx
        else:  # 'wm'
            actions_cond = actions_ctx
            act_sigma_val = self.clean_idx

        obs_sigma_idx = torch.full((B, T_ctx), self.cond_tau_idx, dtype=torch.long, device=self.device)
        act_sigma_idx = torch.full((B, T_ctx), act_sigma_val, dtype=torch.long, device=self.device)
        step_idx = torch.full((B, T_ctx), self.flow_step_idx, dtype=torch.long, device=self.device)

        self.denoiser.forward_step(
            noisy_act=actions_cond,
            noisy_obs=latents_cond,
            obs_sigma_idx=obs_sigma_idx,
            obs_step_idx=step_idx,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx,
            start_step_idx=0,
            update_cache=True,
            cond_class=self.cond_class_t,
        )

        self.current_frame_index = T_ctx

    # ------------------------------------------------------------------
    # one autoregressive step
    # ------------------------------------------------------------------
    @torch.no_grad
    def step(self, actions_t=None):
        """Sample one frame.

        wm mode    : `actions_t` (shape (B, n_act) or (B, 1, n_act)) is required.
                     Returns the predicted image (B, C, H, W), or
                     `(image, rewards)` when the denoiser has a reward head, where
                     `rewards` is (B, L) decoded scalar rewards from the final
                     denoising step's agent token (L = MTP horizon).
        policy mode: `actions_t` must be None.
                     Returns `(image, action)`, or `(image, action, rewards)` when
                     the denoiser has a reward head.
        """
        if self.mode == 'wm':
            assert actions_t is not None, "wm mode requires `actions_t`"
            return self._step_wm(actions_t)
        assert actions_t is None, "policy mode does not accept `actions_t`"
        return self._step_policy()

    def _decode_reward_logits(self, pred_rewards):
        """(B, T, L, K) logits → (B, T, L) scalar rewards in real space."""
        head0 = self.denoiser.model.reward_head.heads[0]
        buckets = head0.buckets.to(device=pred_rewards.device, dtype=torch.float32)
        probs = torch.softmax(pred_rewards.float(), dim=-1)
        symlog_pred = (probs * buckets).sum(dim=-1)
        return torch.sign(symlog_pred) * (torch.exp(torch.abs(symlog_pred)) - 1.0)

    def _step_wm(self, actions_t):
        actions_t = actions_t.to(device=self.device, dtype=self.dtype)
        if actions_t.ndim == 2:
            actions_t = actions_t.unsqueeze(1)  # (B, 1, n_act)

        B, _, N_lat, D_lat = self.current_z.shape
        N = self.num_noise_levels
        stride = N // self.denoising_step_count
        step_size = 1.0 / self.denoising_step_count

        z_obs = torch.randn(B, 1, N_lat, D_lat, device=self.device, dtype=self.dtype)
        step_idx = torch.full((B, 1), self.flow_step_idx, dtype=torch.long, device=self.device)
        act_sigma_idx = torch.full((B, 1), self.clean_idx, dtype=torch.long, device=self.device)

        last_pred_rewards = None
        for k in range(self.denoising_step_count):
            tau_idx = k * stride
            tau = tau_idx / float(N)
            obs_sigma_idx = torch.full((B, 1), tau_idx, dtype=torch.long, device=self.device)

            obs_pred, _, pred_rewards = self.denoiser.forward_step(
                noisy_act=actions_t,
                noisy_obs=z_obs,
                obs_sigma_idx=obs_sigma_idx,
                obs_step_idx=step_idx,
                act_sigma_idx=act_sigma_idx,
                act_step_idx=step_idx,
                start_step_idx=self.current_frame_index,
                update_cache=False,
                cond_class=self.cond_class_t,
            )
            denom = max(1.0 - tau, 1e-5)
            v_obs = (obs_pred - z_obs) / denom
            z_obs = z_obs + v_obs * step_size
            if k == self.denoising_step_count - 1:
                last_pred_rewards = pred_rewards

        # commit cache: noise predicted obs to tau_cond; action stays clean
        cor_z = (1.0 - self.context_cond_tau) * torch.randn_like(z_obs) + self.context_cond_tau * z_obs
        commit_obs_sigma = torch.full((B, 1), self.cond_tau_idx, dtype=torch.long, device=self.device)

        self.denoiser.forward_step(
            noisy_act=actions_t,
            noisy_obs=cor_z,
            obs_sigma_idx=commit_obs_sigma,
            obs_step_idx=step_idx,
            act_sigma_idx=act_sigma_idx,
            act_step_idx=step_idx,
            start_step_idx=self.current_frame_index,
            update_cache=True,
            cond_class=self.cond_class_t,
        )

        imgs_recon = self.tokenizer.decode_step(z_obs,
                                                start_step_idx=self.current_frame_index,
                                                update_cache=True)

        self.current_z = z_obs.clone()
        self.current_act = actions_t.clone()
        self.current_frame_index += 1

        if last_pred_rewards is not None:
            rewards = self._decode_reward_logits(last_pred_rewards)[:, 0]  # (B, L)
            return imgs_recon[:, 0, ...], rewards
        return imgs_recon[:, 0, ...]

    def _step_policy(self):
        B, _, N_lat, D_lat = self.current_z.shape
        N = self.num_noise_levels
        stride = N // self.denoising_step_count
        step_size = 1.0 / self.denoising_step_count

        z_obs = torch.randn(B, 1, N_lat, D_lat, device=self.device, dtype=self.dtype)
        z_act = torch.randn(B, 1, self.n_act, device=self.device, dtype=self.dtype)
        step_idx = torch.full((B, 1), self.flow_step_idx, dtype=torch.long, device=self.device)

        last_pred_rewards = None
        for k in range(self.denoising_step_count):
            tau_idx = k * stride
            tau = tau_idx / float(N)
            obs_sigma_idx = torch.full((B, 1), tau_idx, dtype=torch.long, device=self.device)
            act_sigma_idx = torch.full((B, 1), tau_idx, dtype=torch.long, device=self.device)

            obs_pred, act_pred, pred_rewards = self.denoiser.forward_step(
                noisy_act=z_act,
                noisy_obs=z_obs,
                obs_sigma_idx=obs_sigma_idx,
                obs_step_idx=step_idx,
                act_sigma_idx=act_sigma_idx,
                act_step_idx=step_idx,
                start_step_idx=self.current_frame_index,
                update_cache=False,
                cond_class=self.cond_class_t,
            )
            act_pred = act_pred.squeeze(-2)  # (B, 1, n_act)

            denom = max(1.0 - tau, 1e-5)
            v_obs = (obs_pred - z_obs) / denom
            v_act = (act_pred - z_act) / denom
            z_obs = z_obs + v_obs * step_size
            z_act = z_act + v_act * step_size
            if k == self.denoising_step_count - 1:
                last_pred_rewards = pred_rewards

        # commit cache: noise both predicted obs and action to tau_cond
        cor_z = (1.0 - self.context_cond_tau) * torch.randn_like(z_obs) + self.context_cond_tau * z_obs
        cor_act = (1.0 - self.context_cond_tau) * torch.randn_like(z_act) + self.context_cond_tau * z_act
        commit_obs_sigma = torch.full((B, 1), self.cond_tau_idx, dtype=torch.long, device=self.device)
        commit_act_sigma = torch.full((B, 1), self.cond_tau_idx, dtype=torch.long, device=self.device)

        self.denoiser.forward_step(
            noisy_act=cor_act,
            noisy_obs=cor_z,
            obs_sigma_idx=commit_obs_sigma,
            obs_step_idx=step_idx,
            act_sigma_idx=commit_act_sigma,
            act_step_idx=step_idx,
            start_step_idx=self.current_frame_index,
            update_cache=True,
            cond_class=self.cond_class_t,
        )

        imgs_recon = self.tokenizer.decode_step(z_obs,
                                                start_step_idx=self.current_frame_index,
                                                update_cache=True)

        self.current_z = z_obs.clone()
        self.current_act = z_act.clone()
        self.current_frame_index += 1

        if last_pred_rewards is not None:
            rewards = self._decode_reward_logits(last_pred_rewards)[:, 0]  # (B, L)
            return imgs_recon[:, 0, ...], z_act[:, 0, ...], rewards
        return imgs_recon[:, 0, ...], z_act[:, 0, ...]