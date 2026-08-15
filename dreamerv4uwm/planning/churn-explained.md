# What `churn` is, and how `rollout_stoch.py` works

A walkthrough of the maths behind [`rollout_stoch.py`](rollout_stoch.py). The short version:

> The current sampler is a **deterministic function** of the initial noise draw. Churn makes
> it stop being one — by *partially re-rolling the dice mid-flight* — without changing the
> distribution it is sampling from and without spending a single extra denoiser forward.

---

## 1. The setup: what the sampler actually integrates

### The interpolant

The model is trained on a straight-line path between noise and data. Writing $\tau \in [0,1]$
for **cleanness** ($\tau = 0$ is pure noise, $\tau = 1$ is clean — the codebase's noise level
is $n = 1-\tau$):

$$x_\tau = (1-\tau)\,\varepsilon + \tau\,x_1, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

where $x_1$ is the clean sample (an action sequence, or a latent state sequence). Every point
on this path is a specific mixture of a specific noise draw and a specific clean target.

### The velocity

Differentiating along the path, $\dfrac{dx_\tau}{d\tau} = x_1 - \varepsilon$. We don't know
$\varepsilon$ at sampling time, but we can eliminate it: from the interpolant,
$\varepsilon = \dfrac{x_\tau - \tau x_1}{1-\tau}$, so

$$\frac{dx_\tau}{d\tau} \;=\; x_1 - \frac{x_\tau - \tau x_1}{1-\tau}
\;=\; \frac{(1-\tau)x_1 - x_\tau + \tau x_1}{1-\tau}
\;=\; \boxed{\;\frac{x_1 - x_\tau}{1-\tau}\;}$$

The denoiser gives us $\hat{x}_\theta(x, \tau) \approx \mathbb{E}[x_1 \mid x_\tau]$, so the
learned velocity field is

$$v_\theta(x,\tau) = \frac{\hat{x}_\theta(x,\tau) - x}{1-\tau}$$

### The Euler step

Integrating that ODE with $K$ steps of size $\Delta\tau = 1/K$ gives exactly the line in
[`rollout.py`](rollout.py):

$$x \;\leftarrow\; x + \frac{\hat{x}_\theta(x,\tau) - x}{1-\tau}\,\Delta\tau$$

```python
z[:, Tc:] = z[:, Tc:] + (z_hat[:, Tc:] - z[:, Tc:]) / denom * dt
```

---

## 2. Why this collapses

**There is no randomness in that recursion.** The only random thing in the whole sampler is
the initial draw $\varepsilon$. So for a fixed context $c$, the sampler is a deterministic map

$$\Phi_c : \varepsilon \;\longmapsto\; a_{1:H}$$

Draw $B$ siblings and you are just evaluating $\Phi_c$ at $B$ points. Whether they come out
different depends entirely on whether $\Phi_c$ *spreads* its input.

And it doesn't. Probability-flow ODEs are empirically **contractive**: they funnel large
regions of noise space onto small regions of output space. So $B$ well-separated $\varepsilon$
draws produce $B$ nearly identical trajectories. That is edge collapse.

> **The key fact that makes a fix possible:** the probability-flow ODE and the reverse SDE
> have the *same marginals*. They describe the same distribution. But the ODE is a
> deterministic transport that can concentrate onto modes, while the SDE genuinely samples.
> So we can add stochasticity back **without changing the target distribution** — unlike
> `ctx_noise` or `action_temp`, which buy diversity by pushing the model off-distribution.

---

## 3. Churn: partially re-rolling the dice

### The idea

Halfway through integration, the point $x$ still "contains" a noise component. If we could
read it out, replace part of it with fresh randomness, and put it back, we would be at a
different point on the *same* path family — heading toward a possibly different outcome.

We can do exactly that, because the interpolant is invertible.

### Step 1 — read out the implied noise

At cleanness $\tau$, the model believes the clean target is $\hat{x}$. Inverting
$x = (1-\tau)\varepsilon + \tau x_1$ with $x_1 \approx \hat{x}$:

$$\hat{\varepsilon} \;=\; \frac{x - \tau \hat{x}}{1-\tau}$$

This is *the noise realisation still living inside $x$* — the part of the original dice roll
that hasn't been resolved into data yet.

### Step 2 — partially resample it

Mix it with a fresh independent draw $\xi \sim \mathcal{N}(0,I)$, using churn
$\eta \in [0,1]$:

$$\varepsilon_{\text{new}} \;=\; \sqrt{1-\eta^2}\;\hat{\varepsilon} \;+\; \eta\,\xi$$

**Why the $\sqrt{1-\eta^2}$?** Because it is exactly what keeps the result standard normal.
If $\hat{\varepsilon} \sim \mathcal{N}(0,I)$ and $\xi \sim \mathcal{N}(0,I)$ independently:

$$\operatorname{Var}(\varepsilon_{\text{new}}) = (1-\eta^2)\,I + \eta^2 I = I \quad\checkmark$$

Any other coefficient would inflate or shrink the noise scale and quietly move us off the
distribution the model was trained on. This is the same variance-preserving mix used in
stochastic DDIM.

The correlation with the original draw is $\operatorname{Corr}(\varepsilon_{\text{new}},
\hat{\varepsilon}) = \sqrt{1-\eta^2}$:

| $\eta$ | correlation with the old noise | meaning |
|---|---|---|
| $0$ | $1$ | nothing changes — the deterministic sampler |
| $0.3$ | $0.95$ | a nudge |
| $0.6$ | $0.80$ | substantial re-roll |
| $1$ | $0$ | the noise realisation is completely forgotten |

### Step 3 — put it back

$$x \;\leftarrow\; (1-\tau)\,\varepsilon_{\text{new}} + \tau\,\hat{x}$$

We are back on the interpolant at the **same cleanness $\tau$**, so integration continues
normally. Nothing about the schedule changed.

### The whole thing

$$\boxed{\;
x \;\longleftarrow\; (1-\tau)\left[\sqrt{1-\eta^2}\,\frac{x - \tau\hat{x}}{1-\tau} + \eta\,\xi\right] + \tau\,\hat{x}
\;}$$

---

## 4. The code

[`_refresh`](rollout_stoch.py#L80) is those three steps verbatim:

```python
keep    = max(1.0 - tau, 1e-5)                     # (1 - tau), guarded against tau -> 1
eps_hat = (x - tau * x_hat) / keep                 # step 1: read out the implied noise
xi      = torch.randn(x.shape, ...)                #         a fresh draw
if orthogonal:
    xi = _proj_out(xi, x_hat - x)                  # (section 6)
eps_new = math.sqrt(max(1.0 - churn**2, 0.0)) * eps_hat + churn * xi   # step 2: mix
return keep * eps_new + tau * x_hat                # step 3: put it back
```

and it is called once per Euler step, *after* the update, at the new cleanness:

```python
z[:, Tc:] = z[:, Tc:] + (z_hat[:, Tc:] - z[:, Tc:]) / denom * dt   # ODE step
a[:, Tc:] = a[:, Tc:] + (a_hat[:, Tc:] - a[:, Tc:]) / denom * dt
cur += dt

eta = _churn_at(cur, churn, churn_tmax)
if eta > 0.0:                       # <- no RNG draw at all when churn is off
    z[:, Tc:] = _refresh(z[:, Tc:], z_hat[:, Tc:], cur, eta, ...)
    a[:, Tc:] = _refresh(a[:, Tc:], a_hat[:, Tc:], cur, eta, ...)
```

---

## 5. Four properties this form was chosen for

**(a) Exact identity at $\eta = 0$.** Substituting $\eta = 0$ gives
$\varepsilon_{\text{new}} = \hat\varepsilon$, so

$$x \leftarrow (1-\tau)\frac{x - \tau\hat x}{1-\tau} + \tau \hat x = x - \tau\hat x + \tau\hat x = x$$

Not "approximately unchanged" — algebraically the identity. Combined with skipping the
`torch.randn` call entirely when `eta == 0`, this means `churn=0` reproduces `rollout.imagine`
**bit-for-bit including the RNG stream**. That is what makes the churn sweep a clean ablation:
the $\eta=0$ column is genuinely the current sampler, not a re-implementation of it.

**(b) It stays on the interpolant.** The output is by construction of the form
$(1-\tau)\varepsilon + \tau\hat x$ with $\varepsilon$ standard normal. Compare the naive
alternative, $x \leftarrow x + \sigma\xi$: that adds noise *on top of* a point whose noise
budget is already spoken for, leaving the manifold and distorting the marginal.

**(c) The perturbation dies out on its own.** The displacement is

$$\Delta x = (1-\tau)\left[\left(\sqrt{1-\eta^2}-1\right)\hat\varepsilon + \eta\,\xi\right]$$

so $\|\Delta x\| \propto (1-\tau)$. In $d$ dimensions its expected size is

$$\mathbb{E}\|\Delta x\|^2 = 2\,(1-\tau)^2\,d\,\bigl(1 - \sqrt{1-\eta^2}\bigr)
\;\;\approx\;\; (1-\tau)^2 d\,\eta^2 \quad \text{for small } \eta$$

Early steps (small $\tau$) get perturbed hard; the last steps are effectively un-churned, so
the sample lands cleanly on the data manifold. No schedule needed — the geometry provides it.

**(d) Identical compute budget.** $K$ denoiser forwards regardless of $\eta$. Karras-style
churn normally rewinds $\tau$ and therefore needs extra steps to re-cover the ground; framing
it as *noise replacement at fixed $\tau$* avoids that. This matters because the whole research
plan insists on comparing at equal `n_denoiser_calls` — here that comes for free.

---

## 6. Orthogonal churn

`churn_orthogonal=True` projects the fresh noise off the local flow direction
$v = \hat{x} - x$ before mixing:

$$\xi_\perp = \xi - \frac{\langle \xi, v\rangle}{\|v\|^2}\,v$$

**Why.** Perturbing *along* $v$ just moves you forward or backward along the path you were
already on — it changes *when* you arrive, not *where*. Only the component orthogonal to $v$
moves you to a different trajectory. So removing the parallel part should buy more diversity
per unit of fidelity damage.

**Honest caveat.** $\xi_\perp$ is no longer exactly $\mathcal{N}(0,I)$: it has zero variance in
one direction, so $\mathbb{E}\|\xi_\perp\|^2 = d-1$ instead of $d$. For $d = H \cdot n_{\text{act}}$
or $d = N_{\text{lat}}D_{\text{lat}}$ that is a shrinkage of $1 - 1/d$, negligible in practice
but not zero. This variant is an approximation of the marginal, where plain churn is not —
which is why it is opt-in rather than the default.

---

## 7. The remaining knobs

| knob | what it does | why you'd touch it |
|---|---|---|
| `churn` | $\eta$, the fraction of noise resampled each step | **the** knob; sweep it |
| `churn_tmax` | only churn while $\tau \le$ this | force the late steps to be pure ODE; mostly redundant given property (c) |
| `churn_orthogonal` | use $\xi_\perp$ instead of $\xi$ | more diversity per unit fidelity cost, at the cost of a slight marginal bias |
| `churn_targets` | `both` / `action` / `state` | isolates *which* prior drives sibling diversity — an experiment, not a tuning knob |

`churn_targets` is worth dwelling on: `imagine` denoises actions and states jointly, so both
have a refreshable noise component. Running `action` and `state` separately answers "is it the
action prior or the state prior that the siblings actually inherit their differences from?" —
which nothing in the codebase currently knows.

---

## 8. Relation to the literature

| this | is essentially |
|---|---|
| noise refreshment at fixed $\tau$ | Karras et al.'s `S_churn`, and "restart sampling", reparameterised so the step schedule is untouched |
| the $\eta$ family | a discretisation of the reverse **SDE** whose marginals match the probability-flow ODE; $\eta = 0$ recovers the ODE exactly |
| the orthogonal variant | the "orthogonal perturbation" of [*Letting Trajectories Spread*](https://arxiv.org/abs/2510.09060) |

The advantage of parameterising by "fraction of noise replaced" rather than by a diffusion
coefficient $g(\tau)$: $\eta$ is dimensionless, lives in $[0,1]$, is comparable across
modalities of very different scale (2-D actions vs 8192-D latents), and has an exact
zero-point.

---

## 9. What churn cannot do

Churn resamples $\varepsilon$. It cannot help if $\varepsilon$ never mattered.

Suppose the posterior $p(a_{1:H} \mid c)$ is genuinely a spike — the model is simply certain
what happens next. Then $\hat{x}_\theta(x,\tau) \approx x_1$ for *every* $x$, and after any
refresh the next Euler step pulls straight back toward the same $x_1$. At $\tau = 1$ you get
that same $x_1$ regardless of how much noise you re-rolled.

So the churn sweep is a **diagnostic**, not just a fix:

- **diversity rises with $\eta$** → the posterior was always wide, the ODE was collapsing it
  (**H2**). Diversity is free and in-distribution; adopt churn and move on.
- **diversity doesn't move** → the posterior really is narrow (**H1**). No sampler can help,
  and every other family (repulsion, guidance meta-actions, oversampling) is a deliberate,
  measurable departure from what the model believes.

This is exactly what `notebooks/E0-churn-diagnostic.ipynb` measures, and what
`notebooks/edge-collapse-visual.ipynb` shows you without any numbers.

> Observed already: against a stub denoiser whose `tanh` head is a contraction — an
> artificial H1 world — churn correctly produces **no** extra diversity. The mechanism does
> not manufacture spread that isn't in the model, which is the behaviour we want from a
> diagnostic. The real checkpoint has not been tested yet; that is what E0 is for.

---

## 10. One-paragraph summary

The sampler walks a straight line from a noise draw to a clean sample, and because the walk is
deterministic, the only source of variety between siblings is where they started. That map
turns out to be contractive, so different starts arrive at the same place. Churn fixes this by
noticing that partway along the walk you can still algebraically recover "how much noise is
left in this point", replace a fraction $\eta$ of it with fresh randomness in a way that
provably preserves the noise's unit scale, and carry on. It costs nothing, it changes no
distribution, it is exactly the old sampler at $\eta = 0$, and it fades out by itself as the
sample becomes clean.
