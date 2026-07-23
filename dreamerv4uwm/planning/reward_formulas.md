# T-centering reward formulas

Reference for the pixel-space PushT rewards in [`reward.py`](reward.py):
`TCenterReward`, `TCenterStraightReward`, and `TCenterAngleReward`. All three share
one pipeline and differ only in an **orientation** term.

> Render note: this file uses `$…$` / `$$…$$` LaTeX math — view it in a Markdown
> preview (VS Code, GitHub) rather than a plain terminal.

---

## The shared pipeline

Each reward maps a batch of latent **states** to scalar rewards, one frame at a time:

1. **Decode** the latent $z$ to an RGB image via the tokenizer.
2. **Segment** the red **T**: threshold in HSV, take the largest connected component,
   giving a binary mask, its centroid $(c_x, c_y)$ (pixels), and area $A$.
3. **Score** the frame with the reward-specific formula below.

The reward is in $[0, 1]$. If no valid T is found (empty mask, or area below
`min_area_frac`), the score is `floor` (default $0$).

---

## `TCenterAngleReward` — centered **and** upright

This is the reward that distinguishes an **upright** T from an **upside-down** one, and
is maximal only when the T is both centered and pointing the right way up.

### Default form (`combine="product"`)

$$
r \;=\; \underbrace{\exp\!\left(-\tfrac{1}{2}\left(\frac{d}{\sigma}\right)^{2}\right)}_{\text{center}}
\;\times\;
\underbrace{\exp\!\left(-\tfrac{1}{2}\left(\frac{\Delta\varphi}{\sigma_\varphi}\right)^{2}\right)}_{\text{orient (upright)}}
$$

Both factors are in $[0,1]$, so $r \in [0,1]$, and $r$ is large **only when both** are —
"centered **and** upright."

### Term 1 — centering

$$
d \;=\; \left\lVert \left(\tfrac{c_x}{W},\, \tfrac{c_y}{H}\right) - \text{center\_xy} \right\rVert_2 ,
\qquad
\text{center} \;=\; \exp\!\left(-\tfrac{1}{2}\left(\frac{d}{\sigma}\right)^{2}\right)
$$

- $(c_x/W,\, c_y/H)$ is the T centroid normalized to $[0,1]$ image coordinates.
- $\text{center\_xy}$ is the target (default $(0.5, 0.5)$, image center).
- $\sigma =$ `sigma` (default $0.25$) sets how quickly reward falls off with distance.

This term alone **is** `TCenterReward` (via `score_t_centered`).

### Term 2 — upright orientation

$$
\Delta\varphi \;=\; \big((\varphi - \varphi_{\text{target}} + \pi)\ \bmod\ 2\pi\big) - \pi
\;\in\; (-\pi, \pi],
\qquad
\text{orient} \;=\; \exp\!\left(-\tfrac{1}{2}\left(\frac{\Delta\varphi}{\sigma_\varphi}\right)^{2}\right)
$$

- $\varphi$ is the T's **full heading** (see below), the direction its crossbar points.
- $\varphi_{\text{target}} =$ `target_heading_deg` (default $-90^\circ$, i.e. crossbar pointing
  **up** in image coordinates, where $y$ increases downward — an upright letter "T").
- $\sigma_\varphi =$ `sigma_heading_deg` (default $25^\circ$) is the angular tolerance.

The wrap is **mod $2\pi$** (full circle). This is the whole point: an upside-down T is
$180^\circ$ away, giving $\Delta\varphi = \pi$ and

$$
\text{orient} = \exp\!\left(-\tfrac{1}{2}\left(\tfrac{\pi}{\sigma_\varphi}\right)^{2}\right) \approx 0
\quad(\sigma_\varphi = 25^\circ),
$$

so a centered-but-flipped T scores $\approx 0$.

### The heading $\varphi$

The heading comes from [`_t_heading`](reward.py#L283) and resolves the $180^\circ$
ambiguity that a raw principal axis cannot:

1. Take the mask pixel coordinates, center them, and form the covariance matrix.
   Its major eigenvector $\mathbf{v}$ is the T's **principal (stem) axis** — but with an
   arbitrary sign, i.e. only defined up to $180^\circ$.
2. Split the mask at its centroid along $\mathbf{v}$ into two halves. The **crossbar**
   half has a larger spread *perpendicular* to $\mathbf{v}$ than the stem half (the
   crossbar is wider). Point the heading toward the wider half:
   $\mathbf{h} = \mathbf{v}$ if the $+\mathbf{v}$ side is wider, else $-\mathbf{h} = -\mathbf{v}$.
3. $\varphi = \operatorname{atan2}(h_y, h_x) \in (-\pi, \pi]$.

So $\varphi$ points toward the crossbar (the flat top of an upright T). In image
coordinates ($y$ down): upright $\Rightarrow \varphi \approx -90^\circ$, inverted
$\Rightarrow \varphi \approx +90^\circ$.

### Alternative form (`combine="weighted_sum"`)

$$
r \;=\; \frac{w_c \cdot \text{center} + w_o \cdot \text{orient}}{w_c + w_o}
$$

with $w_c =$ `w_center`, $w_o =$ `w_orient` (defaults $0.5, 0.5$). This is more forgiving
than the product — it does **not** require both terms to be high — so prefer the product
when you want a strict "centered AND upright" reward.

---

## The reward family, side by side

All three share the centering term; they differ in the orientation factor $\text{orient}$:

| Reward | orientation factor | angle wrap | target (default) | upright vs. upside-down |
|---|---|---|---|---|
| `TCenterReward` | $\text{orient} \equiv 1$ | — | — | ignores orientation entirely |
| `TCenterStraightReward` | $\exp\!\big(\!-\tfrac12(\Delta\theta/\sigma_\theta)^2\big)$ | **mod $\pi$** (axis) | $\theta_{\text{target}}=90^\circ$ (vertical) | **same score** (axis only) |
| `TCenterAngleReward` | $\exp\!\big(\!-\tfrac12(\Delta\varphi/\sigma_\varphi)^2\big)$ | **mod $2\pi$** (heading) | $\varphi_{\text{target}}=-90^\circ$ (crossbar up) | **distinguished** (flip $\to 0$) |

For `TCenterStraightReward`, $\theta$ is the second-moment principal **axis** in
$(-\tfrac\pi2, \tfrac\pi2]$ and $\Delta\theta$ wraps mod $\pi$, so $+90^\circ$ and
$-90^\circ$ are the same "vertical" axis — an upright and an upside-down T are
indistinguishable. `TCenterAngleReward` replaces the axis with the full heading and
wraps mod $2\pi$ to tell them apart.

---

## Parameters (constructor keyword args)

Forwarded to `score_t_centered_angle`; defaults in parentheses.

| Param | Meaning |
|---|---|
| `center_xy` $= (0.5, 0.5)$ | target centroid, normalized image coords |
| `sigma` $= 0.25$ | centering Gaussian width $\sigma$ |
| `target_heading_deg` $= -90$ | upright heading $\varphi_{\text{target}}$ (crossbar up, $y$-down) |
| `sigma_heading_deg` $= 25$ | heading tolerance $\sigma_\varphi$ (degrees) |
| `combine` $=$ `"product"` | `"product"` (AND) or `"weighted_sum"` |
| `w_center`, `w_orient` $= 0.5, 0.5$ | weights for `weighted_sum` |
| `floor` $= 0.0$ | score when no valid T is found |
| segmentation gates | `s_min`, `v_min`, `hue_lo`, `hue_hi`, `min_area_frac`, … |

**Calibration.** $\varphi_{\text{target}} = -90^\circ$ assumes the *decoded* upright T's
crossbar points up. If your decoded upright frame reads a different heading, decode a
known-upright frame, print `debug['heading_deg']`, and set `target_heading_deg` to it.
The heading resolver assumes the crossbar is visibly wider than the stem (true for a
clean T; can get noisy on very blurry decodes).

---

## Worked example (defaults)

| T state | center | $\varphi$ | orient | $r$ (product) |
|---|---|---|---|---|
| centered, upright | $\approx 1$ | $-90^\circ$ | $\approx 1$ | $\approx 1$ |
| centered, upside-down | $\approx 1$ | $+90^\circ$ | $\approx 0$ | $\approx 0$ |
| off-center, upright | $\approx 0.3$ | $-90^\circ$ | $\approx 1$ | $\approx 0.3$ |
| no T found | — | — | — | `floor` $(=0)$ |

---

## Code pointers

- [`score_t_centered_angle`](reward.py#L318) — the scorer (the formula above)
- [`_t_heading`](reward.py#L283) — the full heading $\varphi$
- [`_heading_alignment`](reward.py#L311) — the mod-$2\pi$ Gaussian orient term
- [`TCenterAngleReward`](reward.py#L490) — the reward class (decode → score per frame)
- [`score_t_centered`](reward.py#L157) — the shared centering term
