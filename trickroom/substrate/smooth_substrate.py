"""Smooth differentiable substrate for Two-Room planning.

Replaces the hard if/else R3 wall rule in python_substrate.py with sigmoid
approximations, making the dynamics differentiable w.r.t. both actions and
the door-position prior θ.

This enables:
  1. Gradient-based planning (action optimisation via torch.autograd)
  2. Online parameter estimation: infer θ from wall-crossing observations
  3. Tracing V(θ) as a continuous curve to locate the crossover point

Mathematical structure
----------------------
The original R3 rule (CollideWall) is:

    push_left  = [started_left] ∧ [x_prop ∈ wall_band] ∧ ¬[y_prop ∈ door_band(θ)]
    push_right = ¬[started_left] ∧ [x_prop ∈ wall_band] ∧ ¬[y_prop ∈ door_band(θ)]
    x_new = push_left·99.5 + push_right·124.5 + (1−push_left−push_right)·x_prop

The smooth surrogate replaces each logical indicator with a product of sigmoids:

    φ_door(y, θ) = σ((y − (θ−w/2))/τ) · σ(((θ+w/2)−y)/τ)      ∈ (0,1)
    φ_wall(x)    = σ((x − effective_low)/τ) · σ((effective_high−x)/τ)
    φ_left(x0)   = σ((WALL_CENTER − x0)/τ)

    push_left  = φ_left · φ_wall · (1 − φ_door)
    push_right = (1 − φ_left) · φ_wall · (1 − φ_door)
    x_new = push_left·99.5 + push_right·124.5 + (1−push_left−push_right)·x_prop

As τ→0, this recovers the exact discontinuous rule. For planning, τ=1–2 gives
smooth but faithful gradients.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ── geometry constants (match python_substrate.py exactly) ─────────────────
IMG_SIZE = 224
BORDER_SIZE = 14
WALL_CENTER = 112.0
WALL_HALF = 5.0          # wall_thickness // 2
AGENT_RADIUS = 7.0
AGENT_SPEED = 5.0
EFFECTIVE_LOW = WALL_CENTER - WALL_HALF - AGENT_RADIUS   # 100.0
EFFECTIVE_HIGH = WALL_CENTER + WALL_HALF + AGENT_RADIUS  # 124.0
CLIP_LOW = BORDER_SIZE + AGENT_RADIUS                    # 21.0
CLIP_HIGH = IMG_SIZE - BORDER_SIZE - AGENT_RADIUS        # 203.0
DOOR_HALF_EXTENT = 14.0
DOOR_MARGIN = 1.75
DOOR_BAND = DOOR_HALF_EXTENT + DOOR_MARGIN               # 15.75

PUSH_LEFT_X = EFFECTIVE_LOW - 0.5    # 99.5
PUSH_RIGHT_X = EFFECTIVE_HIGH + 0.5  # 124.5


# ── smooth primitives ───────────────────────────────────────────────────────

def _phi_interval(x: torch.Tensor, lo: float, hi: float, tau: float) -> torch.Tensor:
    """Soft indicator: ≈1 if x ∈ (lo, hi), ≈0 outside. Shape: same as x."""
    return torch.sigmoid((x - lo) / tau) * torch.sigmoid((hi - x) / tau)


def _phi_door(y_prop: torch.Tensor, door_y: torch.Tensor, tau: float) -> torch.Tensor:
    """Soft indicator: ≈1 if y_prop is within the door band centred at door_y.

    y_prop  : (...,)
    door_y  : (...,)   — the learnable prior / observed value
    """
    return _phi_interval(y_prop, door_y - DOOR_BAND, door_y + DOOR_BAND, tau)  # type: ignore[arg-type]


# ── single differentiable step ──────────────────────────────────────────────

def smooth_step(
    agent_pos: torch.Tensor,
    action: torch.Tensor,
    door_y: torch.Tensor,
    tau: float = 1.5,
) -> torch.Tensor:
    """One Two-Room step, fully differentiable in (agent_pos, action, door_y).

    Parameters
    ----------
    agent_pos : (..., 2)  current agent position
    action    : (..., 2)  action in [-1, 1]
    door_y    : (,) or (...,)  door y-coordinate prior (θ)
    tau       : temperature; lower = sharper. 1.5 is a good default.

    Returns
    -------
    new_pos : (..., 2)
    """
    action = torch.clamp(action, -1.0, 1.0)
    proposed = agent_pos + action * AGENT_SPEED

    # R2: CollideBorder (hard clamp — differentiable via straight-through-ish grad)
    proposed = torch.clamp(proposed, CLIP_LOW, CLIP_HIGH)

    x_start = agent_pos[..., 0]
    x_prop = proposed[..., 0]
    y_prop = proposed[..., 1]

    # R3: CollideWall — smooth surrogate
    phi_left = torch.sigmoid((WALL_CENTER - x_start) / tau)       # ≈1 if started left
    phi_wall = _phi_interval(x_prop, EFFECTIVE_LOW, EFFECTIVE_HIGH, tau)
    phi_door = _phi_door(y_prop, door_y, tau)

    push_left = phi_left * phi_wall * (1.0 - phi_door)
    push_right = (1.0 - phi_left) * phi_wall * (1.0 - phi_door)
    free = 1.0 - push_left - push_right

    new_x = push_left * PUSH_LEFT_X + push_right * PUSH_RIGHT_X + free * x_prop
    new_y = y_prop

    return torch.stack([new_x, new_y], dim=-1)


# ── multi-step rollout ──────────────────────────────────────────────────────

def smooth_rollout(
    agent_pos: torch.Tensor,
    actions: torch.Tensor,
    door_y: torch.Tensor,
    tau: float = 1.5,
) -> torch.Tensor:
    """Roll out T steps. Returns final agent position.

    agent_pos : (..., 2)
    actions   : (..., T, 2)
    door_y    : scalar or (...,)
    """
    T = actions.shape[-2]
    pos = agent_pos
    for t in range(T):
        pos = smooth_step(pos, actions[..., t, :], door_y, tau)
    return pos


def smooth_rollout_trajectory(
    agent_pos: torch.Tensor,
    actions: torch.Tensor,
    door_y: torch.Tensor,
    tau: float = 1.5,
) -> torch.Tensor:
    """Like smooth_rollout but returns all intermediate positions.

    Returns: (..., T+1, 2)  where index 0 is the initial position.
    """
    T = actions.shape[-2]
    traj = [agent_pos]
    pos = agent_pos
    for t in range(T):
        pos = smooth_step(pos, actions[..., t, :], door_y, tau)
        traj.append(pos)
    return torch.stack(traj, dim=-2)


# ── cost function ───────────────────────────────────────────────────────────

def smooth_cost(
    agent_pos: torch.Tensor,
    actions: torch.Tensor,
    target_pos: torch.Tensor,
    door_y: torch.Tensor,
    tau: float = 1.5,
) -> torch.Tensor:
    """Final L2 distance to target after rolling out actions. Differentiable.

    agent_pos  : (..., 2)
    actions    : (..., T, 2)
    target_pos : (..., 2)
    door_y     : scalar or (...,)

    Returns: (...,)   cost = ||final_pos − target||
    """
    final = smooth_rollout(agent_pos, actions, door_y, tau)
    return torch.norm(final - target_pos, dim=-1)


# ── Costable-conforming wrapper ─────────────────────────────────────────────

class SmoothSubstrateCostModel:
    """Drop-in replacement for SubstrateCostModel with gradient support.

    Conforms to the Costable protocol (get_cost signature), but internally
    uses the smooth PyTorch dynamics instead of numpy physics.

    The key new capability: pass `theta` (door_y) as a torch.Tensor with
    requires_grad=True to obtain ∂cost/∂θ.
    """

    def __init__(
        self,
        door_y: float = 112.0,
        tau: float = 1.5,
        device: Optional[str] = None,
    ):
        self.tau = tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.door_y = torch.tensor(float(door_y), device=self.device)

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cost for CEM candidate action sequences.

        Parameters
        ----------
        info_dict          : dict with 'observation' key, shape (B, S, T, 10)
        action_candidates  : (B, S, H, action_block * 2)
        theta              : optional door_y override (for gradient estimation)

        Returns
        -------
        cost : (B, S)
        """
        obs = info_dict["observation"]                    # (B, S, T, 10)
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if isinstance(action_candidates, np.ndarray):
            action_candidates = torch.as_tensor(
                action_candidates, dtype=torch.float32, device=self.device
            )

        # Use most recent observation (last time step)
        state = obs[..., -1, :]                           # (B, S, 10)
        agent_pos = state[..., 0:2]                       # (B, S, 2)
        target_pos = state[..., 2:4]                      # (B, S, 2)

        # Reshape actions: (B, S, H, action_block*2) → (B, S, H*action_block, 2)
        B, S, H, act_dim = action_candidates.shape
        actions = action_candidates.reshape(B, S, H * (act_dim // 2), 2)

        door = theta if theta is not None else self.door_y

        cost = smooth_cost(agent_pos, actions, target_pos, door, self.tau)
        return cost


# ── gradient-based planner ──────────────────────────────────────────────────

def plan_gradient(
    agent_pos: torch.Tensor,
    target_pos: torch.Tensor,
    door_y: torch.Tensor,
    horizon: int = 30,
    n_iter: int = 200,
    lr: float = 0.05,
    tau: float = 1.5,
    verbose: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """Optimise an action sequence via gradient descent on the smooth cost.

    Parameters
    ----------
    agent_pos  : (2,)
    target_pos : (2,)
    door_y     : scalar tensor — door prior
    horizon    : number of steps to plan
    n_iter     : gradient steps
    lr         : Adam learning rate
    tau        : smoothness temperature

    Returns
    -------
    actions : (horizon, 2)  optimised actions in [-1, 1]
    losses  : list of cost values per iteration
    """
    # Initialise actions at zero
    actions_raw = torch.zeros(horizon, 2, device=agent_pos.device, requires_grad=True)
    opt = torch.optim.Adam([actions_raw], lr=lr)
    losses: list[float] = []

    for i in range(n_iter):
        opt.zero_grad()
        actions = torch.tanh(actions_raw)                 # maps R → (-1,1) smoothly
        cost = smooth_cost(agent_pos, actions, target_pos, door_y, tau)
        cost.backward()
        opt.step()
        losses.append(float(cost.detach()))
        if verbose and i % 50 == 0:
            print(f"  iter {i:4d}  cost={losses[-1]:.2f}")

    actions_final = torch.tanh(actions_raw.detach())
    return actions_final, losses


# ── online parameter estimation ─────────────────────────────────────────────

def estimate_theta(
    trajectories: list[np.ndarray],
    init_theta: float = 112.0,
    n_steps: int = 300,
    lr: float = 1.0,
    tau: float = 1.5,
    verbose: bool = False,
) -> tuple[float, list[float]]:
    """Infer door_y from observed wall-crossing trajectories via gradient descent.

    Each trajectory is a sequence of 10-dim observations. The loss is the
    sum of squared distances between predicted and observed agent positions
    after replaying the action-free dynamics with the door prior θ.

    Parameters
    ----------
    trajectories : list of (T, 10) obs arrays — one per episode
    init_theta   : starting guess for door_y (default 112 = wrong prior)
    n_steps      : gradient steps
    lr           : learning rate
    tau          : smoothness temperature

    Returns
    -------
    theta_hat : estimated door_y
    losses    : loss per step
    """
    theta = torch.tensor(float(init_theta), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    losses: list[float] = []

    # Pre-convert trajectories
    trajs_t = [torch.as_tensor(t, dtype=torch.float32) for t in trajectories]

    for step in range(n_steps):
        opt.zero_grad()
        total_loss = torch.tensor(0.0)

        for traj in trajs_t:
            # traj: (T, 10)
            agent_pos = traj[0, 0:2]
            T = len(traj) - 1
            for t in range(T):
                action_approx = (traj[t + 1, 0:2] - traj[t, 0:2]) / AGENT_SPEED
                action_approx = torch.clamp(action_approx, -1.0, 1.0)
                predicted = smooth_step(agent_pos, action_approx, theta, tau)
                observed = traj[t + 1, 0:2]
                total_loss = total_loss + torch.sum((predicted - observed) ** 2)
                agent_pos = predicted                     # use predicted (not observed) to propagate θ grad

        total_loss.backward()
        opt.step()
        with torch.no_grad():
            theta.clamp_(CLIP_LOW, CLIP_HIGH)             # keep in valid arena range
        losses.append(float(total_loss.detach()))
        if verbose and step % 50 == 0:
            print(f"  step {step:4d}  θ={float(theta):.2f}  loss={losses[-1]:.4f}")

    return float(theta.detach()), losses


# ── V(θ) sweep utility ──────────────────────────────────────────────────────

def value_vs_theta(
    obs_batch: np.ndarray,
    action_sequences: np.ndarray,
    theta_range: tuple[float, float] = (49.0, 112.0),
    n_points: int = 64,
    tau: float = 1.5,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute expected cost E[cost(θ)] across a sweep of door priors.

    Parameters
    ----------
    obs_batch        : (N, 10)     initial observations for N episodes
    action_sequences : (N, T, 2)   action sequences to evaluate
    theta_range      : (lo, hi)    sweep range for door_y
    n_points         : number of θ values
    tau              : smoothness temperature

    Returns
    -------
    thetas : (n_points,)  θ values
    values : (n_points,)  mean cost at each θ
    """
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(action_sequences, dtype=torch.float32, device=device)

    agent_pos = obs_t[:, 0:2]                             # (N, 2)
    target_pos = obs_t[:, 2:4]                            # (N, 2)

    thetas = np.linspace(theta_range[0], theta_range[1], n_points)
    values = np.zeros(n_points)

    with torch.no_grad():
        for i, th in enumerate(thetas):
            door = torch.tensor(th, dtype=torch.float32, device=device)
            cost = smooth_cost(agent_pos, act_t, target_pos, door, tau)
            values[i] = float(cost.mean())

    return thetas, values
