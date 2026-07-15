# Forklift RL Nav — Project Briefing

## Goal
Train PPO policy to navigate to a goal while avoiding obstacles.
Eventually deploy as a ROS2 node replacing the bug-algorithm nav node.

## Current status (2026-07-15)
- Retrained from scratch with two simultaneous fixes: (1) reset() now forces an
  obstacle onto the direct spawn->goal line in ~50% of episodes (was fully random,
  so avoidance was often optional); (2) turn-first reward shaping in step() —
  bonus when well-aligned and moving, penalty when misaligned and moving fast.
  3-stage curriculum collapsed to 0 -> 1 -> 3 obstacles (dropped the old 2-obstacle
  stage). evaluate.py: 15/20 (target was 13+/20, lower bar since task is genuinely
  harder now). debug_episode.py x3 confirms both: obstacle navigation with
  min_lidar down to ~0.5m while lin_vel stays >0.05 (not stalling), and a clear
  turn-first arc (large angle_to_goal + high ang_vel + low lin_vel early, lin_vel
  rising to max once aligned).
- IMPORTANT lesson on shaping magnitude: the first attempt at the turn-first
  reward used +-0.05 bonus/penalty and measurably did NOT change behavior —
  across 60 eval episodes, lin_vel stayed pinned at max regardless of
  angle_to_goal. Root cause: it exactly canceled the existing unconditional
  `lin_vel_frac * 0.05` forward bonus, netting zero marginal incentive to slow
  down. Had to raise it to +0.1 / -0.25 before it actually shifted behavior
  (verified via the lin_vel-by-angle_to_goal-bucket check below). Lesson: don't
  trust a "small" shaping term is doing anything just because training completes
  and eval numbers look fine — check whether it's actually bigger than whatever
  unconditional/competing reward term it's supposed to be nudging against.
- Useful verification technique beyond debug_episode.py (which only surfaces
  FAILED episodes, biasing toward worst-case behavior): run ~60 episodes,
  bucket every step's lin_vel by abs(angle_to_goal) (<0.3 / 0.3-1.0 / >1.0), and
  compare mean lin_vel per bucket. A working turn-first policy shows a clear
  downward gradient; a non-functional one shows ~flat lin_vel across buckets.
- Prior status (2026-07-13, before this session): evaluate.py filtered to
  obstructed-path episodes only, ~15/20 average (range 12-19/20), 0 timeouts,
  but obstacle placement was still fully random (only ~50%+ of episodes were
  actually obstructed by chance) and there was no explicit turn-first shaping.

## Files
- forklift_env.py — Gymnasium env, 24 lidar rays, normalized obs, action space [-1,1].
  Has `path_blocked()` (True if an obstacle sits near the direct spawn→goal line —
  use this to filter/eval on episodes that actually require avoidance).
- train.py — 4-stage curriculum (0→1→2→3 obstacles), low ent_coef (0.005-0.01),
  net_arch=[128,128], log_std_init=-1.0, ~2.4M total timesteps.
- evaluate.py — 20 episodes filtered by path_blocked(), reports success/collision/timeout.
- debug_episode.py — searches for a FAILED path_blocked() episode, prints full
  trace, flags steps where min_lidar<1.5 and lin_vel<0.05 as "STALL".

## Already ruled out (don't retry these)
- Reward tuning alone (previous session)
- Single-jump curriculum (0→3 obstacles)
- Obstacle overlap (fixed with rejection sampling)
- Observation normalization (already done)
- Lidar aliasing (fixed, now 24 rays)
- Action space too narrow (fixed, now symmetric [-1,1])
- ent_coef=0.01 alone, without the reward-shaping below (previous session)
- High ent_coef (0.04-0.1) to force exploration — with clipped (non-squashed)
  Gaussian PPO actions, this is exploitable: the policy can inflate log_std for
  free entropy reward once the mean saturates near ±1, since clipping absorbs the
  extra variance. Watch the SB3 `std` log — it should stay in single digits. We
  saw it run away to 1140 (ent_coef 0.05-0.1) and even 40 at ent_coef 0.02 with
  enough updates; only stabilized (~0.5) once dropped to 0.005-0.01 combined with
  log_std_init=-1.0. Performance got measurably worse while std was runaway.
- Reward bearing-to-nearest-obstacle computed from the nearest *lidar ray index* —
  the index is quantized to 15° bins (24 rays) and flips between neighboring rays
  under tiny heading changes, feeding a chattering signal back into the policy
  (visible as ang_vel alternating +/-max every single step, net-zero heading
  change). Fixed by computing bearing to the nearest obstacle's true (x,y) center
  instead — continuous, no aliasing.
- Proximity penalty alone (distance-based, even quadratic) — a policy can point
  away from an obstacle while orbiting it at constant range indefinitely. Needed
  a separate `clearance_gain` term rewarding an actual increase in min_obstacle_dist
  step over step while in the danger zone, not just correct orientation.
- Obstacle spawn distribution `o_dist ~ U(1.0, 4.5)` from the origin — allowed
  obstacles within ~0.6m surface clearance of spawn, an almost-unreactable start.
  Raised lower bound to 1.6.
- No goal↔obstacle clearance check in forklift_env.reset() itself — obstacles could
  spawn nearly on top of the goal (visualize_policy.py already worked around this
  with its own rejection sampling, but evaluate.py/debug_episode.py/training did
  not have that protection). Fixed in reset() directly: reject obstacle if within
  obstacle_radius+0.5 of the goal.

## Reward shaping that fixed the freeze-near-obstacles failure mode
In forklift_env.step(), when min_obstacle_dist < danger_dist (1.8m):
- quadratic proximity penalty (ramps up hard near collision_dist)
- `clearance_gain` reward: `(min_obstacle_dist - prev_min_obstacle_dist) * 2.0`
- stall penalty when lin_vel_frac < stall_vel_frac (0.25, i.e. lin_vel < 0.05):
  scaled 4x — this is what makes "stop near the obstacle" no longer the safe move
- bearing-to-nearest-obstacle-center reward (orient away), scaled by closeness
Plus a small unconditional `lin_vel_frac * 0.05` reward every step, so stopping is
never free even outside the danger zone.

## Target
13+/20 successes on evaluate.py (obstructed-path episodes only) without disabling
obstacles or trivializing the task — lowered from 14+/20 on 2026-07-15 since
forced on-path obstacle placement + turn-first behavioral constraints make the
task genuinely harder. Currently at 15/20 (2026-07-15) with both fixes applied.
Run train.py and evaluate.py (multiple times — there's real run-to-run variance)
to verify each fix before concluding it worked. For behavioral checks (not just
the success count), don't rely on debug_episode.py alone — it only ever prints
FAILED episodes, which biases toward worst-case behavior. Cross-check with a
bucketed lin_vel-vs-angle_to_goal sweep across many episodes (see above).