# Forklift RL Nav — Project Briefing

## Goal
Train PPO policy to navigate to a goal while avoiding obstacles.
Eventually deploy as a ROS2 node replacing the bug-algorithm nav node.

## Current status
- Success: ~2/20, Collisions: ~5/20, Timeouts: ~15/20
- Agent moves but stalls/circles near obstacles instead of navigating around

## Files
- forklift_env.py — Gymnasium env, 24 lidar rays, normalized obs, action space [-1,1]
- train.py — 3-stage curriculum (0→1→3 obstacles), ent_coef=0.01, 350k timesteps
- evaluate.py — 20 episodes, reports success/collision/timeout
- debug_episode.py — step-by-step pos/vel/lidar for 2 episodes

## Already ruled out (don't retry these)
- Reward tuning alone
- Single-jump curriculum (0→3 obstacles)
- Obstacle overlap (fixed with rejection sampling)
- Observation normalization (already done)
- Lidar aliasing (fixed, now 24 rays)
- Action space too narrow (fixed, now symmetric [-1,1])
- ent_coef=0.01 (already in train.py)

## Target
14+/20 successes on evaluate.py without disabling obstacles or trivializing the task.
Run train.py and evaluate.py to verify each fix before concluding it worked.