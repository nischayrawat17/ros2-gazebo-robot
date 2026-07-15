import numpy as np
from stable_baselines3 import PPO
from forklift_env import ForkliftNavEnv

env = ForkliftNavEnv(num_obstacles=3)
model = PPO.load("forklift_ppo")

MAX_ATTEMPTS = 50

for attempt in range(MAX_ATTEMPTS):
    obs, _ = env.reset()
    if not env.path_blocked():
        continue

    goal_x, goal_y = env.goal_x, env.goal_y
    obstacles = list(env.obstacles)

    trace = []
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        min_lidar = float(np.min(env._get_lidar()))
        angle_to_goal = np.arctan2(env.goal_y - env.y, env.goal_x - env.x) - env.theta
        angle_to_goal = float(np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal)))
        trace.append((step, env.x, env.y, env.theta, env.lin_vel, env.ang_vel, min_lidar, angle_to_goal))
        step += 1

    success = bool(terminated and reward > 0)
    if success:
        continue  # keep searching for a FAILED episode

    print(f"=== Failed episode (attempt {attempt + 1}) ===")
    print(f"Goal: ({goal_x:.2f}, {goal_y:.2f})")
    print(f"Obstacles: {[(round(ox, 2), round(oy, 2)) for ox, oy in obstacles]}")
    print()

    violations = 0
    for s, x, y, theta, lin_vel, ang_vel, min_lidar, angle_to_goal in trace:
        stalling = min_lidar < 1.5 and lin_vel < 0.05
        violations += stalling
        flag = "  <-- STALL near obstacle" if stalling else ""
        print(f"step {s:3d}  pos=({x:5.2f},{y:5.2f})  theta={theta:5.2f}  "
              f"lin_vel={lin_vel:.2f}  ang_vel={ang_vel:.2f}  min_lidar={min_lidar:.2f}  "
              f"angle_to_goal={angle_to_goal:5.2f}{flag}")

    if terminated:
        print(f"\nResult: COLLISION (final reward={reward:.1f})")
    else:
        print("\nResult: TIMEOUT")

    print(f"\nStall violations (min_lidar<1.5 and lin_vel<0.05): {violations}/{len(trace)} steps")
    break
else:
    print(f"No failed obstructed-path episode found in {MAX_ATTEMPTS} attempts.")
