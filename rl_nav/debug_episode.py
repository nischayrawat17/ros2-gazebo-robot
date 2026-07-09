import numpy as np
from stable_baselines3 import PPO
from forklift_env import ForkliftNavEnv

env = ForkliftNavEnv(num_obstacles=3)
model = PPO.load("forklift_ppo")

for ep in range(2):
    obs, _ = env.reset()
    print(f"=== Episode {ep+1} ===")
    print(f"Goal: ({env.goal_x:.2f}, {env.goal_y:.2f})")
    print(f"Obstacles: {[(round(ox,2), round(oy,2)) for ox,oy in env.obstacles]}")
    print()

    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < 60:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        min_lidar = float(np.min(env._get_lidar()))
        print(f"step {step:3d}  pos=({env.x:5.2f},{env.y:5.2f})  theta={env.theta:5.2f}  "
              f"lin_vel={env.lin_vel:.2f}  ang_vel={env.ang_vel:.2f}  min_lidar={min_lidar:.2f}")
        step += 1

    if terminated and reward > 0:
        print("Result: SUCCESS\n")
    elif terminated:
        print(f"Result: COLLISION (final reward={reward:.1f})\n")
    else:
        print("Result: TIMEOUT\n")