from stable_baselines3 import PPO
from forklift_env import ForkliftNavEnv

env = ForkliftNavEnv()
model = PPO.load("forklift_ppo")

successes, collisions, timeouts = 0, 0, 0
n_episodes = 20
for ep in range(n_episodes):
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
    if terminated and reward > 0:
        successes += 1
    elif terminated:
        collisions += 1
    else:
        timeouts += 1

print(f"Success: {successes}/{n_episodes}  Collisions: {collisions}  Timeouts: {timeouts}")