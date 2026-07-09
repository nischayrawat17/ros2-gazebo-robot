from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from forklift_env import ForkliftNavEnv

check_env(ForkliftNavEnv(num_obstacles=0))

# Stage 1: goal-reaching, 4 parallel envs
env1 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=0), n_envs=4)
model = PPO("MlpPolicy", env1, ent_coef=0.05, verbose=1)
model.learn(total_timesteps=200_000)
print("Stage 1 done — goal-reaching baseline trained.")

# Stage 2: one obstacle, high entropy to explore going-around strategies
env2 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=1), n_envs=4)
model.set_env(env2)
model.ent_coef = 0.1
model.learn(total_timesteps=600_000)
print("Stage 2 done — single-obstacle avoidance trained.")

# Stage 3: two obstacles — stop before catastrophic forgetting (~400k sweet spot)
env3 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=2), n_envs=4)
model.set_env(env3)
model.ent_coef = 0.05
model.learn(total_timesteps=400_000)
print("Stage 3 done — two-obstacle navigation trained.")

model.save("forklift_ppo")
print("Training done. Model saved as forklift_ppo.zip")
