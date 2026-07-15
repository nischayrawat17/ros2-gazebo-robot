from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from forklift_env import ForkliftNavEnv

check_env(ForkliftNavEnv(num_obstacles=0))

# Stage 1: goal-reaching, 4 parallel envs. Alignment reward shaping (turn-first)
# is unconditional in forklift_env.step(), so it's active from this stage on.
env1 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=0), n_envs=4)
model = PPO("MlpPolicy", env1, ent_coef=0.005, verbose=1,
            policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0))
model.learn(total_timesteps=200_000)
print("Stage 1 done — goal-reaching baseline trained.")

# Stage 2: one obstacle — explore going-around strategies. From here on, reset()
# forces an obstacle onto the direct spawn->goal line in ~50% of episodes, so
# avoidance is actually required instead of being optional most of the time.
env2 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=1), n_envs=4)
model.set_env(env2)
model.ent_coef = 0.01
model.learn(total_timesteps=1_000_000)
print("Stage 2 done — single-obstacle avoidance trained.")

# Stage 3: three obstacles — matches evaluate.py/debug_episode.py's default env,
# so the policy actually practices the scene it gets judged on
env3 = make_vec_env(lambda: ForkliftNavEnv(num_obstacles=3), n_envs=4)
model.set_env(env3)
model.ent_coef = 0.005
model.learn(total_timesteps=1_200_000)
print("Stage 3 done — three-obstacle navigation trained.")

model.save("forklift_ppo")
print("Training done. Model saved as forklift_ppo.zip")
