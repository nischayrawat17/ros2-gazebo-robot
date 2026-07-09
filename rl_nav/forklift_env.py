import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ForkliftNavEnv(gym.Env):
    def __init__(self, num_obstacles=3):
        super().__init__()
        self.max_linear_vel = 0.2
        self.max_angular_vel = 0.5
        self.dt = 0.1
        self.max_steps = 500
        self.goal_radius = 0.3

        self.num_rays = 24
        self.lidar_max_range = 5.0
        self.num_obstacles = num_obstacles
        self.obstacle_radius = 0.4
        self.collision_dist = 0.5

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32)

        obs_dim = 4 + self.num_rays
        self.observation_space = spaces.Box(
            low=np.full(obs_dim, -2.0, dtype=np.float32),
            high=np.full(obs_dim, 2.0, dtype=np.float32),
            dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.lin_vel, self.ang_vel = 0.0, 0.0
        self.step_count = 0

        angle = self.np_random.uniform(-np.pi, np.pi)
        dist = self.np_random.uniform(2.0, 5.0)
        self.goal_x = dist * np.cos(angle)
        self.goal_y = dist * np.sin(angle)
        self._prev_dist = dist

        self.obstacles = []
        attempts = 0
        min_spacing = 2 * self.obstacle_radius + 0.5
        while len(self.obstacles) < self.num_obstacles and attempts < 50:
            attempts += 1
            o_angle = self.np_random.uniform(-np.pi, np.pi)
            o_dist = self.np_random.uniform(1.0, 4.5)
            ox, oy = o_dist * np.cos(o_angle), o_dist * np.sin(o_angle)
            too_close = any(np.hypot(ox - px, oy - py) < min_spacing for (px, py) in self.obstacles)
            if not too_close:
                self.obstacles.append((ox, oy))

        return self._get_obs(), {}

    def step(self, action):
        raw_lin = float(np.clip(action[0], -1.0, 1.0))
        raw_ang = float(np.clip(action[1], -1.0, 1.0))
        self.lin_vel = (raw_lin + 1.0) / 2.0 * self.max_linear_vel
        self.ang_vel = raw_ang * self.max_angular_vel

        self.theta += self.ang_vel * self.dt
        self.x += self.lin_vel * np.cos(self.theta) * self.dt
        self.y += self.lin_vel * np.sin(self.theta) * self.dt

        dist_to_goal = np.hypot(self.goal_x - self.x, self.goal_y - self.y)
        lidar = self._get_lidar()
        min_obstacle_dist = float(np.min(lidar))

        reward = (self._prev_dist - dist_to_goal) * 10.0 - 0.05
        self._prev_dist = dist_to_goal

        if min_obstacle_dist < 1.0:
            reward -= (1.0 - min_obstacle_dist) * 0.5

        terminated = False
        if dist_to_goal < self.goal_radius:
            reward += 100.0
            terminated = True
        elif min_obstacle_dist < self.collision_dist:
            reward -= 100.0
            terminated = True

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        if truncated:
            reward -= dist_to_goal * 5.0
        return self._get_obs(), reward, bool(terminated), bool(truncated), {}

    def _get_lidar(self):
        readings = []
        for i in range(self.num_rays):
            ray_angle = self.theta + (2 * np.pi * i / self.num_rays)
            dx, dy = np.cos(ray_angle), np.sin(ray_angle)
            min_dist = self.lidar_max_range
            for (ox, oy) in self.obstacles:
                ocx, ocy = self.x - ox, self.y - oy
                b = 2 * (dx * ocx + dy * ocy)
                c = ocx**2 + ocy**2 - self.obstacle_radius**2
                disc = b**2 - 4 * c
                if disc >= 0:
                    t = (-b - np.sqrt(disc)) / 2
                    if 0 < t < min_dist:
                        min_dist = t
            readings.append(min_dist)
        return np.array(readings, dtype=np.float32)

    def _get_obs(self):
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        dist = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx) - self.theta
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        lidar = self._get_lidar()

        base = np.array([
            dist / 10.0,
            angle / np.pi,
            self.lin_vel / self.max_linear_vel,
            self.ang_vel / self.max_angular_vel
        ], dtype=np.float32)
        lidar_norm = lidar / self.lidar_max_range
        return np.concatenate([base, lidar_norm]).astype(np.float32)