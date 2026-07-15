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
        self.danger_dist = 1.8
        self.stall_vel_frac = 0.25

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
        # half of episodes force at least one obstacle onto the direct spawn->goal
        # line, so avoidance is actually required instead of being optional most
        # of the time under the old fully-random placement.
        force_on_path = self.num_obstacles > 0 and self.np_random.uniform() < 0.5
        forced_placed = not force_on_path
        ux, uy = self.goal_x / dist, self.goal_y / dist
        perp_x, perp_y = -uy, ux
        while len(self.obstacles) < self.num_obstacles and attempts < 50:
            attempts += 1
            if not forced_placed:
                t = self.np_random.uniform(1.0, dist - 0.8)
                offset = self.np_random.uniform(-0.3, 0.3)
                ox, oy = ux * t + perp_x * offset, uy * t + perp_y * offset
            else:
                o_angle = self.np_random.uniform(-np.pi, np.pi)
                # lower bound gives >=1.2m surface clearance from spawn (o_dist - obstacle_radius),
                # so every episode has some reaction room instead of spawning already
                # inside (or right at the edge of) the collision-avoidance envelope
                o_dist = self.np_random.uniform(1.6, 4.5)
                ox, oy = o_dist * np.cos(o_angle), o_dist * np.sin(o_angle)
            too_close = any(np.hypot(ox - px, oy - py) < min_spacing for (px, py) in self.obstacles)
            # also keep clear of the goal itself — otherwise reaching it means
            # threading between goal_radius and obstacle_radius with near-zero
            # margin, which isn't "steering around an obstacle" so much as an
            # unsolvable spawn. visualize_policy.py already rejection-samples for
            # this same reason; enforcing it here covers eval/training too.
            too_close_to_goal = np.hypot(ox - self.goal_x, oy - self.goal_y) < self.obstacle_radius + 0.5
            if not too_close and not too_close_to_goal:
                self.obstacles.append((ox, oy))
                if not forced_placed:
                    forced_placed = True

        self._prev_min_obstacle_dist = float(np.min(self._get_lidar()))
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
        angle_to_goal = np.arctan2(self.goal_y - self.y, self.goal_x - self.x) - self.theta
        angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))
        lidar = self._get_lidar()
        min_obstacle_dist = float(np.min(lidar))
        lin_vel_frac = self.lin_vel / self.max_linear_vel

        reward = (self._prev_dist - dist_to_goal) * 10.0 - 0.05
        self._prev_dist = dist_to_goal

        # small constant incentive to keep moving forward, so stopping is never "safe"
        reward += lin_vel_frac * 0.05

        # turn-first shaping: nudge to rotate onto heading before driving, rather
        # than driving while pointed the wrong way. The penalty has to be bigger
        # than it looks: at moderate misalignment (~1.0-1.3 rad) cos(angle) is
        # still positive, so driving at full speed still banks a small amount of
        # the goal-progress term above plus the unconditional forward bonus. A
        # +-0.05 nudge only cancels that forward bonus (net zero vs turning), which
        # measurably failed to change behavior (lin_vel stayed pinned at max
        # regardless of angle_to_goal across 60 eval episodes). 0.25 is sized so
        # that driving fast while abs(angle_to_goal)>1.0 nets worse than stopping
        # to turn (which nets ~ -0.05/step from the fixed per-step cost alone),
        # while staying an order of magnitude below the +-100 terminal reward.
        if abs(angle_to_goal) < 0.3 and self.lin_vel > 0.05:
            reward += 0.1
        if abs(angle_to_goal) > 1.0 and self.lin_vel > 0.1:
            reward -= 0.25

        if min_obstacle_dist < self.danger_dist:
            closeness = (self.danger_dist - min_obstacle_dist) / self.danger_dist  # 0..~0.67

            # quadratic proximity penalty — ramps up sharply near collision_dist so
            # a straight-line approach can't out-earn it with goal-progress reward
            reward -= (closeness ** 2) * 4.0

            # reward actually increasing separation while in the danger zone. Pointing
            # away (below) isn't enough on its own — a policy can point away from an
            # obstacle while circling it at constant range forever. This term is what
            # tells the agent the maneuver is *working*, not just aimed correctly.
            clearance_gain = min_obstacle_dist - self._prev_min_obstacle_dist
            reward += clearance_gain * 2.0

            # much larger penalty for crawling/stalling instead of steering past —
            # this is what used to make "stop near the obstacle" look like the safe move
            if lin_vel_frac < self.stall_vel_frac:
                reward -= (self.stall_vel_frac - lin_vel_frac) * 4.0

            # reward pointing away from the nearest obstacle — this gives a direct
            # gradient toward turning, instead of leaving "steer around" as an
            # indirect side-effect of the collision penalty alone. Use the true
            # obstacle center (continuous in x/y) rather than the nearest lidar ray
            # index: the ray index is quantized to 15-degree bins and flips between
            # neighboring rays under tiny heading changes, which fed a chattering
            # left/right signal back into the policy (observed as ang_vel alternating
            # +/-max every single step instead of committing to one direction).
            nx, ny = min(self.obstacles, key=lambda o: np.hypot(o[0] - self.x, o[1] - self.y))
            bearing = np.arctan2(ny - self.y, nx - self.x) - self.theta
            bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
            reward += -np.cos(bearing) * 0.3 * closeness

        self._prev_min_obstacle_dist = min_obstacle_dist

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

    def path_blocked(self, margin=0.6):
        """True if some obstacle lies close enough to the straight line from
        spawn to goal that reaching the goal requires steering around it."""
        sx, sy = 0.0, 0.0
        dx, dy = self.goal_x - sx, self.goal_y - sy
        seg_len2 = dx * dx + dy * dy
        for (ox, oy) in self.obstacles:
            t = ((ox - sx) * dx + (oy - sy) * dy) / seg_len2
            t = max(0.0, min(1.0, t))
            cx, cy = sx + t * dx, sy + t * dy
            if np.hypot(ox - cx, oy - cy) < self.obstacle_radius + margin:
                return True
        return False

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