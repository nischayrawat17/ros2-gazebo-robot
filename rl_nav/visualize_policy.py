import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
from stable_baselines3 import PPO
from forklift_env import ForkliftNavEnv

# ── config ───────────────────────────────────────────────────────────────────
N_EPISODES     = 8
STEP_DELAY     = 60      # ms per frame — increase if video feels too fast
EPISODE_COLORS = [
    '#00ffcc','#00b4d8','#90e0ef','#48cae4',
    '#f72585','#7209b7','#3a86ff','#ffbe0b'
]

# ── load model ────────────────────────────────────────────────────────────────
env   = ForkliftNavEnv(num_obstacles=3)
model = PPO.load("forklift_ppo")

# ── run all episodes, fix goal-in-obstacle ────────────────────────────────────
print("Running episodes...")
episodes = []
for ep in range(N_EPISODES):
    while True:
        obs, _ = env.reset()
        goal_ok = all(
            np.hypot(env.goal_x - ox, env.goal_y - oy) > env.obstacle_radius + 0.5
            for (ox, oy) in env.obstacles
        )
        if goal_ok:
            break

    traj, lidar_frames = [(env.x, env.y, env.theta)], [env._get_lidar()]
    obstacles_snap = list(env.obstacles)
    goal_snap      = (env.goal_x, env.goal_y)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        traj.append((env.x, env.y, env.theta))
        lidar_frames.append(env._get_lidar())
    success = bool(terminated and reward > 0)
    episodes.append({
        "traj": traj, "lidar": lidar_frames,
        "obstacles": obstacles_snap,
        "goal": goal_snap, "success": success
    })
    print(f"  Ep {ep+1}: {'✓ SUCCESS' if success else '✗ FAIL'}")

# ── flatten into frame list ───────────────────────────────────────────────────
frames = []
for ep_idx, ep in enumerate(episodes):
    for step_idx in range(len(ep["traj"])):
        frames.append((ep_idx, step_idx))

print(f"Total frames: {len(frames)} → ~{len(frames)*STEP_DELAY//1000}s video")

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9), facecolor='#080c14')
ax.set_facecolor('#080c14')
ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
ax.set_aspect('equal')
ax.tick_params(colors='#3d5a80', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#1a2744')
ax.grid(True, color='#0e1726', linewidth=0.6, linestyle='--')
ax.set_title(
    "PPO Warehouse Forklift — Learned Navigation Policy\n"
    "Curriculum RL  ·  Stable-Baselines3  ·  Virginia Tech",
    color='white', fontsize=12, fontweight='bold', pad=14
)
ax.set_xlabel("x  (m)", color='#3d5a80', fontsize=9)
ax.set_ylabel("y  (m)", color='#3d5a80', fontsize=9)

# spawn marker
ax.plot(0, 0, 's', color='#f0e68c', markersize=9, zorder=5)
ax.annotate('SPAWN', (0.1, 0.15), color='#f0e68c', fontsize=7.5)

# ── ghost trail artists (one per episode, shown only after completion) ─────────
ghost_lines = []
for ep_idx in range(N_EPISODES):
    color = EPISODE_COLORS[ep_idx % len(EPISODE_COLORS)]
    for lw, alpha in [(4, 0.04), (2, 0.07), (1, 0.12)]:
        line, = ax.plot([], [], color=color, linewidth=lw,
                        alpha=alpha, zorder=2)
    ghost_lines.append(line)  # keep only the top layer ref per episode

# ── dynamic artists ───────────────────────────────────────────────────────────
lidar_lines = [ax.plot([], [], color='#00ffcc', alpha=0.12,
                        linewidth=0.7, zorder=3)[0]
               for _ in range(env.num_rays)]

trail_glow1, = ax.plot([], [], linewidth=5,   alpha=0.10, zorder=3)
trail_glow2, = ax.plot([], [], linewidth=2.5, alpha=0.20, zorder=4)
trail_main,  = ax.plot([], [], linewidth=1.2, alpha=0.90, zorder=5)

robot_dot,   = ax.plot([], [], marker=(3, 0, 0), markersize=13,
                        color='white', zorder=8)
arrow_ann    = ax.annotate(
    "", xy=(0,0), xytext=(0,0),
    arrowprops=dict(arrowstyle="-|>", color='#00ffcc',
                    lw=1.8, mutation_scale=14), zorder=9)

obs_patches  = []
goal_star,   = ax.plot([], [], '*', markersize=18, zorder=7)
goal_glow,   = ax.plot([], [], '*', markersize=28, alpha=0.20, zorder=6)

ep_text   = ax.text(0.03, 0.97, '', transform=ax.transAxes,
                     color='white', fontsize=12, fontweight='bold',
                     va='top', family='monospace')
suc_text  = ax.text(0.97, 0.97, '', transform=ax.transAxes,
                     color='#39d353', fontsize=12, fontweight='bold',
                     va='top', ha='right', family='monospace')
res_text  = ax.text(0.50, 0.04, '', transform=ax.transAxes,
                     color='white', fontsize=13, fontweight='bold',
                     va='bottom', ha='center', family='monospace')
step_text = ax.text(0.03, 0.03, '', transform=ax.transAxes,
                     color='#3d5a80', fontsize=8,
                     va='bottom', family='monospace')

legend_elements = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='#39d353',
           markersize=13, label='Goal', linestyle='None'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc2222',
           markersize=10, label='Obstacle', linestyle='None'),
    Line2D([0],[0], color='#00ffcc', lw=1.2, alpha=0.6, label='Lidar rays'),
    Line2D([0],[0], color='white',   lw=1.5,             label='Active path'),
    Line2D([0],[0], color='#aaaaaa', lw=1,   alpha=0.3,  label='Completed paths'),
]
ax.legend(handles=legend_elements, loc='lower right',
          facecolor='#0d1526', edgecolor='#1a2744',
          labelcolor='white', fontsize=8.5, framealpha=0.85)

completed_episodes = set()

def update(frame_idx):
    global obs_patches

    ep_idx, step_idx = frames[frame_idx]
    ep    = episodes[ep_idx]
    traj  = ep["traj"]
    color = EPISODE_COLORS[ep_idx % len(EPISODE_COLORS)]
    is_last_step = (step_idx == len(traj) - 1)

    # mark episode complete on its last frame
    if is_last_step:
        completed_episodes.add(ep_idx)

    # ── ghost trails: only show for already-completed episodes ────────────────
    for past_idx in range(N_EPISODES):
        if past_idx in completed_episodes and past_idx != ep_idx:
            xs = [t[0] for t in episodes[past_idx]["traj"]]
            ys = [t[1] for t in episodes[past_idx]["traj"]]
            ghost_lines[past_idx].set_data(xs, ys)
        else:
            ghost_lines[past_idx].set_data([], [])

    # ── obstacles ─────────────────────────────────────────────────────────────
    for p in obs_patches:
        p.remove()
    obs_patches = []
    for (ox, oy) in ep["obstacles"]:
        glow = patches.Circle((ox, oy), env.obstacle_radius + 0.18,
                               color='#ff4444', alpha=0.12, zorder=3)
        core = patches.Circle((ox, oy), env.obstacle_radius,
                               color='#cc2222', alpha=0.85, zorder=4)
        ax.add_patch(glow); ax.add_patch(core)
        obs_patches += [glow, core]

    # ── goal ──────────────────────────────────────────────────────────────────
    gx, gy = ep["goal"]
    goal_star.set_data([gx], [gy]); goal_star.set_color('#39d353')
    goal_glow.set_data([gx], [gy]); goal_glow.set_color('#39d353')

    # ── active trail ──────────────────────────────────────────────────────────
    xs = [t[0] for t in traj[:step_idx+1]]
    ys = [t[1] for t in traj[:step_idx+1]]
    for line in (trail_glow1, trail_glow2, trail_main):
        line.set_data(xs, ys)
        line.set_color(color)

    # ── robot ─────────────────────────────────────────────────────────────────
    x, y, theta = traj[step_idx]
    robot_dot.set_data([x], [y])
    robot_dot.set_color(color)
    robot_dot.set_marker((3, 0, np.degrees(theta) - 90))
    arrow_len = 0.4
    arrow_ann.xy     = (x + arrow_len * np.cos(theta),
                         y + arrow_len * np.sin(theta))
    arrow_ann.xytext = (x, y)
    arrow_ann.arrowprops['color'] = color

    # ── lidar ─────────────────────────────────────────────────────────────────
    for i, (line, dist) in enumerate(zip(lidar_lines, ep["lidar"][step_idx])):
        ray_angle = theta + (2 * np.pi * i / env.num_rays)
        line.set_data([x, x + dist * np.cos(ray_angle)],
                      [y, y + dist * np.sin(ray_angle)])

    # ── text ──────────────────────────────────────────────────────────────────
    successes = sum(episodes[i]["success"] for i in completed_episodes)
    ep_text.set_text(f"EP  {ep_idx+1} / {N_EPISODES}")
    suc_text.set_text(f"✓  {successes} / {len(completed_episodes)}")
    step_text.set_text(f"step {step_idx:>3d}  |  obstacles: {len(ep['obstacles'])}")

    if is_last_step:
        if ep["success"]:
            res_text.set_text("✓  GOAL REACHED")
            res_text.set_color('#39d353')
        else:
            res_text.set_text("✗  FAILED")
            res_text.set_color('#ff4444')
    else:
        res_text.set_text("")

    return (trail_glow1, trail_glow2, trail_main,
            robot_dot, goal_star, goal_glow,
            ep_text, suc_text, res_text, step_text,
            *lidar_lines, *ghost_lines)

print("Rendering...")
ani = FuncAnimation(fig, update, frames=len(frames),
                    interval=STEP_DELAY, blit=False)
writer = FFMpegWriter(fps=24, metadata=dict(title="Forklift RL Nav"), bitrate=2400)
ani.save("forklift_nav_demo.mp4", writer=writer, dpi=150)
print("Saved → forklift_nav_demo.mp4")
plt.close()