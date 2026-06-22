# Forklift Pallet-Avoidance System (ROS2 + Gazebo)

A simulated autonomous forklift built from scratch in ROS2 Humble and Gazebo Classic, navigating a warehouse environment and avoiding pallets, shelving, and other obstacles using real-time lidar sensing.

## Project Overview
This project simulates a differential-drive forklift operating inside a warehouse — the kind of automation problem found in real industrial AGV (Automated Guided Vehicle) systems. The forklift uses a 360° lidar sensor to detect nearby obstacles and react in real time, with camera-based pallet detection and autonomous goal navigation layered in next.

## Project Phases
- [x] Phase 0 — WSL2 + Ubuntu 22.04 + ROS2 Humble environment setup
- [x] Phase 1 — ROS2 core concepts (nodes, topics, workspace)
- [x] Phase 2 — URDF robot description, visualized in RViz
- [x] Phase 3 — Gazebo physics simulation with differential drive plugin
- [x] Phase 4 — ROS2 publisher node for robot motion control
- [x] Phase 5 — Lidar sensor integration + subscriber-based obstacle avoidance
- [x] Phase 6 — Forklift redesign + integration into a warehouse simulation environment
- [ ] Phase 7 — Camera-based pallet detection (OpenCV) + ROS2 action server for autonomous navigation
- [ ] Phase 8 — Documentation, demo video, final polish

## Stack
- ROS2 Humble
- Gazebo Classic 11
- Python (rclpy)
- URDF / SDF
- OpenCV (Phase 7+)

## Environment Credit
This project uses the [AWS RoboMaker Small Warehouse World](https://github.com/aws-robotics/aws-robomaker-small-warehouse-world) as the simulated environment. The robot model, control nodes, and all ROS2 logic in this repo are custom-built for this project.

## Notes
Built and tested in WSL2 (Ubuntu 22.04) using software rendering (`LIBGL_ALWAYS_SOFTWARE=1`) due to GPU passthrough limitations in WSL2.
