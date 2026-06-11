---
layout: default
title: "Project Submission"
---

# Project Submission

**Team Members:** 
- Minseok Lee (이민석) - 22100504
- Sunghwan Shim (심성환) - 22631005
- Hyunmo Kang (강현모) - 22100026

**Title:** Deep Reinforcement Learning Locomotion for Quadruped Robots: A Sim-to-Real Approach

**Summary:**

This project focuses on developing a robust locomotion controller for a quadruped robot (Unitree Go2) using Deep Reinforcement Learning (DRL) within the NVIDIA Isaac Sim environment. Traditional kinematic approaches to quadruped locomotion often struggle with complex dynamics, unpredictable terrains, and sudden velocity changes. To overcome this, we chose to implement a Deep Feedforward Network trained via Proximal Policy Optimization (PPO) using the SKRL framework. 

In this project, we detail the end-to-end pipeline of training this neural network policy in a massively parallelized simulation. First, we explain the state representations (proprioceptive sensor data) and the design of the reward functions critical for stable gait generation. Second, we discuss the optimization process, highlighting how spawning thousands of agents simultaneously accelerates the convergence of the deep models and prevents local minima. Finally, we demonstrate the inference phase, where the trained weights (`best_agent.pt`) are deployed to dynamically control the robot's 12-DoF joint torques in real-time. By isolating the deep learning locomotion component from the broader autonomy stack, this project provides a deep dive into the algorithmic foundations of modern robotic control, ultimately paving the way for seamless Sim-to-Real deployment.
