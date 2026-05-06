# Deep RL Locomotion for Unitree Go2 (Isaac Sim)

![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)
![IsaacLab](https://img.shields.io/badge/IsaacLab-4.1.0-green.svg)
![DeepLearning](https://img.shields.io/badge/Deep_Learning-PyTorch-red.svg)

This repository contains the **Deep Reinforcement Learning (RL) locomotion training logic** required to drive the Unitree Go2 quadruped robot in NVIDIA Isaac Sim. 

This project isolates the core Deep Learning algorithms responsible for teaching the robot how to walk, maintain balance, and respond to velocity commands, completely independent of external navigation stacks (like ROS 2 or SLAM).

---

## 🧠 Deep Learning Foundations

The core architecture of this repository is deeply rooted in the foundational concepts of Modern Practical Deep Networks. The robot's "brain" is designed and optimized based on two critical pillars of Deep Learning:

### 1. Deep Feedforward Networks (Chapter 6)
The agent's policy (the brain, `best_agent.pt`) is structured as a **Multi-Layer Perceptron (MLP)**, a classic Deep Feedforward Network.
* **Input Layer (State):** Receives massive streams of proprioceptive data from the robot in real-time, including joint positions, joint velocities, base orientation (gravity vector), and the user's target velocity commands.
* **Hidden Layers:** Transforms these raw sensory inputs through non-linear activation functions (e.g., ELU/ReLU), extracting complex spatial and kinetic representations of the robot's state.
* **Output Layer (Action):** Outputs a continuous probability distribution mapping to target joint positions or torques for all 12 motors simultaneously.

### 2. Optimization for Training Deep Models (Chapter 8)
Training a quadruped to walk from scratch is a highly non-convex optimization problem. We utilize **PPO (Proximal Policy Optimization)** via the SKRL library to update the network weights.
* **Massive Parallelization:** We spawn **4,096 robots simultaneously** in Isaac Sim. This generates massive batch sizes, significantly reducing the variance of the gradient estimates and providing stable optimization.
* **Gradient Descent & Reward Maximization:** The Adam optimizer updates the network's weights iteratively. It strictly follows the gradients calculated via backpropagation to maximize the cumulative reward (e.g., matching target velocity, minimizing joint energy, and penalizing falls).

---

## ⚙️ Training Logic & Workflow

### The Environment (`flat_env_cfg.py`)
Before training, the environment dictates *what* the neural network should optimize for:
- **Rewards:** Positive points for following the target velocity (X, Y, Yaw). Negative points (penalties) for high energy consumption, base collisions, or joint limit violations.
- **Randomization:** We apply domain randomization (pushing the robot, adding noise to observations) to ensure the optimized weights are robust and prevent overfitting.

### Step 1: Training the Deep Neural Network
To begin the optimization process, run the training script. This will launch Isaac Sim in headless mode (no UI) to dedicate 100% of GPU resources to tensor calculations and network backpropagation.

```bash
./train_go2.sh
```
* **Process:** 4,096 agents interact with the physics engine. Every few seconds, the SKRL library collects the rollouts, calculates the gradients, and steps the optimizer to update the MLP weights.

### Step 2: Evaluating the Feedforward Network
Once optimization is complete, the updated weights are saved as a PyTorch file (`best_agent.pt`). To test how well the Feedforward Network maps real-time states to motor actions, run the play script:

```bash
./play.sh
```
* **Process:** This launches Isaac Sim with a UI. The network runs in `torch.inference_mode()`, meaning weights are frozen. It simply performs forward passes: taking in the current robot state + your keyboard velocity commands, and outputting motor torques in real-time.

---

## 📂 Repository Structure (Locomotion Focus)
* `scripts/reinforcement_learning/skrl/train.py` : The main script that configures the PPO optimizer and network architecture.
* `scripts/reinforcement_learning/skrl/play.py` : The inference script to test the Feedforward network.
* `train_go2.sh` / `play.sh` : Shell wrappers for launching Isaac Lab tasks.