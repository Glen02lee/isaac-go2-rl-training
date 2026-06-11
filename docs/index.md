---
layout: default
title: "Deep RL Locomotion for Unitree Go2 (Isaac Sim)"
---

# Deep RL Locomotion for Unitree Go2 (Isaac Sim)

![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)
![IsaacLab](https://img.shields.io/badge/IsaacLab-4.1.0-green.svg)
![DeepLearning](https://img.shields.io/badge/Deep_Learning-PyTorch-red.svg)

---

## 👥 Team Members
- **Minseok Lee (이민석)** - 22100504 ([GitHub](https://github.com/Glen02lee))
- **Sunghwan Shim (심성환)** - 22631005 ([GitHub](https://github.com/hwan129))
- **Hyunmo Kang (강현모)** - 22100026 ([GitHub](https://github.com/hmkang012))

---

## 🎥 Project Presentation Video (YouTube)

<div align="center">
  <a href="https://youtu.be/mHaLuUE3lUo">
    <img src="https://img.youtube.com/vi/mHaLuUE3lUo/0.jpg" alt="Project Presentation Video" width="600">
  </a>
  <br><br>
  <p><i>Click the image above or use the link below to watch the full 7-minute presentation on YouTube:</i></p>
  <p><b>🔗 Video Link: <a href="https://youtu.be/mHaLuUE3lUo">https://youtu.be/mHaLuUE3lUo</a></b></p>
  <p><b>📄 Presentation Slides: <a href="./Isaac%20Go2%20RL%20Training.pdf">Isaac Go2 RL Training.pdf</a></b> / <b><a href="file:///C:/Users/USER/Desktop/캡스톤/go2/isaac-go2-rl-training/docs/Isaac%20Go2%20RL%20Training.pdf">Local Path</a></b></p>
</div>

---

## 📌 Table of Contents
1. [Introduction](#1-introduction)
2. [Task and Method](#2-task-and-method)
   - [A. State and Action Space](#a-state-and-action-space)
   - [B. Network Architecture (Chapter 6)](#b-network-architecture-chapter-6-deep-feedforward-networks)
   - [C. Optimization Strategy (Chapter 8)](#c-optimization-strategy-chapter-8-optimization-for-deep-models)
3. [Experiments & Implementation](#3-experiments--implementation)
   - [A. Prerequisites](#a-prerequisites-mandatory)
   - [B. Training the Model](#b-training-the-model)
   - [C. Inference (Testing)](#c-inference-testing)
   - [D. Codebase Mapping](#d-codebase-mapping)
4. [Results & Analysis](#4-results--analysis)
   - [A. Optimization and Regularization (Chapter 7)](#a-optimization-and-regularization-chapter-7-regularization)
   - [B. Visual Demonstrations](#b-visual-demonstrations)
5. [Conclusion & Future Work](#5-conclusion--future-work)
6. [References](#6-references)

---

## 1. Introduction

Controlling a quadrupedal robot involves coordinating a high-dimensional joint space (12 Degrees of Freedom) based on continuous multi-sensor feedback (IMUs, joint encoders, and contact sensors). Using traditional model-based control methods to mathematically model and integrate all these sensor dynamics is extremely complex and time-consuming, especially when dealing with unpredictable or rough terrains.

Furthermore, quadrupedal robots are highly sophisticated, expensive hardware. Directly testing unverified control algorithms or raw policies on physical robots in the real world is incredibly risky. Falls, collisions, and motor overloads can easily result in catastrophic hardware damage, significant financial loss, and safety hazards. 

To resolve these challenges, this project applies **Deep Reinforcement Learning (DRL)** in a high-fidelity physics simulator (**NVIDIA Isaac Sim** using the **Isaac Lab** framework). This approach allows the robot to learn optimal locomotion "reflexes" through massive, safe trial-and-error simulation, completely eliminating any risk to physical hardware. By training a deep neural network policy to map sensory feedback directly to motor commands, we establish a safe, stable, and cost-effective pipeline that bridges the gap between simulation and real-world deployment (Sim-to-Real).

---

## 2. Task and Method

We formulate the locomotion problem as a Reinforcement Learning task under a Markov Decision Process (MDP). The agent interacts with the Isaac Sim physics environment to learn an optimal policy $\pi_\theta(a|s)$ parameterized by the weights $\theta$ of a deep neural network.

### A. State and Action Space
* **State Space ($s_t$):** High-dimensional proprioceptive observations including the gravity vector (body orientation), joint positions and velocities of all 12 motors, and the user's velocity commands (linear and angular).
* **Action Space ($a_t$):** 12-dimensional continuous joint target angles, which are subsequently mapped to motor torques via Proportional-Derivative (PD) controllers.
* **Reward Function ($r_t$):** A multi-objective reward function that balances velocity tracking accuracy, base stability (keeping the body flat), and energy conservation (penalizing high joint torque changes and falls).

### B. Network Architecture (Chapter 6: Deep Feedforward Networks)
To approximate the control function, we design a Multi-Layer Perceptron (MLP) that processes state inputs and outputs motor commands in real-time.

```mermaid
graph LR
    %% Colors and Styles
    classDef inputLayer fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef hiddenLayer fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef outputLayer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef tensor fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5;

    %% Nodes
    I_State["Input Tensor<br>(Observation State)"]:::tensor
    
    subgraph MLP [Deep Feedforward Policy Network]
        direction LR
        L1["Input Layer<br>Dim: [N]"]:::inputLayer
        H1["Hidden Layer 1<br>Dim: [512]<br>Activation: ELU"]:::hiddenLayer
        H2["Hidden Layer 2<br>Dim: [256]<br>Activation: ELU"]:::hiddenLayer
        H3["Hidden Layer 3<br>Dim: [128]<br>Activation: ELU"]:::hiddenLayer
        O1["Output Layer<br>Dim: [12]"]:::outputLayer
    end

    O_Action["Output Tensor<br>(Joint Torques)"]:::tensor

    %% Connections
    I_State -->|e.g., Gravity, Joint Vel,<br>Command Vel| L1
    L1 --> H1
    H1 --> H2
    H2 --> H3
    H3 --> O1
    O1 -->|Mapped to 12 Motors| O_Action
```

* **Hidden Layers:** Three dense layers with **ELU (Exponential Linear Unit)** activations are utilized to handle negative inputs and avoid the vanishing gradient problem, allowing the network to capture complex, non-linear relationships.

### C. Optimization Strategy (Chapter 8: Optimization for Deep Models)
Training the policy network starting from a random initialization is a severely non-convex optimization problem with a sparse reward landscape. To achieve convergence and stability, we utilize:
* **Proximal Policy Optimization (PPO) Loss:** A surrogate objective function that clips probability updates, ensuring the policy does not deviate too far from the previous iteration, preventing catastrophic performance drops.
* **Adam Optimizer:** An adaptive learning rate optimization algorithm that computes individual learning rates for each weight based on running estimates of the first and second moments of the gradients.
* **Massively Parallel Rollouts:** We leverage GPU-accelerated simulation to spawn **4,096 parallel agents** in Isaac Sim. This generates extremely large and diverse mini-batches of experiences, significantly reducing gradient variance and accelerating backpropagation convergence.

```mermaid
flowchart TD
    %% Styling
    classDef env fill:#e8f4f8,stroke:#0277bd,stroke-width:2px,color:#000;
    classDef opt fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000;
    classDef weight fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;

    subgraph Isaac_Sim [NVIDIA Isaac Sim]
        Agents[Parallel Agents 1 to 4,096*]:::env
    end

    Batch[Massive Mini-Batch<br>State, Action, Reward]:::env
    Loss[PPO Surrogate Loss<br>Reward Maximization]:::loss
    Adam[Adam Optimizer<br>Gradient Descent]:::opt
    Weights[(MLP Weights)]:::weight

    Isaac_Sim -->|Collect Rollouts| Batch
    Batch --> Loss
    Loss -->|Backpropagation| Adam
    Adam -->|Update| Weights
    Weights -.->|Deploy Updated Policy| Isaac_Sim
```

---

## 3. Experiments & Implementation

### A. Prerequisites (Mandatory)
The training scripts run inside the Omniverse Python environment.
1. **NVIDIA Isaac Sim (5.1.0):** The physics and rendering simulator.
2. **Isaac Lab (4.1.0):** The RL wrapper framework. `isaaclab.sh` must be configured in your path.

### B. Training the Model
To initiate backpropagation and optimize network weights, run:
```bash
./train_go2.sh
```
* **Mechanism:** The script starts Isaac Sim in headless mode. The **SKRL** library coordinates the training loop, feeding state inputs to the MLP, computing the PPO loss, and updating weights using the Adam optimizer.

### C. Inference (Testing)
To test the generalized capabilities of the network:
```bash
./play.sh
```
* **Mechanism:** The model weights are frozen using `torch.inference_mode()`. Feedforward passes map sensory inputs directly to motor targets at 50 Hz.

### D. Codebase Mapping
* **[train.py](file:///C:/Users/USER/Desktop/캡스톤/go2/isaac-go2-rl-training/scripts/reinforcement_learning/skrl/train.py):** Sets up PPO hyperparameters, initializes the MLP, and runs the gradient descent loop.
* **[play.py](file:///C:/Users/USER/Desktop/캡스톤/go2/isaac-go2-rl-training/scripts/reinforcement_learning/skrl/play.py):** Implements real-time feedforward inference using `best_agent.pt`.

---

## 4. Results & Analysis

### A. Optimization and Regularization (Chapter 7: Regularization)
Deep RL is highly prone to instability; policies may overfit to recent exploration paths and "forget" how to walk (catastrophic forgetting). To prevent this, we apply **Early Stopping** (a core concept in Chapter 7).

```mermaid
xychart-beta
    title "Optimization & Early Stopping (Chapter 7 & 8)"
    x-axis "Training Iterations" [0, 1000, 2000, 3000, 4000, 5000]
    y-axis "Cumulative Reward" 0 --> 100
    bar [10, 40, 75, 95, 80, 65]
    line [10, 40, 75, 95, 80, 65]
```

The system continuously tracks episodic performance and automatically saves the checkpoint that achieved the **highest historical reward** as `best_agent.pt`. This acts as a regularization mechanism, extracting the most generalized model before performance degrades.

### B. Visual Demonstrations

<div align="center">
  <table style="width:100%; text-align:center;">
    <tr>
      <td><b>1. Early Training Phase<br><i>(Struggles to balance)</i></b></td>
      <td><b>2. Massively Parallel RL<br><i>(Gradient Stability)</i></b></td>
    </tr>
    <tr>
      <td><img src="./docs/images/early_training.gif" width="350" alt="Early Training Phase"><br><i>The agent struggles to maintain balance.</i></td>
      <td><img src="./docs/images/parallel_training.png" width="350" alt="Parallel Training"><br><i>4,096 agents trained simultaneously.</i></td>
    </tr>
  </table>

  <br><hr style="width:70%;"><br>

  <h3>3. SLAM Integration & Autonomous Navigation</h3>
  <table style="width:100%; text-align:center;">
    <tr>
      <td><b>3-1. ROS 2 RTAB-Map SLAM</b></td>
      <td><b>3-2. Autonomous Navigation (Nav2)</b></td>
    </tr>
    <tr>
      <td><img src="./docs/images/nav_slam.gif" width="350" alt="SLAM"><br><i>Mapping the environment in real-time.</i></td>
      <td><img src="./docs/images/주행테스트 성공.webm.gif" width="350" alt="Navigation"><br><i>Path planning combined with Deep RL locomotion.</i></td>
    </tr>
  </table>
</div>

---

## 5. Conclusion & Future Work

By utilizing a three-layer deep feedforward network trained via Proximal Policy Optimization (PPO), we successfully enabled the Unitree Go2 robot to learn stable locomotion behaviors. Deploying massive parallel rollouts in Isaac Sim minimized gradient variance, enabling quick optimization of a non-convex loss function. 

<div align="center">
  <img src="./docs/images/shaking%20hands.gif" width="400" alt="Real Go2 Waving"><br>
  <i>The real Unitree Go2 robot performing a physical action.</i>
</div>

**Future Work (Sim-to-Real):**
Our next milestone is deploying `best_agent.pt` directly onto physical hardware. To address the "reality gap" (differences in friction, motor latencies, and mass distributions), we will implement **Domain Randomization** during training, adding Gaussian noise to physical properties to ensure the learned weights are highly robust in the real world!

---

## 6. References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 6: Feedforward Networks, Chapter 7: Regularization, Chapter 8: Optimization).
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
3. Liang, J., Makoviychuk, V., Handa, A., Chentanez, N., Macklin, M., & Fox, D. (2018). GPU-accelerated robotic simulation for distributed reinforcement learning. *arXiv preprint arXiv:1810.05762*.
4. SKRL: Synthesizing Reinforcement Learning (https://github.com/ToniRV/skrl).
