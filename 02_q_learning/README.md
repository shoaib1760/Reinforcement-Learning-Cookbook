# 🧭 Module 02: Value-Based RL & Tabular Q-Learning

Value-Based Reinforcement Learning focuses on estimating the expected cumulative reward of taking actions in given states. This module implements **Tabular Q-Learning**, a model-free, off-policy Temporal Difference (TD) control algorithm.

---

## 📖 Mathematical Formulation

The core of Q-Learning is learning the Action-Value function (or Q-function) $Q(s, a)$, which represents the expected long-term return for taking action $a$ in state $s$ and following the optimal policy thereafter:

$$Q^*(s, a) = \mathbb{E} \left[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \;\middle|\; S_t = s, A_t = a \right]$$

### The Q-Learning Update Rule:
During training, we update our estimate of $Q(s,a)$ iteratively as transition steps are observed in the environment:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \underbrace{\left( R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right)}_{\text{Temporal Difference (TD) Error}}$$

Where:
- $\alpha \in (0, 1]$ is the learning rate.
- $\gamma \in [0, 1)$ is the discount factor.
- $R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)$ is the **TD Target**.

### State Discretization
For environments with continuous state spaces (such as the position and velocity in MountainCar), tabular Q-learning requires converting these continuous metrics into discrete buckets. This maps an infinite state space onto a finite Q-table.

---

## 🤖 Connection to GenAI Practices: Reasoning and MCTS

While value-based tabular methods seem simple, the underlying framework of calculating "state quality" is a vital pillar in advanced reasoning models:

1. **Value Networks**: In reasoning architectures (such as AlphaGo or OpenAI's "Strawberry"/o1 series), standard autoregressive generation is augmented with search. A **Value Model** is trained to act as a critic, scoring intermediate thoughts or steps ($V(s)$ or $Q(s,a)$).
2. **Search-guided Generation**: Just as Q-learning finds the best path by selecting $\max_a Q(s,a)$, reasoning models use search algorithms like **Monte Carlo Tree Search (MCTS)** or Beam Search guided by a Value Model to explore different generation paths (chains of thought) and select the path that yields the highest expected accuracy.

---

## 🛠️ Module Files

- [`q_learning_delivery_rider.ipynb`](file:///d:/Reading_Resources/Master/RIL/lab/02_q_learning/q_learning_delivery_rider.ipynb): Focuses on implementing a basic tabular Q-learning environment from scratch (a delivery rider grid-world task), showing state transition matrices and Q-table convergence.
- [`q_learning_mountaincar.ipynb`](file:///d:/Reading_Resources/Master/RIL/lab/02_q_learning/q_learning_mountaincar.ipynb): Details continuous state discretization and tuning hyperparameters ($\epsilon$-greedy decay, learning rates) to solve the classic MountainCar-v0 task using Gym.
