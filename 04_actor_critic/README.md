# 🎭 Module 04: Actor-Critic Architectures & A2C

While Policy Gradient methods like REINFORCE learn a policy directly, they suffer from high variance because Monte Carlo returns $G_t$ can fluctuate wildly between episodes. **Actor-Critic** methods address this by combining policy-based and value-based RL. 

This module implements **Advantage Actor-Critic (A2C)**, which uses a value function baseline to stabilize training.

---

## 📖 Mathematical Formulation

An Actor-Critic agent splits the learning task into two components:
1. **The Actor**: Parametrizes the policy $\pi_\theta(a \mid s)$, determining which actions to take.
2. **The Critic**: Estimates the state-value function $V_\phi(s)$, evaluating the quality of the states the actor reaches.

### The Advantage Function
To determine if an action was better or worse than average, we compute the **Advantage** $A(s,a)$:
$$A(s, a) = Q(s, a) - V(s)$$

In A2C, we approximate the Advantage using the 1-step bootstrap Temporal Difference (TD) error:
$$A(s_t, a_t) \approx R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### Loss Functions
We optimize both networks simultaneously:
- **Actor Loss** (Policy Gradient weighted by Advantage):
  $$\mathcal{L}_{\text{actor}}(\theta) = -\frac{1}{B} \sum_{t=1}^B \log \pi_\theta(a_t \mid s_t) A(s_t, a_t)$$

- **Critic Loss** (Mean Squared Error between value estimate and TD target):
  $$\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{B} \sum_{t=1}^B \left( R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2$$

---

## 🤖 Connection to GenAI Practices: PPO, DPO, and GRPO

The Actor-Critic framework represents the gold standard architecture for LLM alignment:

### 1. PPO (Proximal Policy Optimization)
The standard RLHF alignment pipeline uses a four-model Actor-Critic setup:
- **Actor Model** (Policy $\pi_\theta$): The LLM generating responses.
- **Critic Model** (Value Network $V_\phi$): Evaluates the expected return of the response given the prompt prefix.
- **Reference Model** (Frozen SFT): Computes KL-divergence to ensure the Actor does not drift too far from natural language.
- **Reward Model**: Evaluates the final preference score.
The Actor's updates are optimized using token-level advantages computed by the Critic, making PPO a direct successor of the A2C architecture.

### 2. DPO (Direct Preference Optimization)
Recognizing that maintaining the Critic and Reward networks in GPU memory is highly resource-intensive, **DPO** bypasses the Actor-Critic cycle. By mathematically substituting the reward function with the policy likelihood ratio, DPO optimizes the LLM directly on preference pairs using standard cross-entropy, eliminating the need for a separate Critic network.

### 3. GRPO (Group Relative Policy Optimization)
Introduced in models like DeepSeek-R1, **GRPO** eliminates the Critic model entirely to save GPU memory. Instead of a value network estimating $V(s)$, GRPO samples a group of outputs (e.g., $G = 4$ or $8$) for a single prompt, computes their rewards, and normalizes them across the group. The average group reward serves as the baseline, and individual relative performance serves as the Advantage:
$$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
This allows training massive reasoning models using policy gradients without allocating memory for a secondary Critic model.

---

## 🛠️ Module Files

- [`a2c_mountaincar.ipynb`](file:///d:/Reading_Resources/Master/RIL/lab/04_actor_critic/a2c_mountaincar.ipynb): Contains the joint Actor-Critic neural network architecture, advantage calculations, bootstrap targets, and learning curves demonstrating faster and more stable convergence on MountainCar-v0 compared to REINFORCE.
