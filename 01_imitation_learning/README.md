# 🎯 Module 01: Imitation Learning & Behavioral Cloning

Imitation Learning (IL) bypasses the need for designing complex reward functions by training an agent to directly mimic the actions of an expert. This module explores **Behavioral Cloning (BC)**, the most straightforward form of Imitation Learning.

---

## 📖 Mathematical Formulation

In Behavioral Cloning, we treat policy learning as a standard Supervised Learning task. Given a dataset of expert demonstrations $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$:

- For **discrete actions**, we optimize a classification objective (Cross-Entropy loss):
  $$\mathcal{L}(\theta) = - \frac{1}{N} \sum_{i=1}^N \log \pi_\theta(a_i \mid s_i)$$

- For **continuous actions**, we optimize a regression objective (Mean Squared Error loss):
  $$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \| \pi_\theta(s_i) - a_i \|^2$$

where $\pi_\theta$ represents our neural network policy parameterized by weights $\theta$.

---

## 🤖 Connection to GenAI Practices: Supervised Fine-Tuning (SFT)

Imitation learning is the primary mechanism behind **Supervised Fine-Tuning (SFT)** of Large Language Models (LLMs):

1. **The Parallel**: In LLM SFT, the prompt acts as the environment state ($s$), and the expert-generated response acts as the expert action ($a$). The LLM is trained via behavioral cloning using next-token prediction (Cross-Entropy loss) to mimic the text distribution of human annotators or strong teacher models (e.g., GPT-4).
2. **The Limitation (Distribution/Covariate Shift)**: 
   - Traditional BC suffers from **compounding error**: when the agent makes a minor mistake $\epsilon$ at time $t$, it enters a state $s_{t+1}$ that was not visited by the expert. Because the policy was never trained on $s_{t+1}$, its behavior becomes highly unpredictable, leading to a cascade of failures.
   - In Generative AI, this directly explains **hallucination compounding**. If an LLM generates a slightly incorrect or off-topic token, its prompt prefix drifts out of the distribution of its SFT training corpus, making it highly susceptible to generating further nonsense or hallucinations.

```
Classic BC:  State s0 ---> Action a0' (Error ε) ---> Unseen State s1' ---> Complete Failure
LLM SFT:     Prompt s0 --> Typo/Error (Error ε) ---> Unseen Context s1' -> Hallucination Cascade
```

---

## 🛠️ Module Files

- [`behavioral_cloning_basics.ipynb`](file:///d:/Reading_Resources/Master/RIL/lab/01_imitation_learning/behavioral_cloning_basics.ipynb): Contains a step-by-step implementation of behavioural cloning on a simple grid movement task where an agent learns to approach a target position using rule-based expert training data.
- [`behavioral_cloning_reflection.md`](file:///d:/Reading_Resources/Master/RIL/lab/01_imitation_learning/behavioral_cloning_reflection.md): Written lab reflection covering behavioral cloning, distribution shifts, and mitigation strategies such as DAgger (Dataset Aggregation) and noise injection.
