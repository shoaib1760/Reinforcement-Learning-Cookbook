# 🏎️ Module 05: MountainCar Benchmark: Comparative Study

This module serves as a comprehensive benchmark, applying all four algorithmic families analyzed in this cookbook to the classic **MountainCar-v0** control problem. 

MountainCar-v0 is an iconic reinforcement learning testbed because its optimal path requires driving the car *away* from the goal (climbing the opposite hill) to build gravity-based momentum, presenting a difficult sparse-reward scenario.

---

## 📊 Summary of Results and Trade-offs

The notebook compare the performance of:
1. **Behavioral Cloning (IL)**
2. **Tabular Q-Learning (Value-Based)**
3. **REINFORCE (Policy-Based)**
4. **Advantage Actor-Critic (Actor-Critic)**

Here is a summary of how these methods perform on the MountainCar-v0 benchmark:

| Metric | Behavioral Cloning (BC) | Tabular Q-Learning | REINFORCE | A2C (Actor-Critic) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Type** | Supervised (Regression) | Tabular (Look-up Table) | Deep Neural Network | Joint Actor & Critic Networks |
| **Data Source** | Expert Demonstrations | Active Exploration | Active Exploration | Active Exploration |
| **Tuning Overhead** | Low | Medium (Discretization bins) | High (Learning rate, seeds) | Medium (Stable baseline) |
| **Sample Efficiency**| High (relative to expert size) | Low (Thousands of steps) | Medium | High (Fast convergence) |
| **Training Stability**| High (standard supervised) | Guaranteed (eventual) | Low (High variance) | High (Variance reduced) |
| **Performance** | Limited to expert accuracy | High (dependent on grid density)| Highly variable | **High & Stable** |

---

## 🎬 Agent Demonstration

A visualization of a fully trained agent solving the MountainCar environment can be viewed here:
- [**MountainCar Demo Video**](file:///d:/Reading_Resources/Master/RIL/lab/assets/mountaincar_demo.mp4) (Moved to local assets directory).

---

## 🧠 Key Takeaways

1. **Reward Shaping is Crucial**: Without modifying the environment's default sparse reward (which yields $-1$ per step), deep policy gradient methods struggle to find the goal. Introducing a potential-based reward using the car's height and velocity drastically improves training speed.
2. **Variance Reduction Works**: Moving from REINFORCE to A2C dramatically improves stability, demonstrating that subtracting a baseline (Critic estimate) is key to scaling policy gradient updates.
3. **Imitation vs. RL**: Behavioral Cloning reaches high performance instantly but cannot exceed the expert's quality. RL methods take longer but have the potential to discover more optimal, alternative velocity patterns.

---

## 🛠️ Module Files

- [`mountaincar_master_comparison.ipynb`](file:///d:/Reading_Resources/Master/RIL/lab/05_comprehensive_mountaincar/mountaincar_master_comparison.ipynb): The master notebook containing implementation blocks for all four methods side-by-side, comparison plots (rewards vs. episodes), and qualitative summaries.
