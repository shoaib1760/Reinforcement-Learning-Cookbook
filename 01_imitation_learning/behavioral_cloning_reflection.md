# Imitation Learning: Behavioral Cloning - Lab Report

**1. What is imitation learning?**
Imitation learning is a machine learning approach where an agent learns how to perform a task by observing and mimicking the behavior of an expert (often a human or a rule-based algorithm), rather than learning through trial-and-error rewards.

**2. What is behavioral cloning?**
Behavioral cloning is the simplest form of imitation learning. It treats the problem as a standard supervised learning task: the model is trained using the expert's observed states as inputs and the expert's chosen actions as the target labels.

**3. What are the inputs and outputs in this lab?**
- **Inputs (State `X`):** The agent's current position and the target's position.
- **Outputs (Action `y`):** The movement decision to make (`+1` for moving right, `-1` for moving left).

**4. Did the agent successfully imitate the expert?**
Yes, the agent successfully learned the expert's rule, achieving 100% test accuracy and effectively reaching the target during the small simulation testing phase.

**5. What is distribution shift?**
Distribution shift occurs when the data a model encounters during testing or deployment is significantly different from the data distribution it was trained on (e.g., the model was trained on positions between `-10` and `10`, but tested on a position like `50`).

**6. Why can imitation learning fail in unseen situations?**
Behavioral cloning models only know what they have been explicitly shown. When they encounter an unseen situation or make a small mistake that leads to an unfamiliar state, they do not know how to recover, often causing a chain reaction of errors.

**7. How could we improve this model?**
- **Increase Dataset Diversity:** Train the model on a wider and more varied range of states (e.g., larger starting/target positions).
- **DAgger (Dataset Aggregation):** Let the agent play in the environment, record the states where it is uncertain or makes mistakes, ask the expert for the correct actions in those specific states, and retrain the model.
- **Add Noise during Training:** Introduce slight random noise to the expert's trajectories to teach the model how to correct itself and recover from minor errors.
