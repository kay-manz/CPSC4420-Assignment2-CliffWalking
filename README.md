# CPSC 4420 — Assignment 2: Q-Learning in CliffWalking

This project implements **tabular Q-learning from scratch** for Gymnasium's `CliffWalking-v1` environment. It compares learning rate (alpha), discount factor (gamma), and exploration rate (epsilon), evaluates learned greedy policies, and investigates robustness to stochastic action execution (optional extra credit).

## Environment and method

- Environment: `CliffWalking-v1` (4 x 12 grid; 48 discrete states, 4 actions).
- Goal: reach the destination without falling into the cliff.
- Rewards: -1 per normal step; -100 for stepping into the cliff (which resets the agent to the start).
- Agent: tabular Q-learning, Q-table initialized to zeros, epsilon-greedy action selection.
- Training: 5,000 episodes per configuration, maximum 200 steps per episode.
- Baseline: alpha = 0.5, gamma = 0.9, epsilon = 0.1 (constant exploration).

## Run in Google Colab

1. Open `CPSC4420_Assignment2_CliffWalking.ipynb` in Google Colab (or click **Open in Colab** from the GitHub notebook view, if offered).
2. Run the first setup cell to install dependencies:

   ```python
   %pip install "gymnasium[toy-text]" numpy matplotlib imageio imageio-ffmpeg
   ```

3. Select **Runtime > Run all**. The notebook runs the experiments, produces plots, evaluates policies, and records an MP4 video. It may take a few minutes.
4. Find exported figures and the video in the `results/` directory. Colab-generated files need to be downloaded before the runtime resets.

For a local Jupyter notebook, install the packages listed in `requirements.txt` and open the notebook with Jupyter. The Colab-specific download cell at the end is optional and is intended only for Colab.

## Experimental results

**Baseline training:** average reward improved from -93.00 (first 100 episodes) to -50.29 (last 100); best episode reward was -13.

| Experiment | Setting | Mean reward, last 100 training episodes |
|---|---:|---:|
| Learning rate | alpha = 0.10 | -46.85 |
| Learning rate | alpha = 0.50 | -50.29 |
| Learning rate | alpha = 0.90 | -48.39 |
| Discount factor | gamma = 0.50 | -47.19 |
| Discount factor | gamma = 0.90 | -50.29 |
| Discount factor | gamma = 0.99 | -50.29 |
| Exploration rate | epsilon = 0.05 | -23.15 |
| Exploration rate | epsilon = 0.10 | -50.29 |
| Exploration rate | epsilon = 0.30 | -160.71 |

**Policy evaluation:** ten trained Q-tables were evaluated over 100 greedy episodes each. All attained -13 mean reward, 100% success, 13 mean steps and zero cliff falls in the deterministic environment. The baseline was retained as the final policy; it was **tied**, not uniquely better.

**Stochastic action noise (extra credit):** with probability `p`, the environment replaces the intended action with a different valid action. At 10% noise, the noise-trained policy achieved mean evaluation reward -25.63 versus -63.28 for the original baseline. At 25% noise, the corresponding values were -259.62 versus -171.51. Each comparison used 200 evaluation episodes. High-noise training was not reliably successful; the result was checked with additional evaluation seeds.

These results describe the reported training runs; they should not be interpreted as universal rankings across random seeds.

## Repository layout

```text
CPSC4420_Assignment2_CliffWalking.ipynb
README.md
requirements.txt
results/
    baseline_reward.png
    learning_rate_comparison.png
    discount_factor_comparison.png
    exploration_rate_comparison.png
    stochastic_training_comparison.png
    stochastic_robustness_comparison.png
    best_agent.mp4
```

The `results/` folder includes the required reward plots and a rendered video of the learned greedy policy. The video uses the standard deterministic environment. Experiment methodology, figures, and conclusions are discussed in the accompanying PDF report submitted through Canvas.
