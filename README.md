# Pedestrian Street Crossing - Reinforcement Learning

This project implements a reinforcement learning solution for a pedestrian street crossing environment using the REINFORCE algorithm.

## Overview

The project trains an agent to make decisions in a street crossing scenario using policy gradient methods. The agent learns to navigate the environment optimally through reinforcement learning.

## Project Structure

- **train.py** - Main training script using REINFORCE algorithm
- **reinforce.py** - REINFORCE algorithm implementation
- **env.py** - Custom environment for pedestrian street crossing
- **visualize.py** - Visualization utilities
- **demo_rollout.py** - Demo rollout script
- **test_env_stats.py** - Environment statistics testing
- **ac_pedestrian.pt** - Trained model weights

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pedestrian-street-crossing-rl.git
cd pedestrian-street-crossing-rl
```

2. Install required packages:
```bash
pip install torch numpy matplotlib
```

## Usage

### Training

Run the training script to train the agent:
```bash
python train.py
```

### Visualization

Visualize the trained agent:
```bash
python visualize.py
```

### Demo Rollout

Run a demo rollout with the trained model:
```bash
python demo_rollout.py
```

## Environment

The environment simulates a pedestrian street crossing scenario where the agent must learn to:
- Observe the current state
- Make decisions about crossing
- Receive rewards for successful actions

## Algorithm

This project uses the **REINFORCE** (Policy Gradient) algorithm:
- Model-free policy gradient method
- Direct optimization of the policy
- Uses baseline for variance reduction

## Author

Your Name

## License

MIT License

---

For more details, refer to the individual Python files and docstrings.
