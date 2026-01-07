# Poker RL

A Reinforcement Learning environment for Poker (Texas Hold'em), featuring a Deep Q-Network (DQN) agent and a PyGame-based visualization.

## Overview

This project implements a custom Poker environment where AI agents can be trained to play No-Limit Texas Hold'em. It includes:

- A fully functional Poker environment (`poker_env.py`).
- A generic Agent interface with Random and Deep Learning implementations (`agent.py`).
- A training script using PyTorch to train a DQN agent (`train_rl.py`).
- A graphical user interface (GUI) built with PyGame to visualize the game (`gui.py`).
- An interactive application with replay controls and menu system (`poker_app.py`).

## Installation

Ensure you have Python 3.13 or higher installed.

1.  Clone the repository.
2.  Install the required dependencies.

You can install dependencies using `pip`:

```bash
pip install numpy pygame torch pandas matplotlib
```

Or using `uv` if you prefer the lockfile:

```bash
uv sync
```

## Usage

### 1. Training the AI

To train the reinforcement learning agent, run:

```bash
python train_rl.py
```

This script will train a DQN agent over 200,000 episodes (by default) and save the model weights to `poker_dqn.pth` every 5,000 episodes.

### 2. Running Experiments

To run comparative experiments with different configurations:

```bash
uv run run_experiments.py
```

This will train 4 different configurations and save results to `docs/experiments/`.

### 3. Interactive UI

To play or watch the AI in action with an interactive interface:

```bash
uv run poker_app.py
```

Features:

- **Menu System**: Choose between 1v1 (AI vs Random) or 4-bot simulation
- **Replay Controls**: Navigate through game history with arrow keys or mouse
- **Resizable Window**: Automatically scales interface while maintaining aspect ratio
- **Mouse Support**: Click menu options or use keyboard navigation

## Results

We conducted four training experiments (20,000 episodes each):

| Configuration            | Avg Reward | Win Rate  | Peak Win Rate |
| ------------------------ | ---------- | --------- | ------------- |
| **Baseline** (4p, h=512) | **772.2**  | 65.3%     | 100.0%        |
| Long Decay (slow ε)      | 448.3      | 51.8%     | 73.0%         |
| Big Network (h=1024)     | 734.0      | 65.0%     | 86.0%         |
| Heads-Up (2p)            | 421.4      | **74.4%** | 94.0%         |

**Key Findings**:

- The **Baseline** configuration (4 players, 512 hidden units) achieves the best overall performance
- Heads-Up mode has higher win rate but lower average reward due to different strategic dynamics
- Increasing network size to 1024 provides marginal improvement
- Slower epsilon decay underperforms the standard schedule

For detailed analysis, see `docs/report.pdf` or run:

```bash
uv run analyze_experiments.py
```

## Project Structure

```
poker_ai/
├── agent.py              # Agent interface and implementations (Random, DQN)
├── card.py               # Card data structure
├── deck.py               # Deck management
├── evaluator.py          # Hand evaluation logic
├── gui.py                # PyGame visualization
├── player.py             # Player data structure
├── poker_app.py          # Interactive application with menu and replay
├── poker_env.py          # Poker environment (game logic)
├── settings.py           # Global constants
├── train_rl.py           # Training script
├── run_experiments.py    # Comparative experiment runner
├── analyze_experiments.py# Result analysis and visualization
├── docs/
│   ├── report.pdf        # Technical report
│   ├── experiments_comparison.png
│   └── experiments/      # Experiment logs and models
└── README.md
```

## Dependencies

- **numpy**: For numerical operations.
- **pygame**: For the game interface.
- **torch**: For the neural network and reinforcement learning.
