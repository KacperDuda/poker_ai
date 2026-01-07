# Poker RL

A Reinforcement Learning environment for Poker (Texas Hold'em), featuring a Deep Q-Network (DQN) agent and a PyGame-based visualization.

## Overview

This project implements a custom Poker environment where AI agents can be trained to play No-Limit Texas Hold'em. It includes:

- A fully functional Poker environment (`poker_env.py`).
- A generic Agent interface with Random and Deep Learning implementations (`agent.py`).
- A training script using PyTorch to train a DQN agent (`train_rl.py`).
- A graphical user interface (GUI) built with PyGame to visualize the game (`gui.py`).

## Installation

Ensure you have Python 3.13 or higher installed.

1.  Clone the repository.
2.  Install the required dependencies.

You can install dependencies using `pip`:

```bash
pip install numpy pygame torch
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

This script will train a DQN agent over 200,000 episodes (by default) and save the model weights to `poker_dqn.pth` every 5,000 episodes. The training process uses a "Teacher" strategy (rule-based) for initial exploration.

### 2. Running a Demo

To watch the trained AI playing against random bots:

```bash
python run_demo.py
```

This script creates a 10-game session. If `poker_dqn.pth` is present, Player 0 will use the pre-trained AI. Otherwise, it defaults to a random agent.

### 3. UI Simulation

For a quick test of the GUI with random players:

```bash
python run_ui.py
```

## Project Structure

- **poker_env.py**: Core logic for the Texas Hold'em poker environment.
- **agent.py**: Agent definitions including `RandomAgent`, `DeepAgent` and the `PokerNet` neural network architecture.
- **train_rl.py**: Main training loop for the DQN agent using experience replay and target networks.
- **run_demo.py**: Script to run a visual demonstration of the AI.
- **gui.py**: PyGame class for rendering the poker table, cards, and player stats.
- **evaluator.py, card.py, deck.py**: Helper classes for card management and hand evaluation.

## Dependencies

- **numpy**: For numerical operations.
- **pygame**: For the game interface.
- **torch**: For the neural network and reinforcement learning.
