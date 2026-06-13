# Multi-Agent Reinforcement Learning for Energy-Aware and Fault-Resilient Scheduling in Industrial IoT Manufacturing Systems

## Background

Modern manufacturing environments are distributed and dynamic, with heterogeneous machines, unpredictable workloads, and strict efficiency requirements. Traditional centralized control struggles to adapt when conditions change, while single-agent RL models lose autonomy and require retraining whenever the environment shifts.

MARL offers an alternative: each machine, or cluster of machines, is managed by its own agent, with agents coordinating through fixed communication protocols. This project uses the 100 prisoners hats puzzle as a controlled testbed (a Dec-POMDP with partial observability, sequential decision-making, and a fixed one-bit communication channel) to compare training strategies before applying insights to industrial scheduling problems.

## This Implementation

Each prisoner is modeled as an agent with its own actor and critic network. Critics are trained with access to the full global state (centralized training), while actors act using only local observations — the hats they can see and the announcements they've heard (decentralized execution). Discrete hat-color actions are handled via Gumbel-Softmax with a straight-through estimator, and training proceeds through a curriculum of increasing prisoner counts (`N = 5, 7, 10, 15, 20`).

The implementation includes CUDA-forced training, atomic checkpointing, and full auto-resume, so long runs can be safely interrupted and continued.

## Results

In the accompanying report, PPO, DDRQN, and MADDPG were each tested on a 5-prisoner version of the puzzle:

| Algorithm | Success Rate |
|---|---|
| PPO | 66% |
| DDRQN | 93% |
| MADDPG (this repo) | 51% |

DDRQN performed best overall and was used for the full-scale 100-prisoner experiment, reaching a 99.5% success rate. MADDPG, implemented here, is included as a CTDE comparison baseline and its lower performance reflects the added difficulty of adapting a continuous-action algorithm to this discrete, sequential puzzle via Gumbel-Softmax relaxation.
