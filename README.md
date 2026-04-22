# Agent Transferability Evaluator

A modular Python framework designed to quantify the reality-gap and evaluate the transferability of autonomous agents. 

While highly applicable to Reinforcement Learning (RL), the underlying mathematical models and software architecture are strictly paradigm-agnostic. 
The framework natively supports evaluating agents trained via Evolutionary Algorithms, Imitation Learning, or classical Control Theory techniques. 
Whether transitioning an agent from simulation to the real world, modifying the underlying physics of an environment, or altering the task's reward function, this repository provides the formal tools required to benchmark model generalization.

---

## Use Cases

This framework is developed for machine learning researchers and control engineers who require the ability to:

* **Quantify the Environment Transferability Gap:** Measure performance degradation when moving an agent from a source environment to a destination environment, such as transitioning from a simulated environment to a real physical environment.
* **Benchmark Generalization Techniques:** Evaluate the effectiveness of robustness mitigations—such as Domain Randomization or Observation Noise—by composing them with base learning algorithms and analyzing the resulting transferability metrics.
* **Evaluate Task Transferability Gap:** Test how an agent, optimized for a specific objective, performs when evaluated under a distinctly different behavior quality function within the same underlying environment.

---

## Theoretical Framework and Modeling

This repository strictly separates the definitions of environments, learning algorithms, environment wrappers, and evaluation metrics, strictly adhering to a formal mathematical framework.

### 1. Key Concepts
* **Agent:** An agent is defined as a discrete-time dynamical system $(f_{a,state}, f_{a,out}, s_{a}^{(0)})$ characterized by an input space, output space, and state space.
* **Environment:** An environment is similarly defined as a discrete-time dynamical system $(f_{e,state}, f_{e,out}, s_{e}^{(0)})$.
* **Task:** A task is formalized as a tuple $(e, S_{e,init}, f_q)$, where $e$ is an environment, $S_{e,init}$ is a set of initial states, and $f_q$ is a behavior quality function.
* **Learning Technique ($L$):** A procedure that solves an optimization problem to maximize the behavior quality function, returning a learned agent $a^\ast$.

### 2. Transferability Metrics
The framework systematically measures two distinct types of transferability:

* **Environment Transferability ($\delta_{e \to e'}$):** This metric represents the effectiveness-gap of a learned agent $a^\ast$ when evaluated on a destination environment $e'$ compared to its source training environment $e$. 
    * To facilitate this, an environment transformation triplet $(\phi_A, \phi_O, \phi_S)$ is utilized to map actions, observations, and states, making the destination environment compatible with the agent. 
    * The resulting transformed environment is denoted as $\phi(e', \phi_A, \phi_O, \phi_S)$.
    * Formal Definition: $\delta_{e \to e'} = f_q(a^\ast, \phi(e', \phi_A, \phi_O, \phi_S), \phi_S(S_{e,test})) - f_q(a^\ast, e, S_{e,test})$.
* **Task Transferability ($\delta_{f_q \to f_q'}$):** This metric measures the effectiveness difference of the agent $a^\ast$ when evaluated using a new behavior quality function $f_q'$ compared against reference quality values.
    * Formal Definition: $\delta_{f_q \to f_q'} = \frac{f_q'(a^\ast, e, S_{e,test})}{f_q'^\ast} - \frac{f_q(a^\ast, e, S_{e,test})}{f_q^\ast}$.

### 3. Software Architecture
* **Paradigm-Agnostic `Agent` Protocol:** The evaluation engine requires the agent to implement a standard prediction method, mapping inputs to outputs as defined by the discrete-time dynamical system formalization.
* **`LearningTechnique` Protocol:** A standardized interface encapsulating the learning procedure $L$, requiring a `train(env)` method that returns a trained agent, alongside pipeline tracking configurations.
* **Decorator Pattern for Environment Transformations:** Environment transformations and reality-gap mitigations are modeled as decorators. They wrap a base learning technique, inject necessary environment transformations $(\phi_A, \phi_O, \phi_S)$ during the training phase, and return a cohesive evaluation pipeline.
* **`TransferabilityEvaluator`:** The core mathematical engine that computes the exact values for $\delta_{e \to e'}$ and $\delta_{f_q \to f_q'}$ through synchronized testing state trajectories.

---

## Usage Examples

The `usage/` directory contains fully executable scripts demonstrating the practical application of the evaluation framework. 
The following snippets highlight the core logic utilized in these scripts to measure both environment and task transferability.

### Example 1: Measuring Environment Transferability (Benchmarking Mitigations)

The following example demonstrates how to evaluate the environment transferability ($\delta_{e \to e'}$) of a Stable-Baselines3 PPO agent, testing whether Domain Randomization mitigates the reality-gap between two different physical configurations.

```python
import gymnasium as gym
from stable_baselines3 import PPO

from learning_techniques.learning_technique import create_sb3_technique
from learning_techniques.rg_mitigation_techniques import DomainRandomizationMitigation
from transferability import transferability_comparator
from environments.env_parameters_handler import apply_env_params

# 1. Define transformation mappings (phi_A, phi_O), identity in this case
def phi_action_identity(action): return action
def phi_observation_inv_identity(obs): return obs

# 2. Setup Source (e) and Destination (e') Environments
src_config = {'gravity': 8.3, 'masscart': 1.1}
dst_config = {'gravity': 9.8, 'masscart': 1.0}

env_src = gym.make("CartPole-v1")
apply_env_params(env_src.unwrapped, src_config)

env_dst = gym.make("CartPole-v1")
apply_env_params(env_dst.unwrapped, dst_config)

# 3. Create the Base Learning Algorithm (L)
base_ppo = create_sb3_technique(PPO, name="PPO", timesteps=10000, policy="MlpPolicy", device='cpu')

# 4. Compose Learning Techniques
experiment_LTs = [
    base_ppo,  # Baseline
    DomainRandomizationMitigation(
        base_technique=base_ppo, 
        nominal_params=src_config, 
        variation_pct=0.15
    )          # Mitigation
]

# 5. Execute the Comparator to measure delta_{e -> e'}
results = transferability_comparator.compare_environment_transferability(
    env=env_src,
    env_prime=env_dst,
    phi_action=phi_action_identity,
    phi_observation_inverse=phi_observation_inv_identity,
    learning_techniques=experiment_LTs,
    n_repetitions=3,
    n_eval_episodes=5,
    save_fig=True,
    exp_name="results/environment_transferability_dr_benchmark"
)
```

### Example 2: Measuring Task Transferability

This example evaluates the task transferability ($\delta_{f_q \to f_q'}$) of an agent trained on a standard objective ($f_q$) when evaluated against a modified behavior quality function ($f_q'$) within the same environment.

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN

from learning_techniques.learning_technique import create_sb3_technique
from transferability.transferability_comparator import compare_task_transferability

# 1. Define the New Task Quality Function (f_q')
def centered_cartpole_reward(observation, original_reward, done, info):
    """Modified behavior quality function penalizing drift from the center."""
    cart_pos = observation[0]
    positional_penalty = abs(cart_pos) / 2.4
    new_reward = original_reward * (1.0 - positional_penalty)
    return float(np.clip(new_reward, 0.0, 1.0))

# 2. Initialize the Base Environment (e)
env_base = gym.make("CartPole-v1")

# 3. Define the Algorithms for Comparison
experiment_LTs = [
    create_sb3_technique(PPO, name="PPO", timesteps=10000, policy="MlpPolicy", device='cpu'),
    create_sb3_technique(DQN, name="DQN", timesteps=10000, policy="MlpPolicy")
]

# 4. Execute the Task Comparator to measure delta_{f_q -> f_q'}
results = compare_task_transferability(
    env=env_base,
    reward_fn_prime=centered_cartpole_reward,
    reference_quality=500.0,         # f_q^\ast
    reference_quality_prime=500.0,   # f_q'^\ast
    learning_techniques=experiment_LTs,
    n_repetitions=2,
    n_eval_episodes=5,
    save_fig=True,
    exp_name="results/task_transferability_ppo_vs_dqn"
)
```

---

## Installation

### Requirements
* Python 3.8+
* `gymnasium`
* `stable-baselines3`
* `numpy`
* `matplotlib`

Clone the repository and install the specified dependencies:

```bash
git clone https://github.com/elsalibymichel/RealityGap.git
cd RealityGap

# 1. Install external dependencies
pip install -r requirements.txt

# 2. Install the framework in editable mode
pip install -e .
```

### Running an Example

Once the dependencies are installed, you can execute the provided benchmark scripts directly from the terminal to see the evaluation framework in action and generate the resulting comparison plots.

To run the environment transferability benchmark:

```bash
cd usage
python environment_transferability_example.py
```

To run the task transferability benchmark:

```bash
cd usage
python task_transferability_example.py
```

---

## Repository Structure

```text
├── environments/
│   ├── environment_transformation.py   # State/Action/Observation mapping (phi_A, phi_O, phi_S)
│   └── env_parameters_handler.py       # Dynamic physics recalculation routines
├── learning_techniques/
│   ├── learning_technique.py           # Protocol definitions and SB3 adapter
│   ├── rg_mitigation_technique.py      # Abstract Mitigation Decorator class
│   ├── rg_mitigation_wrappers.py       # Core Gymnasium observation/action wrappers
│   └── rg_mitigation_techniques.py     # Concrete implementations (DR, Noise)
├── misc/
│   └── agent.py                        # Paradigm-agnostic Agent protocol specification
├── transferability/
│   ├── transferability_evaluator.py    # Mathematical engine for delta calculations
│   └── transferability_comparator.py   # Training loop execution and visualization
└── usage/
    ├── environment_transferability_example.py
    └── task_transferability_example.py
```
