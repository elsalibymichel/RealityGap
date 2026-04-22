from typing import Any, Dict

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN

from learning_techniques.learning_technique import create_sb3_technique
from transferability.transferability_comparator import compare_task_transferability


#  Define the New Task Quality Function (f_q'), just for demonstration purposes
def centered_cartpole_reward(observation: Any, original_reward: float, done: bool, info: Dict[str, Any]) -> float:
    """
    New task: Penalize the agent for drifting away from the center of the track.
    CartPole track limits are -2.4 to 2.4.
    """
    cart_pos = observation[0]
    positional_penalty = abs(cart_pos) / 2.4
    new_reward = original_reward * (1.0 - positional_penalty)
    return float(np.clip(new_reward, 0.0, 1.0))



if __name__ == "__main__":
    env_base = gym.make("CartPole-v1")

    experiment_LTs = [
        create_sb3_technique(PPO, name="PPO", timesteps=3000, policy="MlpPolicy", device='cpu'),
        create_sb3_technique(DQN, name="DQN", timesteps=3000, policy="MlpPolicy")
    ]

    print("Starting Task Transferability Study: Standard vs Centered CartPole...")
    reference_quality = 500.0

    results = compare_task_transferability(
        env=env_base,
        reward_fn_prime=centered_cartpole_reward,
        reference_quality=reference_quality,
        reference_quality_prime=reference_quality,
        learning_techniques=experiment_LTs,
        n_repetitions=30,
        n_eval_episodes=100,
        save_fig=True,
        exp_name="results/task_transferability_ppo_vs_dqn"
    )