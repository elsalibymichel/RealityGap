from typing import Any, Dict, Protocol, Type

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm

from misc.agent import Agent


class LearningTechnique(Protocol):
    """
    Protocol defining the structure of a learning technique.
    A learning technique encompasses the RL algorithm and any environment
    modifications (like mitigations) applied during the training phase.
    """

    @property
    def name(self) -> str:
        """Returns the descriptive name of the learning technique pipeline."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Returns a dictionary of hyperparameters for the entire pipeline."""
        ...

    def train(self, env: gym.Env) -> Agent:
        """
        Trains the agent on the provided environment and returns the trained model.

        Args:
            env: The base gymnasium environment to train on.

        Returns:
            A trained model conforming to the Agent protocol (must implement `predict`).
        """
        ...


def create_sb3_technique(
        algo_class: Type[BaseAlgorithm],
        name: str,
        timesteps: int,
        policy: str,
        **algo_kwargs: Any
) -> LearningTechnique:
    """
    A factory function that wraps any standard Stable-Baselines3 algorithm
    into a compliant LearningTechnique.

    Args:
        algo_class: The SB3 class (e.g., PPO, DQN, SAC).
        name: The descriptive name for the evaluator.
        timesteps: Total timesteps for the `.learn()` phase.
        policy: The policy string.
        **algo_kwargs: Any additional arguments to pass to the SB3 algorithm
                       (e.g., learning_rate, device).
    """

    class SB3Technique:
        @property
        def name(self) -> str:
            return name

        def get_config(self) -> Dict[str, Any]:
            return {
                "algorithm": algo_class.__name__,
                "name": name,
                "timesteps": timesteps,
                "policy": policy,
                **algo_kwargs
            }

        def train(self, env: gym.Env) -> Agent:
            model = algo_class(policy, env, verbose=0, **algo_kwargs)
            model.learn(total_timesteps=timesteps)
            return model

    return SB3Technique()