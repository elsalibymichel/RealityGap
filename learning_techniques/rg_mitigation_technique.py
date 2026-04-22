from abc import ABC, abstractmethod
from typing import Any, Dict
import gymnasium as gym

from learning_techniques.learning_technique import LearningTechnique
from misc.agent import Agent


class RGMitigationTechnique(LearningTechnique, ABC):
    """
    Base class for reality gap mitigation techniques.
    Acts as a Decorator over a base LearningTechnique.
    """
    def __init__(self, base_technique: LearningTechnique):
        self.base_technique = base_technique

    @property
    def name(self) -> str:
        """Dynamically generates the name based on the wrapper and base technique."""
        return f"{self.base_technique.name} + {self._mitigation_name()}"

    @abstractmethod
    def _mitigation_name(self) -> str:
        raise NotImplementedError("Subclasses must implement _mitigation_name.")

    @abstractmethod
    def wrap_environment(self, env: gym.Env) -> gym.Env:
        raise NotImplementedError("Subclasses must implement wrap_environment.")

    def _get_local_config(self) -> Dict[str, Any]:
        return {}

    def get_config(self) -> Dict[str, Any]:
        config = self.base_technique.get_config()
        config.update(self._get_local_config())
        return config

    def train(self, env: gym.Env) -> Agent:
        wrapped_env = self.wrap_environment(env)
        return self.base_technique.train(wrapped_env)