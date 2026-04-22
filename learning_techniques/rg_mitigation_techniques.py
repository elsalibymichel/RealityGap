from typing import Dict, Any
import gymnasium as gym
from learning_techniques.rg_mitigation_technique import RGMitigationTechnique
from learning_techniques.learning_technique import LearningTechnique
from learning_techniques.rg_mitigation_wrappers import DomainRandomizationWrapper, ObservationNoiseWrapper

class DomainRandomizationMitigation(RGMitigationTechnique):
    """Applies Domain Randomization during training."""
    def __init__(
            self,
            base_technique: LearningTechnique,
            nominal_params: Dict[str, float],
            variation_pct: float = 0.15
    ):
        super().__init__(base_technique)
        self.nominal_params = nominal_params
        self.variation_pct = variation_pct

    def _mitigation_name(self) -> str:
        return "Domain Randomization"

    def wrap_environment(self, env: gym.Env) -> gym.Env:
        return DomainRandomizationWrapper(
            env,
            nominal_params=self.nominal_params,
            variation_pct=self.variation_pct
        )

    def _get_local_config(self) -> Dict[str, Any]:
        return {
            "dr_nominal_params": self.nominal_params,
            "dr_variation_pct": self.variation_pct
        }


class ObservationNoiseMitigation(RGMitigationTechnique):
    """Applies Observation Noise during training."""
    def __init__(self, base_technique: LearningTechnique, noise_std: float = 0.01):
        super().__init__(base_technique)
        self.noise_std = noise_std

    def _mitigation_name(self) -> str:
        return "Observation Noise"

    def wrap_environment(self, env: gym.Env) -> gym.Env:
        return ObservationNoiseWrapper(env, noise_std=self.noise_std)

    def _get_local_config(self) -> Dict[str, Any]:
        return {"obs_noise_std": self.noise_std}