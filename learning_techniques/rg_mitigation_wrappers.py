import gymnasium as gym
import numpy as np
from typing import Any, Dict, Optional, Tuple, Callable
from environments.env_parameters_handler import get_derived_parameters_updater



class DomainRandomizationWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for Domain Randomization.

    This wrapper randomizes physical parameters of the environment's underlying
    model at each reset. It is used to increase the robustness of RL agents
    against model inaccuracies and environment variations.
    """

    def __init__(
            self,
            env: gym.Env,
            nominal_params: Dict[str, float],
            variation_pct: float = 0.15
    ) -> None:
        """
        Initializes the DomainRandomizationWrapper.

        Args:
            env: The gymnasium environment to wrap.
            nominal_params: Mapping of parameter names to their base float values.
            variation_pct: The percentage of variation to apply at each reset.
        """
        super().__init__(env)
        self.nominal_params = nominal_params
        self.variation_pct = variation_pct
        self.update_physics_fn: Optional[Callable[[gym.Env], None]] = \
            get_derived_parameters_updater(self.unwrapped)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and applies new random physical parameters.

        Args:
            seed: The random seed for the reset.
            options: Additional options for the reset.

        Returns:
            A tuple containing the initial observation and environment info.
        """
        # Randomize base parameters within the specified percentage range
        for key, nominal_value in self.nominal_params.items():
            low = nominal_value * (1.0 - self.variation_pct)
            high = nominal_value * (1.0 + self.variation_pct)
            new_value = float(np.random.uniform(low, high))
            setattr(self.unwrapped, key, new_value)
        # Update dependent variables (e.g., moments of inertia, total mass)
        if self.update_physics_fn:
            self.update_physics_fn(self.unwrapped)
        return super().reset(seed=seed, options=options)



class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that injects Gaussian noise into observations.

    Simulates sensor inaccuracy or noisy data channels by adding zero-mean
    Gaussian noise to the observation vector during step() and reset().
    """

    def __init__(self, env: gym.Env, noise_std: float = 0.01) -> None:
        """
        Initializes the ObservationNoiseWrapper.

        Args:
            env: The gymnasium environment to wrap.
            noise_std: Standard deviation of the Gaussian noise.
        """
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Processes the observation by adding Gaussian noise.

        Args:
            observation: The raw observation from the environment.

        Returns:
            The noisy observation, cast to the original input data type.
        """
        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=observation.shape
        )
        noisy_observation = observation + noise
        return noisy_observation.astype(observation.dtype)



if __name__ == "__main__":

    print("--- Testing Domain Randomization: CartPole-v1 ---")
    env_cp = gym.make("CartPole-v1")
    dr_cp = DomainRandomizationWrapper(
        env_cp,
        nominal_params={'masscart': 1.0, 'masspole': 0.1}
    )
    for i in range(3):
        dr_cp.reset()
        m_c = getattr(dr_cp.unwrapped, 'masscart', 0.0)
        m_p = getattr(dr_cp.unwrapped, 'masspole', 0.0)
        t_m = getattr(dr_cp.unwrapped, 'total_mass', 0.0)
        print(f"Reset {i + 1} | mass_cart: {m_c:.3f} | mass_pole: {m_p:.3f} | total_mass: {t_m:.3f}")



    print("--- Testing Observation Noise: CartPole-v1 ---")
    base_env = gym.make("CartPole-v1")
    noisy_env = ObservationNoiseWrapper(base_env, noise_std=0.1)
    clean_obs, _ = base_env.reset(seed=42)
    noisy_obs, _ = noisy_env.reset(seed=42)
    print(f"Clean Initial Obs: {clean_obs}")
    print(f"Noisy Initial Obs: {noisy_obs}")
    print(f"Abs Difference:    {np.abs(noisy_obs - clean_obs)}")