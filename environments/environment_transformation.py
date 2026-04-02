from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym



class EnvironmentTransformationWrapper(gym.Wrapper):
    r"""
    A wrapper that applies the environment transformation ($\phi$) to a destination
    environment ($e'$) to make it compatible with the source agent ($a$).
    """

    def __init__(
            self,
            env_prime: gym.Env,
            phi_action: Optional[Callable[[Any], Any]] = None,
            phi_observation_inverse: Optional[Callable[[Any], Any]] = None,
            phi_state: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        r"""
        Initializes the EnvironmentTransformationWrapper.

        Args:
            env_prime ($e'$): The destination environment.
            phi_action ($\phi_A$): Maps source action ($A$) $\to$ destination action ($A'$).
                Defaults to the identity function if None.
            phi_observation_inverse ($\phi_O^{-1}$): Maps destination observation ($O'$) $\to$
                source observation ($O$). Inverse is required to present $O$ to the agent.
                Defaults to the identity function if None.
            phi_state ($\phi_S$): Optional mapping for state initialization.
                Applied to the 'state' key within the reset options dict.
        """
        super().__init__(env_prime)
        # Fallbacks to identity functions if mappings are not provided
        self.phi_action = phi_action if phi_action is not None else lambda x: x
        self.phi_observation_inverse = (
            phi_observation_inverse if phi_observation_inverse is not None else lambda x: x
        )
        self.phi_state = phi_state

    def step(
            self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        action_prime = self.phi_action(action)
        observation_prime, reward, terminated, truncated, info = self.env.step(action_prime)
        observation = self.phi_observation_inverse(observation_prime)
        return observation, float(reward), terminated, truncated, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Assuming that the initial state (if present) is passed via a 'state' key in options
        if self.phi_state is not None and options is not None:
            if "state" in options:
                transformed_options = options.copy()
                # Apply the transformation phi_S to the source state
                transformed_options["state"] = self.phi_state(options["state"])
                options = transformed_options

        observation_prime, info = self.env.reset(seed=seed, options=options)
        observation = self.phi_observation_inverse(observation_prime)
        return observation, info