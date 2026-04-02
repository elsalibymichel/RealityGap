import warnings
from typing import Any, Callable, Dict

import gymnasium as gym



def update_cartpole(env: gym.Env) -> None:
    """
    Recalculates the derived mass and length parameters for the CartPole environment.
    This is required because CartPole computes these only once in its __init__ method.
    If the base parameters are changed dynamically, the derived parameters must be
    recalculated to keep the physics simulation accurate.

    Args:
        env: The unwrapped CartPole environment instance.
    """
    if (
            hasattr(env, "masspole")
            and hasattr(env, "masscart")
            and hasattr(env, "length")
    ):
        env.total_mass = env.masspole + env.masscart
        env.polemass_length = env.masspole * env.length

def update_pendulum(env: gym.Env) -> None:
    """
    No-operation update function for the Pendulum environment.
    The Pendulum environment does not require derived parameter recalculation.

    Args:
        env: The unwrapped Pendulum environment instance.
    """
    pass



# Registry mapping environment class names to their specific update functions
PHYSICS_UPDATER_REGISTRY: Dict[str, Callable[[gym.Env], None]] = {
    "CartPoleEnv": update_cartpole,
    "PendulumEnv": update_pendulum,
}



def get_derived_parameters_updater(
        env_unwrapped: gym.Env,
) -> Callable[[gym.Env], None]:
    """
    Retrieves the specific derived-parameter update function for the given environment.

    Issues a warning if the environment is not found in the registry, and
    returns a safe, no-operation fallback function to prevent crashes.

    Args:
        env_unwrapped: The base gymnasium environment instance.

    Returns:
        A callable function that updates the environment's derived physics.
    """
    env_class_name = env_unwrapped.__class__.__name__
    update_physics_fn = PHYSICS_UPDATER_REGISTRY.get(env_class_name)
    if update_physics_fn is None:
        available_envs = list(PHYSICS_UPDATER_REGISTRY.keys())
        warnings.warn(
            f"Environment '{env_class_name}' is not in the physics updater "
            f"registry {available_envs}. Falling back to a no-op function."
        )
        return lambda x: None
    return update_physics_fn



def apply_env_params(env_unwrapped: gym.Env, params: Dict[str, Any]) -> None:
    """
    Applies a dictionary of new physical parameters to the base environment and
    triggers the recalculation of any derived parameters.

    Args:
        env_unwrapped: The base gymnasium environment instance.
        params: A dictionary mapping parameter names to their new values.
    """
    for key, value in params.items():
        setattr(env_unwrapped, key, value)
    update_physics_fn = get_derived_parameters_updater(env_unwrapped)
    update_physics_fn(env_unwrapped)