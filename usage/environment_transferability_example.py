from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO, DQN

from learning_techniques.learning_technique import create_sb3_technique
from learning_techniques.rg_mitigation_techniques import DomainRandomizationMitigation, ObservationNoiseMitigation
from transferability import transferability_comparator
from environments.env_parameters_handler import apply_env_params



def phi_action_identity(action): return action
def phi_observation_inv_identity(obs): return obs



if __name__ == "__main__":
    ####################################################################################
    # Options: "rg_mitigations" or "rl_algorithms"
    experiment_type = "rg_mitigations"
    ####################################################################################

    src_config = {'gravity': 8.3, 'masscart': 1.1, 'masspole': 0.15, 'length': 0.45, 'force_mag': 9.7}
    dst_config = {'gravity': 9.8, 'masscart': 1.0, 'masspole': 0.1, 'length': 0.5, 'force_mag': 10.0}
    env_src = gym.make("CartPole-v1")
    apply_env_params(env_src.unwrapped, src_config)
    env_dst = gym.make("CartPole-v1")
    apply_env_params(env_dst.unwrapped, dst_config)
    base_ppo = create_sb3_technique(PPO, name="PPO", timesteps=3000, policy="MlpPolicy", device='cpu')
    base_dqn = create_sb3_technique(DQN, name="DQN", timesteps=3000, policy="MlpPolicy")
    if experiment_type == "rg_mitigations":
        experiment_LTs = [
            DomainRandomizationMitigation(
                base_technique=base_ppo,
                nominal_params=src_config,
                variation_pct=0.15
            ),
            ObservationNoiseMitigation(
                base_technique=base_ppo,
                noise_std=0.01
            ),
            base_ppo
        ]
        print("Starting Mitigation Transferability Study...")
    elif experiment_type == "rl_algorithms":
        experiment_LTs = [
            base_ppo,
            base_dqn,
        ]
        print("Starting Algorithm Transferability Study...")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    results = transferability_comparator.compare_environment_transferability(
        env=env_src,
        env_prime=env_dst,
        phi_action=phi_action_identity,
        phi_observation_inverse=phi_observation_inv_identity,
        learning_techniques=experiment_LTs,
        n_repetitions=30,
        n_eval_episodes=100,
        save_fig=True,
        exp_name=f"results/comparison_{experiment_type}"
    )