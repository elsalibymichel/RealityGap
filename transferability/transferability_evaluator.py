import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from misc.agent import Agent
from environments.environment_transformation import EnvironmentTransformationWrapper



class TransferabilityEvaluator:
    r"""
    Tool to measure Environment Transferability ($\delta_{e \to e'}$).
    """

    def __init__(self, agent: Agent) -> None:
        """
        Initializes the RealityGapEvaluator.

        Args:
            agent: The trained agent. Must conform to the Agent protocol, which requires a predict method.
        """
        self.agent = agent

    def evaluate_quality(
            self,
            env: gym.Env,
            n_episodes: int,
            seeds: Optional[List[int]] = None,
            render: bool = False,
    ) -> float:
        """
        Evaluates the agent's quality (average reward) in a given environment.

        Args:
            env: The gymnasium environment to evaluate.
            n_episodes: The number of episodes to run.
            seeds: An optional list of seeds for each episode.
            render: Whether to render the environment during evaluation.

        Returns:
            The average reward obtained over the evaluated episodes.
        """
        total_rewards: List[float] = []
        assert seeds is None or len(seeds) == n_episodes, "Length of seeds list must match n_episodes when seeds are provided."
        for i in range(n_episodes):
            current_seed = seeds[i] if seeds is not None else None
            observation, _ = env.reset(seed=current_seed)
            done = False
            truncated = False
            episode_reward = 0.0
            while not (done or truncated):
                action, _ = self.agent.predict(observation, deterministic=True)
                observation, reward, done, truncated, _ = env.step(action)
                episode_reward += float(reward)
                if render:
                    env.render()
            total_rewards.append(episode_reward)
        return float(np.mean(total_rewards))

    def measure_environment_transferability(
            self,
            env_src: gym.Env,
            env_wrapped_dst: gym.Env,
            n_episodes: int,
            seeds: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        r"""
        Measures environment transferability ($\delta_{e \to e'}$).

        Args:
            env_src ($e$): The training environment.
            env_wrapped_dst ($\phi(e')$): The destination environment ALREADY wrapped
                with the compatibility transformation.
            n_episodes: Number of episodes to average $f_q$.
            seeds: Optional list of seeds for each episode to ensure identical
                trajectories in source and destination. If None, random seeds will be used,
                which may lead to a non-zero transferability gap even for identical environments.

        Returns:
            A dictionary containing the transferability metrics.
        """
        # Measure Baseline Effectiveness on Source (e), f_q(a*, e)
        fq_source = self.evaluate_quality(
            env_src, n_episodes=n_episodes, seeds=seeds
        )
        # Measure Transformed Effectiveness on Destination (phi(e')), f_q(a*, phi(e'))
        fq_dest = self.evaluate_quality(
            env_wrapped_dst, n_episodes=n_episodes, seeds=seeds
        )
        # Calculate Gap delta_e->e' = f_q(a*, phi(e')) - f_q(a*, e)
        delta = fq_dest - fq_source
        percentage = (delta / abs(fq_source)) * 100 if fq_source != 0 else 0.0
        return {
            "f_q_e": fq_source,
            "f_q_e_prime": fq_dest,
            "delta_transferability": delta,
            "delta_transferability_percentage": percentage,
        }

    def measure_task_transferability(
            self,
            env: gym.Env,
            reward_fn_prime: Callable[[Any, float, bool, Dict[str, Any]], float],
            reference_quality: float,
            reference_quality_prime: float,
            n_episodes: int,
    ) -> Dict[str, float]:
        r"""
        Measures Task Transferability ($\delta_{f_q \to f_q'}$).

        Args:
            env ($e$): The environment.
            reward_fn_prime ($f_q'$): A function to calculate the NEW quality/reward
                given (obs, original_reward, done, info).
            reference_quality ($f_q^*$): Reference quality for the original task.
            reference_quality_prime ($f_q'^*$): Reference quality for the new task.
            n_episodes: Number of evaluation runs.

        Returns:
            A dictionary containing the task transferability metrics.
        """
        total_rewards_original: List[float] = []
        total_rewards_prime: List[float] = []
        for _ in range(n_episodes):
            observation, _ = env.reset()
            done = False
            truncated = False
            ep_reward_orig = 0.0
            ep_reward_prime = 0.0
            while not (done or truncated):
                action, _ = self.agent.predict(observation, deterministic=True)
                next_obs, reward, done, truncated, info = env.step(action)
                # Accumulate Original f_q
                ep_reward_orig += float(reward)
                # Calculate and Accumulate New f_q'
                r_prime = reward_fn_prime(next_obs, float(reward), done, info)
                ep_reward_prime += r_prime
                observation = next_obs
            total_rewards_original.append(ep_reward_orig)
            total_rewards_prime.append(ep_reward_prime)
        avg_fq_original = float(np.mean(total_rewards_original))
        avg_fq_prime = float(np.mean(total_rewards_prime))
        term1 = avg_fq_prime / reference_quality_prime if reference_quality_prime != 0 else 0.0
        term2 = avg_fq_original / reference_quality if reference_quality != 0 else 0.0
        delta = term1 - term2
        return {
            "f_q_original": avg_fq_original,
            "f_q_prime": avg_fq_prime,
            "normalized_term_prime": term1,
            "normalized_term_original": term2,
            "delta_task_transferability": delta,
        }



if __name__ == "__main__":

    print("--- Testing Transferability Gap on Identical Environments ---")
    class DummyAgent(Agent):
        def predict(
                self,
                observation: np.ndarray,
                state: Optional[Any] = None,
                episode_start: Optional[Any] = None,
                deterministic: bool = True
        ) -> Tuple[int, None]:
            action = 0 if observation[0] < 0 else 1
            return action, None

    dummy_agent = DummyAgent()
    evaluator = TransferabilityEvaluator(agent=dummy_agent)
    env_source = gym.make("CartPole-v1")
    env_dest = gym.make("CartPole-v1")
    n_test_episodes = 100

    # Test with random seeds
    print(f"\nMeasuring {n_test_episodes} episodes with random seeds...")
    results_random = evaluator.measure_environment_transferability(
        env_src=env_source,
        env_wrapped_dst=env_dest,
        n_episodes=n_test_episodes,
    )
    print("Results (avg):")
    print(f"  Quality on Source (fq_e):      {results_random['f_q_e']:.2f}")
    print(f"  Quality on Dest (fq_e_prime):  {results_random['f_q_e_prime']:.2f}")
    print(f"  Delta Transferability:         {results_random['delta_transferability']:.2f}")
    print(f"  Delta Transferability %:       {results_random['delta_transferability_percentage']:.2f}")
    if results_random["delta_transferability"] != 0.0:
        print("-> SUCCESS! The trajectories were not identical.")
    else:
        print("-> ERROR: The trajectories were perfectly identical, which should not happen with random seeds.")

    # Test with assigned seeds
    evaluation_seeds = [
        int(np.random.randint(0, sys.maxsize)) for _ in range(n_test_episodes)
    ]
    print(f"\nMeasuring {n_test_episodes} repetitions with assigned seeds...")
    results_seeded = evaluator.measure_environment_transferability(
        env_src=env_source,
        env_wrapped_dst=env_dest,
        seeds=evaluation_seeds,
        n_episodes=n_test_episodes,
    )
    print("Results (avg):")
    print(f"  Quality on Source (fq_e):      {results_seeded['f_q_e']:.2f}")
    print(f"  Quality on Dest (fq_e_prime):  {results_seeded['f_q_e_prime']:.2f}")
    print(f"  Delta Transferability:         {results_seeded['delta_transferability']:.2f}")
    if results_seeded["delta_transferability"] == 0.0:
        print("-> SUCCESS! The trajectories were perfectly identical.")
    else:
        print("-> ERROR: The trajectories were not identical, which should not happen when using the same seeds.")



    print("\n--- Testing Task Transferability Gap ---")
    # Define a new reward function that penalizes the agent for moving away from the center
    def custom_reward_fn(observation: Any, original_reward: float, done: bool, info: Dict[str, Any]) -> float:
        cart_pos = observation[0]
        positional_penalty = abs(cart_pos) / 2.4
        new_reward = original_reward * (1.0 - positional_penalty)
        return float(np.clip(new_reward, 0.0, 1.0))
    ref_quality_orig = 500.0
    ref_quality_prime = 500.0
    print("Measuring task gap between standard CartPole and Centered-CartPole...")
    results_task = evaluator.measure_task_transferability(
        env=env_source,
        reward_fn_prime=custom_reward_fn,
        reference_quality=ref_quality_orig,
        reference_quality_prime=ref_quality_prime,
        n_episodes=n_test_episodes
    )
    print("Results:")
    print(f"  Avg Quality Original Task (f_q):      {results_task['f_q_original']:.2f}")
    print(f"  Avg Quality New Task (f_q_prime):     {results_task['f_q_prime']:.2f}")
    print(f"  Normalized Original (f_q / f_q*):     {results_task['normalized_term_original']:.4f}")
    print(f"  Normalized New (f_q' / f_q'*):        {results_task['normalized_term_prime']:.4f}")
    print(f"  Delta Task Gap (Task Transferability):{results_task['delta_task_transferability']:.4f}")
    if results_task["f_q_original"] > results_task["f_q_prime"]:
        print("-> SUCCESS! The new task correctly applied penalties, resulting in a negative delta.")
    else:
        print("-> Note: Check reward math, original performance should be higher or equal.")



    print("\n--- Testing state transformation (phi_state) with CartPole ---")
    # Standard CartPole ignores options
    # Here a wrapper to force the simulator to accept injected states
    class InjectableCartPole(gym.Wrapper):
        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            if options is not None and "state" in options:
                # Force the internal simulator state
                self.unwrapped.state = np.array(options["state"], dtype=np.float32)
                # Re-calculate observation based on injected state
                observation = np.array(self.unwrapped.state, dtype=np.float32)
            return observation, info
    dest_env_injectable = InjectableCartPole(gym.make("CartPole-v1"))
    # Source agent only knows a 1D state
    # Destination CartPole requires a 4D state:
    def phi_state(source_state: List[float]) -> np.ndarray:
        cart_pos = source_state[0]
        # Translate 1D source state to 4D destination state
        return np.array([cart_pos, 0.0, 0.0, 0.0], dtype=np.float32)
    wrapped_dest_env = EnvironmentTransformationWrapper(
        env_prime=dest_env_injectable,
        phi_state=phi_state,
    )
    # Request a reset using the 1D Source State format
    source_initial_state = [1.5]
    print(f"Requested Source State (1D): {source_initial_state}")
    obs, _ = wrapped_dest_env.reset(options={"state": source_initial_state})
    print(f"Destination Environment Actual State (4D): {dest_env_injectable.unwrapped.state}")
    if np.allclose(dest_env_injectable.unwrapped.state, [1.5, 0.0, 0.0, 0.0]):
        print("-> SUCCESS! phi_state correctly translated the 1D state to the 4D environment.")
    else:
        print("-> ERROR: State was not transformed correctly.")