import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from learning_techniques.learning_technique import LearningTechnique
from transferability.transferability_evaluator import TransferabilityEvaluator
from environments.environment_transformation import EnvironmentTransformationWrapper



class EnvTransferabilityResult(TypedDict):
    delta_transferability: List[float]
    delta_transferability_percentage: List[float]
    f_q_e: List[float]
    f_q_e_prime: List[float]
    config: Dict[str, Any]


class TaskTransferabilityResult(TypedDict):
    delta_task_transferability: List[float]
    f_q_original: List[float]
    f_q_prime: List[float]
    normalized_term_original: List[float]
    normalized_term_prime: List[float]
    config: Dict[str, Any]



def compare_environment_transferability(
        env: gym.Env,
        env_prime: gym.Env,
        phi_action: Callable[[Any], Any],
        phi_observation_inverse: Callable[[Any], Any],
        learning_techniques: List[LearningTechnique],
        n_repetitions: int,
        n_eval_episodes: int,
        save_fig: bool = True,
        eval_seeds: Optional[List[int]] = None,
        exp_name: Optional[str] = None
) -> Dict[str, EnvTransferabilityResult]:
    """
    Compares different RL configurations to evaluate their impact on transferability between two environments.

    Args:
        env: The source environment (training environment).
        env_prime: The destination environment (target environment).
        phi_action: Transformation function for actions (maps A -> A').
        phi_observation_inverse: Inverse transformation for observations (maps O' -> O).
        learning_techniques: List of the learning techniques to compare.
        n_repetitions: Number of times to repeat the training and evaluation process
            to account for stochasticity.
        n_eval_episodes: Number of episodes to run during each evaluation phase.
        save_fig: Whether to save the resulting comparison plot as 'results.png'.
        eval_seeds: Optional list of integer seeds to synchronize the initial states of
            the source and destination environments across evaluation repetitions. If set to None,
            the initial states will be not synchronized.
        exp_name: Optional name for the experiment.

    Returns:
        A dictionary mapping each learning technique name to its collected transferability metrics.
    """
    if eval_seeds is not None:
        if len(eval_seeds) != n_eval_episodes:
            raise ValueError("When seeds is provided, its length must match n_repetitions.")
    else:
        eval_seeds = [None] * n_eval_episodes
    results: Dict[str, EnvTransferabilityResult] = {
        technique.name: {
            "delta_transferability": [],
            "delta_transferability_percentage": [],
            "f_q_e": [],
            "f_q_e_prime": [],
            "config": technique.get_config()
        }
        for technique in learning_techniques
    }
    for technique in learning_techniques:
        name = technique.name
        for i in range(n_repetitions):
            print(f"Run {i + 1}/{n_repetitions} for LT: {name}...")
            env_src_train = copy.deepcopy(env)
            env_src_eval = copy.deepcopy(env)
            # Train the model via the protocol
            model = technique.train(env_src_train)
            # Wrap destination environment for transferability evaluation
            env_dst_wrapped = EnvironmentTransformationWrapper(
                env_prime=env_prime,
                phi_action=phi_action,
                phi_observation_inverse=phi_observation_inverse,
            )
            # Evaluate the agent's transferability
            rg_evaluator = TransferabilityEvaluator(agent=model)
            res = rg_evaluator.measure_environment_transferability(
                env_src=env_src_eval,
                env_wrapped_dst=env_dst_wrapped,
                n_episodes=n_eval_episodes,
                seeds=eval_seeds,
            )
            results[name]["delta_transferability"].append(res["delta_transferability"])
            results[name]["delta_transferability_percentage"].append(
                res["delta_transferability_percentage"]
            )
            results[name]["f_q_e"].append(res["f_q_e"])
            results[name]["f_q_e_prime"].append(res["f_q_e_prime"])
            env_src_train.close()
            env_src_eval.close()
            env_dst_wrapped.close()
    _print_summary_statistics(results)
    _plot_results(results, save_fig, exp_name)
    return results


def compare_task_transferability(
        env: gym.Env,
        reward_fn_prime: Callable[[Any, float, bool, Dict[str, Any]], float],
        reference_quality: float,
        reference_quality_prime: float,
        learning_techniques: List[LearningTechnique],
        n_repetitions: int,
        n_eval_episodes: int,
        save_fig: bool = True,
        exp_name: Optional[str] = None
) -> Dict[str, TaskTransferabilityResult]:
    """
    Compares different RL configurations to evaluate their impact on task transferability
    within the same environment but with a modified behavior quality function.

    Args:
        env: The base environment ($e$).
        reward_fn_prime: A function to calculate the new quality/reward ($f_q'$).
        reference_quality: Reference quality for the original task ($f_q^*$).
        reference_quality_prime: Reference quality for the new task ($f_q'^*$).
        learning_techniques: List of the learning techniques to compare.
        n_repetitions: Number of times to repeat the training/eval process.
        n_eval_episodes: Number of episodes to run during evaluation.
        save_fig: Whether to save the resulting comparison plot.
        exp_name: Optional name for the experiment file.

    Returns:
        A dictionary mapping each learning technique name to its task transferability metrics.
    """
    results: Dict[str, TaskTransferabilityResult] = {
        technique.name: {
            "delta_task_transferability": [],
            "f_q_original": [],
            "f_q_prime": [],
            "normalized_term_original": [],
            "normalized_term_prime": [],
            "config": technique.get_config()
        }
        for technique in learning_techniques
    }
    for technique in learning_techniques:
        name = technique.name
        for i in range(n_repetitions):
            print(f"Run {i + 1}/{n_repetitions} for LT: {name}...")
            env_train = copy.deepcopy(env)
            env_eval = copy.deepcopy(env)
            # Train the model
            model = technique.train(env_train)
            # Evaluate the agent's task transferability
            evaluator = TransferabilityEvaluator(agent=model)
            res = evaluator.measure_task_transferability(
                env=env_eval,
                reward_fn_prime=reward_fn_prime,
                reference_quality=reference_quality,
                reference_quality_prime=reference_quality_prime,
                n_episodes=n_eval_episodes,
            )
            results[name]["delta_task_transferability"].append(res["delta_task_transferability"])
            results[name]["f_q_original"].append(res["f_q_original"])
            results[name]["f_q_prime"].append(res["f_q_prime"])
            results[name]["normalized_term_original"].append(res["normalized_term_original"])
            results[name]["normalized_term_prime"].append(res["normalized_term_prime"])
            env_train.close()
            env_eval.close()
    _print_task_summary_statistics(results)
    _plot_task_results(results, save_fig, exp_name)
    return results



def _print_summary_statistics(results: Dict[str, Any]):
    """Prints a formatted summary of the results to the console."""
    for name, metrics in results.items():
        print(f"\n=== {name} ===")
        print(f"Source Quality f_q(e): Mean = {np.mean(metrics['f_q_e']):.2f}, SD = {np.std(metrics['f_q_e']):.2f}")
        print(
            f"Dest Quality f_q(phi(e')): Mean = {np.mean(metrics['f_q_e_prime']):.2f}, SD = {np.std(metrics['f_q_e_prime']):.2f}")
        print(
            f"Transferability delta(e -> e'): Mean = {np.mean(metrics['delta_transferability']):.2f}, SD = {np.std(metrics['delta_transferability']):.2f}")
        print(
            f"Transferability % delta(e -> e')%: Mean = {np.mean(metrics['delta_transferability_percentage']):.2f}%, SD = {np.std(metrics['delta_transferability_percentage']):.2f}")

def _plot_results(results: Dict[str, Any], save_fig: bool, exp_name: Optional[str]) -> None:
    """Generates a visual summary of the transferability metrics."""
    n_LTs = len(results)
    fig, axes = plt.subplots(n_LTs, 1, figsize=(12, 4 * n_LTs), squeeze=False)
    fig.suptitle("Transferability Metrics Comparison", fontsize=18, fontweight="bold", y=0.98)
    for idx, (name, metrics) in enumerate(results.items()):
        ax = axes[idx, 0]
        stats_rows = [
            (r"Source Quality $f_q(e)$:", np.mean(metrics['f_q_e']), np.std(metrics['f_q_e']), ""),
            (r"Dest Quality $f_q(\Phi(e'))$:", np.mean(metrics['f_q_e_prime']), np.std(metrics['f_q_e_prime']), ""),
            (r"Transferability $\delta_{e \to e'}$:", np.mean(metrics['delta_transferability']),
             np.std(metrics['delta_transferability']), ""),
            (r"Transferability % $\delta_{e \to e'} \%$:", np.mean(metrics['delta_transferability_percentage']),
             np.std(metrics['delta_transferability_percentage']), "%")
        ]
        ax.set_title(f"LT: {name}", fontsize=14, fontweight="bold", loc="left", color="#333333")
        ax.axis("off")
        rect = plt.Rectangle((0, 0), 1, 1, fill=True, color="#f7f7f7", alpha=0.3, transform=ax.transAxes, zorder=-1)
        ax.add_patch(rect)
        start_y = 0.85
        delta_y = 0.20
        for i, (label, mean, std, unit) in enumerate(stats_rows):
            current_y = start_y - (i * delta_y)
            ax.text(0.05, current_y, label, fontsize=12, ha="left", va="center")
            ax.text(0.60, current_y, f"Mean = {mean:.1f}{unit}", fontsize=12, ha="right", va="center")
            ax.text(0.95, current_y, f"SD = {std:.1f}{unit}", fontsize=12, ha="right", va="center")
            if i < len(stats_rows) - 1:
                ax.axhline(y=current_y - (delta_y / 2), color="gray", linestyle=":", linewidth=0.5, xmin=0.05,
                           xmax=0.95)
    # noinspection PyTypeChecker
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_fig:
        base_filename = exp_name if exp_name else "results"
        filename = f"{base_filename}.png"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_filename}({counter}).png"
            counter += 1
        plt.savefig(filename)
        print(f"\nPlot saved at {filename}")
    plt.show()



def _print_task_summary_statistics(results: Dict[str, Any]) -> None:
    """Prints a formatted summary of the task transferability results to the console."""
    for name, metrics in results.items():
        print(f"\n=== {name} ===")
        print(
            f"Orig Task Quality f_q: Mean = {np.mean(metrics['f_q_original']):.2f}, SD = {np.std(metrics['f_q_original']):.2f}")
        print(
            f"New Task Quality f_q': Mean = {np.mean(metrics['f_q_prime']):.2f}, SD = {np.std(metrics['f_q_prime']):.2f}")
        print(
            f"Task Transferability delta(f_q -> f_q'): Mean = {np.mean(metrics['delta_task_transferability']):.4f}, SD = {np.std(metrics['delta_task_transferability']):.4f}")

def _plot_task_results(results: Dict[str, Any], save_fig: bool, exp_name: Optional[str]) -> None:
    """Generates a visual summary of the task transferability metrics."""
    n_LTs = len(results)
    fig, axes = plt.subplots(n_LTs, 1, figsize=(12, 3.5 * n_LTs), squeeze=False)
    fig.suptitle("Task Transferability Metrics Comparison", fontsize=18, fontweight="bold", y=0.98)
    for idx, (name, metrics) in enumerate(results.items()):
        ax = axes[idx, 0]
        stats_rows = [
            (r"Orig Task Quality $f_q(a^*, e)$:", np.mean(metrics['f_q_original']),
             np.std(metrics['f_q_original'])),
            (r"New Task Quality $f_q'(a^*, e)$:", np.mean(metrics['f_q_prime']), np.std(metrics['f_q_prime'])),
            (r"Task Transferability $\delta_{f_q \to f_q'}$:", np.mean(metrics['delta_task_transferability']),
             np.std(metrics['delta_task_transferability']))
        ]
        ax.set_title(f"LT: {name}", fontsize=14, fontweight="bold", loc="left", color="#333333")
        ax.axis("off")
        rect = plt.Rectangle((0, 0), 1, 1, fill=True, color="#f7f7f7", alpha=0.3, transform=ax.transAxes, zorder=-1)
        ax.add_patch(rect)
        start_y = 0.80
        delta_y = 0.25
        for i, (label, mean, std) in enumerate(stats_rows):
            current_y = start_y - (i * delta_y)
            ax.text(0.05, current_y, label, fontsize=12, ha="left", va="center")
            fmt = ".4f" if i == 2 else ".1f"
            ax.text(0.60, current_y, f"Mean = {mean:{fmt}}", fontsize=12, ha="right", va="center")
            ax.text(0.95, current_y, f"SD = {std:{fmt}}", fontsize=12, ha="right", va="center")
            if i < len(stats_rows) - 1:
                ax.axhline(y=current_y - (delta_y / 2), color="gray", linestyle=":", linewidth=0.5, xmin=0.05,
                           xmax=0.95)
    # noinspection PyTypeChecker
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_fig:
        base_filename = exp_name if exp_name else "task_transferability_results"
        filename = f"{base_filename}.png"
        counter = 1
        while os.path.exists(filename):
            filename = f"{base_filename}({counter}).png"
            counter += 1
        plt.savefig(filename)
        print(f"\nPlot saved at {filename}")
    plt.show()