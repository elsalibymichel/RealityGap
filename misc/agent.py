from typing import Any, Optional, Protocol, Tuple


class Agent(Protocol):
    """
    Structural type for an agent compatible with the TransferabilityEvaluator.

    Theoretically, this represents a discrete-time dynamical system
    (f_{a,state}, f_{a,out}, s_a^{(0)}).
    """

    def predict(
            self,
            observation: Any,
            state: Optional[Any] = None,
            episode_start: Optional[Any] = None,
            deterministic: bool = True
    ) -> Tuple[Any, Optional[Any]]:
        """
        Computes the action and the next state.

        Args:
            observation: The current observation from the environment (I_a).
            state: The current internal state of the agent (s_a).
            episode_start: Indicator if the episode just started.
            deterministic: Whether to use deterministic or stochastic actions.

        Returns:
            A tuple containing the action (O_a) and the next state (S_a).
        """
        ...