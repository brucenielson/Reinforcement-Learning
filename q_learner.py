from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from q_table import QModel


class QLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None)\
            -> None:
        super(QLearner, self).__init__(environment, num_states, num_actions, max_episodes)
        # Report states and actions in env
        # Create model
        self._q_model: QModel = QModel(num_states, num_actions)
        self._min_alpha: float = 0.05
        self._alpha: float = 0.1  # Learning Rate (alpha)

    def update_model(self, state: int, action: int, reward: float, new_state: int, done: bool = False) -> None:
        self._q_model.save_history(state, action, reward, new_state, done)
        self._q_model.update_q_model(state, action, reward, new_state, done, self._gamma, self._alpha)

    def print_progress(self, converge_count: int, score: float, avg_score: float):
        print("Episode:", self._episode, "Last High:", converge_count, "Epsilon:", round(self._epsilon, 4),
              "Alpha:", round(self._alpha, 4), "Score:", round(score, 2), "Avg Score:", round(avg_score, 2))

    # Getter for alpha
    @property
    def min_alpha(self) -> float:
        return self._min_alpha

    # Setter for alpha (discount factor)
    @min_alpha.setter
    def min_alpha(self, min_alpha: float) -> None:
        # Set discount factor
        self._min_alpha = min_alpha

    # Getter for alpha
    @property
    def alpha(self) -> float:
        return self._alpha

    # Setter for alpha (discount factor)
    @alpha.setter
    def alpha(self, alpha: float) -> None:
        # Set discount factor
        self._alpha = alpha

    # Overridden Getter for the model being used
    @property
    def q_model(self) -> QModel:
        return self._q_model
