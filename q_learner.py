from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from q_table import QModel


class QLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None)\
            -> None:
        super(QLearner, self).__init__(environment, num_states, num_actions, max_episodes)
        # Report states and actions in env
        # Create model
        self.q_model = QModel(num_states, num_actions)
        self.min_alpha: float = 0.05
        self.alpha: float = 0.1                                 # Learning Rate (alpha)

    def update_model(self, state: int, action: int, reward: float, new_state: int, done: bool = False) -> None:
        self.q_model.save_history(state, action, reward, new_state, done)
        self.q_model.update_q_model(state, action, reward, new_state, done, self.gamma, self.alpha)

    def print_progress(self, converge_count: int, score: float, avg_score: float):
        print("Episode:", self.episode, "Last High:", converge_count, "Epsilon:", round(self.epsilon, 4),
              "Alpha:", round(self.alpha, 4), "Score:", round(score, 2), "Avg Score:", round(avg_score, 2))

    # Set the minimum learning rate (alpha) to not decay below
    def set_min_alpha(self, min_alpha: float) -> None:
        # Don't decay below this alpha
        self.min_alpha = min_alpha

    # Set learning rate (alpha)
    def set_alpha(self, alpha: float) -> None:
        # Set learning rate
        self.alpha = alpha
