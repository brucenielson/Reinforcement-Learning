from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from dqn_model import DQNModel


class DQNLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None,
                 lr: float = 0.001) -> None:
        super(DQNLearner, self).__init__(environment, num_states, num_actions, max_episodes)
        # Report states and actions in env
        print("States: ", num_states, "Actions: ", num_actions)
        # Create model
        self.q_model = DQNModel(num_states, num_actions, lr=lr)

    def update_model(self, state: int, action: int, reward: float, new_state: int, done: bool = False) -> None:
        self.q_model.save_history(state, action, reward, new_state, done)
        self.q_model.update_q_model(state, action, reward, new_state, done, self.gamma, self.alpha)

    def print_progress(self, converge_count: int, score: float, avg_score: float):
        print("Episode:", self.episode, "Last High:", converge_count, "Epsilon:", round(self.epsilon, 4),
              "Score:", round(score, 2), "Avg Score:", round(avg_score, 2))
