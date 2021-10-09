from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from q_table import QModel


class QLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None)\
            -> None:
        super(QLearner, self).__init__(environment, num_states, num_actions, max_episodes)
        # Report states and actions in env
        print("States: ", num_states, "Actions: ", num_actions)
        # Create model
        self.q_model = QModel(num_states, num_actions)

    def update_model(self, state: int, action: int, reward: float, new_state: int, done: bool = False) -> None:
        self.q_model.save_history(state, action, reward, new_state, done)
        self.q_model.update_q_model(state, action, reward, new_state, done, self.gamma, self.alpha)
