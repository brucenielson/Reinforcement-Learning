from q_learner_contracts import QLearnerContract, UnusedConstructor
from q_table import QTable
import pickle
import numpy as np


class QLearner(QLearnerContract):
    def __init__(self):
        self.average_over = None
        self.min_alpha = None
        self.min_epsilon = None
        raise UnusedConstructor()

    def __int__(self, num_states: int, num_actions: int, num_episodes: int,
                epsilon: float = 0.8, decay=0.9999, gamma: float = 0.99, lr: float = 0.001, alpha: float = 0.1):
        super(QLearnerContract, self).__init__(num_states, num_actions, num_episodes, epsilon, decay, gamma, lr, alpha)
        self.q_model = QTable(num_states, num_actions)

    def save_model(self, file_name: str = "QModel") -> None:
        pickle.dump(self.q_model.get_model(), open(file_name+".pkl", "wb"))

    def load_model(self, file_name: str = "QModel") -> None:
        self.q_model.set_model(pickle.load(open(file_name+".pkl", "rb")))

    def set_average_over(self, value: float) -> None:
        self.average_over = value

    def set_min_alpha(self, value: float) -> None:
        self.min_alpha = value

    def set_min_epsilon(self, value: float) -> None:
        self.min_epsilon = value

    def get_q_value(self, state: int, action: int) -> float:
        return self.q_model.get_q_value(state, action)

    def e_greedy_action(self, state: int, e_greedy: bool = True) -> int:
        # Get a random value from 0.0 to 1.0
        rand_val: float = float(np.random.rand())
        # Grab a random action
        action: int = np.random.randint(0, self.num_actions)
        # If we're doing e_greedy, then get a random action if rand_val < current epsilon
        if not (e_greedy and rand_val < self.epsilon):
            # Take best action instead of random action
            action = int(np.argmax(self.q_model.get_q_state(state)))
        return action

    def update_q_table(self, state: int, action: int, reward: float, new_state: int) -> None:
        self.q_model.update_q_table(state, action, reward, new_state, self.gamma, self.alpha)
