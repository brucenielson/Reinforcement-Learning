from q_learner_contracts import QTableContract, QLearnerContract
import numpy as np


class AlphaRequiredException(Exception):
    super().__init__("The QTable Class requires a learning rate (alpha)")


class QTable(QTableContract):
    def __int__(self, num_states: int, num_actions: int, epsilon: float, gamma: float, alpha: float):
        super(QTableContract, self).__init__(num_states, num_actions, epsilon, gamma, alpha)
        # Create a numpy table to be our Q Table
        self.model: np.ndarray = np.zeros((self.num_states, self.num_actions), dtype=np.single)

    def get_q_value(self, state: int, action: int) -> np.single:
        return self.model[state, action]

    def get_q_state(self, state: int) -> np.ndarray:
        return self.model[state]

    def set_q_value(self, state: int, action: int, value: float) -> None:
        self.model[state, action] = value

    def update_q_table(self, state: int, action: int, reward: float, new_state: object,
                       gamma: float, alpha: float = None) -> None:
        if alpha is None:
            raise AlphaRequiredException()
        self.model[state, action] += alpha * (reward + gamma * np.max(self.model[new_state])
                                              - self.model[state, action])

