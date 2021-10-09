from q_learner_interfaces import IQModelInterface
import numpy as np


# class AlphaRequiredException(Exception):
#     super().__init__("The QTable Class requires a learning rate (alpha)")


class QModel(IQModelInterface):
    def __init__(self, num_states: int, num_actions: int) -> None:
        super(QModel, self).__init__(num_states, num_actions)
        self.model = None
        # Create a numpy table to be our Q Table
        self.model: np.ndarray = np.zeros((self.num_states, self.num_actions), dtype=np.single)
        # No need to store history for a Q Table
        self.ignore_history: bool = True

    def get_model(self) -> np.ndarray:
        return self.model

    def set_model(self, model: np.ndarray) -> None:
        self.model = model

    def get_q_value(self, state: int, action: int) -> float:
        return float(self.model[state, action])

    def get_q_state(self, state: int) -> np.ndarray:
        return self.model[state]

    def set_q_value(self, state: int, action: int, value: float) -> None:
        self.model[state, action] = value

    def update_q_model(self, state: int, action: int, reward: float, new_state: object, done: bool = False,
                       gamma: float = 0.9, alpha: float = 0.1) -> None:
        self.model[state, action] += alpha * (reward + gamma * np.max(self.model[new_state])
                                              - self.model[state, action])

    def q_sparseness(self) -> float:
        zeros = np.sum(self.model == 0)
        divisor = self.num_states * self.num_actions
        return float(zeros) / float(divisor)
