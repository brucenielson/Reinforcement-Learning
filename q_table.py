from q_learner_interfaces import IQModelInterface
import numpy as np
import pickle


# class AlphaRequiredException(Exception):
#     super().__init__("The QTable Class requires a learning rate (alpha)")


class QModel(IQModelInterface):
    def __init__(self, num_states: int, num_actions: int) -> None:
        super(QModel, self).__init__(num_states, num_actions)
        # Create a numpy table to be our Q Table
        self.model: np.ndarray = np.zeros((self.num_states, self.num_actions), dtype=np.single)
        # No need to store history for a Q Table
        self.ignore_history: bool = True

    # Given a state and action, return the current Q value for that state / action combo
    def get_value(self, state: int, action: int) -> float:
        return float(self.model[state, action])

    def get_state(self, state: int) -> np.ndarray:
        return self.model[state]

    def set_value(self, state: int, action: int, value: float) -> None:
        self.model[state, action] = value

    def update_q_model(self, state: int, action: int, reward: float, new_state: object, done: bool = False,
                       gamma: float = 0.9, alpha: float = 0.1) -> None:
        self.model[state, action] += alpha * (reward + gamma * np.max(self.model[new_state])
                                              - self.model[state, action])

    def save_model(self, file_name: str = "QModel") -> None:
        pickle.dump(self.get_model(), open(file_name+".pkl", "wb"))

    def load_model(self, file_name: str = "QModel") -> None:
        self.set_model(pickle.load(open(file_name+".pkl", "rb")))

    def q_sparseness(self) -> float:
        zeros = np.sum(self.model == 0)
        divisor = self.num_states * self.num_actions
        return float(zeros) / float(divisor)
