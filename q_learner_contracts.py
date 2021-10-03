from signal import signal, SIGINT
import numpy as np

# Globals
DEBUG = False
abort = False


class UnusedConstructor(Exception):
    super().__init__("This constructor should not be used. It's just to make clear what variables exist to PyCharm.")


# Setup a way to exit training using CTRL-C
# From: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
def handler(signal_received, frame):
    global abort
    print('CTRL-C detected. Exiting.')
    abort = True


signal(SIGINT, handler)


class QTableContract:
    # Abstract methods
    def __init__(self):
        self.model: object = None
        raise UnusedConstructor

    def __int__(self, num_states: int, num_actions: int):
        self.num_states: int = num_states
        self.num_actions: int = num_actions

    def get_model(self) -> object:
        pass

    def set_model(self, model: object) -> None:
        pass

    # Given a state, return the current Q value for all actions in that state
    def get_q_state(self, state: int) -> object:
        pass

    # Given a state and action, return the current Q value for that state / action combo
    def get_q_value(self, state: int, action: int) -> object:
        pass

    # At a certain 'state' we took 'action' and received 'reward' and ended up in 'new_state'
    # Update the QTable to represent this
    def update_q_table(self, state: int, action: int, reward: float, new_state: object,
                       gamma: float, alpha: float) -> None:
        pass

    # Keep track of every episode (state, action, reward, new_state, done)
    # Where 'done' is if the episode ended or not (for Lunar Lander is the episode complete?)
    # 'done' is optional as it's only used for Deep RL
    def save_history(self, state: int, action: int, reward: float, new_state: object, done: bool) -> None:
        pass


class QLearnerContract:
    def __int__(self, num_states: int, num_actions: int, num_episodes: int,
                epsilon: float, decay, gamma: float, lr: float, alpha: float) -> None:
        self.num_states: int = None         # Number of world states
        self.num_actions: int = None        # Number of actions available per world state
        self.num_episodes = num_episodes    # Number of training episodes to run
        self.epsilon: float = epsilon       # Chance of e-greedy random move
        self.gamma: float = gamma           # Future discount factor
        self.lr = lr                        # Regularization parameter (lr) - used for Deep RL
        self.alpha: float = alpha           # Learning rate - not used for Deep RL

    def e_greedy_action(self, state, e_greedy=True):
        pass

    def save_model(self, file_name="QModel"):
        pass

    def load_model(self, filename="QModel"):
        pass

