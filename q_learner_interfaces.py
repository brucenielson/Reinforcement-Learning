# How to implement interfaces in python
# https://www.godaddy.com/engineering/2018/12/20/python-metaclasses/
# https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
# https://realpython.com/python-interface/
from signal import signal, SIGINT
from abc import ABC, abstractmethod
import math


class UnusedConstructor(Exception):
    def __init__(self):
        super().__init__("This constructor should not be used. It's to make clear what variables exist to PyCharm.")


# Setup a way to exit training using CTRL-C
# From: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
# noinspection PyUnusedLocal
def handler(signal_received, frame):
    print('CTRL-C detected. Exiting.')
    IQLearnerInterface.set_abort(True)


signal(SIGINT, handler)


# Function used to set decay if given max_episodes so that we get to min epsilon just before max_episodes
def calc_decay(max_ep: int, min_epsilon: float, target_percent: float = 0.8) -> float:
    target_episodes: float = float(max_ep) * target_percent
    return math.pow(min_epsilon, 1.0 / target_episodes)


class IQTableInterface(ABC):
    def __init__(self, num_states: int, num_actions: int) -> None:
        self.num_states: int = num_states
        self.num_actions: int = num_actions

    @abstractmethod
    def get_model(self) -> object:
        pass

    @abstractmethod
    def set_model(self, model: object) -> None:
        pass

    # Given a state, return the current Q value for all actions in that state
    @abstractmethod
    def get_q_state(self, state: object) -> object:
        pass

    # Given a state and action, return the current Q value for that state / action combo
    @abstractmethod
    def get_q_value(self, state: object, action: int) -> object:
        pass

    # At a certain 'state' we took 'action' and received 'reward' and ended up in 'new_state'
    # Update the QTable to represent this
    @abstractmethod
    def update_q_table(self, state: object, action: int, reward: float, new_state: object, done: bool = False) -> None:
        pass


class IEnvironmentInterface(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self, action: int):
        pass


class IQLearnerInterface(ABC):
    abort: bool = False     # Abort flag to stop training
    debug: bool = False     # Debug flag to give debug info

    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None):
        self.num_states: int = num_states                       # Number of world states
        self.num_actions: int = num_actions                     # Number of actions available per world state
        self.environment: IEnvironmentInterface = environment   # Environment to do updates
        self.max_episodes: int = max_episodes                   # Number of training episodes to run
        # Set min hyper parameters - don't decay below these values
        self.min_epsilon: float = 0.001
        self.min_alpha: float = 0.05
        # Defaults for hyper parameters
        self.epsilon: float = 0.99                              # Chance of e-greedy random move
        self.decay: float = 0.99                                # Decay rate for epsilon and possibly alpha
        self.gamma: float = 0.9                                 # Future discount factor
        self.alpha: float = 0.1                                 # Learning Rate (alpha)
        # If max_episodes is set, default decay to be 80% of that
        if max_episodes is not None:
            self.decay = calc_decay(max_episodes, self.min_epsilon, target_percent=0.9)
        # Flags
        self.abort: bool = False                                # Abort flag to stop training
        self.debug: bool = False                                # Debug flag to give debug info
        # Stats
        self.report_every_nth = 1                               # Show every nth episode. Defaults to every episode.

    def report_every_nth_episode(self, every_nth: int) -> None:
        self.report_every_nth = every_nth

    def recalculate_decay(self, target_percent: float = 0.9) -> None:
        if self.max_episodes is not None:
            self.decay = calc_decay(self.max_episodes, self.min_epsilon, target_percent=target_percent)

    def set_min_alpha(self, min_alpha: float) -> None:
        # Don't decay below this alpha
        self.min_alpha = min_alpha

    def set_min_epsilon(self, min_epsilon: float) -> None:
        # Don't decay below this epsilon
        self.min_epsilon = min_epsilon

    def set_gamma(self, gamma: float) -> None:
        # Set discount factor
        self.gamma = gamma

    def set_alpha(self, alpha: float) -> None:
        # Set learning rate
        self.alpha = alpha

    def set_decay(self, decay: float) -> None:
        # Set decay
        self.decay = decay

    def set_epsilon(self, epsilon: float) -> None:
        # Set epsilon (chance of random move)
        self.epsilon = epsilon

    @abstractmethod
    def get_e_greedy_action(self, state: object, e_greedy=True) -> int:
        pass

    @abstractmethod
    def save_model(self, file_name: str = "QModel") -> None:
        pass

    @abstractmethod
    def load_model(self, filename: str = "QModel") -> None:
        pass

    @abstractmethod
    def train(self, decay_alpha: bool = True) -> None:
        pass

    @staticmethod
    def set_debug(setting: bool) -> None:
        IQLearnerInterface.debug = setting

    @staticmethod
    def get_debug() -> bool:
        return IQLearnerInterface.debug

    @staticmethod
    def set_abort(setting: bool) -> None:
        IQLearnerInterface.abort = setting

    @staticmethod
    def get_abort() -> bool:
        return IQLearnerInterface.abort
