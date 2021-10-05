# How to implement interfaces in python
# https://www.godaddy.com/engineering/2018/12/20/python-metaclasses/
# https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
# https://realpython.com/python-interface/
from signal import signal, SIGINT
from abc import ABC, abstractmethod


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


class IQTableInterface(ABC):
    def __init__(self, num_states: int, num_actions: int):
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
    def update_q_table(self, state: object, action: int, reward: float, new_state: object) -> None:
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

    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, num_episodes: int,
                 epsilon: float, decay: float, gamma: float):
        self.num_states: int = num_states       # Number of world states
        self.num_actions: int = num_actions     # Number of actions available per world state
        self.num_episodes = num_episodes        # Number of training episodes to run
        self.environment: IEnvironmentInterface = environment          # Environment to do updates
        self.epsilon: float = epsilon           # Chance of e-greedy random move
        self.decay: float = decay               # Decay rate for epsilon and possibly alpha
        self.gamma: float = gamma               # Future discount factor
        self.abort: bool = False                # Abort flag to stop training
        self.debug: bool = False                # Debug flag to give debug info

    @abstractmethod
    def e_greedy_action(self, state, e_greedy=True):
        pass

    @abstractmethod
    def save_model(self, file_name="QModel"):
        pass

    @abstractmethod
    def load_model(self, filename="QModel"):
        pass

    @abstractmethod
    def train(self, decay_alpha=True):
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
