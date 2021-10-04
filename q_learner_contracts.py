from signal import signal, SIGINT


class UnusedConstructor(Exception):
    def __init__(self):
        super().__init__("This constructor should not be used. It's just to make clear what variables exist to PyCharm.")


# Setup a way to exit training using CTRL-C
# From: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
def handler(signal_received, frame):
    print('CTRL-C detected. Exiting.')
    QLearnerContract.set_abort(True)


signal(SIGINT, handler)


class QTableContract:
    # Abstract methods
    def __int__(self, num_states: int, num_actions: int):
        self.num_states: int = num_states
        self.num_actions: int = num_actions

    def get_model(self) -> object:
        pass

    def set_model(self, model: object) -> None:
        pass

    # Given a state, return the current Q value for all actions in that state
    def get_q_state(self, state: object) -> object:
        pass

    # Given a state and action, return the current Q value for that state / action combo
    def get_q_value(self, state: object, action: int) -> object:
        pass

    # At a certain 'state' we took 'action' and received 'reward' and ended up in 'new_state'
    # Update the QTable to represent this
    def update_q_table(self, state: object, action: int, reward: float, new_state: object) -> None:
        pass

    # Keep track of every episode (state, action, reward, new_state, done)
    # Where 'done' is if the episode ended or not (for Lunar Lander is the episode complete?)
    # 'done' is optional as it's only used for Deep RL
    def save_history(self, state: object, action: int, reward: float, new_state: object, done: bool) -> None:
        pass


class QLearnerContract:
    abort: bool = False     # Abort flag to stop training
    debug: bool = False     # Debug flag to give debug info

    def __int__(self, num_states: int, num_actions: int, environment: object, num_episodes: int,
                epsilon: float, decay: float, gamma: float):
        self.num_states: int = 0            # Number of world states
        self.num_actions: int = 0           # Number of actions available per world state
        self.num_episodes = num_episodes    # Number of training episodes to run
        self.epsilon: float = epsilon       # Chance of e-greedy random move
        self.decay: float = decay           # Decay rate for epsilon and possibly alpha
        self.gamma: float = gamma           # Future discount factor
        self.abort: bool = False            # Abort flag to stop training
        self.debug: bool = False            # Debug flag to give debug info

    def e_greedy_action(self, state, e_greedy=True):
        pass

    def save_model(self, file_name="QModel"):
        pass

    def load_model(self, filename="QModel"):
        pass

    def train(self, decay_alpha=True):
        pass

    @staticmethod
    def set_debug(setting: bool) -> None:
        QLearnerContract.debug = setting

    @staticmethod
    def get_debug() -> bool:
        return QLearnerContract.debug

    @staticmethod
    def set_abort(setting: bool) -> None:
        QLearnerContract.abort = setting

    @staticmethod
    def get_abort() -> bool:
        return QLearnerContract.abort


class Environment:
    def __int__(self, learner: QLearnerContract, q_table: QTableContract) -> None:
        self.learner: QLearnerContract = learner
        self.q_table: QTableContract = q_table

    def run_episode(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def step(self, action: int):
        pass
