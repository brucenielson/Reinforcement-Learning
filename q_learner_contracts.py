from signal import signal, SIGINT

# Globals
DEBUG = False
abort = False


# Setup a way to exit training using CTRL-C
# From: https://www.devdungeon.com/content/python-catch-sigint-ctrl-c
def handler(signal_received, frame):
    global abort
    print('CTRL-C detected. Exiting.')
    abort = True


signal(SIGINT, handler)


class QTableContract:
    def __init__(self):
        self.model = None

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    # Abstract methods

    # Given a state, return the current Q value of that state
    def get_q_value(self, state):
        pass

    # At a certain 'state' we took 'action' and received 'reward' and ended up in 'new_state'
    # Update the QTable to represent this
    def update_q_table(self, state, action, reward, new_state):
        pass

    # Keep track of every episode (state, action, reward, new_state, done)
    # Where 'done' is if the episode ended or not (for Lunar Lander is the episode complete?)
    # 'done' is optional as it's only used for Deep RL
    def save_history(self, state, action, reward, new_state, done=None):
        pass


class QLearnerContract:
    def __int__(self, num_states, num_actions, epsilon, gamma, alpha=None):
        self.epsilon = epsilon      # Chance of e-greedy random move
        self.gamma = gamma          # Future discount factor
        self.alpha = alpha          # Learning rate - not used for Deep RL
        self.num_states = None      # Number of world states
        self.num_actions = None     # Number of actions available per world state

