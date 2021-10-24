# How to implement interfaces in python
# https://www.godaddy.com/engineering/2018/12/20/python-metaclasses/
# https://stackoverflow.com/questions/2124190/how-do-i-implement-interfaces-in-python
# https://realpython.com/python-interface/
from signal import signal, SIGINT
from abc import ABC, abstractmethod
import math
import numpy as np
import matplotlib.pyplot as plt


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
    if min_epsilon <= 0.0:
        min_epsilon = 0.000001
    target_episodes: float = float(max_ep) * target_percent
    return math.pow(min_epsilon, 1.0 / target_episodes)


def plot_results(plot, title=None, x_label=None, y_label=None):
    plt.figure()
    plt.plot([i for i in range(0, len(plot))], plot)
    if title is not None:
        plt.title(title)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.show()


class IQModelInterface(ABC):
    def __init__(self) -> None:
        self._history: list = []
        self._ignore_history: bool = False
        self._model: object = None
        self._best_model: object = None

    # For Deep Reinforcement Learning we need a history of all (state, action, reward, new_state, done) tuples to
    # train with. For a Q-table model we can just ignore this by setting the ignore_history flag instance variable.
    def save_history(self, state: int, action: int, reward: float, new_state: int, done: bool, max_history: int = None):
        if not self._ignore_history:
            self._history.append((state, action, reward, new_state, done))
            # If history is very large, we can drop off some of the earlier ones
            if max_history is not None:
                if len(self._history) > max_history:
                    self._history = self._history[1:]

    # Call this function to mark that this is the best model so far
    # It will cause the model class to back up the underlying model
    # Use the method use_best_model to set back to last best model
    @abstractmethod
    def best_model_checkpoint(self):
        pass

    # Set the model to be the best one so far
    @abstractmethod
    def use_best_model(self):
        pass

    # Given a state, return the current Q value for all actions in that state
    @abstractmethod
    def get_state(self, state: object) -> np.ndarray:
        pass

    # At a certain 'state' we took 'action' and received 'reward' and ended up in 'new_state'
    # Update the QTable to represent this
    @abstractmethod
    def update_q_model(self, state: object, action: int, reward: float, new_state: object, done: bool = False) -> None:
        pass

    # Save the model to a file
    @abstractmethod
    def save_model(self, file_name: str = "QModel") -> None:
        pass

    # Load the model from a file
    @abstractmethod
    def load_model(self, file_name: str = "QModel") -> None:
        pass


class IEnvironmentInterface(ABC):
    def __init__(self, environment: object) -> None:
        self._environment: object = environment

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def step(self, action: int):
        pass

    # Getter for _environment
    @property
    def environment(self) -> object:
        return self._environment


class IQLearnerInterface(ABC):
    abort: bool = False     # Abort flag to stop training
    debug: bool = False     # Debug flag to give debug info

    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int,
                 max_episodes: int = None):
        self._num_states: int = num_states                       # Number of world states
        self._num_actions: int = num_actions                     # Number of actions available per world state
        # Report states and actions in env
        if IQLearnerInterface.get_debug():
            print("States: ", num_states, "Actions: ", num_actions)
        self._environment: IEnvironmentInterface = environment   # Environment to do updates
        self._max_episodes: int = max_episodes                   # Number of training episodes to run
        # Set min hyper parameters - don't decay below these values
        self._min_epsilon: float = 0.001
        # Defaults for hyper parameters
        self._epsilon: float = 0.99                              # Chance of e-greedy random move
        self._decay: float = 0.99                                # Decay rate for epsilon and possibly alpha
        self._gamma: float = 0.9                                 # Future discount factor
        # If max_episodes is set, default decay to be 80% of that
        if max_episodes is not None:
            self._decay = calc_decay(max_episodes, self._min_epsilon, target_percent=0.9)
            if IQLearnerInterface.debug:
                print("Setting decay to: ", self._decay)
        # Flags
        self.abort: bool = False                                # Abort flag to stop training
        self.debug: bool = False                                # Debug flag to give debug info
        # Stats
        self._report_every_nth = 1                               # Show every nth episode. Defaults to every episode.
        self._q_model: IQModelInterface or None = None           # Q Model we'll use
        self._episode: int = 0                                   # Count of episodes
        # Track scores, averages, and states across a session of training
        self._scores: list = []
        self._running_average: list = []
        # self._average_blocks: list = []
        # For tracking training values progress and determining best model
        # Default to 100 or to 1/10th of max episodes, whichever is smaller. But don't go below 1.
        self._average_over: int = 100
        if max_episodes is not None:
            self._average_over: int = max(min(100, max_episodes//10), 1)

    # Set this to 1 to show feedback on every training episode. Set higher than 1 to show fewer. e.g. 100 = only report
    # every 100th episode, etc.
    def report_every_nth_episode(self, every_nth: int) -> None:
        self._report_every_nth = every_nth

    # If you reset the min_epsilon you'll need to call recalculate_decay to calculate a new decay rate
    # You can also call this to set a different 'target_percent' (i.e. to get to min_epsilon by something different
    # then 0.9 (90%) of max_episodes
    def recalculate_decay(self, target_percent: float = 0.9) -> None:
        if self._max_episodes is not None:
            self._decay = calc_decay(self._max_episodes, self._min_epsilon, target_percent=target_percent)
            if IQLearnerInterface.debug:
                print("Setting decay to: ", self._decay)

    # Set the minimum epsilon (chance of random move) to not decay below
    def set_min_epsilon(self, min_epsilon: float, recalculate_decay: bool = True) -> None:
        # Don't decay below this epsilon
        self._min_epsilon = min_epsilon
        if recalculate_decay:
            self.recalculate_decay()

    # Getter for gamma
    @property
    def gamma(self) -> float:
        return self._gamma

    # Setter for gamma (discount factor)
    @gamma.setter
    def gamma(self, gamma: float) -> None:
        # Set discount factor
        self._gamma = gamma

    # Getter for decay
    @property
    def decay(self) -> float:
        return self._decay

    # Setter to manually set what decay rate you want instead of letting the learner calculate it off of max_episodes
    @decay.setter
    def decay(self, decay: float) -> None:
        # Set decay
        self._decay = decay

    # Getter for epsilon (chance of random move)
    @property
    def epsilon(self) -> float:
        return self._epsilon

    # Set the starting rate of making a random move (i.e. epsilon)
    @epsilon.setter
    def epsilon(self, epsilon: float) -> None:
        # Set epsilon (chance of random move)
        self._epsilon = epsilon

    # Getter for the model being used
    @property
    def q_model(self) -> IQModelInterface:
        return self._q_model

    # Get next action based on current model / random move if e_greedy
    def get_action(self, state: int, e_greedy: bool = True) -> int:
        # Get a random value from 0.0 to 1.0
        rand_val: float = float(np.random.rand())
        if e_greedy and rand_val < self._epsilon:
            # Grab a random action
            action: int = np.random.randint(0, self._num_actions)
        else:
            # Take best action instead of random action
            action = int(np.argmax(self.q_model.get_state(state)))
        return action

    # Save the model to a file
    def save_model(self, file_name: str = "QModel") -> None:
        self.q_model.save_model(file_name)

    # Load the model from a file
    def load_model(self, file_name: str = "QModel") -> None:
        self.q_model.load_model(file_name)

    def run_episode(self, render: bool = False, no_learn: bool = False) -> float:
        state: int = self._environment.reset()
        score: float = 0
        done: bool = False
        if no_learn:
            self._epsilon = 0.0
        # Loop until episode is done
        while not done:
            # If we're rendering the environment, display it
            if render:
                self._environment.render()
            # Pick an action
            action: int
            if not no_learn:
                action = self.get_action(state)
            else:
                action = self.get_action(state, e_greedy=False)
            # Take action and advance environment
            new_state: int
            reward: float
            done: bool
            info: object
            new_state, reward, done, info = self._environment.step(action)
            # Collect reward
            score += reward
            # If we are learning, update Q Table
            if not no_learn:
                self.update_model(state, action, reward, new_state, done)
            # New state becomes current state
            state = new_state

        # If we're rendering environment live, then show score. Otherwise, just return it
        if render:
            print("Score:", round(score, 2))
        return score

    def train(self, decay_alpha=True, every_nth_average: int = 100, max_converge_count: int = 50) -> None:
        score: float = 0
        avg_score: float = 0.0
        printed_episode: bool = False
        best_avg_score: float = -float('inf')
        converge_count: int = 0
        # Check if we converged.
        # We define converged as max_converge_count rounds without improvement once we reach min_epsilon
        while (self._max_episodes is None or self._episode < self._max_episodes) and \
                (converge_count < max_converge_count or self._epsilon > self._min_epsilon):
            # Reset score for this episode and reset if we printed an update for this episode
            score = 0
            printed_episode = False
            self._episode += 1
            # Reset score for this new episode
            # Run an episode
            score = round(self.run_episode(), 2)
            # Save off score
            self._scores.append(score)
            # Get current average score. Take the last 'average_over' amount
            avg_score = float(np.mean(self._scores[-self._average_over:]))
            # Track averages
            self._running_average.append(avg_score)
            # Convergence criteria
            best_avg_score = max(best_avg_score, avg_score)
            if avg_score < best_avg_score:
                converge_count += 1
            else:
                converge_count = 0
                # Checkpoint best model
                self._q_model.best_model_checkpoint()
            # Show results of a this round
            if self._episode % self._report_every_nth == 0:
                self.print_progress(converge_count, score, avg_score)
                printed_episode = True
            # Decay after each episode
            self._epsilon = round(max(self._epsilon * self._decay, self._min_epsilon), 5)
            # TODO: I'm not sure this is the right way to handle alpha decay - maybe I should call am abstract function
            if decay_alpha and hasattr(self, '_alpha') and hasattr(self, '_min_alpha'):
                # noinspection PyAttributeOutsideInit
                self._alpha = round(max(self._alpha * self._decay, self._min_alpha), 5)
            # Track every Nth average score to make the final graph look more readable
            # if self._episode % every_nth_average == 0:
            #     self._average_blocks.append(avg_score)
            # If we abort, just break the train loop and move on
            if IQLearnerInterface.get_abort():
                # Reset abort
                IQLearnerInterface.set_abort(False)
                # Then break out of training loop so we can move on
                break

        # Print final episode if not already printed
        if not printed_episode:
            self.print_progress(converge_count, score, avg_score)
        # Set model to be the best one we found so far (based on avg_score)
        # self._q_model.use_best_model()

    def render_episode(self) -> float:
        return self.run_episode(render=True, no_learn=True)

    def set_average_over(self, value: float) -> None:
        self._average_over = value

    def get_average_score(self, iterations=100) -> float:
        scores = []
        for i in range(iterations):
            score = self.run_episode(render=False, no_learn=True)
            scores.append(score)
        return round(float(np.mean(scores)), 2)

    def graph_trained_agent(self, n_iterations=10):
        scores = []
        for i in range(n_iterations):
            score = self.run_episode(no_learn=True)
            scores.append(score)
        avg = float(sum(scores))/float(len(scores))
        print("Graphed Average Trained Score: {:.2f}".format(round(avg, 2)))
        plot_results(scores, title="Trained Scores for Gamma: " + str(self.gamma) + " Decay: " + str(self.decay),
                     x_label="Episodes", y_label="Scores")

    def show_graphs(self, n_iterations=100):
        plot_results(self._scores, title="Training Scores for Gamma: " + str(self.gamma) + " Decay: " + str(self.decay),
                     x_label="Episodes", y_label="Scores")
        plot_results(self._running_average, title="Training Scores Running Average for Gamma: " + str(self.gamma) +
                                                  " Decay: " + str(self.decay), x_label="Episodes", y_label="Scores")
        # PlotResults(self.average_blocks, title="Block Averages", x_label="Average Block", y_label="Scores")
        self.graph_trained_agent(n_iterations=n_iterations)

    def render_episodes(self, episodes: int):
        for i in range(episodes):
            self.render_episode()

    # For this model, what do you want to print out for each progress update?
    # Parameters are converge_count (how long since we saw an improvement), score for current episode
    # and avg_score as determined by every_nth_average parameter passed to train method
    @abstractmethod
    def print_progress(self, converge_count: int, score: float, avg_score: float):
        pass

    # Passing an update tuple to the model to update the model
    @abstractmethod
    def update_model(self, state: object, action: int, reward: float, new_state: object, done: bool = False) -> None:
        pass

    # Set class debug flag
    @staticmethod
    def set_debug(setting: bool) -> None:
        IQLearnerInterface.debug = setting

    # Get class debug flag
    @staticmethod
    def get_debug() -> bool:
        return IQLearnerInterface.debug

    # Set class abort flag (to manually end training)
    @staticmethod
    def set_abort(setting: bool) -> None:
        IQLearnerInterface.abort = setting

    # Get class abort flag (to manually end training)
    @staticmethod
    def get_abort() -> bool:
        return IQLearnerInterface.abort
