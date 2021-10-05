from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from q_table import QTable
import pickle
import numpy as np


class QLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, num_episodes: int,
                 epsilon: float = 0.8, decay: float = 0.9999, gamma: float = 0.99, alpha: float = 0.1):
        super(QLearner, self).__init__(environment, num_states, num_actions, num_episodes,
                                                 epsilon, decay, gamma)
        # Create model
        self.q_model = QTable(num_states, num_actions)
        # Set learning rate (alpha)
        self.alpha: float = alpha
        # Track scores, averages, and states across a session of training
        self.scores: list = []
        self.running_average: list = []
        self.average_blocks: list = []
        # For tracking training values
        self.average_over: int = 0
        self.min_alpha: float = 0.0
        self.min_epsilon: float = 0.0
        self.episode: int = 0

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

    def run_episode(self, render: bool = False, no_learn: bool = False):
        state: int = self.environment.reset()
        score: float = 0
        done: bool = False
        if no_learn:
            self.epsilon = 0.0
            self.alpha = 0.0

        # Loop until episode is done
        while not done:
            # If we're rendering the environment, display it
            if render:
                self.environment.render()
            # Pick an action
            action: int = self.e_greedy_action(state)
            # Take action and advance environment
            new_state: int
            reward: float
            done: bool
            info: object
            new_state, reward, done, info = self.environment.step(action)
            # Collect reward
            score += reward
            # Save history if DQN
            # self.q_model.save_history(state, action, reward, new_state, done)
            # If we are learning, update Q Table
            if not no_learn:
                self.q_model.update_q_table(state, action, reward, new_state, self.gamma, self.alpha)
            # New state becomes current state
            state = new_state

        # If we're rendering environment live, then show score. Otherwise, just return it
        if render:
            print("Score:", round(score, 2))
        return score

    def train(self, decay_alpha=True, every_nth_average: int = 10, max_converge_count: int = 25) -> None:
        best_avg_score: float = -1000
        converge_count: int = 0
        for i in range(self.num_episodes):
            self.episode = i
            # Reset score for this new episode
            score: float = 0
            # Run an episode
            score += self.run_episode()
            # Decay after each episode
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            if decay_alpha:
                self.alpha = max(self.alpha * self.decay, self.min_alpha)
            # Save off score
            self.scores.append(score)
            # Get current average score. Take the last 'average_over' amount
            avg_score: float = float(np.mean(self.scores[-self.average_over:]))
            # Track averages
            self.running_average.append(avg_score)
            # Convergence criteria
            best_avg_score = max(best_avg_score, avg_score)
            if avg_score < best_avg_score:
                converge_count += 1
            else:
                converge_count = 0
            # Show results of a this round
            print("Episode:", self.episode, "Last High:", converge_count, "Epsilon:", round(self.epsilon, 4), "Alpha:",
                  round(self.alpha, 4), "Score:", round(score, 2), "Avg Score:", round(avg_score, 2))
            # Track every Nth average score to make the final graph look more readable
            if i % every_nth_average == 0:
                self.average_blocks.append(avg_score)
            # Check if we converged.
            # We define converged as 50 rounds without improvement once we reach min_epsilon
            # Alternatively, if we abort, just break the train loop and move on
            if (converge_count >= max_converge_count and self.epsilon > self.min_epsilon) \
                    or IQLearnerInterface.get_abort():
                # Reset abort
                IQLearnerInterface.set_abort(False)
                # Then break out of training loop so we can move on
                break
