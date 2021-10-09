from q_learner_interfaces import IQLearnerInterface, IEnvironmentInterface
from q_table import QModel
import pickle
import numpy as np


class QLearner(IQLearnerInterface):
    def __init__(self, environment: IEnvironmentInterface, num_states: int, num_actions: int, max_episodes: int = None)\
            -> None:
        super(QLearner, self).__init__(environment, num_states, num_actions, max_episodes)
        # Report states and actions in env
        print("States: ", num_states, "Actions: ", num_actions)
        # Create model
        self.q_model = QModel(num_states, num_actions)
        # Track scores, averages, and states across a session of training
        self.scores: list = []
        self.running_average: list = []
        self.average_blocks: list = []
        # For tracking training values
        self.average_over: int = 100
        self.min_alpha: float = 0.05
        self.min_epsilon: float = 0.001
        self.episode: int = 0

    def save_model(self, file_name: str = "QModel") -> None:
        pickle.dump(self.q_model.get_model(), open(file_name+".pkl", "wb"))

    def load_model(self, file_name: str = "QModel") -> None:
        self.q_model.set_model(pickle.load(open(file_name+".pkl", "rb")))

    def set_average_over(self, value: float) -> None:
        self.average_over = value

    def update_model(self, state: int, action: int, reward: float, new_state: int, done: bool = False) -> None:
        # Save history if DQN
        self.q_model.save_history(state, action, reward, new_state, done)
        self.q_model.update_q_model(state, action, reward, new_state, done, self.gamma, self.alpha)

    def run_episode(self, render: bool = False, no_learn: bool = False) -> float:
        state: int = self.environment.reset()
        score: float = 0
        done: bool = False
        if no_learn:
            self.epsilon: float = 0.0
            self.alpha: float = 0.0

        # Loop until episode is done
        while not done:
            # If we're rendering the environment, display it
            if render:
                self.environment.render()
            # Pick an action
            action: int = self.get_action(state)
            # Take action and advance environment
            new_state: int
            reward: float
            done: bool
            info: object
            new_state, reward, done, info = self.environment.step(action)
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
        best_avg_score: float = -float('inf')
        converge_count: int = 0
        while self.max_episodes is None or self.episode <= self.max_episodes:
            self.episode += 1
            # Reset score for this new episode
            score: float = 0
            # Run an episode
            score += self.run_episode()
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
            if self.episode % self.report_every_nth == 0:
                print("Episode:", self.episode, "Last High:", converge_count, "Epsilon:", round(self.epsilon, 4),
                      "Alpha:", round(self.alpha, 4), "Score:", round(score, 2), "Avg Score:", round(avg_score, 2))
            # Decay after each episode
            self.epsilon = max(self.epsilon * self.decay, self.min_epsilon)
            if decay_alpha:
                self.alpha = max(self.alpha * self.decay, self.min_alpha)
            # Track every Nth average score to make the final graph look more readable
            if self.episode % every_nth_average == 0:
                self.average_blocks.append(avg_score)
            # Check if we converged.
            # We define converged as max_converge_count rounds without improvement once we reach min_epsilon
            # Alternatively, if we abort, just break the train loop and move on
            if (converge_count >= max_converge_count and self.epsilon <= self.min_epsilon) \
                    or IQLearnerInterface.get_abort():
                # Reset abort
                IQLearnerInterface.set_abort(False)
                # Then break out of training loop so we can move on
                break

    def render_episode(self) -> float:
        return self.run_episode(render=True, no_learn=True)

    def get_average_score(self, iterations=100) -> float:
        scores = []
        for i in range(iterations):
            score = self.run_episode(render=False, no_learn=True)
            scores.append(score)
        return round(float(np.mean(scores)), 2)
