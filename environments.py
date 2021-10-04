from q_learner_contracts import QLearnerContract, QTableContract, Environment
import gym


class OpenGymEnvironmentRequired(Exception):
    super().__init__("Must pass a valid Open Gym environment to OpenGymEnvironment")


class OpenGymEnvironment(Environment):
    def __int__(self, learner: QLearnerContract, q_table: QTableContract, open_gym_env: object = None) -> None:
        super().__init__(learner, q_table)
        if open_gym_env is None:
            raise OpenGymEnvironmentRequired
        self.open_gym_env: object = open_gym_env

    def run_episode(self, render: bool = False, no_learn: bool = False):
        state: object = self.reset()
        score: float = 0
        done: bool = False
        if no_learn:
            self.learner.epsilon = 0.0
            self.learner.alpha = 0.0

        # Loop until episode is done
        while not done:
            # If we're rendering the environment, display it
            if render:
                self.render()
            # Pick an action
            action: int = self.learner.e_greedy_action(state)
            # Take action and advance environment
            new_state: object
            reward: float
            done: bool
            info: object
            new_state, reward, done, info = self.step(action)
            # Collect reward
            score += reward
            # Save history
            self.q_table.save_history(state, action, reward, new_state, done)
            # If we are learning, update Q Table
            if not no_learn:
                self.q_table.update_q_table(state, action, reward, new_state)
            # New state becomes current state
            state = new_state

        # If we're rendering environment live, then show score. Otherwise, just return it
        if render:
            print("Score:", round(score, 2))
        return score

    def reset(self) -> object:
        return self.open_gym_env.reset()

    def render(self) -> None:
        self.open_gym_env.render()

    def step(self, action: int) -> (object, float, bool, object):
        return self.open_gym_env.step(action)
