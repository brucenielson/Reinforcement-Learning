from q_learner_interfaces import IQLearnerInterface, IQTableInterface, IEnvironmentInterface
import gym


class OpenGymEnvironmentRequired(Exception):
    def __init__(self):
        super().__init__("Must pass a valid Open Gym environment to OpenGymEnvironment")


class OpenGymEnvironmentInterface(IEnvironmentInterface):
    def __init__(self, open_gym_env: object = None) -> None:
        if open_gym_env is None:
            raise OpenGymEnvironmentRequired
        self.open_gym_env: object = open_gym_env

    def reset(self) -> object:
        return self.open_gym_env.reset()

    def render(self) -> None:
        self.open_gym_env.render()

    def step(self, action: int) -> (object, float, bool, object):
        return self.open_gym_env.step(action)
