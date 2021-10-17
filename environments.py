from q_learner_interfaces import IEnvironmentInterface


class OpenGymEnvironmentRequired(Exception):
    def __init__(self):
        super().__init__("Must pass a valid Open Gym environment to OpenGymEnvironment")


class OpenGymEnvironmentInterface(IEnvironmentInterface):
    def __init__(self, open_gym_env: object = None) -> None:
        if open_gym_env is None:
            raise OpenGymEnvironmentRequired
        super(OpenGymEnvironmentInterface, self).__init__(open_gym_env)

    def reset(self) -> object:
        # noinspection PyUnresolvedReferences
        return self._environment.reset()

    def render(self) -> None:
        # noinspection PyUnresolvedReferences
        self._environment.render()

    def step(self, action: int) -> (object, float, bool, object):
        # noinspection PyUnresolvedReferences
        return self._environment.step(action)
