from q_learner_interfaces import IEnvironmentInterface


class OpenGymEnvironmentRequired(Exception):
    def __init__(self):
        super().__init__("Must pass a valid Open Gym environment to OpenGymEnvironment")


# noinspection PyUnresolvedReferences
class OpenGymEnvironmentInterface(IEnvironmentInterface):
    def __init__(self, open_gym_env: object = None) -> None:
        if open_gym_env is None:
            raise OpenGymEnvironmentRequired
        super(OpenGymEnvironmentInterface, self).__init__(open_gym_env)
        self._num_actions = self._environment.action_space.n
        self._is_state_space_discrete = hasattr(self._environment.observation_space, 'n')
        if self._is_state_space_discrete:
            # If states space is discrete, get the total number of states
            self._num_states = self._environment.observation_space.n
        else:
            # if state space is continuous, get the total number of variables we'll track for our state space
            self._num_states = self._environment.observation_space.shape[0]

    def reset(self) -> object:
        # noinspection PyUnresolvedReferences
        return self._environment.reset()

    def render(self) -> None:
        # noinspection PyUnresolvedReferences
        self._environment.render()

    def step(self, action: int) -> (object, float, bool, object):
        # noinspection PyUnresolvedReferences
        return self._environment.step(action)

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def is_state_space_discrete(self) -> int:
        return self._is_state_space_discrete
