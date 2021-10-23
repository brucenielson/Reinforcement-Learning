from dqn_learner import DQNLearner
from dqn_model import DQNModel
from q_learner import QLearner
from q_table import QModel
from environments import OpenGymEnvironmentInterface
import gym
import dqn_model
import q_learner_interfaces
from gym.envs.toy_text.taxi import TaxiEnv


def set_seed(seed: int, environment) -> None:
    # noinspection PyUnresolvedReferences
    environment.seed(seed)
    dqn_model.random.seed(seed)
    dqn_model.os.environ['PYTHONHASHSEED'] = str(seed)
    dqn_model.tf.random.set_seed(seed)
    q_learner_interfaces.np.random.seed(seed)


class LunarLanderLearner(DQNLearner):
    def __init__(self, max_episodes: int = None, lr: float = 0.001, seed: int = None) -> None:
        env = gym.make('LunarLander-v2')
        if seed is not None:
            set_seed(seed, env)
        environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
        super(DQNLearner, self).__init__(environment, environment.num_states, environment.num_actions, max_episodes)
        # Create model
        self._q_model = DQNModel(environment.num_states, environment.num_actions, lr=lr)


class TaxiLearner(QLearner):
    def __init__(self, max_episodes: int = None, seed: int = None) -> None:
        env: TaxiEnv = TaxiEnv()
        if seed is not None:
            set_seed(seed)
        environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
        super(TaxiLearner, self).__init__(environment, environment.num_states, environment.num_actions, max_episodes)
        # Create model
        self._q_model = QModel(environment.num_states, environment.num_actions)
