from gym.envs.toy_text import TaxiEnv
from gym.envs.toy_text.taxi import TaxiEnv
import numpy as np
from q_learner import QLearner
from environments import OpenGymEnvironmentInterface


def taxi(seed=42):
    env: TaxiEnv = TaxiEnv()
    environment = OpenGymEnvironmentInterface(env)
    np.random.seed(seed)
    env.seed(seed)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner: QLearner = QLearner(environment, num_states, num_actions, 2000000, epsilon=0.99, decay=0.999997,
                                   gamma=0.9, alpha=0.5)
    q_learner.train()
    print("Final Epsilon", round(q_learner.epsilon, 4))
    print("Final Alpha:", round(q_learner.alpha, 4))
    return q_learner.q_model


ql = taxi()
