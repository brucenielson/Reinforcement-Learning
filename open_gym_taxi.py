from gym.envs.toy_text.taxi import TaxiEnv
import numpy as np
from q_learner import QLearner


def taxi(seed=42):
    env = TaxiEnv()
    np.random.seed(seed)
    env.seed(seed)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner = QLearner(num_states, num_actions, num_episodes=2000000,
                         epsilon=0.99, decay=0.999997, gamma=0.9, alpha=0.5)
    q_learner.Train()
    print("Final Epsilon", round(q_learner.epsilon, 4))
    print("Final Alpha:", round(q_learner.alpha, 4))
    return q_learner.q_model
