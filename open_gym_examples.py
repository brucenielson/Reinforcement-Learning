from gym.envs.toy_text.taxi import TaxiEnv
import gym
import numpy as np
from q_learner import QLearner
from dqn_learner import DQNLearner
from environments import OpenGymEnvironmentInterface
import time
import random


def lunar_lander(seed=42):
    start_time = time.time()

    environment = gym.make('LunarLander-v2')
    np.random.seed(seed)
    environment.seed(seed)
    random.seed(seed)
    num_actions = environment.action_space.n
    # How to get number of states for reinforcement learning
    num_states = environment.observation_space.shape[0]
    lr = 0.001
    dqn_learner = DQNLearner(environment, num_states, num_actions, max_episodes=100, lr=lr)
    dqn_learner.set_gamma(0.99)
    dqn_learner.train()
    print("Final Epsilon", round(dqn_learner.epsilon, 3))
    print("Final Alpha:", round(dqn_learner.alpha, 3))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    for i in range(4):
        dqn_learner.render_episode()
    # dqn.ShowGraphs()
    dqn_learner.save_model()
    print("Average Score:", dqn_learner.get_average_score(10))
    return dqn_learner.q_model


def taxi(seed=42):
    start_time = time.time()
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    np.random.seed(seed)
    env.seed(seed)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner: QLearner = QLearner(environment, num_states, num_actions, 10000)
    q_learner.set_alpha(0.5)
    q_learner.report_every_nth_episode(1)
    q_learner.train()
    print("Final Epsilon", round(q_learner.epsilon, 4))
    print("Final Alpha:", round(q_learner.alpha, 4))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    # for i in range(1):
    #     q_learner.render_episode()
    # q_learner.ShowGraphs()
    q_learner.save_model()
    print("Q Sparseness: ", q_learner.q_model.q_sparseness())
    print("Average Score:", q_learner.get_average_score(100))
    return q_learner.q_model


def load_taxi():
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n
    q_learner: QLearner = QLearner(environment, num_states, num_actions)
    q_learner.load_model("TaxiQModel")
    # How to render better in a notebook: https://casey-barr.github.io/open-ai-taxi-problem/
    # for i in range(4):
    #     q_learner.render_episode()
    print("Q Sparseness: ", q_learner.q_model.q_sparseness())
    print("Average Score:", q_learner.get_average_score(10000))
    return q_learner


def taxi_more_training():
    start_time = time.time()
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner: QLearner = QLearner(environment, num_states, num_actions, 1000000)
    q_learner.load_model("TaxiQModel")
    q_learner.train()
    print("Final Epsilon", round(q_learner.epsilon, 4))
    print("Final Alpha:", round(q_learner.alpha, 4))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    # for i in range(1):
    #     q_learner.render_episode()
    # q_learner.ShowGraphs()
    q_learner.save_model()
    print("Average Score:", q_learner.get_average_score(10000))
    return q_learner


# ql = lunar_lander()
ql = taxi()
# ql = load_taxi()
# ql = taxi_more_training()
# Taxi scores: https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
