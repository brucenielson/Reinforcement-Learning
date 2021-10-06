from gym.envs.toy_text.taxi import TaxiEnv
import numpy as np
from q_learner import QLearner
from environments import OpenGymEnvironmentInterface
import time


def taxi(seed=42):
    start_time = time.time()
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    np.random.seed(seed)
    env.seed(seed)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner: QLearner = QLearner(environment, num_states, num_actions, max_episodes=10000)
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
    print("Average Score:", q_learner.get_average_score(100000))
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


ql = taxi()
# ql = load_taxi()
# ql = taxi_more_training()
# Taxi scores: https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
