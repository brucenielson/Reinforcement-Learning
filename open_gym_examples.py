from gym.envs.toy_text.taxi import TaxiEnv
import numpy as np
from q_learner import QLearner
from environments import OpenGymEnvironmentInterface
import time

from open_gym_environments import LunarLanderLearner


def lunar_lander(seed=43):
    start_time = time.time()
    LunarLanderLearner.set_debug(True)
    learner = LunarLanderLearner(max_episodes=500, lr=0.001, seed=seed)
    # learner.decay = 0.98
    learner.epsilon = 1.0
    learner.gamma = 0.99
    learner.set_average_over(100)
    # learner.recalculate_decay(0.5)
    learner.train()
    print("Final Epsilon", round(learner.epsilon, 3))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    for i in range(4):
        learner.render_episode()
    learner.save_model()
    # print("Average Score:", dqn_learner.get_average_score(100))
    learner.show_graphs()
    return learner


def load_lunar_lander(seed: int = 43):
    learner = LunarLanderLearner(seed=seed)
    learner.load_model("BestQModel")
    for i in range(5):
        learner.render_episode()
    # print("Average Score:", dqn_learner.get_average_score(10))
    return learner


def taxi(seed: int = 42):
    # Taxi scores: https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
    start_time = time.time()
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    np.random.seed(seed)
    env.seed(seed)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    QLearner.set_debug(True)
    q_learner: QLearner = QLearner(environment, num_states, num_actions, 5000)
    q_learner.alpha = 0.5
    q_learner.report_every_nth_episode(100)
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
    print("Q Sparseness: ", q_learner.q_model.q_sparseness())
    print("Average Score:", q_learner.get_average_score(100))
    return q_learner.q_model


def taxi_more_training():
    start_time = time.time()
    env: TaxiEnv = TaxiEnv()
    environment: OpenGymEnvironmentInterface = OpenGymEnvironmentInterface(env)
    num_actions: int = env.action_space.n
    num_states: int = env.observation_space.n

    q_learner: QLearner = QLearner(environment, num_states, num_actions, 1000)
    q_learner.load_model("TaxiQModel")
    print("Q Sparseness: ", q_learner.q_model.q_sparseness())
    q_learner.train()
    print("Final Epsilon", round(q_learner.epsilon, 4))
    print("Final Alpha:", round(q_learner.alpha, 4))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    # for i in range(1):
    #     q_learner.render_episode()
    # q_learner.ShowGraphs()
    q_learner.save_model()
    print("Average Score:", q_learner.get_average_score(100))
    return q_learner


ql = lunar_lander()
# ql = load_lunar_lander()
# ql1 = taxi()
# ql2 = load_taxi()
# ql = taxi_more_training()


# https://gym.openai.com/envs/#classic_control
