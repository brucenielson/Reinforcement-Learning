from gym.envs.toy_text.taxi import TaxiEnv
from q_learner import QLearner
from environments import OpenGymEnvironmentInterface
import time

from open_gym_environments import LunarLanderLearner, TaxiLearner


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
    learner.render_episodes(4)
    learner.save_model()
    # print("Average Score:", dqn_learner.get_average_score(100))
    learner.show_graphs()
    return learner


def load_lunar_lander(seed: int = 43):
    learner = LunarLanderLearner(seed=seed)
    learner.load_model("BestQModel")
    learner.render_episodes(5)
    # print("Average Score:", dqn_learner.get_average_score(10))
    return learner


def taxi(seed: int = 42):
    start_time = time.time()
    # Best Taxi scores: https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
    TaxiLearner.set_debug(True)
    learner: TaxiLearner = TaxiLearner(10000, seed=seed)
    learner.alpha = 0.5
    learner.report_every_nth_episode(100)
    learner.train()
    print("Final Epsilon", round(learner.epsilon, 4))
    print("Final Alpha:", round(learner.alpha, 4))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    # learner.render_episodes(3)
    learner.save_model()
    learner.show_graphs()
    print("Q Sparseness: ", learner.q_model.q_sparseness())
    print("Average Score:", learner.get_average_score(10000))
    return learner


def load_taxi(seed: int = 42):
    learner: TaxiLearner = TaxiLearner(5000, seed=seed)
    learner.load_model("TaxiQModel")
    # How to render better in a notebook: https://casey-barr.github.io/open-ai-taxi-problem/
    # learner.render_episodes(3)
    print("Q Sparseness: ", learner.q_model.q_sparseness())
    print("Average Score:", learner.get_average_score(10000))
    return learner


def taxi_more_training():
    start_time = time.time()
    learner: QLearner = TaxiLearner(1000)
    learner.load_model("TaxiQModel")
    print("Q Sparseness: ", learner.q_model.q_sparseness())
    learner.train()
    print("Final Epsilon", round(learner.epsilon, 4))
    print("Final Alpha:", round(learner.alpha, 4))
    end_time = time.time()
    print("Total run time: ", round(end_time-start_time))
    learner.show_graphs()
    learner.save_model()
    print("Average Score:", learner.get_average_score(10000))
    return learner


# ql = lunar_lander()
ql = load_lunar_lander()
# ql1 = taxi()
# ql2 = load_taxi()
# ql = taxi_more_training()


# https://gym.openai.com/envs/#classic_control
