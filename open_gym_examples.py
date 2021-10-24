from gym.envs.toy_text.taxi import TaxiEnv
from q_learner import QLearner
from environments import OpenGymEnvironmentInterface
import time

from open_gym_environments import LunarLanderLearner, TaxiLearner, CartPoleLearner


def train_loop(learner, verbose=True, save_model=True, show_graphs=True, render=True, render_num=4, get_average=True,
               average_over=100):
    start_time = time.time()
    learner.train()
    if verbose:
        print("Final Epsilon", round(learner.epsilon, 3))
    end_time = time.time()
    if verbose:
        print("Total run time: ", round(end_time-start_time))
    if render:
        learner.render_episodes(render_num)
    if save_model:
        learner.save_model()
    if get_average:
        print("Average Score (over "+str(average_over)+"):", learner.get_average_score(average_over))
    if show_graphs:
        learner.show_graphs()
    return learner


def lunar_lander(seed=43):
    LunarLanderLearner.set_debug(True)
    learner = LunarLanderLearner(max_episodes=500, lr=0.001, seed=seed)
    # learner.decay = 0.98
    learner.epsilon = 1.0
    learner.gamma = 0.99
    learner.set_average_over(100)
    # learner.recalculate_decay(0.5)
    return train_loop(learner)


def load_lunar_lander_training(seed: int = 43):
    learner = LunarLanderLearner(seed=seed)
    learner.load_model("BestLLModel")
    learner.render_episodes(5)
    print("Average Score:", learner.get_average_score(10))
    return learner


def taxi(seed: int = 42):
    # Best Taxi scores: https://medium.com/@anirbans17/reinforcement-learning-for-taxi-v2-edd7c5b76869
    TaxiLearner.set_debug(True)
    learner: TaxiLearner = TaxiLearner(10000, seed=seed)
    learner.alpha = 0.5
    learner.report_every_nth_episode(100)
    learner = train_loop(learner, render=False, average_over=10000)
    print("Q Sparseness: ", learner.q_model.q_sparseness())
    return learner


def load_taxi(seed: int = 42):
    learner: TaxiLearner = TaxiLearner(seed=seed)
    learner.load_model("TaxiQModel")
    # How to render better in a notebook: https://casey-barr.github.io/open-ai-taxi-problem/
    # learner.render_episodes(3)
    print("Q Sparseness: ", learner.q_model.q_sparseness())
    print("Average Score:", learner.get_average_score(10000))
    return learner


def taxi_more_training():
    # Do additional training on an already trained model
    learner: TaxiLearner = TaxiLearner(1000)
    learner.load_model("TaxiQModel")
    return train_loop(learner, render=False, average_over=10000)


def cart_pole(seed: int = 42):
    CartPoleLearner.set_debug(True)
    learner: CartPoleLearner = CartPoleLearner(100, seed=seed, lr=0.001)
    learner.set_min_epsilon(0.0, recalculate_decay=False)
    return train_loop(learner, get_average=False)


def load_cart_pole(seed: int = 43):
    learner = CartPoleLearner(seed=seed)
    learner.load_model("BestCPModel")
    # learner.epsilon = 0.001
    learner.render_episodes(5)
    print("Average Score:", learner.get_average_score(10))
    learner.graph_trained_agent(n_iterations=100)
    return learner


# ql = lunar_lander_training()
# ql = load_lunar_lander()
# ql = taxi()
# ql = load_taxi()
# ql = taxi_more_training()
ql = cart_pole()
# ql = load_cart_pole()
