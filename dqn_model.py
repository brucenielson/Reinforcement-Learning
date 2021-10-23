from q_learner_interfaces import IQModelInterface
from tensorflow.keras import activations, Sequential, layers, optimizers
from tensorflow.keras.models import load_model
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DQNModel(IQModelInterface):
    def __init__(self, num_states: int, num_actions: int, lr: float = 0.001):
        super(DQNModel, self).__init__()
        act_relu = activations.relu
        act_linear = activations.linear
        top_layer = 150
        middle_layer = 120
        # Create Network: Default Parameters
        # from https://towardsdatascience.com/solving-lunar-lander-openaigym-reinforcement-learning-785675066197
        model: Sequential = Sequential()
        layer = layers.Dense(top_layer, input_dim=num_states, activation=act_relu)
        model.add(layer)
        layer = layers.Dense(middle_layer, activation=act_relu)
        model.add(layer)
        layer = layers.Dense(num_actions, activation=act_linear)
        model.add(layer)
        opt = optimizers.Adam(learning_rate=lr)
        model.compile(loss='mse', optimizer=opt)
        self._model: Sequential = model
        self._batch_size: int = 100
        # We want to track history for deep reinforcement learning
        self._ignore_history = False

    # Set the batch_size for the mini batches to train on
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        self._batch_size = size

    def get_state(self, state: np.ndarray):
        state = np.reshape(state, (1, 8))
        return self._model.predict(state)

    def update_q_model(self, state: int, action: int, reward: float, new_state: object, done: bool = False,
                       gamma: float = 0.9) -> None:
        # Preform Replay
        row_count = self.batch_size
        if len(self._history) < row_count:
            return

        # Column names
        state = 0
        action = 1
        reward = 2
        next_state = 3
        done = 4
        # Get samples in mini-batches
        samples = random.sample(self._history, row_count)
        # Separate into separate arrays
        states_array = [sample[state] for sample in samples]
        actions_array = [sample[action] for sample in samples]
        rewards_array = [sample[reward] for sample in samples]
        next_states_array = [sample[next_state] for sample in samples]
        done_array = [sample[done] for sample in samples]
        # Turn into arrays
        states_array = np.array(states_array)
        actions_array = np.array(actions_array)
        rewards_array = np.array(rewards_array)
        next_states_array = np.array(next_states_array)
        done_array = (1.0 - np.array(done_array))

        # train on states_array
        x = states_array

        # Create y (i.e. labels for supervised learning)
        predicted_values = self._model.predict_on_batch(states_array)
        next_predicted_values = self._model.predict_on_batch(next_states_array)
        actual_values = rewards_array + gamma * np.amax(next_predicted_values, axis=1) * done_array

        predicted_values[list(range(row_count)), actions_array] = actual_values
        y = predicted_values

        # Update network
        self._model.fit(x, y, epochs=1, verbose=0)

    def save_model(self, file_name: str = "QModel") -> None:
        self._model.save(file_name+".h5")

    def load_model(self, file_name: str = "QModel") -> None:
        self._model = load_model(file_name+".h5")

    def best_model_checkpoint(self):
        self._best_model = self._model.get_weights()

    def use_best_model(self):
        self._model.set_weights(self._best_model)
