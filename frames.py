import tensorflow as tf
from tensorflow.keras import layers
from gym.utils import colorize
# other necessary imports
from configs import TIME_SEQUENCE_SIZE

class frame_1(tf.keras.Model):
    def __init__(self, actions_size: int, features_size: int, sequence_size: int = TIME_SEQUENCE_SIZE):
        super(frame_1, self).__init__()
        self.conv1d_input = layers.Conv1D(32, 3, padding = "same", activation = "relu", input_shape=(None, sequence_size, features_size))
        self.conv1d_hidden = layers.Conv1D(64, 3, padding="same", activation="relu")
        self.rnn_connector = layers.GRU(128)
        self.dense_1 = layers.Dense(64, activation="relu")
        self.dense_2 = layers.Dense(actions_size)
        self.output_layer = layers.Dense(actions_size)

    def __str__(self):
        return "f1"
    @staticmethod
    def short_str():
        return "f1"

    @tf.function
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """ inputs : tf.Tensor, (batch_size, num_states, features)
			returns : N-D tensor (batch_size, features)
		"""
		# pass inputs on model and return the output value Tensor
        x = self.conv1d_input(inputs)
        x = self.conv1d_hidden(x)
        x = self.rnn_connector(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.output_layer(x)
