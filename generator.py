import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.random_input = tf.keras.layers.Dense(32)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False, gan_training=False):
        gru_training = training#not gan_training if training is True else False
        gan_training = training#gan_training if training is True else False
        x = inputs['dataset_inputs']
        x = self.embedding(x, training=gru_training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=gru_training)
        y = self.random_input(inputs['random_noise'], training=gan_training)
        x = tf.keras.layers.concatenate([x, y])
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x