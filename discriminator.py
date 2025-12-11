import tensorflow as tf

class MaxLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxLayer, self).__init__()
        self.beta = 1e10

    def call(self, inputs):
        x_range = tf.range(inputs.shape.as_list()[-1], dtype=inputs.dtype)
        return tf.reduce_sum(tf.nn.softmax(inputs*self.beta) * x_range, axis=-1)

class PoolingClassifier(tf.keras.Model):
    def __init__(self, encoder):
        super(PoolingClassifier, self).__init__()
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.encoder = encoder
        self.embedding = None
        self.dense_tanh = tf.keras.layers.Dense(64, activation='tanh')
        self.dense_output = tf.keras.layers.Dense(1)
        self.model_name = ""
        self.max_layer = MaxLayer()

    def call(self, inputs, use_max_layer = False, training=False):
        x = inputs
        if use_max_layer:
            x = self.max_layer(x)
        elif self.encoder is not None:
            x = self.encoder(x)
        x = self.embedding(x, training=training)
        x = self.pooling(x, training=training)
        x = self.dense_tanh(x, training=training)
        x = self.dense_output(x, training=training)
        return x

class LinearDiscriminator(PoolingClassifier):
    def __init__(self, encoder=None):
        super(LinearDiscriminator, self).__init__(encoder)
        self.embedding = LinearReplacement(64)

        
class EmbeddingDiscriminator(PoolingClassifier):
    def __init__(self, encoder=None, vocab_size=0):
        super(EmbeddingDiscriminator, self).__init__(encoder)
        self.embedding = EmbeddedLinearReplacement(vocab_size, 64)

class LinearReplacement(tf.keras.layers.Layer):
    def __init__(self, num_outputs, use_bias=True, activation=None):
        super(LinearReplacement, self).__init__()
        self.dense_output = tf.keras.layers.Dense(num_outputs, use_bias=use_bias, activation=activation)

    def call(self, inputs, training):
        outputs = tf.expand_dims(inputs, -1)
        return self.dense_output(outputs, training=training)
    

class EmbeddedLinearReplacement(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, use_bias=True):
        super(EmbeddedLinearReplacement, self).__init__()
        self.dense_embedding = tf.keras.layers.Dense(num_outputs, use_bias=use_bias)
        self.embedding = tf.keras.layers.Embedding(num_inputs, num_outputs)
        self.dense_output = tf.keras.layers.Dense(num_outputs, use_bias=use_bias)

    def call(self, inputs, training):
        expanded_inputs = tf.expand_dims(inputs, -1)
        linear_result = self.dense_embedding(expanded_inputs, training=training)
        embedding_result = self.embedding(inputs, training=training)
        concatenate = tf.keras.layers.concatenate([linear_result, embedding_result])
        return self.dense_output(concatenate, training=training)