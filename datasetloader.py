import tensorflow_datasets as tfds
import tensorflow as tf
import time

class IMDBLoader():
    def __init__(self, configuration):
        super(IMDBLoader, self).__init__()
        self.config = configuration
        self.encorder = None
        self.train_data = None
        self.test_data = None
        self.batch_size = -1

    def load(self):
        print("Loading IMDB dataset...")
        start_time = time.perf_counter()
        dataset = tfds.load('imdb_reviews', as_supervised=True)

        train, test = dataset['train'], dataset['test']

        for data in train.take(1):
            print(data)


        self.train_dataset = train.shuffle(self.config.buffer_size).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = test.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)

        self.encoder = tf.keras.layers.TextVectorization(max_tokens=self.config.vocab_size, ragged=True)
        self.encoder.adapt(self.train_dataset.map(lambda text, label: text))

        # Setup if data does not require non-ragged tensor
            # self.masked_encoder = tf.keras.layers.TextVectorization(max_tokens=self.config.vocab_size)
            # self.masked_encoder.adapt(self.train_dataset.map(lambda text, label: text))

        end_time = time.perf_counter()

        if self.config.print_loading_time:
            print("Time taken to load IMDB Dataset: {}".format(end_time - start_time))

        return self