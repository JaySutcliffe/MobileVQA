import numpy as np
import tensorflow as tf
from data_generator import VQA_data_generator


class Lstm_cnn_trainer():
    # The size of a question and image
    max_question_length = 26
    image_width = 224
    image_height = 224
    vocabulary_size = 12915

    image_inputs = None
    question_inputs = None
    model = None
    train_generator = None
    val_generator = None

    # Model configuration variables
    batch_size = 10
    embedding_size = 200
    rnn_size = 512
    dense_hidden_size = 1024
    output_size = 1000

    def create_image_processing_model(self):
        """
        Creates the image processing part of the VQA model
        :return: A Keras model with VGG19 CNN to extract image features
        """
        return tf.keras.Sequential([
            tf.keras.applications.VGG19(
                include_top=True, weights='imagenet', input_tensor=self.image_inputs),
            # tf.math.l2_normalize(axis=None, epsilon=1e-12, name=None)
        ])

    def create_question_processing_model(self):
        """
        Creates a model that performs question processing
        :return: A TensorFlow Keras model using bidirectional LSTM layers
        """
        return tf.keras.models.Sequential([
            self.question_inputs,
            tf.keras.layers.Embedding(self.vocabulary_size,
                                      input_length=self.max_question_length,
                                      output_dim=self.embedding_size),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size,
                                                               return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)),
            tf.keras.layers.Dense(self.output_size, activation='relu')
        ])

    def create_model(self):
        """
        Creates a VQA model combining an image and question model
        :return: LSTM+CNN VQA model
        """
        image_model = self.create_image_processing_model()
        question_model = self.create_question_processing_model()
        linked = tf.keras.layers.multiply([image_model.output, question_model.output])
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(linked)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self):
        """

        :return:
        """
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.model.fit_generator(generator=self.train_generator,
                                 validation_data=self.val_generator,
                                 use_multiprocessing=True,
                                 workers=6)

    def __init__(self, input_json, input_h5):
        self.image_inputs = tf.keras.Input(shape=(self.image_width, self.image_height, 3))
        self.question_inputs = tf.keras.Input(shape=(self.max_question_length,))
        self.train_generator = VQA_data_generator(input_json, input_h5)
        self.val_generator = VQA_data_generator(input_json, input_h5, train=False)
        self.model = self.create_model()


if __name__ == '__main__':
    vqa = Lstm_cnn_trainer('data/data_prepro.json', 'data/data_prepro.h5')
