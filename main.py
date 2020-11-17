import sys
import tensorflow as tf
from data_generator import VQA_data_generator


class Lstm_cnn_trainer():
    # The size of a question and image
    max_question_length = 26
    image_width = 224
    image_height = 224
    vocabulary_size = 12915

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
        model = tf.keras.applications.VGG19(
            include_top=True,
            weights='imagenet',
            input_tensor=self.image_inputs)

        norm = tf.keras.layers.LayerNormalization()(model.layers[-2].output)
        outputs = tf.keras.layers.Dense(self.dense_hidden_size)(norm)
        return tf.keras.Model(inputs=self.image_inputs, outputs=outputs,
                              name=__class__.__name__ + "_model")

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
            tf.keras.layers.Dense(self.dense_hidden_size, activation='relu')
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

    def train_model(self, checkpoint_path):
        """
        Trains the model and then outputs it to the file entered
        """
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         verbose=1)

        self.model.fit(x=self.train_generator,
                       validation_data=self.val_generator,
                       callbacks=[cp_callback])

    def __init__(self, input_json, input_h5):
        self.image_inputs = tf.keras.Input(shape=(self.image_width, self.image_height, 3))
        self.question_inputs = tf.keras.Input(shape=(self.max_question_length))
        self.train_generator = VQA_data_generator(input_json, input_h5)
        self.val_generator = VQA_data_generator(input_json, input_h5, train=False)
        self.model = self.create_model()


if __name__ == '__main__':
    vqa = Lstm_cnn_trainer(sys.argv[1], sys.argv[2])
    vqa.train_model(sys.argv[3])
