import sys
import tensorflow as tf
from data_generator import VQA_data_generator
from cnn import get_normalised_vgg19_model

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
    batch_size = 32
    embedding_size = 200
    rnn_size = 512
    dense_hidden_size = 1024
    output_size = 1000

    def create_question_processing_model(self):
        """
        Creates a model that performs question processing
        :return: A TensorFlow Keras model using bidirectional LSTM layers
        """
        forward_layer1 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)
        forward_layer2 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)
        backward_layer1 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True,
                                               go_backwards=True)
        backward_layer2 = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True,
                                               go_backwards=True)

        """
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size,
                                                           return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size)),
        """
        return tf.keras.models.Sequential([
            self.question_inputs,
            tf.keras.layers.Embedding(self.vocabulary_size,
                                      input_length=self.max_question_length,
                                      output_dim=self.embedding_size),

            tf.keras.layers.Bidirectional(forward_layer1, backward_layer=backward_layer1),
            tf.keras.layers.Bidirectional(forward_layer2, backward_layer=backward_layer2),
            tf.keras.layers.Dense(self.dense_hidden_size, activation='relu')
        ])


    def create_model(self):
        """
        Creates a VQA model combining an image and question model
        :return: LSTM+CNN VQA model
        """
        if self.__train_cnn:
            image_model = get_normalised_vgg19_model()(self.image_inputs)
        else:
            image_model = self.image_inputs

        image_model_output = \
            tf.keras.layers.Dense(self.dense_hidden_size, activation='relu')(image_model)
        question_model = self.create_question_processing_model()
        linked = tf.keras.layers.multiply([image_model_output, question_model.output])
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(linked)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, checkpoint_path, save_path):
        """
        Trains the model and then outputs it to the file entered
        """
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         verbose=1)

        print(self.model.summary())

        self.model.fit(x=self.train_generator,
                       validation_data=self.val_generator,
                       epochs=1,
                       callbacks=[cp_callback])

        self.model.save(save_path)

    def __init__(self, input_json, input_h5, train_cnn=False,
                 train_feature_file=None,
                 valid_feature_file=None):
        if train_cnn:
            self.image_inputs = tf.keras.Input(shape=(self.image_width, self.image_height, 3))
        else:
            self.image_inputs = tf.keras.Input(shape=(4096,))
        self.question_inputs = tf.keras.Input(shape=(self.max_question_length,))
        if not train_cnn:
            if train_feature_file is None:
                self.train_generator = VQA_data_generator(input_json, input_h5)
            else:
                self.train_generator = VQA_data_generator(
                    input_json, input_h5, feature_file=train_feature_file)
            if valid_feature_file is None:
                self.val_generator = VQA_data_generator(input_json, input_h5, train=False)
            else:
                self.val_generator = VQA_data_generator(
                    input_json, input_h5, train=False,
                    feature_file=valid_feature_file)
        self.__train_cnn = train_cnn
        self.model = self.create_model()
        print(self.model.summary())


if __name__ == '__main__':
    vqa = Lstm_cnn_trainer(sys.argv[1], sys.argv[2], train_feature_file=sys.argv[3], valid_feature_file=sys.argv[4])
    vqa.train_model(sys.argv[5], sys.argv[6])
