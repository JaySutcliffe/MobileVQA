import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from functools import partial

from attention_layers import ModularCoAttention, MultiModalAttention
from cnn import Feature_extracted_mobilenet_1by1, Feature_extracted_mobilenet_3by3
from data_generator import VQA_data_generator
from soft_data_generator import VQA_soft_data_generator


class Lstm_cnn_trainer():
    # The size of a question and image
    max_question_length = 26
    image_feature_size = 1280

    embedding_matrix = None
    model = None
    train_generator = None
    val_generator = None

    # Model configuration variables
    max_epochs = 30
    patience = 3
    batch_size = 500
    embedding_size = 300
    rnn_size = 512
    dense_hidden_size = 1024
    output_size = 1000
    learning_rate = 0.0003

    image_inputs = tf.keras.Input(shape=(image_feature_size,))
    question_inputs = tf.keras.Input(shape=(max_question_length,))

    def set_embedding_matrix(self, input_glove_npy):
        self.embedding_matrix = np.load(input_glove_npy)

    def create_question_processing_model(self):
        """
        Creates a model that performs question processing

        Returns:
             A TensorFlow Keras model using bidirectional LSTM layers
        """
        forward_layer1 = tf.keras.layers.LSTM(self.rnn_size,
                                              input_shape=(self.max_question_length,),
                                              return_sequences=True)
        forward_layer2 = tf.keras.layers.LSTM(self.rnn_size,
                                              input_shape=(self.rnn_size,))
        backward_layer1 = tf.keras.layers.LSTM(self.rnn_size,
                                               input_shape=(self.max_question_length,),
                                               return_sequences=True,
                                               go_backwards=True)
        backward_layer2 = tf.keras.layers.LSTM(self.rnn_size,
                                               input_shape=(self.rnn_size,),
                                               go_backwards=True)

        return tf.keras.models.Sequential([
            self.question_inputs,
            tf.keras.layers.Embedding(self.embedding_matrix.shape[0],
                                      self.embedding_size,
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_question_length),
            tf.keras.layers.Bidirectional(forward_layer1, backward_layer=backward_layer1),
            tf.keras.layers.Bidirectional(forward_layer2, backward_layer=backward_layer2)
        ])

    def create_model(self):
        """
        Creates a VQA model combining an image and question model

        Returns:
            Keras LSTM+CNN VQA model
        """
        image_model_output = tf.keras.layers.Dense(
            self.dense_hidden_size, activation='tanh')(self.image_inputs)
        question_model = self.create_question_processing_model()
        question_dense = tf.keras.layers.Dense(
            self.dense_hidden_size, activation='tanh')(question_model.output)
        linked = tf.keras.layers.multiply([image_model_output, question_dense])
        next = tf.keras.layers.Dense(self.output_size, activation="tanh")(linked)
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs],
                              outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path):
        """
        Trains the model and then outputs it to the file entered
        """

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size)
        self.val_generator = VQA_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()


def apply_pruning_to_dense_embedding(layer, params):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **params)
    if isinstance(layer, tf.keras.layers.Embedding):
        return tfmot.sparsity.keras.prune_low_magnitude(layer, **params)
    return layer


class Pruned_lstm_cnn_trainer(Lstm_cnn_trainer):
    initial_sparsity = 0.5
    final_sparsity = 0.8

    def train_model(self, save_path):
        end_step = self.train_generator.__len__() * self.max_epochs

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=self.initial_sparsity,
                                                                     final_sparsity=self.final_sparsity,
                                                                     begin_step=0,
                                                                     end_step=end_step),
        }

        self.model = tf.keras.models.clone_model(
            self.model,
            clone_function=partial(apply_pruning_to_dense_embedding, params=pruning_params),
        )

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience),
                     tfmot.sparsity.keras.UpdatePruningStep()]

        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=callbacks)

        # Removes training variables
        self.model = tfmot.sparsity.keras.strip_pruning(self.model)

        self.model.save(save_path)
        return history


class Soft_lstm_cnn_trainer(Lstm_cnn_trainer):
    output_size = 3000

    def create_model(self):
        """
        Creates a VQA model combining an image and question model

        Returns:
             LSTM+CNN VQA model
        """
        image_model = self.image_inputs

        image_model_output = \
            tf.keras.layers.Dense(self.dense_hidden_size, activation='tanh')(image_model)
        question_model = self.create_question_processing_model()
        question_dense = tf.keras.layers.Dense(self.dense_hidden_size, activation='tanh')(question_model.output)
        linked = tf.keras.layers.multiply([image_model_output, question_dense])
        next = tf.keras.layers.Dense(self.output_size, activation="tanh")(linked)
        # Difference from before is the sigmoid output
        outputs = tf.keras.layers.Dense(self.output_size, activation="sigmoid")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_soft_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size)
        self.val_generator = VQA_soft_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()


class Attention_trainer(Lstm_cnn_trainer):
    image_inputs = tf.keras.Input(shape=(3, 3, 1280))

    def create_model(self):
        """
        Creates a VQA model combining an image and question model applying simple
        attention over the image features directed by the question features

        Returns:
             Attention VQA model
        """
        image_features = tf.keras.layers.Reshape((9, 1280))(self.image_inputs)
        question_model = self.create_question_processing_model()
        question_dense_features = tf.keras.layers.Dense(self.dense_hidden_size, activation='tanh')(
            question_model.output)

        question_stack = tf.keras.layers.RepeatVector(9)(question_model.output)
        non_linear_input = tf.keras.layers.concatenate([image_features, question_stack], axis=-1)
        y = tf.keras.layers.Dense(512, activation="tanh")(non_linear_input)
        g = tf.keras.layers.Dense(512, activation="sigmoid")(non_linear_input)
        attention_input = tf.keras.layers.multiply([y, g])
        attention_output = tf.keras.layers.Dense(1)(attention_input)
        attention_output = tf.keras.layers.Reshape((9,))(attention_output)
        attention_output = tf.keras.layers.Dense(9, activation="softmax", use_bias=False)(attention_output)
        attention_image_features = tf.keras.layers.Dot(axes=(1, 1))([attention_output, image_features])

        attention_final_dense = tf.keras.layers.Dense(self.dense_hidden_size, activation="tanh")(
            attention_image_features)
        linked = tf.keras.layers.multiply([attention_final_dense, question_dense_features])
        next = tf.keras.layers.Dense(self.output_size, activation="tanh")(linked)
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")


def non_linear_layer(size, x):
    #y_til = tf.keras.layers.Dense(size, activation='tanh')(x)
    #g = tf.keras.layers.Dense(size, activation='sigmoid')(x)
    #return tf.keras.layers.multiply([y_til, g])
    return tf.keras.layers.Dense(size, activation='tanh')(x)

class Soft_attention_trainer(Lstm_cnn_trainer):
    output_size = 3000
    dense_hidden_size = 1024
    image_inputs = tf.keras.Input(shape=(3, 3, 1280))

    def create_model(self):
        """
        Creates a VQA model combining an image and question model

        Returns:
            Attention VQA model
        """
        image_features = tf.keras.layers.Reshape((9, 1280))(self.image_inputs)
        #image_features = tf.keras.layers.LayerNormalization(axis=-1)(image_features)
        question_model = self.create_question_processing_model()
        question_dense = non_linear_layer(self.dense_hidden_size, question_model.output)

        question_stack = tf.keras.layers.RepeatVector(9)(question_model.output)
        non_linear_input = tf.keras.layers.concatenate([image_features, question_stack], axis=-1)
        attention_input = non_linear_layer(self.dense_hidden_size, non_linear_input)
        attention_output = tf.keras.layers.Dense(1, use_bias=False)(attention_input)
        attention_output = tf.keras.layers.Reshape((1, 9))(attention_output)
        attention_output = tf.nn.softmax(attention_output, axis=-1)
        attention_image_features = tf.reduce_sum(tf.matmul(attention_output, image_features), axis=-1)

        attention_final_dense = non_linear_layer(self.dense_hidden_size, attention_image_features)
        linked = tf.keras.layers.multiply([attention_final_dense, question_dense])
        next = non_linear_layer(self.dense_hidden_size, linked)
        outputs = tf.keras.layers.Dense(self.output_size, activation="sigmoid")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_soft_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size)
        self.val_generator = VQA_soft_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()


class Full_attention_trainer(Lstm_cnn_trainer):
    image_inputs = tf.keras.Input(shape=(3, 3, 1280))
    output_size = 3000
    max_question_length = 14
    question_inputs = tf.keras.Input(shape=(max_question_length,))
    batch_size = 64

    def create_question_processing_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Embedding(self.embedding_matrix.shape[0],
                                      self.embedding_size,
                                      weights=[self.embedding_matrix],
                                      input_length=self.max_question_length),
            tf.keras.layers.LSTM(1280, return_sequences=True)
        ])

    def create_model(self):
        """
        Creates a VQA model combining an image and question model
        :return: Attention VQA model
        """
        x = tf.keras.layers.Reshape((9, 1280))(self.image_inputs)
        #x = tf.keras.layers.Dense(512, activation='tanh')(x)
        #x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        y = self.create_question_processing_model()(self.question_inputs)
        y, x = ModularCoAttention(6, 1280, 8, 4*1280)(y, x)
        outputs = MultiModalAttention(1280, self.output_size)(y, x)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path, checkpoint_path):
        """
        Trains the model and then outputs it to the file entered
        """

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_best_only=True)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback, model_checkpoint_callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_soft_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size, max_length=self.max_question_length)
        self.val_generator = VQA_soft_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size, max_length=self.max_question_length)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()


if __name__ == '__main__':
    input_json = "data/data_prepro.json"
    input_h5 = "data/data_prepro.h5"
    input_glove_npy = "D:/Part2Project/word_embeddings.npy"
    train_feature_file = "D:/Part2Project/train_new.npy"
    valid_feature_file = "D:/Part2Project/val_new.npy"
    #train_feature_file = "D:/Part2Project/train30002.npy"
    #valid_feature_file = "D:/Part2Project/val30002.npy"
    output = "D:/Part2Project/saved_model/lstm_cnn_model"

    tf.keras.backend.clear_session()
    vqa = Soft_attention_trainer(input_json, input_h5, input_glove_npy,
                                 train_feature_object=Feature_extracted_mobilenet_3by3(train_feature_file),
                                 valid_feature_object=Feature_extracted_mobilenet_3by3(valid_feature_file))

    """
    vqa = Pruned_lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                                  train_feature_object=Feature_extracted_mobilenet_1by1(train_feature_file),
                                  valid_feature_object=Feature_extracted_mobilenet_1by1(valid_feature_file))
    """
    history = vqa.train_model(output)