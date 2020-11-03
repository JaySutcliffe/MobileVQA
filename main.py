import json
import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras

def right_align(seq,lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N-lengths[i]:N-1]=seq[i][0:lengths[i]-1]
    return v

class Lstm_cnn_trainer():
    # The size of a question and inage
    max_question_length = 26
    image_width = 224
    image_height = 224

    image_inputs = None
    question_inputs = None

    # Input files from preprocessing
    input_json = None
    input_h5 = None

    # For training
    train_dataset = {}
    train_data = {}

    # For testing
    test_dataset = {}
    test_data = {}
    test_input_json = None
    test_input_questions_h5 = None

    model = None

    # Model configuration variables
    embedding_size = 200
    rnn_size = 512
    rnn_layer_count = 2
    dim_image = 4096
    dense_hidden_size = 1024
    output_size = 1000
    img_norm = 1

    def get_data(self):
        self.train_dataset = {}
        self.train_data = {}
        # load json file
        print('loading json file...')
        with open(self.input_json) as data_file:
            data = json.load(data_file)
        for key in data.keys():
            self.train_dataset[key] = data[key]

        # load h5 file
        print('loading h5 file...')
        with h5py.File(self.input_h5, 'r') as hf:
            tem = hf.get('ques_train')
            self.train_data['question'] = np.array(tem) - 1

            tem = hf.get('ques_length_train')
            self.train_data['length_q'] = np.array(tem)

            tem = hf.get('img_pos_train')
            self.train_data['img_list'] = np.array(tem) - 1

            tem = hf.get('answers')
            self.train_data['answers'] = np.array(tem) - 1

        print('question aligning')
        self.train_data['question'] = right_align(self.train_data['question'],
                                                  self.train_data['length_q'])

        print(self.train_data['question'][1])
        print(self.train_data['answers'][1])

    def get_data_test(self):
        self.test_dataset = {}
        self.test_data = {}
        # load json file
        print('loading json file...')
        with open(self.input_json) as data_file:
            data = json.load(data_file)
        for key in data.keys():
            self.test_dataset[key] = data[key]

        # load h5 file
        print('loading h5 file...')
        with h5py.File(self.input_h5, 'r') as hf:
            tem = hf.get('ques_test')
            self.test_data['question'] = np.array(tem) - 1

            tem = hf.get('ques_length_test')
            self.test_data['length_q'] = np.array(tem)

            tem = hf.get('img_pos_test')
            self.test_data['img_list'] = np.array(tem) - 1

            tem = hf.get('question_id_test')
            self.test_data['ques_id'] = np.array(tem)

        print('question aligning')
        self.test_data['question'] = right_align(
            self.test_data['question'], self.test_data['length_q'])

    #def get_images(self):

    #    for i in range(0, len())
    #        img = image.load_img(img_path, target_size=(224, 224))
    #        x = image.img_to_array(img)
    #        x = np.expand_dims(x, axis=0)
    #        x = tf.keras.applications.vgg19.preprocess_input(x)

    def create_image_processing_model(self):
        """
        Creates the image processing part of the VQA model
        :return: A Keras model with VGG19 CNN to extract image features
        """
        return tf.keras.models.Sequential([
            self.image_inputs,
            tf.keras.applications.VGG19(
                include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                pooling=None, classes=1000, classifier_activation='softmax'
            )

            #tem = np.sqrt(np.sum(np.multiply(self.image_feature, self.image_feature), axis=1))
            #self.train_image_feature = np.divide(img_feature, np.transpose(np.tile(tem, (4096, 1))))
        ])


    def create_question_processing_model(self):
        """
        Creates a model that performs question processing
        :return: A TensorFlow Keras model using bidirectional LSTM layers
        """
        inputs = self.question_inputs
        question_layers = tf.keras.layers.Embedding(input_dim=self.embedding_size,
                                                    output_dim=self.rnn_size, mask_zero=True)(inputs)
        question_layers = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.rnn_size))(question_layers)
        outputs = tf.keras.layers.Dense(self.output_size, activation="relu")(question_layers)
        return tf.keras.Model(inputs=inputs, outputs=outputs)


    def create_model(self):
        """
        Creates a VQA model combining an image and question model
        :return: LSTM+CNN VQA model
        """
        image_model = self.create_image_processing_model()
        question_model = self.create_question_processing_model()
        linked = tf.keras.layers.multiply([image_model.output, question_model.output])
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(linked)

        return keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                            name=__class__.__name__ + "_model")

    def train_model(self):
        """

        :return:
        """
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        #self.model.fit(self.train_data_question, train_labels, epochs=1)


    def __init__(self, input_json, input_h5):
        self.image_inputs = tf.keras.Input(shape=(self.image_width, self.image_height, 3))
        self.question_inputs = tf.keras.Input(shape=(self.max_question_length))
        self.model = self.create_model()


        self.input_json = input_json
        self.input_h5 = input_h5
        self.get_data()
        self.get_data_test()



if __name__ == '__main__':
    vqa = Lstm_cnn_trainer('data/data_prepro.json','data/data_prepro.h5')