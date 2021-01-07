import numpy as np
import tensorflow as tf
import h5py
import json
from cnn import preprocess_image, get_normalised_vgg19_model


def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N - lengths[i]:N - 1] = seq[i][0:lengths[i] - 1]
    return v


class VQA_data_generator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __get_data(self):
        self.__dataset = {}
        self.__data = {}
        with open(self.input_json) as data_file:
            data = json.load(data_file)
        for key in data.keys():
            self.__dataset[key] = data[key]

        with h5py.File(self.input_h5, 'r') as hf:
            temp = hf.get('ques_' + self.__mode)
            self.__data['questions'] = np.array(temp) - 1

            temp = hf.get('ques_length_' + self.__mode)
            self.__data['length_q'] = np.array(temp)

            temp = hf.get('img_pos_' + self.__mode)
            self.__data['img_list'] = np.array(temp) - 1

            if self.__train:
                temp = hf.get('answers')
            else:
                temp = hf.get('question_id_test')

            self.__data['answers'] = np.array(temp) - 1

        self.__data['questions'] = right_align(self.__data['questions'],
                                               self.__data['length_q'])

    def __init__(self, input_json, input_h5, train=True, train_cnn=False,
                 batch_size=32, shuffle=True, feature_file=None):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_json = input_json
        self.input_h5 = input_h5
        self.__train = True
        self.__train_cnn = train_cnn
        if train:
            self.__mode = 'train'
        else:
            self.__mode = 'test'
        self.__get_data()
        #self.on_epoch_end()

        self.__unique_features = None
        if not self.__train_cnn:
            if feature_file is None:
                self.__cnn_model = get_normalised_vgg19_model()
            else:
                self.__unique_features = np.load(feature_file)

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.__data['questions']) / self.batch_size))
        return 3

    def __getitem__(self, idx):
        questions = np.array(self.__data['questions'][
                             idx * self.batch_size:(idx + 1) * self.batch_size])
        image_list = self.__data['img_list'][
                     idx * self.batch_size:(idx + 1) * self.batch_size]
        answers = np.array(self.__data['answers'][
                           idx * self.batch_size:(idx + 1) * self.batch_size])

        if self.__train_cnn:
            images_preprocessed = np.array([preprocess_image(
                self.__dataset['unique_img_' + self.__mode][i])
                for i in image_list])
            return [images_preprocessed, questions], answers

        if self.__unique_features is None:
            images_preprocessed = np.array([preprocess_image(
                self.__dataset['unique_img_' + self.__mode][i])
                for i in image_list])
            image_features = self.__cnn_model.predict(images_preprocessed)
        else:
            image_features = np.array(
                [self.__unique_features[i] for i in image_list])
            image_features = image_features.reshape((image_features.shape[0], -1))

        return [image_features, questions], answers

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            perm = np.random.permutation(len(self.__data['questions']))
            self.__data['questions'] = \
                [self.__data['questions'][i] for i in perm]
            self.__data['answers'] = \
                [self.__data['answers'][i] for i in perm]
            self.__data['img_list'] = \
                [self.__data['img_list'][i] for i in perm]


if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=True,
                                 train_cnn=False, feature_file='D:/Part2Project/train2.npy')
    [image_features, questions], answers = vqa_gen.__getitem__(0)
    print(questions[0])