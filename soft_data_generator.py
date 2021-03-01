import numpy as np
import tensorflow as tf
import h5py
import json
from data_generator import align


def soft_score(occurences):
    if occurences == 0:
        return 0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1


def string_to_sparse_answer_vector(string):
    sparse_vector = {}
    string = string.decode('ascii')
    answer_mapping = string.split(";")

    if answer_mapping == ['']:
        return sparse_vector
    for answer in answer_mapping:
        a = answer.split(",")
        index = int(a[0]) - 1
        occurrences = int(a[1])
        sparse_vector[index] = soft_score(occurrences)
    return sparse_vector


def sparse_to_full(sparse_vector):
    vector = np.zeros(3000)
    for answer, score in sparse_vector.items():
        vector[answer] = score
    return vector


class Feature_extracted_mobilenet_1by1():
    def get(self, i):
        feat = self.__feature_file[i]
        # feat = normalize(feat)
        feat = feat.reshape(1280, )
        return feat

    def __init__(self, feature_file):
        self.__feature_file = np.load(feature_file)


class VQA_soft_data_generator(tf.keras.utils.Sequence):
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
            self.__data['questions'] = np.array(temp)

            temp = hf.get('ques_length_' + self.__mode)
            self.__data['length_q'] = np.array(temp)

            temp = hf.get('img_pos_' + self.__mode)
            self.__data['img_list'] = np.array(temp) - 1

            temp = hf.get('ans_more_' + self.__mode)
            self.__data['answers'] = np.array([string_to_sparse_answer_vector(string) for string in temp])

        self.__data['questions'] = align(self.__data['questions'],
                                         self.__data['length_q'])

    def __init__(self, input_json, input_h5, train=True,
                 batch_size=500, shuffle=True, feature_object=None):
        'Initialization'
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_json = input_json
        self.input_h5 = input_h5
        if train:
            self.__mode = 'train'
        else:
            self.__mode = 'test'
        self.__get_data()
        self.on_epoch_end()
        self.__unique_features = feature_object

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.__data['questions']) / self.batch_size))

    def __getitem__(self, idx):
        questions = np.array(self.__data['questions'][
                             idx * self.batch_size:(idx + 1) * self.batch_size])
        image_list = self.__data['img_list'][
                     idx * self.batch_size:(idx + 1) * self.batch_size]
        answers_sparse = self.__data['answers'][
                           idx * self.batch_size:(idx + 1) * self.batch_size]
        answers = [sparse_to_full(answer) for answer in answers_sparse]
        image_features = np.array(
            [self.__unique_features.get(i) for i in image_list])

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
    vqa_gen = VQA_soft_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                      feature_object=
                                      Feature_extracted_mobilenet_1by1('D:/Part2Project/val4.npy'))
    [image_features, questions], answers = vqa_gen.__getitem__(0)
    print(answers[0])
