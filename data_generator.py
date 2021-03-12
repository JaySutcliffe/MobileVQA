import numpy as np
import tensorflow as tf
import h5py
import json


def right_align(seq, lengths):
    v = np.zeros(np.shape(seq))
    N = np.shape(seq)[1]
    for i in range(np.shape(seq)[0]):
        v[i][N - lengths[i]:N - 1] = seq[i][0:lengths[i] - 1]
    return v


def align(seq, lengths, max_length=26):
    v = np.zeros((np.shape(seq)[0], max_length))
    for i in range(np.shape(seq)[0]):
        v[i][:min(lengths[i], max_length)] = seq[i][:min(lengths[i], max_length)]
    return v


class VQA_data_generator(tf.keras.utils.Sequence):
    def __get_data(self):
        """
        Loads the questions, images and answers from the input dataset
        """
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

            # Subtract 1 based on indexing
            temp = hf.get('img_pos_' + self.__mode)
            self.__data['img_list'] = np.array(temp) - 1

            temp = hf.get('ans_' + self.__mode)
            self.__data['answers'] = np.array(temp) - 1

        # Removes questions with answers outside of the highest entered anwers
        self.__data['questions'] = self.__data['questions'][self.__data['answers'] < self.answer_count]
        self.__data['length_q'] = self.__data['length_q'][self.__data['answers'] < self.answer_count]
        self.__data['img_list'] = self.__data['img_list'][self.__data['answers'] < self.answer_count]
        self.__data['answers'] = self.__data['answers'][self.__data['answers'] < self.answer_count]


        # Aligns questions to the left or right
        self.__data['questions'] = align(self.__data['questions'],
                                         self.__data['length_q'])

    def __init__(self, input_json, input_h5, train=True,
                 batch_size=500, shuffle=True, feature_object=None, answer_count=1000):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_json = input_json
        self.input_h5 = input_h5
        self.answer_count = answer_count
        self.__train = True
        if train:
            self.__mode = 'train'
        else:
            self.__mode = 'test'
        self.__get_data()
        self.on_epoch_end()
        self.__unique_features = feature_object

    def __len__(self):
        return int(np.floor(len(self.__data['questions']) / self.batch_size))

    def __getitem__(self, idx):
        questions = np.array(self.__data['questions'][
                             idx * self.batch_size:(idx + 1) * self.batch_size])
        image_list = self.__data['img_list'][
                     idx * self.batch_size:(idx + 1) * self.batch_size]
        answers = np.array(self.__data['answers'][
                           idx * self.batch_size:(idx + 1) * self.batch_size])

        image_features = np.array(
            [self.__unique_features.get(i) for i in image_list])

        return [image_features, questions], answers

    def on_epoch_end(self):
        # Simple shuffling
        if self.shuffle:
            perm = np.random.permutation(len(self.__data['questions']))
            self.__data['questions'] = \
                [self.__data['questions'][i] for i in perm]
            self.__data['answers'] = \
                [self.__data['answers'][i] for i in perm]
            self.__data['img_list'] = \
                [self.__data['img_list'][i] for i in perm]


if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                 feature_object='D:/Part2Project/val.npy')
    [image_features, questions], answers = vqa_gen.__getitem__(0)
    print(questions[0])