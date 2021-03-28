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
    def get_data(self):
        """
        Loads the questions, images and answers from the input dataset
        """

        # Unused but done for debugging purposes
        # if the indexes need to be converted to words
        # for debugging etc.
        with open(self.input_json) as data_file:
            d = json.load(data_file)
        for key in d.keys():
            self.dataset[key] = d[key]

        with h5py.File(self.input_h5, 'r') as hf:
            temp = hf.get('ques_' + self.mode)
            self.data['questions'] = np.array(temp)

            temp = hf.get('ques_length_' + self.mode)
            self.data['length_q'] = np.array(temp)

            # Subtract 1 based on indexing
            temp = hf.get('img_pos_' + self.mode)
            self.data['img_list'] = np.array(temp) - 1

            temp = hf.get('ans_' + self.mode)
            self.data['answers'] = np.array(temp) - 1

        # Removes questions with answers outside of the highest entered anwers
        self.data['questions'] = self.data['questions'][self.data['answers'] < self.answer_count]
        self.data['length_q'] = self.data['length_q'][self.data['answers'] < self.answer_count]
        self.data['img_list'] = self.data['img_list'][self.data['answers'] < self.answer_count]
        self.data['answers'] = self.data['answers'][self.data['answers'] < self.answer_count]


        # Aligns questions to the left or right
        self.data['questions'] = align(self.data['questions'],
                                       self.data['length_q'])

    def __init__(self, input_json, input_h5, train=True,
                 batch_size=500, shuffle=True, feature_object=None, answer_count=1000):
        self.data = {}
        self.dataset = {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_json = input_json
        self.input_h5 = input_h5
        self.answer_count = answer_count
        self.train = train
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'
        self.get_data()
        self.on_epoch_end()
        self.unique_features = feature_object

    def __len__(self):
        return int(np.floor(len(self.data['questions']) / self.batch_size))

    def __getitem__(self, idx):
        questions = np.array(self.data['questions'][
                             idx * self.batch_size:(idx + 1) * self.batch_size])
        image_list = self.data['img_list'][
                     idx * self.batch_size:(idx + 1) * self.batch_size]
        answers = np.array(self.data['answers'][
                           idx * self.batch_size:(idx + 1) * self.batch_size])

        image_features = np.array(
            [self.unique_features.get(i) for i in image_list])

        return [image_features, questions], answers

    def on_epoch_end(self):
        # Simple shuffling
        if self.shuffle:
            perm = np.random.permutation(len(self.data['questions']))
            self.data['questions'] = \
                [self.data['questions'][i] for i in perm]
            self.data['answers'] = \
                [self.data['answers'][i] for i in perm]
            self.data['img_list'] = \
                [self.data['img_list'][i] for i in perm]


if __name__ == '__main__':
    vqa_gen = VQA_data_generator('data/data_prepro.json', 'data/data_prepro.h5', train=False,
                                 feature_object='D:/Part2Project/val30002.npy')
    [image_features, questions], answers = vqa_gen.__getitem__(0)
    print(questions[0])