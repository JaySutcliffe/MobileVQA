import numpy as np
import h5py
import json
from data_generator import align, VQA_data_generator


def soft_score(occurrences):
    if occurrences == 0:
        return 0
    elif occurrences == 1:
        return 0.3
    elif occurrences == 2:
        return 0.6
    elif occurrences == 3:
        return 0.9
    else:
        return 1


def get_to_sparse_answer_vector(string, actual_answer):
    """
    Returns a sparse vector mapping each answer to its soft score

    Parameters:
        string (str): The string representation of the answer to occurrence mapping
        actual_answer (str): The ground truth answer provided in the dataset

    Returns:
        A dict object mapping answers with non-zero soft scores to soft scores
    """
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
    sparse_vector[actual_answer - 1] = 1
    return sparse_vector


def sparse_to_full(sparse_vector):
    """
    Takes in a sparse answer vector generated by get_to_sparse_answer_vector

    Parameters:
        sparse_vector (dict): The dictionary mapping index to soft score

    Returns:
        A (3000,) numpy array each index mapping to answer soft score
    """
    vector = np.zeros(3000)
    for answer, score in sparse_vector.items():
        answer = answer - 1
        if answer < 3000:
            vector[answer] = score
    return vector


class VQA_soft_data_generator(VQA_data_generator):

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

            temp_ans_more = hf.get('ans_more_' + self.__mode)
            temp_ans = hf.get('ans_' + self.__mode)
            self.__data['answers'] = [get_to_sparse_answer_vector(string, ans)
                                      for string, ans in zip(temp_ans_more, temp_ans)]

        self.__data['questions'] = align(self.__data['questions'],
                                         self.__data['length_q'], max_length=14)

    def __init__(self, input_json, input_h5, train=True,
                 batch_size=500, shuffle=True, feature_object=None):
        super().__init__(self, input_json, input_h5, train, batch_size, shuffle, feature_object)

    def __len__(self):
        return int(np.floor(len(self.__data['questions']) / self.batch_size))

    def __getitem__(self, idx):
        questions = np.array(self.__data['questions'][
                             idx * self.batch_size:(idx + 1) * self.batch_size])
        image_list = self.__data['img_list'][
                     idx * self.batch_size:(idx + 1) * self.batch_size]
        answers_sparse = self.__data['answers'][
                         idx * self.batch_size:(idx + 1) * self.batch_size]
        answers = np.array([sparse_to_full(answer) for answer in answers_sparse])
        image_features = np.array(
            [self.__unique_features.get(i) for i in image_list])

        return [image_features, questions], answers