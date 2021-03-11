import argparse
import json
import h5py
#import tensorflow as tf
#import tflite_runtime.interpreter as tflite
import numpy as np
from data_generator import align


def evaluate_model(model_path, input_json, input_h5, feature_object, max_length):
    dataset = {}
    data = {}
    with open(input_json) as data_file:
        d = json.load(data_file)
    for key in d.keys():
        dataset[key] = d[key]

    with h5py.File(input_h5, 'r') as hf:
        temp = hf.get('ques_test')
        data['questions'] = np.array(temp)

        temp = hf.get('ques_length_test')
        data['length_q'] = np.array(temp)

        # Subtract 1 based on indexing
        temp = hf.get('img_pos_test')
        data['img_list'] = np.array(temp) - 1

        temp = hf.get('ans_test')
        data['answers'] = np.array(temp) - 1

    # Aligns questions to the left or right
    data['questions'] = align(data['questions'], data['length_q'], max_length=max_length)

    questions = np.array(data['questions'])
    image_features = np.array([feature_object.get(i) for i in data['img_list']])

    # interpreter = tf.lite.Interpreter(model_path=model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='data/data_prepro.json',
                        help='the path to the json file')
    parser.add_argument('--input_h5', default='data/data_prepro.h5',
                        help='the path to the h5 file')
    parser.add_argument('--model_path', default='data/vqa.tflite',
                        help='the path to the tne Tensorflow Lite mod')
    parser.add_argument('--feature_object', type="int", default='2',
                        help='1 for VGG19, 2 MobileNetv2, 3 MobileNetv2 3x3')
    parser.add_argument('--max_length', type="int", default='14',
                        help='26 normally, 14 for attention models')

    args = parser.parse_args()
    params = vars(args)
    main(params['model_path'],
         params['input_json'],
         params['input_h5'],
         params['feature_object'],
         params['max_length'])

