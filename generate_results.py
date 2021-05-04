import argparse
import json
import h5py
import tensorflow as tf
import numpy as np

from cnn import Feature_extracted_mobilenet_1by1
from cnn import Feature_extracted_mobilenet_3by3
from data_generator import align


def store_results(model_path, input_json, input_h5,
                  feat_object, max_length, result):
    dataset = {}
    data = {}
    answers_with_ids = []
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

        temp = hf.get('question_id_test')
        data['question_id_test'] = np.array(temp)

    # Aligns questions to the left or right
    data['questions'] = align(data['questions'], data['length_q'], max_length=max_length)

    questions = np.array(data['questions'])
    image_features = np.array([feat_object.get(i) for i in data['img_list']])

    interpreter = tf.lite.Interpreter(model_path=model_path)
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    interpreter.allocate_tensors()

    for i in range(questions.shape[0]):
        img_feat = np.expand_dims(image_features[i], 0).astype(np.float32)
        ques = np.expand_dims(questions[i], 0).astype(np.float32)
        print(img_feat.shape)
        interpreter.set_tensor(2, ques)
        interpreter.set_tensor(3, img_feat)
        interpreter.invoke()
        answer_index = int(np.argmax(output()))
        answer = dataset['ix_to_ans'][str(answer_index + 1)]
        answers_with_ids.append({'answer': answer, 'question_id': int(data['question_id_test'][i])})

        if i % 1000 == 0:
            print(i)

    json.dump(answers_with_ids, open(result, 'w'))


def store_results_keras_model(model, input_json, input_h5,
                              feat_object, max_length, result):
    dataset = {}
    data = {}
    answers_with_ids = []
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

        temp = hf.get('question_id_test')
        data['question_id_test'] = np.array(temp)

    # Aligns questions to the left or right
    data['questions'] = np.array(align(data['questions'], data['length_q'], max_length=max_length))

    image_features = np.array([feat_object.get(i) for i in data['img_list']])
    predictions = model.predict([image_features, data['questions']])

    for i in range(predictions.shape[0]):
        answer_index = int(np.argmax(predictions[i]))
        answer = dataset['ix_to_ans'][str(answer_index + 1)]
        answers_with_ids.append({'answer': answer, 'question_id': int(data['question_id_test'][i])})

    json.dump(answers_with_ids, open(result, 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='data/data_prepro.json',
                        help='the path to the json file')
    parser.add_argument('--input_h5', default='data/data_prepro.h5',
                        help='the path to the h5 file')
    parser.add_argument('--model_path',
                        default='D:/Downloads/basic_attention_vqa.tflite',
                        help='the path to the tne Tensorflow Lite mod')
    parser.add_argument('--feature_file', default='D:/Part2Project/val_new.npy',
                        help='the file containing the test images')
    parser.add_argument('--feature_object', type=int, default=3,
                        help='1 for VGG19, 2 MobileNetv2, 3 MobileNetv2 3x3')
    parser.add_argument('--max_length', type=int, default=26,
                        help='26 normally, 14 for attention models')
    parser.add_argument('--output_json', default='data/basic_attention_test_results.json',
                        help='output json file to store the results')

    args = parser.parse_args()
    params = vars(args)

    if params['feature_object'] == 2:
        feature_object = Feature_extracted_mobilenet_1by1(params['feature_file'])
    elif params['feature_object'] == 3:
        feature_object = Feature_extracted_mobilenet_3by3(params['feature_file'])
        print(feature_object.get(0).shape)

    store_results(params['model_path'],
                  params['input_json'],
                  params['input_h5'],
                  feature_object,
                  params['max_length'],
                  params['output_json'])

    """
    model = tf.keras.models.load_model("D:/Part2Project/full_attention_model")
    store_results_keras_model(model,
                              params['input_json'],
                              params['input_h5'],
                              feature_object,
                              params['max_length'],
                              params['output_json'])
    """