import argparse
import sys
import tensorflow as tf
import json
import numpy as np


def preprocess_image(model_number, image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    if model_number == 1:
        return tf.keras.applications.vgg19.preprocess_input(image_array)
    else:
        return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)


def get_normalised_vgg19_model():
    """
    Creates the image processing part of the VQA model
    :return: A Keras model with VGG19 CNN to extract image features
    """
    model = tf.keras.applications.VGG19(
        include_top=True,
        weights='imagenet')

    outputs = tf.keras.layers.LayerNormalization()(model.layers[-2].output)
    return tf.keras.Model(model.input, outputs=outputs)


def get_mobilenet_v2():
    model = tf.keras.applications.MobileNetV2(include_top=True,
                                              weights='imagenet')

    outputs = model.layers[-2].output
    return tf.keras.Model(model.input, outputs=outputs)


def get_mobilenet_v2_3by3():
    model = tf.keras.applications.MobileNetV2(include_top=False,
                                              weights='imagenet')
    avg_pool = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    outputs = avg_pool(model.layers[-2].output)
    return tf.keras.Model(model.input, outputs=outputs)

def preprocess_all_images(model_number, input_json, train, result):
    mode = "test"
    if train:
        mode = "train"

    if model_number == 1:
        model = get_normalised_vgg19_model()
    else:
        model = get_mobilenet_v2()

    dataset = {}
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    unique_images_ids = dataset['unique_img_' + mode]
    image_features = []
    for i in range(0, len(unique_images_ids)):
        print(i)
        image_preprocessed = np.array([preprocess_image(model_number, unique_images_ids[i])])
        prediction = model.predict(image_preprocessed)
        image_features.append(prediction)

    with open(result, "wb") as f:
        np.save(f, image_features)


def preprocess_all_images_7by7(model_number, input_json, train, result):
    mode = "test"
    if train:
        mode = "train"

    if model_number == 1:
        model = get_normalised_vgg19_model()
    elif model_number == 2:
        model = get_mobilenet_v2()
    else:
        model = get_mobilenet_v2_3by3()

    dataset = {}
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    unique_images_ids = dataset['unique_img_' + mode]
    image_features = []
    j = 0
    for i in range(0, len(unique_images_ids)):
        print(i)
        image_preprocessed = np.array([preprocess_image(model_number, unique_images_ids[i])])
        image_features.append(model.predict(image_preprocessed))
        j += 1
        if j >= 30000:
            j = 0
            np.savez(result+str(i//30000), *image_features)
            image_features = []
    np.savez(result + str(i // 30000), *image_features)


def main(model, input_json, train_result, test_result):
    preprocess_all_images(model, input_json, True, train_result)
    preprocess_all_images(model, input_json, False, test_result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', help='input json file output from preprocess_questions.py')
    parser.add_argument('--train_result', help='where to place and what to call the output training npy file')
    parser.add_argument('--test_result', help='where to place and what to call the output testing npy file')
    parser.add_argument('--model', help='1 is normalised VGG19, 2 is mobilenet, 3 is mobilnet 3x3', type=int)

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict

    main(params['model'], params['input_json'], params['train_result'], params['test_result'])