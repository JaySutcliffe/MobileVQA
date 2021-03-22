import argparse
import sys
import tensorflow as tf
import json
import numpy as np


def preprocess_image(model_number, image_path):
    """
    Performs the necessary preprocessing on an image,
    dependent on the CNN being used

    Parameters:
        model_number (int): 1 for VGG19, 2 Mobilenet, 3 stripped MobileNet
        image_path (str): The image location on disc

    Returns:
        Preprocessed image based on image path entered
    """
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

    Returns:
         A Keras model with VGG19 CNN to extract image features
    """
    model = tf.keras.applications.VGG19(
        include_top=True,
        weights='imagenet')

    outputs = tf.keras.layers.LayerNormalization()(model.layers[-2].output)
    return tf.keras.Model(model.input, outputs=outputs)


def get_mobilenet_v2():
    """
    Gets MobileNetv2 trained on ImageNet removing the final 1000
    classifier layer so transfer learning can be performed.

    Returns:
         MobileNetv2 Keras Model with a 1x1x1280 output vector
    """
    model = tf.keras.applications.MobileNetV2(include_top=True,
                                              weights='imagenet')
    outputs = model.layers[-2].output
    return tf.keras.Model(model.input, outputs=outputs)


def get_mobilenet_v2_3by3():
    """
    Gets Mobilenetv2 but removes the two final layers leading to a 3x3x1280
    output.

    Returns:
        MobileNetv2 with no top layers and a average pool added to produce
        3x3x1280 features
    """
    model = tf.keras.applications.MobileNetV2(include_top=False,
                                              weights='imagenet',
                                              input_shape=(224, 224, 3))
    avg_pool = tf.keras.layers.AveragePooling2D(
        pool_size=(3, 3), strides=2, padding='valid', data_format=None)
    outputs = avg_pool(model.layers[-1].output)
    return tf.keras.Model(model.input, outputs=outputs)


def process_all_images(model_number, input_json, train, result):
    """
    Processes all images writing the image features to the file location result

    Parameters:
        model_number (int): 1 for VGG19, 2 MobileNet, 3 stripped MobileNet
        input_json (str): input json file to read the image locations from
        train (boolean): True when training data
        result (str): resulting npy file location

    Returns:
        MobileNetv2 with no top layers and a average pool added to produce
        3x3x1280 features
    """
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
    for i in range(0, len(unique_images_ids)):
        print(i)
        image_preprocessed = np.array([preprocess_image(model_number, unique_images_ids[i])])
        prediction = model.predict(image_preprocessed)
        image_features.append(prediction)

    with open(result, "wb") as f:
        np.save(f, image_features)


def process_all_images_7by7(model_number, input_json, train, result):
    """No longer used. With this function I tested storing 7x7x1280 MobileNet
    image features in separate archived files"""
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
            np.savez(result + str(i // 30000), *image_features)
            image_features = []
    np.savez(result + str(i // 30000), *image_features)


class Feature_extracted_mobilenet_1by1:
    """
    Class to extract 1x1x1280 image features from the MobileNetv2 model
    """

    def get(self, i):
        feat = self.feature_file[i]
        #temp = np.sqrt(np.sum(np.multiply(feat, feat), axis=1))
        #img_feature = np.divide(feat, np.transpose(np.tile(temp, (1280, 1))))
        feat = feat.reshape(1280,)
        return feat

    def __init__(self, feature_file):
        self.feature_file = np.load(feature_file)


class Feature_extracted_mobilenet_3by3:
    """
    Class to extract 1x1x1280 image features from the stripped MobileNetv2 model
    """

    def get(self, i):
        feat = self.feature_file[i]
        return feat.reshape((3, 3, 1280))

    def __init__(self, feature_file):
        self.feature_file = np.load(feature_file)


class Feature_extracted_mobilenet_7by7:
    """
    My attempts at 7x7x1280 feature extraction
    """

    def get(self, i):
        if i < 30000:
            return self.feature_file1["arr_" + str(i)]
        elif i < 60000:
            return self.feature_file2["arr_" + str(i - 30000)]
        else:
            return self.feature_file3["arr_" + str(i - 60000)]

    def __init__(self, feature_file1, feature_file2, feature_file3=None):
        self.ff = feature_file1
        self.feature_file1 = np.load(feature_file1)
        self.feature_file2 = np.load(feature_file2)
        self.feature_file3 = None
        if feature_file3 is not None:
            self.feature_file3 = np.load(feature_file3)


def main(model, input_json, train_result, test_result):
    """
    Simple function to process both training and validation data
    """
    process_all_images(model, input_json, True, train_result)
    process_all_images(model, input_json, False, test_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', help='input json file output from preprocess_questions.py')
    parser.add_argument('--train_result', help='where to place and what to call the output training npy file')
    parser.add_argument('--test_result', help='where to place and what to call the output testing npy file')
    parser.add_argument('--model', help='1 is normalised VGG19, 2 is mobilenet, 3 is MobileNetv2 3x3', type=int)

    args = parser.parse_args()
    params = vars(args)
    main(params['model'], params['input_json'], params['train_result'], params['test_result'])
