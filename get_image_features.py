import json
import tensorflow as tf
import numpy as np
import time

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')


def get_image_feature(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    temp = tf.keras.preprocessing.image.img_to_array(image)
    temp = np.expand_dims(temp, axis=0)
    temp = tf.keras.applications.vgg19.preprocess_input(temp)
    image_feature = vgg(temp)
    image_feature_normalised = \
        tf.math.l2_normalize(image_feature, axis=None, epsilon=1e-12, name=None)
    return image_feature_normalised


def main(input_json, output_npy):
    dataset = {}
    # load json file

    print('Loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    for image_path in dataset['unique_img_train']:
        get_image_feature(image_path)
        # To be stored in a file


if __name__ == '__main__':
    main('data/data_prepro.json', 'D:/Part2Project/VQAv2/image_prepro.npy')
