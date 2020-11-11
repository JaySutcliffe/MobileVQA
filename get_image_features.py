import json
import tensorflow as tf
import numpy as np
import time

# Currently this file is not in use and may be removed later

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image


def get_image_feature(image_path):
    #image_feature = vgg(temp)
    #image_feature_normalised = \
    #    tf.math.l2_normalize(image_feature, axis=None, epsilon=1e-12, name=None)
    #return image_feature_normalised
    return None

def main(input_json, output_npy):
    dataset = {}
    # load json file
    print('Loading json file...')
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    for i in range(0, len(dataset['unique_img_train']), 1000):
        cut = dataset['unique_img_train'][
              i:min(len(dataset['unique_img_train']), i+1000)]
        preprocessed_images = np.vectorize(get_image_feature, cut)


    for image_path in dataset['unique_img_train']:
        get_image_feature(image_path)
        # To be stored in a file


if __name__ == '__main__':
    main('data/data_prepro.json', 'D:/Part2Project/VQAv2/image_prepro.npy')
