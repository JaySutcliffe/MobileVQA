import sys
import tensorflow as tf
import json
import numpy as np

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return tf.keras.applications.vgg19.preprocess_input(image_array)


def get_normalised_vgg19_model():
    """
    Creates the image processing part of the VQA model
    :return: A Keras model with VGG19 CNN to extract image features
    """
    model = tf.keras.applications.VGG19(
        include_top=True,
        weights='imagenet')

    outputs = tf.keras.layers.LayerNormalization()(model.layers[-2].output)
    # return tf.keras.Model(model.input, outputs=outputs)
    return model

def preprocess_all_images(input_json, input_h5, train, result):
    mode = "test"
    if train:
        mode = "train"

    dataset = {}
    with open(input_json) as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    cnn = get_normalised_vgg19_model()
    unique_images_ids = dataset['unique_img_' + mode]
    image_features = []
    for i in range(0, len(unique_images_ids)):
        print(i)
        image_preprocessed = np.array([preprocess_image(unique_images_ids[i])])
        image_features.append(cnn.predict(image_preprocessed))

    np.save(result, image_features)

def save_as_txt(input_npy, result):
    arr = np.load(input_npy)
    arr = arr.reshape(82575, -1)
    np.savetxt(result, arr[:1000])

def main(input_json, train_result, test_result):
    preprocess_all_images(input_json, True, train_result)
    preprocess_all_images(input_json, False, test_result)

if __name__=="__main__":
    main(sys.argv[2], sys.argv[3], sys.argv[4])
