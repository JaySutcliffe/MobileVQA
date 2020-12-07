import tensorflow as tf

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
