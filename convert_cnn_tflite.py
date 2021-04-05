import tensorflow as tf

from cnn import get_mobilenet_v2_3by3, get_mobilenet_v2


def convert_mobilenet(output_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(get_mobilenet_v2())
    lite_model = converter.convert()

    with open(output_file, 'wb') as f:
        f.write(lite_model)


def convert_mobilenet_f16(output_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(get_mobilenet_v2())
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    lite_model = converter.convert()

    with open(output_file, 'wb') as f:
        f.write(lite_model)


def convert_mobilenet_3by3(output_file):
    converter = tf.lite.TFLiteConverter.from_keras_model(get_mobilenet_v2_3by3())
    lite_model = converter.convert()

    with open(output_file, 'wb') as f:
        f.write(lite_model)

if __name__ == '__main__':
    convert_mobilenet_f16("D:/Part2Project/mobilenet_f16.tflite")