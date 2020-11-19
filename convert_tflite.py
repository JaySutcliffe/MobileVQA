import sys
import tensorflow as tf

if __name__ == '__main__':
    converter = tf.lite.TFLiteConverter.from_saved_model(sys.argv[1])
    lite_model = converter.convert()

    # Save the model.
    with open(sys.argv[2], 'wb') as f:
        f.write(lite_model)