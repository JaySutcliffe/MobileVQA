import tensorflow as tf

from cnn import Feature_extracted_mobilenet_1by1
from data_generator import VQA_data_generator

if __name__ == '__main__':
    input_json = "data/data_prepro.json"
    input_h5 = "data/data_prepro.h5"
    # Example values found it more convenient to just edit code than set up file as a script
    # This was simply to experiment with Int8 optimisation rather to generate any necessary
    # files.
    train_feature_file = "D:/Part2Project/train30002.npy"
    output = "D:/Downloads/lstm_cnn_model2"
    vqa_gen = VQA_data_generator(input_json, input_h5, train=False,
                                 feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), batch_size=100)

    def representative_dataset():
        [image_features, questions], answers = vqa_gen.__getitem__(0)
        yield [image_features, questions]


    converter = tf.lite.TFLiteConverter.from_saved_model(output)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    #converter.representative_dataset = representative_dataset
    lite_model = converter.convert()

    # Save the model.
    with open("D:/Downlaods/lstm_vqa_i8.tflite", 'wb') as f:
        f.write(lite_model)