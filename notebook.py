# -*- coding: utf-8 -*-
"""Copy of VQA Github Version.ipynb

Automatically generated by Colaboratory.
"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/.../MobileVQA # Cloning the git repo
# %cd MobileVQA

from google.colab import drive

drive.mount('/content/drive/')

!pip install tensorflow_model_optimization

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import h5py
import numpy as np
import json
import sys
import tensorflow_model_optimization as tfmot
import os
import zipfile
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
tf.__version__

from cnn import Feature_extracted_mobilenet_1by1
from cnn import Feature_extracted_vgg19
from cnn import Feature_extracted_mobilenet_3by3

"""
This code was to experiment, train and create TensorFlow Lite files for models
"""

input_json = "/content/drive/My Drive/VQA/VQA_new/data_prepro.json"
input_h5 = "/content/drive/My Drive/VQA/VQA_new/data_prepro.h5"
input_glove_npy = "/content/drive/My Drive/VQA/word_embeddings.npy"
train_feature_file = "/content/drive/My Drive/VQA/VQA_new/train30002.npy"
valid_feature_file = "/content/drive/My Drive/VQA/VQA_new/val30002.npy"
train_feature_file_vgg19 = "/content/drive/My Drive/VQA/trainVGG19.npy"
valid_feature_file_vgg19 = "/content/drive/My Drive/VQA/valVGG19.npy"
train_feature_file_3by3 = "/content/drive/My Drive/VQA/train_new.npy"
valid_feature_file_3by3 = "/content/drive/My Drive/VQA/val_new.npy"
output = "/content/drive/My Drive/VQA/saved_model/lstm_cnn_model3"
output_vgg19 = "/content/drive/My Drive/VQA/saved_model/lstm_cnn_model_vgg19"
output_soft = "/content/drive/My Drive/VQA/saved_model/soft_lstm_cnn_model"
output_attention =  "/content/drive/My Drive/VQA/saved_model/attention_model2"
output_soft_attention =  "/content/drive/My Drive/VQA/saved_model/soft_attention_model"
output_full = "/content/drive/My Drive/VQA/saved_model/full_attention_model"
output_pruned = "/content/drive/My Drive/VQA/saved_model/lstm_cnn_pruned"
full_checkpoint = "/content/drive/My Drive/VQA/saved_model/full_checkpoint"
result_full = "/content/drive/My Drive/VQA/full_test_result.json"
result_attention = "/content/drive/My Drive/VQA/attention_test_result.json"

from trainers import Lstm_cnn_trainer

tf.keras.backend.clear_session()
vqa = Lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), 
                       valid_feature_object=Feature_extracted_mobilenet_1by1(valid_feature_file),
                       normalise=True)
history = vqa.train_model(output)

from trainers import Lstm_cnn_trainer

tf.keras.backend.clear_session()
vqa = Lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_vgg19(train_feature_file_vgg19), 
                       valid_feature_object=Feature_extracted_vgg19(valid_feature_file_vgg19),
                       normalise=True, vgg19=True)
history = vqa.train_model(output_vgg19)

import pandas as pd
from matplotlib import pyplot as plt

loss = {'loss': history.history['loss'],
        'val_loss' : history.history['val_loss']}

pd.DataFrame(loss).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 3)
plt.show()

from google.colab import drive
drive.mount('/content/drive')

acc = {'accuracy': history.history['accuracy'],
        'val_accuracy' : history.history['val_accuracy']}

pd.DataFrame(acc).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa2.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output)
converter.experimental_new_converter = True
lite_model = converter.convert()
# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa2.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_dy.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_dy.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_dy.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_f16.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_f16.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_f16.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

from data_generator import VQA_data_generator

vqa_gen = VQA_data_generator(input_json, input_h5, train=False,
                             feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), batch_size=100)
def representative_dataset():
  [image_features, questions], answers = vqa_gen.__getitem__(0)
  yield [image_features, questions]

converter = tf.lite.TFLiteConverter.from_saved_model(output)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
converter.representative_dataset = representative_dataset
lite_model = converter.convert()

# Save the model.
with open("/content/drive/My Drive/VQA/lstm_vqa_i8.tflite", 'wb') as f:
  f.write(lite_model)

from trainers import Pruned_lstm_cnn_trainer

tf.keras.backend.clear_session()
vqa = Pruned_lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                              train_feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), 
                              valid_feature_object=Feature_extracted_mobilenet_1by1(valid_feature_file))
vqa.load_model(output)
vqa.final_sparsity = 0.9
history = vqa.train_model(output_pruned)

vqa.model.summary()

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_lstm_cnn3.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_pruned)
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
    f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_dy_lstm_cnn.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_pruned)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
    f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_f16_lstm_cnn.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_pruned)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
    f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_lstm_cnn3.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/pruned_lstm_cnn3.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_dy_lstm_cnn.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/pruned_dy_lstm_vqa.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/pruned_f16_lstm_cnn.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/pruned_f16_lstm_vqa.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

from trainers import Soft_lstm_cnn_trainer

tf.keras.backend.clear_session()
vqa = Soft_lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), 
                       valid_feature_object=Feature_extracted_mobilenet_1by1(valid_feature_file))
history = vqa.train_model(output_soft)

from trainers import Soft_lstm_cnn_trainer

tf.keras.backend.clear_session()
vqa = Soft_lstm_cnn_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_mobilenet_1by1(train_feature_file), 
                       valid_feature_object=Feature_extracted_mobilenet_1by1(valid_feature_file))
history = vqa.train_model(output_soft)

from trainers import Lstm_cnn_trainer
from soft_data_generator import VQA_soft_data_generator
from data_generator import VQA_data_generator

# Experimenting with attention trainers to see if can train to a higher accuracy
def non_linear_layer(size, x):
    y_til = tf.keras.layers.Dense(size, activation='tanh')(x)
    g = tf.keras.layers.Dense(size, activation='sigmoid')(x)
    return tf.keras.layers.multiply([y_til, g])

class Attention_trainer_test(Lstm_cnn_trainer):
    output_size = 3000
    dense_hidden_size = 1024
    image_inputs = tf.keras.Input(shape=(3, 3, 1280))

    def create_model(self):
        """
        Creates a VQA model combining an image and question model

        Returns:
            Attention VQA model
        """
        image_features = tf.keras.layers.Reshape((9, 1280))(self.image_inputs)
        #image_features = tf.keras.layers.LayerNormalization(axis=-1)(image_features)
        question_model = self.create_question_processing_model()
        question_dense = non_linear_layer(self.dense_hidden_size, question_model.output)

        question_stack = tf.keras.layers.RepeatVector(9)(question_model.output)
        non_linear_input = tf.keras.layers.concatenate([image_features, question_stack], axis=-1)
        attention_input = non_linear_layer(self.dense_hidden_size, non_linear_input)
        attention_output = tf.keras.layers.Dense(1, use_bias=False)(attention_input)
        attention_output = tf.keras.layers.Reshape((1, 9))(attention_output)
        attention_output = tf.nn.softmax(attention_output, axis=-1)
        attention_image_features = tf.reduce_sum(tf.matmul(attention_output, image_features), axis=-1)

        attention_final_dense = non_linear_layer(self.dense_hidden_size, attention_image_features)
        linked = tf.keras.layers.multiply([attention_final_dense, question_dense])
        next = non_linear_layer(self.dense_hidden_size, linked)
        outputs = tf.keras.layers.Dense(self.output_size, activation="softmax")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size, answer_count=3000)
        self.val_generator = VQA_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size, answer_count=3000)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()

class Soft_attention_trainer_test(Lstm_cnn_trainer):
    output_size = 3000
    dense_hidden_size = 1024
    image_inputs = tf.keras.Input(shape=(3, 3, 1280))

    def create_model(self):
        """
        Creates a VQA model combining an image and question model

        Returns:
            Attention VQA model
        """
        image_features = tf.keras.layers.Reshape((9, 1280))(self.image_inputs)
        #image_features = tf.keras.layers.LayerNormalization(axis=-1)(image_features)
        question_model = self.create_question_processing_model()
        question_dense = non_linear_layer(self.dense_hidden_size, question_model.output)

        question_stack = tf.keras.layers.RepeatVector(9)(question_model.output)
        non_linear_input = tf.keras.layers.concatenate([image_features, question_stack], axis=-1)
        attention_input = non_linear_layer(self.dense_hidden_size, non_linear_input)
        attention_output = tf.keras.layers.Dense(1, use_bias=False)(attention_input)
        attention_output = tf.keras.layers.Reshape((1, 9))(attention_output)
        attention_output = tf.nn.softmax(attention_output, axis=-1)
        attention_image_features = tf.reduce_sum(tf.matmul(attention_output, image_features), axis=-1)

        attention_final_dense = non_linear_layer(self.dense_hidden_size, attention_image_features)
        linked = tf.keras.layers.multiply([attention_final_dense, question_dense])
        next = non_linear_layer(self.dense_hidden_size, linked)
        outputs = tf.keras.layers.Dense(self.output_size, activation="sigmoid")(next)

        return tf.keras.Model(inputs=[self.image_inputs, self.question_inputs], outputs=outputs,
                              name=__class__.__name__ + "_model")

    def train_model(self, save_path):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                           loss="binary_crossentropy",
                           metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)
        history = self.model.fit(x=self.train_generator,
                                 validation_data=self.val_generator,
                                 epochs=self.max_epochs,
                                 callbacks=[callback])

        self.model.save(save_path)
        return history

    def __init__(self, input_json, input_h5, input_glove_npy,
                 train_feature_object,
                 valid_feature_object):
        self.train_generator = VQA_soft_data_generator(
            input_json, input_h5, feature_object=train_feature_object,
            batch_size=self.batch_size)
        self.val_generator = VQA_soft_data_generator(
            input_json, input_h5, train=False, feature_object=valid_feature_object,
            batch_size=self.batch_size)
        self.set_embedding_matrix(input_glove_npy)
        self.model = self.create_model()

from trainers import Attention_trainer

tf.keras.backend.clear_session()
vqa = Attention_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_mobilenet_3by3(train_feature_file_3by3), 
                       valid_feature_object=Feature_extracted_mobilenet_3by3(valid_feature_file_3by3))
history = vqa.train_model(output_attention)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/basic_attention_vqa.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_attention)
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

from trainers import Soft_attention_trainer

tf.keras.backend.clear_session()
vqa = Soft_attention_trainer(input_json, input_h5, input_glove_npy,
                             train_feature_object=Feature_extracted_mobilenet_3by3("/content/drive/My Drive/VQA/latest_train2.npy"), 
                             valid_feature_object=Feature_extracted_mobilenet_3by3("/content/drive/My Drive/VQA/latest_val2.npy"))
history = vqa.train_model(output_soft_attention)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/soft_attention_vqa.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_soft_attention)
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/soft_lstm_vqa.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_soft)
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

from trainers import Full_attention_trainer

tf.keras.backend.clear_session()
vqa = Full_attention_trainer(input_json, input_h5, input_glove_npy,
                       train_feature_object=Feature_extracted_mobilenet_3by3(train_feature_file_3by3), 
                       valid_feature_object=Feature_extracted_mobilenet_3by3(valid_feature_file_3by3))
history = vqa.train_model(output_full, full_checkpoint)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_full)
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_dy.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_full)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_dy.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_dy.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_f16.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(output_full)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
lite_model = converter.convert()

# Save the model.
with open(tflite_location, 'wb') as f:
  f.write(lite_model)

tflite_location = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_f16.tflite"
zipped_file = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_f16.zip"
with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_location)

model = tf.keras.models.load_model(output_full)

from generate_results import store_results_keras_model
store_results_keras_model(model, input_json, input_h5, 
              Feature_extracted_mobilenet_3by3(valid_feature_file_3by3), 14,
              result_full)

from generate_results import store_results_keras_model
model = tf.keras.models.load_model(output_attention)
store_results_keras_model(model, input_json, input_h5, 
              Feature_extracted_mobilenet_3by3(valid_feature_file_3by3), 26,
              result_attention)

from generate_results import store_results

model = "/content/drive/My Drive/VQA/tflite_models/pruned_dy_lstm_cnn.tflite"
result_pruned_dy = "/content/drive/My Drive/VQA/store_results_pruned_dy.json"
store_results(model, input_json, input_h5, 
              Feature_extracted_mobilenet_1by1(valid_feature_file), 26,
              result_pruned_dy)

from generate_results import store_results

model = "/content/drive/My Drive/VQA/tflite_models/lstm_vqa_dy.tflite"
result_pruned_dy = "/content/drive/My Drive/VQA/store_results_dy.json"
store_results(model, input_json, input_h5, 
              Feature_extracted_mobilenet_1by1(valid_feature_file), 26,
              result_pruned_dy)

from generate_results import store_results

model = "/content/drive/My Drive/VQA/tflite_models/full_attention_vqa_dy.tflite"
results = "/content/drive/My Drive/VQA/store_results_full_dy.json"
store_results(model, input_json, input_h5, 
              Feature_extracted_mobilenet_3by3(valid_feature_file_3by3), 14,
              results)

