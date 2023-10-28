# Lint as: python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=redefined-outer-name
# pylint: disable=g-bad-import-order
"""Build and train neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
from data_load import DataLoader

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 3, 1])
  return reshaped_data, label


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def build_cnn(seq_length):
  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          activation="relu",
          input_shape=(seq_length, 3, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                             activation="relu"),  # (batch, 42, 1, 16)
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      tf.keras.layers.Flatten(),  # (batch, 224)
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(4, activation="softmax")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "CNN")
  print("Built CNN.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if os.path.exists(os.path.join(model_path, 'weights.h5')):
      print('loading previous weights')
      model.load_weights("./netmodels/CNN/weights.h5")
  return model, model_path


def build_lstm(seq_length):
  """Builds an LSTM in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(22),
          input_shape=(seq_length, 3)),  # output_shape=(batch, 44)
      tf.keras.layers.Dense(4, activation="sigmoid")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "LSTM")
  print("Built LSTM.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return model, model_path


def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
  data_loader = DataLoader(train_data_path,
                           valid_data_path,
                           test_data_path,
                           seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
      data_loader.valid_data, data_loader.test_len, data_loader.test_data


def build_net(args, seq_length):
  if args.model == "CNN":
    model, model_path = build_cnn(seq_length)
  elif args.model == "LSTM":
    model, model_path = build_lstm(seq_length)
  else:
    print("Please input correct model name.(CNN  LSTM)")
  return model, model_path

def eval_net(model, test_data, test_labels):
  print("********** Evaluation test data **********")
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(labels=tf.constant(test_labels),
                                       predictions=tf.constant(pred),
                                       num_classes=4)
  print(confusion)
  print("Loss {}, Accuracy {}".format(loss, acc))

def prepare_and_save_tflite_nets(model, filename, filename_quantized):
  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model to disk
  open(filename, "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()

  # Save the model to disk
  open(filename_quantized, "wb").write(tflite_model)

  basic_model_size = os.path.getsize("model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize("model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)

def train_net(
    model,
    model_path,
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind,
    weights_metric="val_accuracy"):
  """Trains the model."""
  calculate_model_size(model)
  epochs = 50
  batch_size = 64
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for _, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  chckpt_path = os.path.join(model_path, 'weights_best_' + weights_metric + '.ckpt')

  mode = "min"
  if weights_metric == "val_accuracy" or weights_metric == "accuracy":
      mode = "max"
  chckpt_fct = ModelCheckpoint(chckpt_path, monitor=weights_metric, save_best_only=True, mode=mode, save_weights_only=True)
  model.fit(train_data,
            epochs=epochs,
            validation_data=valid_data,
            steps_per_epoch=1000,
            validation_steps=int((valid_len - 1) / batch_size + 1),
            callbacks=[tensorboard_callback, chckpt_fct])
  idx = 0
  print("*********************************************************************************")
  print("Model after last epoch:")
  eval_net(model, test_data, test_labels)
  model.save(os.path.join(model_path, 'weights.h5'))
  prepare_and_save_tflite_nets(model, "model.tflite", "model_quantized.tflite")

  print("*********************************************************************************")
  print("Model of epoch with best " + weights_metric + ":")
  model.load_weights(chckpt_path)
  eval_net(model, test_data, test_labels)
  model.save(os.path.join(model_path, 'weights_best_' + weights_metric + '.h5'))
  prepare_and_save_tflite_nets(model, "model_best_" + weights_metric + ".tflite", "model_quantized_best_" + weights_metric + ".tflite")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m")
  parser.add_argument("--person", "-p")
  parser.add_argument("--weights_metric", default="val_accuracy", choices=["val_accuracy", "val_loss", "accuracy", "loss"])
  args = parser.parse_args()

  seq_length = 128

  print("Start to load data...")
  if args.person == "true":
    train_len, train_data, valid_len, valid_data, test_len, test_data = \
        load_data("./person_split/train", "./person_split/valid",
                  "./person_split/test", seq_length)
  else:
    train_len, train_data, valid_len, valid_data, test_len, test_data = \
        load_data("./data/train", "./data/valid", "./data/test", seq_length)

  print("Start to build net...")
  model, model_path = build_net(args, seq_length)

  print("Start training...")
  train_net(model, model_path, train_data, valid_len, valid_data,
            test_len, test_data, args.model, args.weights_metric)

  print("Training finished!")
