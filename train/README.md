# Gesture Recognition Magic Wand Training Scripts

Code and README taken from tflite-micro project (see [magic_wand example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/magic_wand/train)) and adapted to work with own data captured with an ESP32 and a MPU6050.

## Introduction

The scripts in this directory can be used to train a TensorFlow model that
classifies gestures based on accelerometer data. The code uses Python 3.7 and
TensorFlow 2.0. The resulting model is less than 20KB in size.

The following document contains instructions on using the scripts to train a
model, and capturing your own training data.

This project was inspired by the [Gesture Recognition Magic Wand](https://github.com/jewang/gesture-demo)
project by Jennifer Wang.

## Training

### Dataset

The same three magic gestures as for the original example were used and data from one person was collected. Additionally some negative data was captured as well.

The sample dataset is included in this repository and can be downloaded at the following link:
<https://github.com/stefan-spiss/MagicWand-TFLite-ESP32-MPU6050/data/data.zip>

### Training in Colab

The following [Google Colaboratory](https://colab.research.google.com)
notebook demonstrates how to train the model. It's the easiest way to get
started:

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/stefan-spiss/MagicWand-TFLite-ESP32-MPU6050/blob/main/train/train_magic_wand_model.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/stefan-spiss/MagicWand-TFLite-ESP32-MPU6050/blob/main/train/train_magic_wand_model.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>

If you'd prefer to run the scripts locally, use the following instructions.

### Running the scripts

Use the following command to install the required dependencies:

```shell
pip install -r requirements.txt
```

There are two ways to train the model:

- Random data split, which mixes different people's data together and randomly
  splits them into training, validation, and test sets
- Person data split, which splits the data by person

#### Random data split

Using a random split results in higher training accuracy than a person split,
but inferior performance on new data.

```shell
$ python data_prepare.py

$ python data_split.py

$ python train.py --model CNN --person false
```

#### Person data split

Using a person data split results in lower training accuracy but better
performance on new data.

```shell
$ python data_prepare.py

$ python data_split_person.py

$ python train.py --model CNN --person true
```

#### Model type

In the `--model` argument, you can provide `CNN` or `LSTM`. The CNN model has a
smaller size and lower latency.

## Collecting new data

To obtain new training data use the Arduino script in the gesture_capture folder in the root of this repository (see [readme.md](https://github.com/stefan-spiss/MagicWand-TFLite-ESP32-MPU6050#readme) in the root folder).

## Edit and run the scripts with new data

Edit the following files to include your new gesture names (replacing
"wing", "ring", and "slope")

- `data_load.py`
- `data_prepare.py`
- `data_split.py`

Edit the following files to include your new person names (replacing "stefan" or adding new persons):

- `data_prepare.py`
- `data_split_person.py`

Finally, run the commands described earlier to train a new model.
