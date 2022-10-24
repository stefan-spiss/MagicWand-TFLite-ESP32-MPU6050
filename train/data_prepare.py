# Lint as: python3
# coding=utf-8
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
"""Prepare data for further process.

Read data from "/slope", "/ring", "/wing", "/negative" and save them
in "/data/complete_data" in python dict format.

It will generate a new file with the following structure:
├── data
│   └── complete_data
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import json
import os
import random
import math

LABEL_NAME = "gesture"
DATA_NAME = "accel_ms2_xyz"
folders = ["wing", "ring", "slope"]
names = [
    "stefan",
    "patrick",
    "justin",
    "yeongmi",
    "arthur",
    "kai",
    "nico",
    "filip",
    "lukas",
    "peter"
]


def prepare_original_data(folder, name, data, file_to_read):  # pylint: disable=redefined-outer-name
  """Read collected data from files."""
  if os.path.isfile(file_to_read):
    with open(file_to_read, "r") as f:
      lines = csv.reader(f)
      data_new = {}
      data_new[LABEL_NAME] = folder
      data_new[DATA_NAME] = []
      data_new["name"] = name
      for idx, line in enumerate(lines):  # pylint: disable=unused-variable,redefined-outer-name
        if len(line) == 3:
          if line[2] == "-" and data_new[DATA_NAME] or (folder == "negative" and len(data_new[DATA_NAME]) == 120):
            data.append(data_new)
            data_new = {}
            data_new[LABEL_NAME] = folder
            data_new[DATA_NAME] = []
            data_new["name"] = name
          elif line[2] != "-":
            data_new[DATA_NAME].append([float(i) for i in line[0:3]])
      data.append(data_new)

def generate_negative_data(data, number_samples):  # pylint: disable=redefined-outer-name
  """Generate negative data labeled as 'negative6~8'."""
  # Big movement -> around straight line
  for i in range(number_samples//3):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 30
    start_y = (random.random() - 0.5) * 30
    start_z = (random.random() - 0.5) * 30
    x_increase = (random.random() - 0.5)
    y_increase = (random.random() - 0.5)
    z_increase = (random.random() - 0.5)
    for j in range(128):
      dic[DATA_NAME].append([
          start_x + j * x_increase + (random.random() - 0.5) * 0.6,
          start_y + j * y_increase + (random.random() - 0.5) * 0.6,
          start_z + j * z_increase + (random.random() - 0.5) * 0.6
      ])
    data.append(dic)
  # Random
  for i in range(number_samples//3):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    for j in range(128):
      dic[DATA_NAME].append([(random.random() - 0.5) * 50,
                             (random.random() - 0.5) * 50,
                             (random.random() - 0.5) * 50])
    data.append(dic)
  # Stay still
  for i in range(number_samples//3):
    if i > 80:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative8"}
    elif i > 60:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative7"}
    else:
      dic = {DATA_NAME: [], LABEL_NAME: "negative", "name": "negative6"}
    start_x = (random.random() - 0.5) * 2
    start_y = (random.random() - 0.5) * 2
    start_z = (random.random() - 0.5) * 2
    for j in range(128):
      dic[DATA_NAME].append([
          start_x + (random.random() - 0.5) * 0.4,
          start_y + (random.random() - 0.5) * 0.4,
          start_z + (random.random() - 0.5) * 0.4
      ])
    data.append(dic)


# Write data to file
def write_data(data_to_write, path):
  with open(path, "w") as f:
    for idx, item in enumerate(data_to_write):  # pylint: disable=unused-variable,redefined-outer-name
      dic = json.dumps(item, ensure_ascii=False)
      f.write(dic)
      f.write("\n")


if __name__ == "__main__":
  data = []  # pylint: disable=redefined-outer-name
  for idx1, folder in enumerate(folders):
    for idx2, name in enumerate(names):
      prepare_original_data(folder, name, data,
                            "./data/%s/output_%s_%s.txt" % (folder, folder, name))
  n_gestures = len(data)
  for idx, name in enumerate(names):
    prepare_original_data("negative", name, data,
                          "./data/negative/output_negative_%s.txt" % (name))
  n_negative = len(data) - n_gestures

  # if there are more than 10% more gesture samples per gesture as negative samples, generate additional negative
  # samples
  if n_gestures - n_negative * len(folders) > n_gestures/len(folders) * 0.1:
    print("not enough negative samples available, creating random data samples")
    generate_negative_data(data, (math.ceil(n_gestures/len(folders)) - n_negative))
  n_negative = len(data) - n_gestures

  print("gesture_data_length: " + str(n_gestures))
  print("negative_data_length: " + str(n_negative))
  print("data_length: " + str(len(data)))

  # if not os.path.exists("./data"):
  #   os.makedirs("./data")
  write_data(data, "./data/complete_data")
